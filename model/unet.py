import torch
import torch.nn as nn
import torch.nn.functional as F


def print_model_summary(model):
    print("Model Summary:")
    total_params = 0

    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name:40} : Params: {param_count}")

    print(f"\nTotal Parameters: {total_params}\n")


def _make_norm(args, num_channels):
    """Optional normalization layer; controlled by config.useGroupNorm."""
    if getattr(args, 'useGroupNorm', False):
        gn_groups = getattr(args, 'gnGroups', 8)
        if num_channels % gn_groups != 0:
            gn_groups = min(gn_groups, num_channels)
            while num_channels % gn_groups != 0 and gn_groups > 1:
                gn_groups -= 1
        return nn.GroupNorm(num_groups=max(1, gn_groups), num_channels=num_channels)
    return nn.Identity()


class FiLMModulator(nn.Module):
    """Per-position feature-wise linear modulation.

    Takes a bio prior (B, L_bio, C_bio) and produces per-channel γ,β that
    modulate a conv feature map (B, C_out, L_feat). Bio is average-pooled
    along the position axis to match L_feat. The MLP's last layer is zero-
    initialized so γ≈1, β≈0 at the start of training (FiLM acts as identity).
    """
    def __init__(self, bio_channels, out_channels, hidden=64):
        super().__init__()
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(bio_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * out_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat, bio):
        # feat: (B, C, L_feat); bio: (B, L_bio, C_bio)
        L_feat = feat.shape[-1]
        bio_t = bio.transpose(1, 2)  # (B, C_bio, L_bio)
        if bio_t.shape[-1] != L_feat:
            bio_t = F.adaptive_avg_pool1d(bio_t, L_feat)
        bio_pos = bio_t.transpose(1, 2)  # (B, L_feat, C_bio)
        gb = self.mlp(bio_pos)  # (B, L_feat, 2*C)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = (gamma + 1.0).transpose(1, 2)  # (B, C, L_feat)
        beta = beta.transpose(1, 2)
        return gamma * feat + beta


class CodebookAttention(nn.Module):
    """Cross-attention from the bottleneck latent to a learned codebook of K prototypes.

    The codebook is a learned (K, D) matrix; the latent (B, C, L_bot) attends to it
    via multi-head attention, and the attended summary is added back to the latent
    as a residual. Codebook entries learn end-to-end to summarize the training
    haplotype/segment distribution — a differentiable, panel-like memory.
    """
    def __init__(self, dim, codebook_size=256, num_heads=4, dropout=0.0):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # Zero-init output projection so the attention starts as a no-op residual
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, latent):
        # latent: (B, C, L_bot)
        B = latent.shape[0]
        q = latent.transpose(1, 2)  # (B, L_bot, C)
        kv = self.codebook.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return (q + attn_out).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.use_film = bool(getattr(args, 'useFiLM', False)) and bool(getattr(args, 'bioAware', False))
        bio_channels = getattr(args, 'bioChannels', 1) if getattr(args, 'bioAware', False) else 0

        # When FiLM is enabled, bio is fed via FiLM modulation rather than channel concat
        if getattr(args, 'bioAware', False) and not self.use_film:
            current_channels = args.vocabSize + bio_channels
        else:
            current_channels = args.vocabSize

        for i in range(args.depth):
            next_channels = args.nchannels * 2 ** i
            padding = (args.kernelSize - 1) // 2

            conv = nn.Conv1d(current_channels, next_channels,
                           kernel_size=args.kernelSize,
                           stride=args.stride,
                           padding=padding)

            block = nn.ModuleDict({
                'conv': conv,
                'norm': _make_norm(args, next_channels),
            })
            if self.use_film:
                block['film'] = FiLMModulator(bio_channels, next_channels)
            if i != args.depth - 1:
                block['pool'] = nn.MaxPool1d(kernel_size=2, stride=2)
                block['drop'] = nn.Dropout(args.dropoutRate)
            self.layers.append(block)

            current_channels = next_channels

    def forward(self, x, bio=None):
        skip_connections = []
        for layer in self.layers[:-1]:
            h = layer['norm'](layer['conv'](x))
            if self.use_film and bio is not None:
                h = layer['film'](h, bio)
            h = F.relu(h)
            skip_connections.append(h)
            x = layer['drop'](layer['pool'](h))

        last = self.layers[-1]
        h = last['norm'](last['conv'](x))
        if self.use_film and bio is not None:
            h = last['film'](h, bio)
        h = F.relu(h)
        return h, skip_connections


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.use_film = bool(getattr(args, 'useFiLM', False)) and bool(getattr(args, 'bioAware', False))
        bio_channels = getattr(args, 'bioChannels', 1) if getattr(args, 'bioAware', False) else 0
        current_channels = args.nchannels * 2 ** (args.depth - 1)
        self.output_channels = 2 if getattr(args, 'useHaploidHeads', False) else args.vocabSize

        for i in range(args.depth - 1, -1, -1):
            next_channels = args.nchannels * 2 ** max(i-1, 0) if i > 0 else self.output_channels
            padding = (args.kernelSize - 1) // 2

            if i == 0:  # Final layer (no norm/film on output logits)
                self.layers.append(nn.ModuleDict({
                    'conv': nn.Conv1d(current_channels, next_channels,
                                    kernel_size=args.kernelSize,
                                    stride=args.stride,
                                    padding=padding)
                }))
            else:
                block = nn.ModuleDict({
                    'up': nn.Upsample(scale_factor=2, mode='nearest'),
                    'conv': nn.Conv1d(current_channels, next_channels,
                                    kernel_size=args.kernelSize,
                                    stride=args.stride,
                                    padding=padding),
                    'norm': _make_norm(args, next_channels),
                    'drop': nn.Dropout(args.dropoutRate)
                })
                if self.use_film:
                    block['film'] = FiLMModulator(bio_channels, next_channels)
                self.layers.append(block)
            current_channels = next_channels

    def forward(self, x, skip_connections, bio=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer['up'](x)
            h = layer['norm'](layer['conv'](x))
            if self.use_film and bio is not None:
                h = layer['film'](h, bio)
            h = F.relu(h)
            x = layer['drop'](h + skip_connections[-(i+1)])

        x = self.layers[-1]['conv'](x)
        return x


class BiUNet(nn.Module):
    def __init__(self, args):
        super(BiUNet, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.mask_value = 0
        self.useHaploidHeads = getattr(args, 'useHaploidHeads', False)
        self.use_film = bool(getattr(args, 'useFiLM', False)) and bool(getattr(args, 'bioAware', False))
        self.use_codebook = bool(getattr(args, 'useCodebook', False))
        if self.use_codebook:
            dim = args.nchannels * 2 ** (args.depth - 1)  # bottleneck channels
            self.codebook_attn = CodebookAttention(
                dim=dim,
                codebook_size=getattr(args, 'codebookSize', 256),
                num_heads=getattr(args, 'codebookNumHeads', 4),
                dropout=getattr(args, 'codebookDropout', 0.0),
            )

    def forward(self, x, bio=None):
        # x: (B, L, C_in). When use_film is True, C_in = vocabSize and bio is passed separately.
        # When False (default), bio (if present) is already concatenated by train.py/test.py.
        x = x.transpose(1, 2)
        latent, skip_connections = self.encoder(x, bio=bio if self.use_film else None)
        if self.use_codebook:
            latent = self.codebook_attn(latent)
        reconstructed = self.decoder(latent, skip_connections, bio=bio if self.use_film else None)
        if not self.useHaploidHeads:
            reconstructed[:, self.mask_value, :] = float('-inf')
        return reconstructed.transpose(1, 2), latent
