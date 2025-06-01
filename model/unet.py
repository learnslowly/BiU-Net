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


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        # Determine initial channels based on bioAware flag
        if hasattr(args, 'bioAware') and args.bioAware:
            current_channels = args.vocabSize + 1  # Add +1 for cM channel
        else:
            current_channels = args.vocabSize

        for i in range(args.depth):
            next_channels = args.nchannels * 2 ** i
            padding = (args.kernelSize - 1) // 2

            conv = nn.Conv1d(current_channels, next_channels,
                           kernel_size=args.kernelSize,
                           stride=args.stride,
                           padding=padding)

            if i != args.depth - 1:
                self.layers.append(nn.ModuleDict({
                    'conv': conv,
                    'pool': nn.MaxPool1d(kernel_size=2, stride=2),
                    'drop': nn.Dropout(args.dropoutRate)
                }))
            else:
                self.layers.append(nn.ModuleDict({'conv': conv}))

            current_channels = next_channels

    def forward(self, x):

        skip_connections = []
        for layer in self.layers[:-1]:  # All but last
            x = F.relu(layer['conv'](x))
            skip_connections.append(x)
            x = layer['drop'](layer['pool'](x))

        x = F.relu(self.layers[-1]['conv'](x))  # Final layer
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = args.nchannels * 2 ** (args.depth - 1)

        for i in range(args.depth - 1, -1, -1):
            next_channels = args.nchannels * 2 ** max(i-1, 0) if i > 0 else args.vocabSize
            padding = (args.kernelSize - 1) // 2

            if i == 0:  # Final layer
                self.layers.append(nn.ModuleDict({
                    'conv': nn.Conv1d(current_channels, next_channels,
                                    kernel_size=args.kernelSize,
                                    stride=args.stride,
                                    padding=padding)
                }))
            else:
                self.layers.append(nn.ModuleDict({
                    'up': nn.Upsample(scale_factor=2, mode='nearest'),
                    'conv': nn.Conv1d(current_channels, next_channels,
                                    kernel_size=args.kernelSize,
                                    stride=args.stride,
                                    padding=padding),
                    'drop': nn.Dropout(args.dropoutRate)
                }))
            current_channels = next_channels

    def forward(self, x, skip_connections):
        for i, layer in enumerate(self.layers[:-1]):  # All but last
            x = layer['up'](x)
            x = F.relu(layer['conv'](x))
            x = layer['drop'](x + skip_connections[-(i+1)])

        x = self.layers[-1]['conv'](x)  # Final layer
        return x

class BiUNet(nn.Module):
    def __init__(self, args):
        super(BiUNet, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.mask_value = 0  # Define the mask value

    def forward(self, x):
        x = x.transpose(1, 2)
        latent, skip_connections = self.encoder(x)
        reconstructed = self.decoder(latent, skip_connections)
        reconstructed[:, self.mask_value, :] = float('-inf') # (batch, channel, segLen), set the mask channel to inf
        return reconstructed.transpose(1, 2), latent
        
