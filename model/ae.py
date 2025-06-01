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


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = config.vocabSize

        for i in range(config.depth):
            next_channels = config.nchannels * 2 ** i
            padding = (config.kernelSize - 1) // 2  # 'same' padding
            #set_trace()
            conv = nn.Conv1d(current_channels, next_channels, kernel_size=config.kernelSize, stride=config.stride, padding=padding)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            relu = nn.ReLU()

            if i == config.depth - 1:
                self.layers.append(nn.Sequential(conv, relu))
            else:
                pool = nn.MaxPool1d(kernel_size=2, stride=2)
                drop = nn.Dropout(config.dropoutRate)
                self.layers.append(nn.Sequential(conv, relu, pool, drop))

            current_channels = next_channels

    def forward(self, x):
        for layer in self.layers:
            #set_trace()
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = config.nchannels * 2 ** (config.depth - 1)

        for i in range(config.depth - 1, -1, -1):
            next_channels = config.nchannels * 2 ** i if i > 0 else config.vocabSize
            padding = (config.kernelSize - 1) // 2  # 'same' padding
            conv = nn.Conv1d(current_channels, next_channels, kernel_size=config.kernelSize, stride=config.stride, padding=padding)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            relu = nn.ReLU()

            if i == 0:
                self.layers.append(nn.Sequential(conv, relu))
            else:
                up = nn.Upsample(scale_factor=2)
                drop = nn.Dropout(config.dropoutRate)
                self.layers.append(nn.Sequential(conv, relu, up, drop))

            current_channels = next_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SCDA(nn.Module):
    def __init__(self, config):
        super(SCDA, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose to match Conv1d input requirements
        latent = self.encoder(x)
        #set_trace()
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.transpose(1, 2)  # Transpose back to original shape
        return reconstructed, latent
