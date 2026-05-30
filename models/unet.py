import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batch_norm=True, dropout_rate=0.0):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.block = nn.Sequential(*layers)
        self.attention = ChannelAttention(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = self.attention(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = out + identity

        return out


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, features=None,
                 use_batch_norm=True, dropout_rate=0.3):
        super(UNet, self).__init__()

        if features is None:
            features = [64, 128, 256]

        self.features = features

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature_count in features:
            self.encoder_blocks.append(
                ConvBlock(current_channels, feature_count, use_batch_norm, dropout_rate)
            )
            current_channels = feature_count

        bottleneck_channels = features[-1] * 2
        self.bottleneck = ConvBlock(
            features[-1], bottleneck_channels, use_batch_norm, dropout_rate
        )

        self.upconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_features = list(reversed(features))

        prev_channels = bottleneck_channels
        for feature_count in reversed_features:
            self.upconv_blocks.append(
                nn.ConvTranspose2d(
                    prev_channels, feature_count,
                    kernel_size=2, stride=2
                )
            )
            self.decoder_blocks.append(
                ConvBlock(feature_count * 2, feature_count, use_batch_norm, dropout_rate)
            )
            prev_channels = feature_count

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_image = x

        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x)
            skip = skip_connections[i]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)

        x = self.final_conv(x)
        x = x + input_image
        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    model = UNet(
        in_channels=3, out_channels=3, features=[64, 128, 256],
        use_batch_norm=True, dropout_rate=0.3
    )

    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)

    print(f'Input shape:  {dummy_input.shape}')
    print(f'Output shape: {output.shape}')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    print(f'Output min: {output.min().item():.4f}')
    print(f'Output max: {output.max().item():.4f}')

    print('\n✅ Model test PASS! U-Net sahi kaam kar raha hai.')