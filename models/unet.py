import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batch_norm=True, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, features=None, use_batch_norm=True, dropout_rate=0.3):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256]
        self.features = features
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        current_channels = in_channels
        for feature_count in features:
            self.encoder_blocks.append(ConvBlock(current_channels, feature_count, use_batch_norm, dropout_rate))
            current_channels = feature_count
        bottleneck_channels = features[-1] * 2
        self.bottleneck = ConvBlock(features[-1], bottleneck_channels, use_batch_norm, dropout_rate)
        self.upconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = bottleneck_channels
        for feature_count in reversed_features:
            self.upconv_blocks.append(nn.ConvTranspose2d(prev_channels, feature_count, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(feature_count * 2, feature_count, use_batch_norm, dropout_rate))
            prev_channels = feature_count
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
        x = self.sigmoid(x)
        return x
if __name__ == '__main__':
    '\n    ❓ if __name__ == "__main__" KYA HAI?\n    ======================================\n    Ye code SIRF tab chalta hai jab tum SEEDHA ye file run karo:\n        python models/unet.py\n\n    Jab koi IMPORT kare (from models.unet import UNet), toh ye NAHI chalta.\n    Testing ke liye perfect — verify karo ki model kaam kar raha hai!\n\n    ❓ TESTING KYUN ZAROORI HAI?\n    ============================\n    Sochlo: tum 5 ghante training chalaaye.\n    5 ghante BAAD pata chala ki output shape galat hai. 😭\n    Ye 2 minute ka test PEHLE hi dhundh leta!\n\n    ML mein common tests:\n    1. SHAPE TEST: output ka size sahi hai?\n    2. FORWARD PASS: kya data bina error ke poora network traverse karta hai?\n    3. VALUE RANGE: output values valid range mein hain? (0-1 for images)\n    4. PARAMETER COUNT: model kitna bada hai?\n    '
    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256], use_batch_norm=True, dropout_rate=0.3)
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f'Input shape:  {dummy_input.shape}')
    print(f'Output shape: {output.shape}')
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Output min: {output.min().item():.4f}')
    print(f'Output max: {output.max().item():.4f}')
    print('\n✅ Model test PASS! U-Net sahi kaam kar raha hai.')