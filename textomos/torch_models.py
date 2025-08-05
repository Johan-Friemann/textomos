import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """See pytorch module documentation for specifics of forward type method
       that is required by the nn.Module interface.

    A 3D UNet double convolution layer. 

    Params:
        in_channels (int): Number of channels in the input data.

        out_channels (int): Number of channels in the output data.

    Keyword params:
        -
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    """See pytorch module documentation for specifics of forward type method
       that is required by the nn.Module interface.

    A 3D UNet. 

    Params:
        in_channels (int): Number of channels in the input data.

        out_channels (int): Number of channels in the output data.

    Keyword params:
        features (list[int]): A list with the number of features per UNet layer.
                              Each entry correspodns to a layar in both the
                              encoder and the decoder.
    """
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder_layers = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for f in features:
            self.encoder_layers.append(DoubleConv(prev_channels, f))
            prev_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(prev_channels, prev_channels * 2)
        decoder_channels = prev_channels * 2

        # Decoder
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose3d(decoder_channels, f, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(DoubleConv(f * 2, f))
            decoder_channels = f

        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        # Encoder path
        for enc in self.encoder_layers:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = enc_features[-(idx + 1)]

            # Pad if necessary (for odd input dimensions)
            if x.shape != skip.shape:
                diffZ = skip.shape[2] - x.shape[2]
                diffY = skip.shape[3] - x.shape[3]
                diffX = skip.shape[4] - x.shape[4]
                x = F.pad(
                    x,
                    [
                        diffX // 2,
                        diffX - diffX // 2,
                        diffY // 2,
                        diffY - diffY // 2,
                        diffZ // 2,
                        diffZ - diffZ // 2,
                    ],
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_layers[idx](x)

        return self.final_conv(x)
