import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
       Encoder (Downsampling) Block for the contracting path of U-Net.

       This module performs spatial downsampling followed by feature extraction:
           MaxPool2d (2x2) -> DoubleConv

       The max pooling operation halves the spatial dimensions while the subsequent
       double convolution increases the feature channel depth. This progressive
       compression captures increasingly abstract and high-level features.
       """

    def __init__(self, in_channels, out_channels):
        """
               Initializes the encoder block.

               :param in_channels: Number of input feature channels.
               :param out_channels: Number of output feature channels (typically 2x input).
               """

        super().__init__()

        # Define the downsampling pipeline: pooling followed by convolutions
        self.maxpool_conv = nn.Sequential(
            # 2x2 max pooling reduces spatial dimensions by half
            nn.MaxPool2d(kernel_size=2),
            # Double convolution for feature extraction
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        """
                Forward pass through the encoder block.

                :param x: Input tensor of shape (batch, in_channels, H, W).
                :return: Output tensor of shape (batch, out_channels, H/2, W/2).
                """

        return self.maxpool_conv(x)


class Up(nn.Module):
    """
       Decoder (Upsampling) Block for the expansive path of U-Net.

       This module performs spatial upsampling and feature fusion:
           Upsample (2x) -> Concatenate(skip) -> DoubleConv

       The upsampling restores spatial resolution while concatenation with the
       corresponding encoder features (skip connection) reintroduces fine-grained
       spatial information that was lost during downsampling. This combination
       enables precise localization in the reconstruction.
       """

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # Choose upsampling method based on bilinear flag
        if bilinear:
            # Bilinear interpolation: faster and uses less memory
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation with skip, channels = in_channels + in_channels/2
            # We use mid_channels to gradually reduce to out_channels
        else:
            # Transposed convolution: learnable upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if bilinear:
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):

        """
               Forward pass through the decoder block with skip connection.

               :param x1: Input tensor from previous decoder layer (lower resolution).
               :param x2: Skip connection tensor from corresponding encoder (higher resolution).
               :return: Upsampled and fused output tensor.
               """

        # Upsample the input from the previous layer
        x = self.up(x)

        # Handle potential size mismatch due to odd dimensions
        # Calculate the difference in height and width
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        # Pad x1 to match x2's dimensions if necessary
        # Padding order: (left, right, top, bottom)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        # Concatenate along the channel dimension
        x = torch.cat([skip, x], dim=1)
        # Apply double convolution to fuse the features
        return self.conv(x)


class OutConv(nn.Module):
    """
       Output Convolution Layer for final prediction.

       This module applies a 1x1 convolution to map the final feature channels
       to the desired number of output channels. For SST reconstruction, this
       produces a single-channel output representing the reconstructed temperature.
       """

    def __init__(self, in_channels, out_channels):
        """
                Initializes the output convolution layer.

                :param in_channels: Number of input feature channels.
                :param out_channels: Number of output channels (1 for SST reconstruction).
                """

        super().__init__()

        # 1x1 convolution for channel-wise projection
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
               Forward pass through the output layer.

               :param x: Input tensor of shape (batch, in_channels, H, W).
               :return: Output tensor of shape (batch, out_channels, H, W).
               """

        return self.conv(x)


class UNet(nn.Module):
    """
       The U-Net consists of a symmetric encoder-decoder structure:
           - Encoder (Contracting Path): Captures context through progressive downsampling
           - Bottleneck: Maximum feature compression with dropout for regularization
           - Decoder (Expansive Path): Enables precise localization through upsampling
           - Skip Connections: Bridge encoder and decoder to preserve spatial details
           """

    def __init__(self, n_channels, n_classes=1,
                 filters=None, bilinear=True, dropout_rate=0.1):
        """
               Initializes the U-Net architecture.

               :param n_channels: Number of input channels (default: 2 for SST+mask).
               :param n_classes: Number of output channels (default: 1 for SST).
               :param filters: List of filter counts for each encoder level.
                               Default: [64, 128, 256, 512].
               :param bilinear: If True, uses bilinear upsampling in decoder.
               :param dropout_rate: Dropout probability in the bottleneck layer.
               """

        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        # Factor to adjust channels when using bilinear upsampling
        # Bilinear mode doesn't halve channels in upsampling, so we compensate
        factor = 2 if bilinear else 1

        # Initial double convolution (no downsampling)
        # Maps input channels to first filter count while preserving spatial dimensions
        self.inc = DoubleConv(n_channels, filters[0])

        # Encoder blocks with progressive downsampling
        # Each block: MaxPool(2x2) -> DoubleConv, doubles channels, halves resolution
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3] // factor)

        # =====================================================================
        # BOTTLENECK
        # =====================================================================

        # Dropout layer for regularization at the bottleneck
        # Prevents overfitting by randomly zeroing features during training
        self.dropout = nn.Dropout(p=dropout_rate)

        # =====================================================================
        # DECODER (Expansive Path)
        # =====================================================================

        # Decoder blocks with progressive upsampling
        # Each block: Upsample(2x) -> Concat(skip) -> DoubleConv
        self.up1 = Up(filters[3] // factor, filters[2], filters[2] // factor, bilinear)
        self.up2 = Up(filters[2] // factor, filters[1], filters[1] // factor, bilinear)
        self.up3 = Up(filters[1] // factor, filters[0], filters[0], bilinear)

        # =====================================================================
        # OUTPUT LAYER
        # =====================================================================

        # Final 1x1 convolution to map to output channels
        self.outc = OutConv(filters[0], n_classes)

    def forward(self, x):
        """
                Forward pass through the complete U-Net.

                The input tensor flows through:
                1. Initial convolution (feature extraction)
                2. Encoder path (progressive downsampling with feature learning)
                3. Bottleneck (maximum compression with dropout)
                4. Decoder path (upsampling with skip connection fusion)
                5. Output convolution (final reconstruction)

                :param x: Input tensor of shape (batch, n_channels, H, W).
                          For SST reconstruction: (batch, 2, 256, 256).
                :return: Output tensor of shape (batch, n_classes, H, W).
                         For SST reconstruction: (batch, 1, 256, 256).
                """

        # =====================================================================
        # ENCODER PATH
        # =====================================================================

        # Initial convolution: extract low-level features
        x1 = self.inc(x)

        # Encoder block 1: first downsampling
        x2 = self.down1(x1)

        # Encoder block 2: second downsampling
        x3 = self.down2(x2)

        # Encoder block 3: third downsampling (to bottleneck)
        x4 = self.down3(x3)

        # =====================================================================
        # BOTTLENECK
        # =====================================================================

        # Apply dropout for regularization
        x4 = self.dropout(x4)

        # =====================================================================
        # DECODER PATH WITH SKIP CONNECTIONS
        # =====================================================================

        # Decoder block 1: upsample and fuse with x3
        x = self.up1(x4, x3)

        # Decoder block 2: upsample and fuse with x2
        x = self.up2(x, x2)

        # Decoder block 3: upsample and fuse with x1
        x = self.up3(x, x1)

        # =====================================================================
        # OUTPUT
        # =====================================================================

        # Final convolution to produce reconstruction
        logits = self.outc(x)
        return logits

    def count_parameters(self):
        """
            Counts the total number of trainable parameters in a model.

            Useful for understanding model complexity and comparing architectures.

            :param model: PyTorch model instance.
            :return: Integer count of trainable parameters.
            """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        """
           Prints a summary of the model architecture.

           :param model: PyTorch model instance.
           :param input_size: Tuple specifying input dimensions (C, H, W).
           """
        print("=" * 60)
        print("U-Net Model Summary")
        print("=" * 60)
        print(f"Input channels:  {self.n_channels}")
        print(f"Output channels: {self.n_classes}")
        print(f"Total trainable parameters: {self.count_parameters():,}")
        print("=" * 60)


if __name__ == '__main__':
    model = UNet(n_channels=26, n_classes=1, bilinear=True)
    model.print_model_summary()
    dummy = torch.randn(2, 26, 53, 97)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
