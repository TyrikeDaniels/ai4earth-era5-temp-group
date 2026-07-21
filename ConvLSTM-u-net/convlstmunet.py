"""ConvLSTM cell and stacked ConvLSTM model for spatiotemporal land-use prediction.

Reference:
    Shi et al. 2015, "Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting", NeurIPS.

The encoder ingests a sequence of multi-channel rasters (B, T, C, H, W) and the
decoder head predicts a per-pixel class probability map for the next-year frame.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch                    # pyright: ignore[reportMissingImports]
import torch.nn as nn           # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingImports]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with 4 gates fused ito a single convolution operation."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ConvLSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            h_prev (torch.Tensor): Previous hidden state of shape (B, C_hidden, H, W).
            c_prev (torch.Tensor): Previous cell state of shape (B, C_hidden, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Current hidden state and cell state.
        """
        combined = torch.cat([x, h_prev], dim=1)  # Concatenate along channel axis
        gates = self.conv(combined)
        
        # Split the gates into their respective components
        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_channels, dim=1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)  # Input gate
        f_gate = torch.sigmoid(f_gate)  # Forget gate
        o_gate = torch.sigmoid(o_gate)  # Output gate
        g_gate = torch.tanh(g_gate)     # Cell candidate

        # Update cell state and hidden state
        c_current = f_gate * c_prev + i_gate * g_gate
        h_current = o_gate * torch.tanh(c_current)

        return h_current, c_current

    def init_state(self, batch_size: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the hidden and cell states to zeros.

        Args:
            batch_size (int): Batch size.
            height (int): Height of the input tensor.
            width (int): Width of the input tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized hidden and cell states.
        """
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM model for spatiotemporal land-use prediction."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias

        # Create a list of ConvLSTM cells for each layer
        self.cells = nn.ModuleList()
        for i in range(len(hidden_channels)):
            in_channels = input_channels if i == 0 else hidden_channels[i - 1]
            self.cells.append(ConvLSTMCell(in_channels, hidden_channels[i], kernel_size, bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the stacked ConvLSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C_out, H, W).
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden and cell states for each layer
        h_states = []
        c_states = []
        for cell in self.cells:
            h, c = cell.init_state(batch_size, height, width)
            h_states.append(h)
            c_states.append(c)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # Get the t-th time step input
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = h_states[i], c_states[i]
                h_current, c_current = cell(x_t, h_prev, c_prev)
                h_states[i], c_states[i] = h_current, c_current
                x_t = h_current  # The output of the current layer is the input to the next layer
            outputs.append(h_current.unsqueeze(1))  # Collect the output of the last layer

        return torch.cat(outputs, dim=1)  # Concatenate along the time dimension

class UNetConvLSTM(nn.Module):
    """UNet architecture with ConvLSTM layers for spatiotemporal land-use prediction."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        # Spatial encoder (applied to each timestep independently)
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], kernel_size, bias)     # 16 channels ; handles 52 x 90 spatial resolution
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], kernel_size, bias) # 32 channels ; handles 26 x 45 spatial resolution
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], kernel_size, bias) # 64 channels ; handles 13 x 23 spatial resolution
        self.pool = nn.MaxPool2d(2)

        # Temporal ConvLSTM encoder/bottleneck (applied across timesteps)
        self.temporal1 = ConvLSTMCell(hidden_channels[0], hidden_channels[0]) # Retain 16 channels
        self.temporal2 = ConvLSTMCell(hidden_channels[2], hidden_channels[2]) # Retain 64 channels (bottleneck)

        # Decoder (upsampling and concatenation with skip connections)
        self.upconv3 = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)      
        self.dec3 = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2 = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)

        self.head = nn.Conv2d(
            hidden_channels[0],
            output_channels,
            kernel_size=1
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True) -> nn.Sequential:
        """Creates a 2-layerconvolutional block with Conv2d, BatchNorm, and ReLU."""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device

        # Initialize hidden states for ConvLSTM layers at bottleneck and first encoder layer
        state1 = self.temporal1.init_state(B, H, W, device)
        state2 = self.temporal2.init_state(B, H//4, W//4, device)

        # Cache last hidden states (1 & 2) for skip connections
        e1_last = e2_last = None
        for t in range(T):
            e1 = self.enc1(x[:, t])            
            state1 = self.temporal1(e1, state1) 
            e1 = state1[0]
            e1_last = e1 # high-resolution feature map for skip connection

            e2 = self.enc2(self.pool(e1)) 
            e2_last = e2  # low-resolution feature map for skip connection

            e3 = self.enc3(self.pool(e2))
            state2 = self.temporal2(e3, state2)
            e3 = state2[0]

        h_t = state2[0]  # Last hidden state from the bottleneck ConvLSTM
        d3 = self.upconv3(h_t)
        d3 = torch.cat([d3, e2_last], dim=1)  # Concatenate with skip connection from encoder
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1_last], dim=1)
        d2 = self.dec2(d2)

        d1 = self.head(d2)
        
        return d1  # Return the final output tensor of shape (B, C_out, H, W)