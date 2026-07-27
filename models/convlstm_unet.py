"""
convlstm_unet.py

UNet-like model with ConvLSTM cells and attention gates for spatiotemporal
prediction. Part of the ai4earth-era5-temp-group project. ConvLSTM cell 
and stacked ConvLSTM model for spatiotemporal land-use prediction.

Reference(s):
    Shi et al. 2015, "Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting", NeurIPS.
    https://github.com/ShivekRanjan/bengaluru-lulc-forecast-/blob/main/src/models/convlstm.py

The encoder ingests a sequence of multi-channel rasters (B, T, C, H, W) and the
decoder head predicts a per-pixel class probability map for the next-year frame.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch         
import torch.nn as nn
import torch.nn.functional as F


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

    def init_state(self, batch_size: int, height: int, width: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Oktay et al. 2018, Attention U-Net)."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: decoder feature map (coarser scale), shape (B, gate_channels, H, W)
            skip: encoder skip-connection feature map, shape (B, skip_channels, H, W)
                  Must already match `gate`'s spatial size (interpolate before calling).
        Returns:
            skip, reweighted elementwise by a learned attention map (same shape as skip).
        """
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        attn = self.relu(g1 + x1)
        attn = self.psi(attn)  # (B, 1, H, W), values in [0, 1]
        return skip * attn


class UNetConvLSTM(nn.Module):
    """UNet architecture with ConvLSTMCell layers, fully separate decoders for
    probability (occurrence) and intensity prediction."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
        use_attention_gates: bool = False,
        temporal_decoder: bool = False
    ):
        super().__init__()
        self.use_attention_gates = use_attention_gates

        self.enc1 = self._conv_block(input_channels, hidden_channels[0], kernel_size, bias)
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], kernel_size, bias)
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], kernel_size, bias)

        self.pool = nn.MaxPool2d(2)

        self.temporal_a = ConvLSTMCell(hidden_channels[0], hidden_channels[0])
        self.temporal_b = ConvLSTMCell(hidden_channels[1], hidden_channels[1])
        self.temporal_c = ConvLSTMCell(hidden_channels[2], hidden_channels[2])

        self.upconv3_a = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)
        self.dec3_a = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2_a = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2_a = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)

        self.upconv3_b = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)
        self.dec3_b = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2_b = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2_b = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)

        self.prob_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], output_channels, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
        )

        self.intensity_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], output_channels, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
        )

        if self.use_attention_gates:
            self.attn3_a = AttentionGate(
                gate_channels=hidden_channels[1], skip_channels=hidden_channels[1],
                inter_channels=hidden_channels[1] // 2,
            )
            self.attn2_a = AttentionGate(
                gate_channels=hidden_channels[0], skip_channels=hidden_channels[0],
                inter_channels=hidden_channels[0] // 2,
            )
            self.attn3_b = AttentionGate(
                gate_channels=hidden_channels[1], skip_channels=hidden_channels[1],
                inter_channels=hidden_channels[1] // 2,
            )
            self.attn2_b = AttentionGate(
                gate_channels=hidden_channels[0], skip_channels=hidden_channels[0],
                inter_channels=hidden_channels[0] // 2,
            )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True, drop_p: float = 0.2) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_p),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_p),
        )

    def _decode_branch(
        self, e3, e2, e1,
        upconv3, dec3, upconv2, dec2,
        attn3, attn2,
        state_d3, state_d2,
    ):
        """Runs one decoder branch for a single timestep. Returns (output, state_d3, state_d2)."""
        d3 = upconv3(e3)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)

        skip2 = attn3(gate=d3, skip=e2) if self.use_attention_gates else e2
        d3 = torch.cat([d3, skip2], dim=1)
        d3 = dec3(d3)

        d2 = upconv2(d3)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)

        skip1 = attn2(gate=d2, skip=e1) if self.use_attention_gates else e1
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = dec2(d2)

        return d2, state_d3, state_d2

    def forward(self, x):
        B, T, _, H, W = x.shape
        device = x.device

        state1 = self.temporal_a.init_state(batch_size=B, height=H, width=W, device=device)
        state2 = self.temporal_b.init_state(batch_size=B, height=H // 2, width=W // 2, device=device)
        state3 = self.temporal_c.init_state(batch_size=B, height=H // 4, width=W // 4, device=device)

        decoder_last_a = None
        decoder_last_b = None

        state_d3_a = state_d2_a = None
        state_d3_b = state_d2_b = None

        for t in range(T):

            e1 = self.enc1(x[:, t])
            state1 = self.temporal_a(e1, state1[0], state1[1])
            e1 = state1[0]

            e2 = self.enc2(self.pool(e1))
            state2 = self.temporal_b(e2, state2[0], state2[1])
            e2 = state2[0]

            e3 = self.enc3(self.pool(e2))
            state3 = self.temporal_c(e3, state3[0], state3[1])
            e3 = state3[0]

            decoder_last_a, _, _ = self._decode_branch(
                e3, e2, e1,
                self.upconv3_a, self.dec3_a, self.upconv2_a, self.dec2_a,
                self.attn3_a if self.use_attention_gates else None,
                self.attn2_a if self.use_attention_gates else None,
                state_d3_a, state_d2_a,
            )

            decoder_last_b, _, _ = self._decode_branch(
                e3, e2, e1,
                self.upconv3_b, self.dec3_b, self.upconv2_b, self.dec2_b,
                self.attn3_b if self.use_attention_gates else None,
                self.attn2_b if self.use_attention_gates else None,
                state_d3_b, state_d2_b,
            )

        rain_logit = self.prob_head(decoder_last_a)
        intensity_pred = self.intensity_head(decoder_last_b)

        return rain_logit, intensity_pred