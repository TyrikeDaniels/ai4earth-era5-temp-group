"""
convlstm_unet.py

UNet-like model with ConvLSTM cells and optional spatiotemporal attention gates for
spatiotemporal prediction. Part of the ai4earth-era5-temp-group project.

Reference(s):
    - Shi et al. 2015, "Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting", NeurIPS.
    - https://github.com/ShivekRanjan/bengaluru-lulc-forecast-/blob/main/src/models/convlstm.py

The encoder ingests a sequence of multi-channel rasters (B, T, C, H, W) and the
decoder heads predict per-pixel occurrence probability and intensity for the next frame.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with 4 gates fused into a single convolution operation."""

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
            x: Input tensor of shape (B, C_in, H, W).
            h_prev: Previous hidden state of shape (B, C_hidden, H, W).
            c_prev: Previous cell state of shape (B, C_hidden, H, W).

        Returns:
            Current hidden state and cell state.
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_channels, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_current = f_gate * c_prev + i_gate * g_gate
        h_current = o_gate * torch.tanh(c_current)

        return h_current, c_current

    def init_state(
        self, batch_size: int, height: int, width: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the hidden and cell states to zeros.

        Args:
            batch_size: Batch size.
            height: Height of the input tensor.
            width: Width of the input tensor.
            device: Device to place the zero states on. Falls back to the
                cell's own parameter device if not given.

        Returns:
            Initialized hidden and cell states.
        """
        target_device = device if device is not None else self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=target_device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=target_device),
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

        self.cells = nn.ModuleList()
        for i in range(len(hidden_channels)):
            in_channels = input_channels if i == 0 else hidden_channels[i - 1]
            self.cells.append(ConvLSTMCell(in_channels, hidden_channels[i], kernel_size, bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the stacked ConvLSTM model.

        Args:
            x: Input tensor of shape (B, T, C_in, H, W).

        Returns:
            Output tensor of shape (B, T, C_out, H, W).
        """
        batch_size, seq_len, _, height, width = x.size()

        h_states = []
        c_states = []
        for cell in self.cells:
            h, c = cell.init_state(batch_size, height, width, device=x.device)
            h_states.append(h)
            c_states.append(c)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = h_states[i], c_states[i]
                h_current, c_current = cell(x_t, h_prev, c_prev)
                h_states[i], c_states[i] = h_current, c_current
                x_t = h_current
            outputs.append(h_current.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class AttentionGate(nn.Module):
    """
    Standard additive attention gate (Attention U-Net style).
    gate:  the coarser, upsampled decoder feature map
    skip:  the encoder feature map at the same resolution
    Returns the skip connection re-weighted by a learned spatial attention mask.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.theta(gate)
        s = self.phi(skip)
        combined = F.relu(g + s, inplace=True)
        attn = torch.sigmoid(self.psi(combined))
        return skip * attn


class UNetConvLSTM(nn.Module):
    """UNet architecture with ConvLSTMCell layers, fully separate decoders for
    probability (occurrence) and intensity prediction, with optional attention gates
    on the skip connections.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
        use_attention_gates: bool = False,
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

        if use_attention_gates:
            self.attn3_a = AttentionGate(hidden_channels[1], hidden_channels[1], hidden_channels[1] // 2)
            self.attn2_a = AttentionGate(hidden_channels[0], hidden_channels[0], hidden_channels[0] // 2)
            self.attn3_b = AttentionGate(hidden_channels[1], hidden_channels[1], hidden_channels[1] // 2)
            self.attn2_b = AttentionGate(hidden_channels[0], hidden_channels[0], hidden_channels[0] // 2)
        else:
            self.attn3_a = self.attn2_a = self.attn3_b = self.attn2_b = None

        self.prob_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1),
        )

        self.intensity_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=1),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True, drop_p: float = 0.15) -> nn.Sequential:
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
        self,
        e3: torch.Tensor, e2: torch.Tensor, e1: torch.Tensor,
        upconv3: nn.Module, dec3: nn.Module,
        upconv2: nn.Module, dec2: nn.Module,
        attn3: Optional[nn.Module], attn2: Optional[nn.Module],
    ) -> torch.Tensor:
        """Runs one decoder branch once, using the final ConvLSTM states."""
        d3 = upconv3(e3)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = attn3(gate=d3, skip=e2) if attn3 is not None else e2
        d3 = torch.cat([d3, skip2], dim=1)
        d3 = dec3(d3)

        d2 = upconv2(d3)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        skip1 = attn2(gate=d2, skip=e1) if attn2 is not None else e1
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = dec2(d2)

        return d2

    def forward(self, x: torch.Tensor):
        B, T, _, H, W = x.shape
        device = x.device

        state1 = self.temporal_a.init_state(batch_size=B, height=H, width=W, device=device)
        state2 = self.temporal_b.init_state(batch_size=B, height=H // 2, width=W // 2, device=device)
        state3 = self.temporal_c.init_state(batch_size=B, height=H // 4, width=W // 4, device=device)

        e1 = e2 = e3 = None

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

        # Decode once, after the ConvLSTM cells have consumed the whole sequence --
        # not once per timestep with all but the last discarded.
        decoder_last_a = self._decode_branch(
            e3, e2, e1,
            self.upconv3_a, self.dec3_a, self.upconv2_a, self.dec2_a,
            self.attn3_a, self.attn2_a,
        )

        decoder_last_b = self._decode_branch(
            e3, e2, e1,
            self.upconv3_b, self.dec3_b, self.upconv2_b, self.dec2_b,
            self.attn3_b, self.attn2_b,
        )

        rain_logit = self.prob_head(decoder_last_a)
        intensity_pred = self.intensity_head(decoder_last_b)

        return rain_logit, intensity_pred