"""
trajgru_unet.py

UNet-like model with TrajGRU cells and attention gates for spatiotemporal
prediction. Part of the ai4earth-era5-temp-group project. NOTE: Incomplete.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch         
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wrap(input_tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Apply optical flow warping to input tensor."""
    B, C, H, W = input_tensor.size()
    
    # Create mesh grid
    xx = torch.arange(0, W, device=DEVICE).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=DEVICE).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # Scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    output = F.grid_sample(input_tensor, vgrid, align_corners=False)
    return output


class TrajGRU(nn.Module):
    """
    Trajectory GRU for spatiotemporal prediction with optical flow warping.
 
    Unlike the original implementation, this does NOT bake batch size or
    spatial resolution into __init__. Instead, like ConvLSTMCell in this
    codebase, it only declares channel counts at construction time and
    computes state shape dynamically in forward() from the actual input
    tensor. This means the same module works at any resolution / batch
    size without reconstruction, and device is always inferred from the
    live tensor rather than a hardcoded global.
    """
 
    def __init__(
        self,
        input_channel: int,
        num_filter: int,
        zoneout: float = 0.0,
        L: int = 5,
        i2h_kernel: Tuple[int, int] = (3, 3),
        i2h_stride: Tuple[int, int] = (1, 1),
        i2h_pad: Tuple[int, int] = (1, 1),
        i2h_dilate: Tuple[int, int] = (1, 1),
        h2h_kernel: Tuple[int, int] = (5, 5),
        h2h_dilate: Tuple[int, int] = (1, 1),
        act_type=torch.tanh,
    ):
        super().__init__()
 
        assert (h2h_kernel[0] % 2 == 1) and (h2h_kernel[1] % 2 == 1), \
            f"Only support odd h2h kernels, got h2h_kernel={h2h_kernel}"
 
        self._num_filter = num_filter
        self._L = L
        self._zoneout = zoneout
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._h2h_kernel = h2h_kernel
        self._h2h_dilate = h2h_dilate
        self._act_type = act_type
 
        # Input-to-hidden: produces reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self._num_filter * 3,
            kernel_size=self._i2h_kernel,
            stride=self._i2h_stride,
            padding=self._i2h_pad,
            dilation=self._i2h_dilate,
        )
 
        # Input to flow
        self.i2f_conv1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )
 
        # Hidden to flow
        self.h2f_conv1 = nn.Conv2d(
            in_channels=self._num_filter,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )
 
        # Generate flow field (L trajectories x 2 channels for x,y displacement)
        self.flows_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self._L * 2,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
        )
 
        # Hidden-to-hidden: processes warped hidden states
        self.ret = nn.Conv2d(
            in_channels=self._num_filter * self._L,
            out_channels=self._num_filter * 3,
            kernel_size=(1, 1),
            stride=1,
        )
 
    def _state_hw(self, in_h: int, in_w: int) -> Tuple[int, int]:
        """Compute output (state) height/width from input height/width,
        given this module's i2h conv settings. Mirrors what BaseConvRNN
        used to precompute once at init, but done fresh per forward call."""
        dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        state_h = (in_h + 2 * self._i2h_pad[0] - dilate_ksize_h) // self._i2h_stride[0] + 1
        state_w = (in_w + 2 * self._i2h_pad[1] - dilate_ksize_w) // self._i2h_stride[1] + 1
        return state_h, state_w
 
    def init_state(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create a zero initial hidden state sized for this input resolution.
        Mirrors ConvLSTMCell.init_state's signature/usage in this codebase."""
        state_h, state_w = self._state_hw(height, width)
        return torch.zeros(
            (batch_size, self._num_filter, state_h, state_w),
            dtype=torch.float,
            device=device,
        )
 
    def _flow_generator(
        self,
        inputs: Optional[torch.Tensor],
        states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Generate optical flow fields from input and hidden state."""
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
 
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)
 
        flows = self.flows_conv(f_conv1)
        # Split into L flow fields of 2 channels each
        flows = torch.split(flows, 2, dim=1)
        return flows
 
    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (S, B, C, H, W) sequence of input frames, or None for generation
            state: (B, num_filter, H', W') initial hidden state. Required if
                   inputs is None (there is no way to infer B/H/W from nothing).
                   If inputs is given and state is None, a zero state sized
                   from `inputs` is created automatically.
            seq_len: Number of timesteps to unroll. Required if inputs is None.
 
        Returns:
            outputs: (S, B, num_filter, H', W') stacked hidden states
            next_h: (B, num_filter, H', W') final hidden state
        """
        prev_h = state

        # input transform
        i2h = self.i2h(inputs)

        i2h_slice = torch.split(
            i2h,
            self._num_filter,
            dim=1
        )

        flows = self._flow_generator(
            inputs,
            prev_h
        )

        wrapped_data = []

        for flow in flows:
            wrapped_data.append(
                wrap(prev_h, -flow)
            )

        wrapped_data = torch.cat(
            wrapped_data,
            dim=1
        )

        h2h = self.ret(wrapped_data)

        h2h_slice = torch.split(
            h2h,
            self._num_filter,
            dim=1
        )

        reset_gate = torch.sigmoid(
            i2h_slice[0] + h2h_slice[0]
        )

        update_gate = torch.sigmoid(
            i2h_slice[1] + h2h_slice[1]
        )

        new_mem = self._act_type(
            i2h_slice[2] + reset_gate * h2h_slice[2]
        )

        next_h = (
            update_gate * prev_h
            +
            (1-update_gate) * new_mem
        )

        return next_h
 
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
        """
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        attn = self.relu(g1 + x1)
        attn = self.psi(attn)  # (B, 1, H, W), values in [0, 1]
        return skip * attn
 
 
class UNetTrajGRU(nn.Module):
    """
    UNet architecture with TrajGRU layers for spatiotemporal precipitation prediction.
    NOTE: Assumes hidden structure is 16->32->64.
 
    Mirrors UNetConvLSTM's forward()-time state initialization pattern exactly:
    batch size, spatial resolution, and device are all read from the live input
    tensor `x` inside forward(), not fixed at construction time. This means the
    module tolerates uneven final batches and doesn't require input_hw to be
    known in advance. (53x97 input works fine here without padding, since state
    size is derived from the encoder feature maps at whatever resolution they
    actually are, not asserted in advance.)
 
    Outputs (rain_logit, intensity_pred) to match compute_loss_a and stay directly
    comparable to the ConvLSTM baseline (UNetConvLSTM).
    """
 
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
        use_attention_gates: bool = False,
        trajgru_L: int = 5,
        trajgru_zoneout: float = 0.0,
    ):
        super().__init__()
        self.use_attention_gates = use_attention_gates
 
        # Spatial encoder (applied to each timestep independently)
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], kernel_size, bias)
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], kernel_size, bias)
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], kernel_size, bias)
 
        self.pool = nn.MaxPool2d(2)
 
        # Temporal TrajGRU encoder/bottleneck (applied across timesteps).
        # No b_h_w here -- state shape is derived per forward() call, same
        # pattern as ConvLSTMCell in UNetConvLSTM.
        self.temporal1 = TrajGRU(hidden_channels[0], hidden_channels[0], L=trajgru_L, zoneout=trajgru_zoneout)
        self.temporal2 = TrajGRU(hidden_channels[1], hidden_channels[1], L=trajgru_L, zoneout=trajgru_zoneout)
        self.temporal3 = TrajGRU(hidden_channels[2], hidden_channels[2], L=trajgru_L, zoneout=trajgru_zoneout)
 
        # Decoder (upsampling and concatenation with skip connections)
        self.upconv3 = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)
        self.dec3 = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2 = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)
 
        # Two heads, matching UNetConvLSTM / compute_loss_a exactly:
        # rain_logit (binary detection) + intensity_pred (regression).
        self.prob_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], out_channels=output_channels, kernel_size=1),
        )
 
        self.intensity_head = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[0], out_channels=output_channels, kernel_size=1),
        )
 
        if self.use_attention_gates:
            self.attn3 = AttentionGate(
                gate_channels=hidden_channels[1], skip_channels=hidden_channels[1],
                inter_channels=hidden_channels[1] // 2,
            )
            self.attn2 = AttentionGate(
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
    
    def forward(self, x):
        B, T, _, H, W = x.shape
        device = x.device

        # initialize states using first frame sizes
        e1 = self.enc1(x[:, 0])
        state1 = self.temporal1.init_state(batch_size=B,height=e1.shape[-2],width=e1.shape[-1],device=device)
        
        e2 = self.enc2(self.pool(e1))
        state2 = self.temporal2.init_state(batch_size=B,height=e2.shape[-2],width=e2.shape[-1],device=device)
        
        e3 = self.enc3(self.pool(e2))
        state3 = self.temporal3.init_state(batch_size=B,height=e3.shape[-2],width=e3.shape[-1],device=device)

        # ---- Encoder + Temporal recurrence ----
        for t in range(T):

            e1 = self.enc1(x[:, t])
            state1 = self.temporal1(e1, state1)
            e1 = state1

            e2 = self.enc2(self.pool(e1))
            state2 = self.temporal2(e2,state2)
            e2 = state2

            e3 = self.enc3(self.pool(e2))
            state3 = self.temporal3(e3,state3)
            e3 = state3

        # last timestep hidden states
        e1_last = state1
        e2_last = state2
        e3_last = state3

        # ---- Decoder ----
        d3 = self.upconv3(e3_last)
        d3 = F.interpolate(d3,size=e2_last.shape[-2:],mode="bilinear",align_corners=False)
        
        skip2 = (self.attn3(gate=d3, skip=e2_last) if self.use_attention_gates else e2_last)

        d3 = torch.cat([d3, skip2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2,size=e1_last.shape[-2:],mode="bilinear",align_corners=False)
        
        skip1 = (self.attn2(gate=d2, skip=e1_last)if self.use_attention_gates else e1_last)

        d2 = torch.cat([d2, skip1], dim=1)
        d2 = self.dec2(d2)

        rain_logit = self.prob_head(d2)
        intensity_pred = self.intensity_head(d2)

        return rain_logit, intensity_pred