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

        i2h_slice = torch.split(i2h, self._num_filter, dim=1)

        flows = self._flow_generator(inputs, prev_h)

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

        return next_h, flows


class UNetTrajGRU(nn.Module):
    """
    UNet architecture with TrajGRU layers for spatiotemporal precipitation prediction.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        bias: bool = True,
        trajgru_L: int = 5,
        trajgru_zoneout: float = 0.0,
    ):
        super().__init__()

        # Spatial encoder (applied to each timestep independently)
        self.enc1 = self._conv_block(input_channels, hidden_channels[0], kernel_size, bias)
        self.enc2 = self._conv_block(hidden_channels[0], hidden_channels[1], kernel_size, bias)
        self.enc3 = self._conv_block(hidden_channels[1], hidden_channels[2], kernel_size, bias)

        self.pool = nn.MaxPool2d(2)

        # Temporal TrajGRU encoder/bottleneck (applied across timesteps)
        self.temporal1 = TrajGRU(hidden_channels[0], hidden_channels[0], L=trajgru_L, zoneout=trajgru_zoneout)
        self.temporal2 = TrajGRU(hidden_channels[1], hidden_channels[1], L=trajgru_L, zoneout=trajgru_zoneout)
        self.temporal3 = TrajGRU(hidden_channels[2], hidden_channels[2], L=trajgru_L, zoneout=trajgru_zoneout)

        # Decoder branches: one for probability output, one for intensity output.
        self.upconv3_a = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)
        self.dec3_a = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2_a = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2_a = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)

        self.upconv3_b = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=2, stride=2)
        self.dec3_b = self._conv_block(hidden_channels[1] + hidden_channels[1], hidden_channels[1], kernel_size, bias)
        self.upconv2_b = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=2, stride=2)
        self.dec2_b = self._conv_block(hidden_channels[0] + hidden_channels[0], hidden_channels[0], kernel_size, bias)

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
    ) -> torch.Tensor:
        """Runs one decoder branch once, using the final encoder states."""
        d3 = upconv3(e3)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        skip2 = e2
        d3 = torch.cat([d3, skip2], dim=1)
        d3 = dec3(d3)

        d2 = upconv2(d3)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        skip1 = e1
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = dec2(d2)

        return d2

    def forward(self, x: torch.Tensor):
        B, T, _, H, W = x.shape
        device = x.device

        # Spatial dims after each 2x2 maxpool -- computed directly, no need
        # to run the encoder on frame 0 just to read off shapes.
        h1, w1 = H, W
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2

        state1 = self.temporal1.init_state(batch_size=B, height=h1, width=w1, device=device)
        state2 = self.temporal2.init_state(batch_size=B, height=h2, width=w2, device=device)
        state3 = self.temporal3.init_state(batch_size=B, height=h3, width=w3, device=device)

        e1 = e2 = e3 = None
        for t in range(T):
            e1 = self.enc1(x[:, t])
            state1, flow1 = self.temporal1(e1, state1)
            e1 = state1

            e2 = self.enc2(self.pool(e1))
            state2, flow2 = self.temporal2(e2, state2)
            e2 = state2

            e3 = self.enc3(self.pool(e2))
            state3, flow3 = self.temporal3(e3, state3)
            e3 = state3

        # Decode once, after the temporal encoder has consumed the whole
        # sequence -- not once per timestep with all but the last discarded.
        decoder_last_a = self._decode_branch(
            e3, e2, e1,
            self.upconv3_a, self.dec3_a,
            self.upconv2_a, self.dec2_a,
        )

        decoder_last_b = self._decode_branch(
            e3, e2, e1,
            self.upconv3_b, self.dec3_b,
            self.upconv2_b, self.dec2_b,
        )

        rain_logit = self.prob_head(decoder_last_a)
        intensity_pred = self.intensity_head(decoder_last_b)

        return (
            rain_logit, 
            intensity_pred, 
            [flow1, flow2, flow3] # for flow field plot
        )