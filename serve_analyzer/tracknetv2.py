"""TrackNetV2-style heatmap detector for tennis ball tracking.

This module adapts TrackNetV2 inference to the detector contract used by the
serve analyzer: return one optional ``(x, y)`` ball center per source frame,
plus video metadata. It does not perform serve selection or timestamp matching.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


TRACKNET_HEIGHT = 288
TRACKNET_WIDTH = 512
TENNIS_BALL_DIAMETER_M = 0.067


def _conv_block(nn, in_channels: int, out_channels: int):
    """Build one Conv-BatchNorm-ReLU block used by TrackNetV2."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _double_conv_block(nn, in_channels: int, out_channels: int):
    """Build two TrackNetV2 convolution blocks."""
    return nn.Sequential(
        _conv_block(nn, in_channels, out_channels),
        _conv_block(nn, out_channels, out_channels),
    )


def _triple_conv_block(nn, in_channels: int, out_channels: int):
    """Build three TrackNetV2 convolution blocks."""
    return nn.Sequential(
        _conv_block(nn, in_channels, out_channels),
        _conv_block(nn, out_channels, out_channels),
        _conv_block(nn, out_channels, out_channels),
    )


def build_tracknetv2_model(in_channels: int = 9, out_channels: int = 3):
    """Create a PyTorch TrackNetV2-style 3-frame heatmap model."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    class TrackNetV2Model(nn.Module):
        """U-Net-like TrackNetV2 architecture for 3 input frames and 3 heatmaps."""

        def __init__(self) -> None:
            super().__init__()
            self.down_block_1 = _double_conv_block(nn, in_channels, 64)
            self.down_block_2 = _double_conv_block(nn, 64, 128)
            self.down_block_3 = _double_conv_block(nn, 128, 256)
            self.bottleneck = _triple_conv_block(nn, 256, 512)
            self.pool = nn.MaxPool2d(2, 2)
            self.upsample = nn.Upsample(scale_factor=2)
            self.up_block_1 = _double_conv_block(nn, 768, 256)
            self.up_block_2 = _double_conv_block(nn, 384, 128)
            self.up_block_3 = _double_conv_block(nn, 192, 64)
            self.predictor = nn.Conv2d(64, out_channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            enc1 = self.down_block_1(x)
            enc2 = self.down_block_2(self.pool(enc1))
            enc3 = self.down_block_3(self.pool(enc2))
            bottleneck = self.bottleneck(self.pool(enc3))
            dec3 = self.up_block_1(torch.cat([self.upsample(bottleneck), enc3], dim=1))
            dec2 = self.up_block_2(torch.cat([self.upsample(dec3), enc2], dim=1))
            dec1 = self.up_block_3(torch.cat([self.upsample(dec2), enc1], dim=1))
            return self.sigmoid(self.predictor(dec1))

    return TrackNetV2Model()


def build_chgyglin_tracknetv2_model():
    """Create the TrackNetV2 architecture used by ChgygLin checkpoints."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    class Conv(nn.Module):
        """Conv-ReLU-BatchNorm block used by ChgygLin/TrackNetV2-pytorch."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding="same"
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.bn(self.act(self.conv(x)))

    class ChgygLinTrackNetV2(nn.Module):
        """VGG-style encoder-decoder matching released ``last.pt`` weights."""

        def __init__(self) -> None:
            super().__init__()
            self.conv2d_1 = Conv(9, 64)
            self.conv2d_2 = Conv(64, 64)
            self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.conv2d_3 = Conv(64, 128)
            self.conv2d_4 = Conv(128, 128)
            self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.conv2d_5 = Conv(128, 256)
            self.conv2d_6 = Conv(256, 256)
            self.conv2d_7 = Conv(256, 256)
            self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.conv2d_8 = Conv(256, 512)
            self.conv2d_9 = Conv(512, 512)
            self.conv2d_10 = Conv(512, 512)
            self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.conv2d_11 = Conv(768, 256)
            self.conv2d_12 = Conv(256, 256)
            self.conv2d_13 = Conv(256, 256)
            self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
            self.conv2d_14 = Conv(384, 128)
            self.conv2d_15 = Conv(128, 128)
            self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
            self.conv2d_16 = Conv(192, 64)
            self.conv2d_17 = Conv(64, 64)
            self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding="same")

        def forward(self, x):
            x = self.conv2d_1(x)
            x1 = self.conv2d_2(x)
            x = self.max_pooling_1(x1)
            x = self.conv2d_3(x)
            x2 = self.conv2d_4(x)
            x = self.max_pooling_2(x2)
            x = self.conv2d_5(x)
            x = self.conv2d_6(x)
            x3 = self.conv2d_7(x)
            x = self.max_pooling_3(x3)
            x = self.conv2d_8(x)
            x = self.conv2d_9(x)
            x = self.conv2d_10(x)
            x = self.up_sampling_1(x)
            x = torch.cat([x, x3], dim=1)
            x = self.conv2d_11(x)
            x = self.conv2d_12(x)
            x = self.conv2d_13(x)
            x = self.up_sampling_2(x)
            x = torch.cat([x, x2], dim=1)
            x = self.conv2d_14(x)
            x = self.conv2d_15(x)
            x = self.up_sampling_3(x)
            x = torch.cat([x, x1], dim=1)
            x = self.conv2d_16(x)
            x = self.conv2d_17(x)
            return torch.sigmoid(self.conv2d_18(x))

    return ChgygLinTrackNetV2()

def build_wasb_tracknetv2_model(in_channels: int = 9, out_channels: int = 3):
    """Create TrackNetV2 matching the nttcom/WASB-SBDT tennis-trained checkpoint.

    Uses Conv→ReLU→BN ordering (bn_first=False) and bilinear nearest upsample.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    class DoubleConv(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oc),
                nn.Conv2d(oc, oc, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oc),
            )

        def forward(self, x):
            return self.double_conv(x)

    class TripleConv(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.triple_conv = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oc),
                nn.Conv2d(oc, oc, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oc),
                nn.Conv2d(oc, oc, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(oc),
            )

        def forward(self, x):
            return self.triple_conv(x)

    class Down(nn.Module):
        def __init__(self, n, ic, oc):
            super().__init__()
            conv = DoubleConv(ic, oc) if n == 2 else TripleConv(ic, oc)
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), conv)

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, n, in1, in2, out):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            conv = DoubleConv(in1 + in2, out) if n == 2 else TripleConv(in1 + in2, out)
            self.conv = conv

        def forward(self, x1, x2):
            x1 = self.up(x1)
            return self.conv(torch.cat([x2, x1], dim=1))

    class OutConv(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.conv = nn.Conv2d(ic, oc, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class WASBTrackNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = Down(2, 64, 128)
            self.down2 = Down(3, 128, 256)
            self.down3 = Down(3, 256, 512)
            self.up1 = Up(3, 512, 256, 256)
            self.up2 = Up(2, 256, 128, 128)
            self.up3 = Up(2, 128, 64, 64)
            self.outc = OutConv(64, out_channels)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            return self.outc(x)

    return WASBTrackNetV2()

def _build_model_for_state_dict(state_dict: Dict[str, object]):
    """Select a supported TrackNetV2 architecture from checkpoint key names."""
    if any(key.startswith("conv2d_1.") for key in state_dict):
        return build_chgyglin_tracknetv2_model()
    if any(key.startswith("inc.") for key in state_dict):
        return build_wasb_tracknetv2_model()
    return build_tracknetv2_model()

def _extract_state_dict(checkpoint: object) -> Dict[str, object]:
    """Return the model state dict from common TrackNetV2 checkpoint shapes."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported TrackNetV2 checkpoint format")


def _strip_state_prefixes(state_dict: Dict[str, object]) -> Dict[str, object]:
    """Normalize common DataParallel/module prefixes from checkpoint keys."""
    stripped: Dict[str, object] = {}
    for key, value in state_dict.items():
        normalized = key
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        stripped[normalized] = value
    return stripped


def load_tracknetv2_model(weights_path: str, device: str = "cpu"):
    """Load TrackNetV2 weights from a PyTorch checkpoint file."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"TrackNetV2 weights not found: {weights_path}")

    checkpoint = torch.load(str(path), map_location=device)
    state_dict = _strip_state_prefixes(_extract_state_dict(checkpoint))
    model = _build_model_for_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _frame_window_to_tensor(frames: List[np.ndarray], device: str):
    """Convert three BGR video frames into one TrackNetV2 input tensor."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    channels: List[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, (TRACKNET_WIDTH, TRACKNET_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        channels.append(np.moveaxis(rgb, 2, 0))
    array = np.concatenate(channels, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device)


def _heatmap_center(
    heatmap: np.ndarray, threshold: float
) -> Optional[Tuple[float, float, float]]:
    """Extract the largest heatmap blob center and approximate diameter."""
    if heatmap.size == 0 or float(np.max(heatmap)) < threshold:
        return None

    mask = (heatmap >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    x, y, width, height = cv2.boundingRect(largest)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    diameter = (width + height) / 2.0
    return center_x, center_y, diameter


def detect_ball_tracknetv2(
    video_path: str,
    weights_path: str,
    conf_threshold: float = 0.5,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    start_frame: int = 0,
    device: str = "cpu",
    progress_interval: int = 100,
) -> Tuple[List[Optional[Tuple[float, float]]], float, int, Optional[float]]:
    """Detect ball centers in a video using TrackNetV2 heatmaps.

    Returns a tuple ``(detections, fps, total_frames, estimated_scale)`` where
    ``detections`` has one entry per original video frame and skipped or
    low-confidence frames are ``None``.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if conf_threshold < 0 or conf_threshold > 1:
        raise ValueError("conf_threshold must be between 0 and 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    model = load_tracknetv2_model(weights_path, device=device)
    scale_x = width / TRACKNET_WIDTH
    scale_y = height / TRACKNET_HEIGHT

    try:
        import torch
    except ImportError as exc:
        raise ImportError("TrackNetV2 detection requires torch") from exc

    detections: List[Optional[Tuple[float, float]]] = [None] * total_frames
    ball_sizes: List[float] = []
    frame_window: Deque[Tuple[int, np.ndarray]] = deque(maxlen=3)

    frame_idx = 0
    with torch.no_grad():
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_window.append((frame_idx, frame))
            if (
                len(frame_window) == 3
                and frame_idx >= start_frame
                and (frame_idx % frame_skip == 0)
            ):
                window_indices = [item[0] for item in frame_window]
                tensor = _frame_window_to_tensor(
                    [item[1] for item in frame_window], device
                )
                raw = model(tensor)
                if isinstance(raw, dict):
                    raw = raw[0]
                output = torch.sigmoid(raw).detach().cpu().numpy()[0]
                for heatmap_idx, source_frame in enumerate(window_indices):
                    if source_frame < start_frame or source_frame >= total_frames:
                        continue
                    if frame_skip > 1 and source_frame % frame_skip != 0:
                        continue
                    center = _heatmap_center(output[heatmap_idx], conf_threshold)
                    if center is None:
                        detections[source_frame] = None
                        continue
                    center_x, center_y, diameter = center
                    detections[source_frame] = (
                        float(center_x * scale_x),
                        float(center_y * scale_y),
                    )
                    ball_sizes.append(float(diameter * (scale_x + scale_y) / 2.0))

            frame_idx += 1
            if frame_idx % progress_interval == 0:
                print(
                    f"  TrackNetV2 processed {frame_idx}/{total_frames} frames "
                    f"({100 * frame_idx / total_frames:.1f}%)"
                )

    cap.release()

    estimated_scale = None
    if len(ball_sizes) >= 10:
        median_diameter_px = float(np.median(ball_sizes))
        if median_diameter_px > 0:
            estimated_scale = TENNIS_BALL_DIAMETER_M / median_diameter_px
            print(
                f"TrackNetV2 estimated scale: {estimated_scale:.6f} m/px "
                f"(from median ball diameter {median_diameter_px:.1f}px)"
            )

    found_count = sum(1 for detection in detections if detection is not None)
    print(
        f"TrackNetV2 detection complete. Found ball in {found_count}/{len(detections)} frames"
    )
    return detections, fps, len(detections), estimated_scale
