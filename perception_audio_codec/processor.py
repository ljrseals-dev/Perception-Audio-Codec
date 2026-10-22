"""Validated audio input and framing helpers for Perception Audio Codec.

This module deliberately does not duplicate the MDCT/psychoacoustic/entropy
codec. It defines the boundary between decoded PCM and that codec pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

ChannelPolicy = Literal["first", "average"]


@dataclass(frozen=True)
class ProcessorConfig:
    """Audio framing configuration.

    Args:
        frame_size: Analysis window length in samples.
        hop_size: Distance between frame starts. Must be <= frame_size.
        channel_policy: Existing prototype behavior is "first" (channel 0).
            "average" explicitly averages channels into mono.
    """

    frame_size: int = 2048
    hop_size: int = 1024
    channel_policy: ChannelPolicy = "first"

    def __post_init__(self) -> None:
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if self.hop_size <= 0 or self.hop_size > self.frame_size:
            raise ValueError("hop_size must be in [1, frame_size]")
        if self.channel_policy not in ("first", "average"):
            raise ValueError("channel_policy must be 'first' or 'average'")


@dataclass(frozen=True)
class AudioFrame:
    """One zero-padded analysis frame and its position in the source signal."""

    index: int
    start_sample: int
    valid_samples: int
    samples: np.ndarray


@dataclass(frozen=True)
class PreparedAudio:
    """Normalized mono PCM plus source metadata useful to a decoder/writer."""

    samples: np.ndarray
    sample_rate: int
    source_channels: int
    source_frames: int


class AudioProcessor:
    """Validate PCM input and split it into deterministic overlapping frames.

    No implicit resampling, loudness normalization, clipping, or channel
    summing is performed unless the caller explicitly selects average mode.
    """

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        self.config = config or ProcessorConfig()

    def prepare(
        self, samples: np.ndarray, sample_rate: int, *, channel_policy: ChannelPolicy | None = None
    ) -> PreparedAudio:
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        data = np.asarray(samples)
        if data.ndim not in (1, 2):
            raise ValueError("audio must be a 1-D mono or 2-D (frames, channels) array")
        if data.shape[0] == 0:
            raise ValueError("audio must contain at least one sample frame")
        if data.ndim == 2 and data.shape[1] == 0:
            raise ValueError("audio must contain at least one channel")

        source_frames = int(data.shape[0])
        source_channels = 1 if data.ndim == 1 else int(data.shape[1])
        policy = channel_policy or self.config.channel_policy
        if policy not in ("first", "average"):
            raise ValueError("channel_policy must be 'first' or 'average'")

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            # PCM integer conventions use the signed full-scale magnitude.
            if info.min < 0:
                scale = float(max(abs(int(info.min)), int(info.max)))
                data = data.astype(np.float64) / scale
            else:
                midpoint = (float(info.max) + 1.0) / 2.0
                data = (data.astype(np.float64) - midpoint) / midpoint
        else:
            data = data.astype(np.float64, copy=False)

        if not np.all(np.isfinite(data)):
            raise ValueError("audio contains NaN or infinite values")

        if data.ndim == 2:
            if policy == "first":
                mono = data[:, 0]
            else:
                mono = np.mean(data, axis=1, dtype=np.float64)
        else:
            mono = data

        # Own the memory: later caller mutation must not change prepared audio.
        mono = np.array(mono, dtype=np.float64, copy=True)
        return PreparedAudio(
            samples=mono,
            sample_rate=int(sample_rate),
            source_channels=source_channels,
            source_frames=source_frames,
        )

    def frames(self, audio: PreparedAudio) -> Iterator[AudioFrame]:
        """Yield frames until every source sample has appeared in a frame.

        The final frame is zero-padded. Empty audio is rejected by prepare(),
        so at least one frame is always emitted.
        """
        samples = audio.samples
        size = self.config.frame_size
        hop = self.config.hop_size
        frame_index = 0
        start = 0
        while start < samples.size:
            end = min(start + size, samples.size)
            valid = end - start
            frame = np.zeros(size, dtype=np.float64)
            frame[:valid] = samples[start:end]
            frame.setflags(write=False)
            yield AudioFrame(
                index=frame_index,
                start_sample=start,
                valid_samples=valid,
                samples=frame,
            )
            frame_index += 1
            start += hop
