"""Deterministic offline encoder for the Perception Audio Codec prototype.

This PAC1 baseline is intentionally separate from the repository's documented
PCD6 format. It uses block DCT-II rather than claiming MDCT/TDAC compatibility.
Codec work belongs on a worker/offline thread, never in an audio callback.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import struct
import zlib

import numpy as np
from scipy.fft import dct, idct

from .processor import AudioProcessor, ChannelPolicy, ProcessorConfig

_MAGIC = b"PAC1"
_VERSION = 1
_HEADER = struct.Struct(">4sHHIIBB")
_FRAME_HEADER = struct.Struct(">I")
_SCALE = struct.Struct(">f")


@dataclass(frozen=True)
class EncoderConfig:
    """Offline encoder settings.

    frame_size must be a power of two. quality is in [0, 1]; higher values
    preserve more transform precision but generally produce larger packets.
    band_count controls the number of independent spectral scale factors.
    """

    frame_size: int = 1024
    band_count: int = 32
    quality: float = 0.75
    channel_policy: ChannelPolicy = "first"

    def __post_init__(self) -> None:
        if self.frame_size < 64 or self.frame_size & (self.frame_size - 1):
            raise ValueError("frame_size must be a power of two >= 64")
        if self.band_count < 1 or self.band_count > self.frame_size:
            raise ValueError("band_count must be in [1, frame_size]")
        if not np.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be finite and in [0, 1]")
        if self.channel_policy not in ("first", "average"):
            raise ValueError("invalid channel_policy")


@dataclass(frozen=True)
class EncodeResult:
    bitstream: bytes
    sample_rate: int
    input_samples: int
    frame_count: int
    original_channels: int
    achieved_bitrate: float


class PerceptionEncoder:
    """Validated PCM -> spectral transform -> band quantization -> PAC1 packet.

    The implementation is deterministic and offline. It is a baseline codec,
    not a production psychoacoustic encoder and not compatible with PCD6.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self._processor = AudioProcessor(
            ProcessorConfig(
                frame_size=self.config.frame_size,
                hop_size=self.config.frame_size,
                channel_policy=self.config.channel_policy,
            )
        )
        self._bands = np.array_split(np.arange(self.config.frame_size), self.config.band_count)
        self._qmax = int(round(127 + self.config.quality * (8191 - 127)))

    def encode(self, samples: np.ndarray, sample_rate: int) -> EncodeResult:
        """Encode mono or multichannel PCM-like samples into a PAC1 byte stream."""
        audio = self._processor.prepare(samples, sample_rate)
        frame_size = self.config.frame_size
        frame_count = (audio.samples.size + frame_size - 1) // frame_size
        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            frame_size,
            audio.sample_rate,
            int(audio.samples.size),
            self.config.band_count,
            0,
        )

        payload = io.BytesIO()
        payload.write(header)
        payload.write(_FRAME_HEADER.pack(frame_count))
        for start in range(0, audio.samples.size, frame_size):
            frame = np.zeros(frame_size, dtype=np.float64)
            segment = audio.samples[start : start + frame_size]
            frame[: segment.size] = segment
            spectrum = dct(frame, type=2, norm="ortho")
            quantized = np.zeros(frame_size, dtype=np.int16)
            scales = []
            for indices in self._bands:
                peak = float(np.max(np.abs(spectrum[indices]))) if indices.size else 0.0
                step = peak / self._qmax if peak > 0.0 else 1.0
                values = np.rint(spectrum[indices] / step)
                values = np.clip(values, -self._qmax, self._qmax)
                quantized[indices] = values.astype(np.int16)
                scales.append(step)
            for scale in scales:
                payload.write(_SCALE.pack(scale))
            payload.write(quantized.astype(">i2", copy=False).tobytes())

        raw = payload.getvalue()
        bitstream = _MAGIC + zlib.compress(raw[len(_MAGIC):], level=6)
        return EncodeResult(
            bitstream=bitstream,
            sample_rate=audio.sample_rate,
            input_samples=int(audio.samples.size),
            frame_count=frame_count,
            original_channels=audio.source_channels,
            achieved_bitrate=(len(bitstream) * 8 * audio.sample_rate / audio.samples.size),
        )

    @staticmethod
    def decode(bitstream: bytes) -> tuple[np.ndarray, int]:
        """Decode PAC1 to float64 mono samples and the stored sample rate."""
        if not isinstance(bitstream, (bytes, bytearray, memoryview)):
            raise TypeError("bitstream must be bytes-like")
        data = bytes(bitstream)
        if len(data) < 5 or data[:4] != _MAGIC:
            raise ValueError("not a PAC1 bitstream")
        try:
            raw = _MAGIC + zlib.decompress(data[4:])
        except zlib.error as exc:
            raise ValueError("corrupt PAC1 compressed payload") from exc
        if len(raw) < _HEADER.size + _FRAME_HEADER.size:
            raise ValueError("truncated PAC1 header")
        magic, version, frame_size, sample_rate, sample_count, band_count, reserved = _HEADER.unpack_from(raw)
        if magic != _MAGIC or version != _VERSION or reserved != 0:
            raise ValueError("unsupported PAC1 version or flags")
        if frame_size < 64 or frame_size & (frame_size - 1):
            raise ValueError("invalid frame size in PAC1 header")
        if not 1 <= band_count <= frame_size or sample_rate <= 0:
            raise ValueError("invalid PAC1 dimensions")
        frame_count = _FRAME_HEADER.unpack_from(raw, _HEADER.size)[0]
        expected_frames = (sample_count + frame_size - 1) // frame_size
        if sample_count == 0 or frame_count != expected_frames:
            raise ValueError("inconsistent PAC1 sample/frame counts")
        bands = np.array_split(np.arange(frame_size), band_count)
        offset = _HEADER.size + _FRAME_HEADER.size
        bytes_per_frame = band_count * _SCALE.size + frame_size * 2
        if len(raw) != offset + frame_count * bytes_per_frame:
            raise ValueError("truncated or trailing PAC1 frame data")
        output = np.zeros(frame_count * frame_size, dtype=np.float64)
        for frame_index in range(frame_count):
            scales = []
            for _ in range(band_count):
                (scale,) = _SCALE.unpack_from(raw, offset)
                offset += _SCALE.size
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("invalid PAC1 scale factor")
                scales.append(scale)
            quantized = np.frombuffer(raw, dtype=">i2", count=frame_size, offset=offset).astype(np.float64)
            offset += frame_size * 2
            spectrum = np.zeros(frame_size, dtype=np.float64)
            for indices, scale in zip(bands, scales):
                spectrum[indices] = quantized[indices] * scale
            start = frame_index * frame_size
            output[start : start + frame_size] = idct(spectrum, type=2, norm="ortho")
        return output[:sample_count], int(sample_rate)
