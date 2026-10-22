"""Validated file-oriented adapter for the repository's PCD6 reference codec.

The authoritative transform, psychoacoustic analysis, quantizer, RLE/Huffman
packing, and rate-control implementation remains in perception_codec_5_6.
This adapter exposes that pipeline through a small Python API without copying
or silently diverging from the reference implementation.

This is an offline API. Do not call it from an audio-device callback.
"""
from __future__ import annotations

import math
import os
from typing import Union

PathLike = Union[str, os.PathLike[str]]


class PCD6Encoder:
    """File-oriented adapter for the repository's MDCT/TDAC PCD6 codec.

    Args:
        target_kbps: Target bitrate in kilobits per second. The underlying
            codec uses closed-loop frame budgeting; achieved bitrate can vary.
    """

    def __init__(self, target_kbps: float = 96.0) -> None:
        try:
            target = float(target_kbps)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_kbps must be a finite positive number") from exc
        if not math.isfinite(target) or target <= 0.0 or target > 10000.0:
            raise ValueError("target_kbps must be finite and in (0, 10000]")
        self.target_kbps = target

    @staticmethod
    def _path(value: PathLike, label: str) -> str:
        try:
            path = os.fspath(value)
        except TypeError as exc:
            raise TypeError(f"{label} must be a path-like value") from exc
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label} must not be empty")
        return path

    def encode_file(self, input_wav: PathLike, output_bit: PathLike) -> None:
        """Encode a WAV file into the existing PCD6 bitstream format."""
        source = self._path(input_wav, "input_wav")
        destination = self._path(output_bit, "output_bit")
        if os.path.abspath(source) == os.path.abspath(destination):
            raise ValueError("input_wav and output_bit must be different paths")
        from perception_codec_5_6 import encode_wav

        encode_wav(source, destination, target_kbps=self.target_kbps)

    def decode_file(self, input_bit: PathLike, output_wav: PathLike) -> None:
        """Decode a PCD6 bitstream into a WAV file."""
        source = self._path(input_bit, "input_bit")
        destination = self._path(output_wav, "output_wav")
        if os.path.abspath(source) == os.path.abspath(destination):
            raise ValueError("input_bit and output_wav must be different paths")
        from perception_codec_5_6 import decode_bit

        decode_bit(source, destination)

    @staticmethod
    def selftest(output_directory: PathLike = ".") -> None:
        """Run the reference codec's transform and end-to-end self-test."""
        directory = PCD6Encoder._path(output_directory, "output_directory")
        from perception_codec_5_6 import selftest

        selftest(directory)
