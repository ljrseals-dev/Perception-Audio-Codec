# PCD6 encoder adapter

The repository's existing `perception_codec_5_6.py` is the authoritative PCD6
implementation: MDCT/TDAC framing, Bark-domain masking, per-band perceptual
quantization, RLE/Huffman frame coding, and closed-loop frame-byte budgeting.
The package adapter in `perception_audio_codec.pcd6.PCD6Encoder` exposes that
implementation without creating a second, incompatible bitstream.

## Usage

```python
from perception_audio_codec.pcd6 import PCD6Encoder

codec = PCD6Encoder(target_kbps=96.0)
codec.encode_file("input.wav", "output.pcd6")
codec.decode_file("output.pcd6", "roundtrip.wav")
codec.selftest("codec-selftest")
```

## Scope and safety

- This adapter is file-oriented and intended for offline or worker-thread use, never for a WASAPI/ASIO real-time callback.
- The reference implementation reads and writes files itself. This API is not an in-memory streaming encoder and does not provide SPSC queues or device I/O.
- The target bitrate is a closed-loop budget, not a guarantee that every file reaches the exact requested bitrate.
- The reference implementation is mono-oriented: stereo WAV input selects channel zero. It does not currently preserve stereo.
- The adapter does not change the PCD6 format or claim compatibility with older PCD5 streams.
- Run the reference self-test and package unit tests before relying on a build. Hardware loopback, sustained-load and perceptual listening tests are separate validation milestones.

PAC1 remains a separate experimental format. Use this adapter when the intended pipeline is the repository's existing PCD6 encoder.
