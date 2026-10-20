# PAC1 offline encoder

The package exposes a deterministic offline baseline:

```python
import numpy as np
from perception_audio_codec import EncoderConfig, PerceptionEncoder

sample_rate = 48_000
t = np.arange(sample_rate, dtype=np.float64) / sample_rate
pcm = 0.4 * np.sin(2 * np.pi * 440 * t)

encoder = PerceptionEncoder(EncoderConfig(quality=0.8))
result = encoder.encode(pcm, sample_rate)
decoded, decoded_rate = encoder.decode(result.bitstream)
```

## Pipeline

1. Validate sample rate, dimensions, finiteness, and input length.
2. Convert integer PCM to normalized floating point and apply the explicit channel policy.
3. Split into deterministic fixed-size blocks, padding only the last block.
4. Apply an orthonormal DCT-II to each block.
5. Divide spectral coefficients into bands and quantize each band using a per-band scale.
6. Serialize versioned frame metadata and coefficients, then compress the payload with zlib.
7. Validate the header and lengths during decoding before allocating the output buffer.

## Important format and quality limits

- This is the **PAC1** baseline format; it is not compatible with the README's documented **PCD6** format.
- The transform is block DCT-II, not MDCT/TDAC. There is no overlap-add reconstruction.
- The band scaling is a simple adaptive quantizer, not a calibrated psychoacoustic masking model or true target-bitrate reservoir.
- `quality` is a quantization precision control in the range `[0, 1]`; it does not promise a fixed bitrate or perceptual quality level.
- Output is mono. Multichannel input defaults to selecting channel 0; `channel_policy="average"` explicitly downmixes.
- Encoding and decoding are offline/worker-thread operations. Do not invoke them from a real-time audio callback.
- The encoder does not resample, apply loudness normalization, or perform hidden file/network activity.

The format is versioned so a future MDCT/TDAC implementation can be added without mislabeling PAC1 data as PCD6. Any production claim requires listening tests, objective quality measurements, bitrate sweeps, malformed-stream fuzzing, and runtime profiling.
