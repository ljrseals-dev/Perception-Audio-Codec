# Processor API: validated PCM and analysis framing

The new `perception_audio_codec` package adds a small, testable boundary for
preparing PCM before the existing MDCT/TDAC codec pipeline. It does **not**
replace or duplicate the repository's transform, masking, quantization, RLE,
Huffman, or bitstream implementation.

## Example

```python
import numpy as np
from perception_audio_codec import AudioProcessor, ProcessorConfig

processor = AudioProcessor(ProcessorConfig(frame_size=2048, hop_size=1024))
prepared = processor.prepare(np.zeros(48000, dtype=np.float32), sample_rate=48000)

for frame in processor.frames(prepared):
    # Feed frame.samples to the codec's analysis/windowing stage.
    # frame.valid_samples identifies real samples in the final padded frame.
    pass
```

## Contract

- Accepts mono arrays shaped `(frames,)` or multichannel arrays shaped
  `(frames, channels)`.
- Rejects empty input, invalid sample rates, unsupported dimensions, and
  NaN/Inf values before codec work starts.
- Converts integer PCM to normalized floating-point samples.
- Preserves the prototype's mono behavior by default: channel zero is selected.
  Averaging channels is explicit via `channel_policy="average"`.
- Frames are deterministic, overlapping according to `hop_size`; the final
  frame is zero-padded and records its actual `valid_samples`.
- Prepared samples own their memory, so caller-side mutation cannot silently
  change data being processed.

This helper does not resample, normalize loudness, limit peaks, or promise
bit-exact round trips. It is a framing/input boundary, not a real-time audio
callback implementation. No codec quality claims are made by these tests.

## Tests

Run with the repository's existing Python dependencies installed:

```bash
python -m unittest discover -s tests -v
```
