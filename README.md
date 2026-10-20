```markdown
# Perception Audio Codec Prototype

![Perception Audio Codec Prototype](image.jpg)

Pure Python, educational and experimental. MDCT/TDAC, a simple psychoacoustic model, perceptual quantization, run-length coding, and Huffman.

This is not a drop-in replacement for MP3, AAC, or Opus. It is a readable prototype of that pipeline: overlap-add filterbank, masking thresholds, a noise allocator, then entropy coding.

## Pipeline

1. **Frame** the waveform with 2048-sample Kaiser–Bessel-derived windows, hopping by 1024.
2. **MDCT** with gain \(\sqrt{2/M}\) on both analysis and synthesis so coefficients stay moderate and overlap-add still reconstructs.
3. **Psychoacoustic model** in Bark: spectral flatness for tonality, asymmetric spreading, and an absolute threshold in the same power units as the MDCT.
4. **Perceptual quantization** per scale-factor band. The step is the coarsest one whose mean-square error stays under the threshold, then widened so indices stay in \(\pm 2047\).
5. **Rate control** re-quantizes the frame until the packed body fits a global bit reservoir at the target bitrate.
6. **RLE + Huffman**, written per frame so the reservoir measures the bytes that are actually stored.
7. **Decode** inverts that path and writes 16-bit PCM.

Bitstreams use the magic `PCD6`. They are not decodable by the older unnormalized `PCD5` tools.

## Requirements

- Python 3
- `numpy`
- `scipy`

## Usage

```bash
python perceptual_codec_5_6.py encode input.wav output.bit [target_kbps]
python perceptual_codec_5_6.py decode input.bit output.wav
python perceptual_codec_5_6.py selftest [outdir]
```

`target_kbps` defaults to 96. `selftest` checks middle-region MDCT perfect reconstruction, then encodes and decodes a two-second tone-plus-noise file and prints SNR, RMS ratio, and achieved bitrate.

## Limits

- Mono only. Stereo input keeps channel 0.
- The masking model is a sketch (fixed spreading slopes, one tonality offset). It will not match a tuned AAC or MP3 encoder.
- Per-frame Huffman tables spend bits that a shared codebook would save. The reservoir accounts for that overhead; low targets get coarse steps.
- Perfect reconstruction holds for the unquantized transform. Quantization noise is what the threshold and the rate loop allow.
```
