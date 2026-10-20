#!/usr/bin/env python3
"""
Iteration 5.6 – Perceptual codec with PR-preserving MDCT normalization,
calibrated thresholds, index-limited quantizer, and closed-loop bit reservoir.

Not compatible with PCD5 decoders: both analysis and synthesis are scaled by
sqrt(2/M). The Kaiser-Bessel-derived window is left unchanged so
w[n]^2 + w[n+M]^2 = 1 still holds.

Usage:
    python perceptual_codec_5_6.py encode input.wav output.bit [target_kbps]
    python perceptual_codec_5_6.py decode input.bit output.wav
    python perceptual_codec_5_6.py selftest [outdir]
"""

import math
import os
import struct
import sys
import heapq
from collections import Counter

import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import kaiser_bessel_derived


MAGIC = b"PCD6"
MAX_IDX = 2047


# =============================================================================
# 1. MDCT / TDAC
# =============================================================================
class MDCT_TDAC:
    """Normalized MDCT. Forward and inverse both apply sqrt(2/M).

    Relative to the unnormalized pair (forward gain 1, inverse gain 2/M) this
    multiplies coefficients by s = sqrt(2/M) and divides the synthesis gain by
    the same s, so overlap-add is unchanged.
    """

    def __init__(self, N=2048, beta=8.0):
        self.N = N
        self.M = N // 2
        self.scale = math.sqrt(2.0 / self.M)
        self.window = kaiser_bessel_derived(N, beta=beta).astype(np.float64)
        self.prev_imdct = np.zeros(self.M, dtype=np.float64)
        n = np.arange(N)
        k = np.arange(self.M)
        self.basis = np.cos(
            (np.pi / self.M) * (n[:, None] + self.M / 2 + 0.5) * (k + 0.5)
        )

    def reset(self):
        self.prev_imdct = np.zeros(self.M, dtype=np.float64)

    def forward(self, x_frame):
        if len(x_frame) != self.N:
            raise ValueError(f"Frame must be {self.N} samples")
        windowed = np.asarray(x_frame, dtype=np.float64) * self.window
        return np.sum(windowed[:, None] * self.basis, axis=0) * self.scale

    def inverse(self, X):
        y = self.scale * np.sum(np.asarray(X, dtype=np.float64)[None, :] * self.basis, axis=1)
        yw = y * self.window
        out = self.prev_imdct + yw[: self.M]
        self.prev_imdct = yw[self.M :].copy()
        return out


# =============================================================================
# 2. Psychoacoustics (power domain of the normalized MDCT)
# =============================================================================
def mdct_bin_freqs(M, fs):
    return (np.arange(M, dtype=np.float64) + 0.5) * (fs / 2.0) / M


def compute_bark(freqs):
    freqs = np.asarray(freqs, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)


def spread_masker(center_bark, masker_power, bark_axis):
    delta = bark_axis - center_bark
    spread_db = np.where(delta >= 0, -27.0 * delta, -10.0 * np.abs(delta))
    spread_db -= np.max(spread_db)
    return masker_power * 10.0 ** (spread_db / 10.0)


def compute_tonality(mdct_coeffs, sfb_limits):
    tonality = np.zeros(len(sfb_limits))
    for i, (s, e) in enumerate(sfb_limits):
        band = mdct_coeffs[s:e]
        if band.size == 0:
            continue
        energy = np.abs(band) ** 2 + 1e-12
        sfm = np.exp(np.mean(np.log(energy))) / np.mean(energy)
        sfm_db = 10.0 * np.log10(sfm + 1e-12)
        tonality[i] = np.clip((sfm_db + 60.0) / 60.0, 0.0, 1.0)
    return tonality


def get_sfb_limits(num_bins):
    widths = [4] * 20 + [8] * 15 + [12] * 10 + [16] * 8 + [24] * 6
    limits, start = [], 0
    while start < num_bins:
        w = widths.pop(0) if widths else 24
        end = min(start + w, num_bins)
        limits.append((start, end))
        start = end
    return limits


def ath_linear_mdct_power(freqs, M):
    """ATH as linear power in the normalized MDCT domain.

    The unnormalized transform used offset 96 + 10*log10(M/2). Coefficients
    here are smaller by s = sqrt(2/M), so power thresholds drop by s^2 and the
    offset grows by another 10*log10(M/2).
    """
    f = np.maximum(freqs, 20.0)
    fk = f / 1000.0
    ath_db = (
        3.64 * fk ** -0.8
        - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2)
        + 1e-3 * fk ** 4
    )
    offset = 96.0 + 20.0 * math.log10(max(M / 2.0, 1.0))
    return 10.0 ** ((ath_db - offset) / 10.0)


# =============================================================================
# 3. Quantization
# =============================================================================
def perceptual_quantize(mdct_coeffs, threshold, sfb_limits, scale_mult=1.0, max_iter=16):
    """Largest step whose MSE is within scale_mult * threshold, then widened
    so every index lies in [-MAX_IDX, MAX_IDX]."""
    n = mdct_coeffs.size
    quant_idx = np.zeros(n, dtype=np.int32)
    step_sizes = np.zeros(len(sfb_limits), dtype=np.float32)
    adjusted = np.maximum(threshold * scale_mult, 0.0)

    for i, (s, e) in enumerate(sfb_limits):
        coeffs = mdct_coeffs[s:e]
        if coeffs.size == 0:
            continue
        peak = float(np.max(np.abs(coeffs)))
        if peak < 1e-12:
            continue
        target = float(adjusted[i])
        lo, hi = 0.0, peak * 2.0
        for _ in range(max_iter):
            delta = 0.5 * (lo + hi)
            if delta <= 0.0:
                break
            err = float(np.mean((coeffs - np.round(coeffs / delta) * delta) ** 2))
            if err <= target:
                lo = delta  # still masked; try coarser
            else:
                hi = delta
        delta = 0.5 * (lo + hi)
        if delta <= 0.0:
            delta = peak / MAX_IDX
        # Alphabet limit: coarsen rather than clip.
        min_delta = peak / MAX_IDX
        if delta < min_delta:
            delta = min_delta
        idx = np.rint(coeffs / delta).astype(np.int32)
        idx = np.clip(idx, -MAX_IDX, MAX_IDX)
        quant_idx[s:e] = idx
        step_sizes[i] = np.float32(delta)
    return quant_idx, step_sizes


# =============================================================================
# 4. RLE + Huffman
# =============================================================================
def run_length_encode(idx):
    rle = []
    i, L = 0, idx.size
    while i < L:
        v = int(idx[i])
        j = i + 1
        while j < L and int(idx[j]) == v:
            j += 1
        rle.append((v, j - i))
        i = j
    return rle


def run_length_decode(rle, length):
    out = np.zeros(length, dtype=np.int32)
    pos = 0
    for v, run in rle:
        end = min(pos + run, length)
        if end > pos:
            out[pos:end] = v
        pos = end
        if pos >= length:
            break
    return out


class HuffmanNode:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None):
        self.freq = freq
        self.symbol = symbol
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_codes(symbols):
    freq = Counter(symbols)
    if not freq:
        return {}
    if len(freq) == 1:
        return {next(iter(freq)): ""}
    heap = [HuffmanNode(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        p = HuffmanNode(a.freq + b.freq)
        p.left, p.right = a, b
        heapq.heappush(heap, p)
    codes = {}

    def walk(node, code):
        if node.symbol is not None:
            codes[node.symbol] = code
            return
        walk(node.left, code + "0")
        walk(node.right, code + "1")

    walk(heap[0], "")
    return codes


def huffman_encode(sequence, codes):
    return "".join(codes[s] for s in sequence)


def huffman_decode(bitstring, root, symbol_count):
    if symbol_count <= 0:
        return []
    if root.symbol is not None:
        return [root.symbol] * symbol_count
    result = []
    node = root
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node is None:
            break
        if node.symbol is not None:
            result.append(node.symbol)
            node = root
            if len(result) == symbol_count:
                break
    return result


def _pack_code(code):
    clen = len(code)
    if clen == 0:
        return clen, b""
    return clen, int(code, 2).to_bytes((clen + 7) // 8, "big")


def _unpack_code(clen, raw):
    if clen == 0:
        return ""
    return format(int.from_bytes(raw, "big"), f"0{clen}b")


def encode_frame_body(steps, rle):
    """Exact on-disk body. Rate control measures this, then writes it."""
    codes = build_huffman_codes(rle)
    bits = huffman_encode(rle, codes)
    buf = bytearray()
    buf += struct.pack(f"<{len(steps)}f", *np.asarray(steps, dtype=np.float32))
    buf += struct.pack("<I", len(codes))
    for (val, run), code in codes.items():
        buf += struct.pack("<iI", int(val), int(run))
        clen, raw = _pack_code(code)
        buf += struct.pack("<H", clen)
        buf += raw
    buf += struct.pack("<I", len(rle))
    buf += struct.pack("<I", len(bits))
    if bits:
        buf += int(bits, 2).to_bytes((len(bits) + 7) // 8, "big")
    return bytes(buf)


def decode_frame_body(f, num_sfb, M, sfb_limits):
    steps = np.array(struct.unpack(f"<{num_sfb}f", f.read(4 * num_sfb)), dtype=np.float64)
    ncodes = struct.unpack("<I", f.read(4))[0]
    root = HuffmanNode(0)
    for _ in range(ncodes):
        val, run = struct.unpack("<iI", f.read(8))
        clen = struct.unpack("<H", f.read(2))[0]
        raw = f.read((clen + 7) // 8)
        code = _unpack_code(clen, raw)
        if not code:
            root.symbol = (val, run)
            continue
        node = root
        for bit in code:
            nxt = node.left if bit == "0" else node.right
            if nxt is None:
                nxt = HuffmanNode(0)
                if bit == "0":
                    node.left = nxt
                else:
                    node.right = nxt
            node = nxt
        node.symbol = (val, run)
    nsym = struct.unpack("<I", f.read(4))[0]
    blen = struct.unpack("<I", f.read(4))[0]
    nbytes = (blen + 7) // 8
    data = f.read(nbytes) if nbytes else b""
    bitstring = "".join(f"{b:08b}" for b in data)[:blen]
    symbols = huffman_decode(bitstring, root, nsym)
    idx = run_length_decode(symbols, M)
    Xq = np.zeros(M, dtype=np.float64)
    for i, (s, e) in enumerate(sfb_limits):
        Xq[s:e] = idx[s:e].astype(np.float64) * float(steps[i])
    return Xq


# =============================================================================
# 5. Encoder
# =============================================================================
def _pad_to_frame_boundary(n, M, N):
    if n < N:
        return N - n
    return (M - (n % M)) % M


def _select_scale(X, thr, sfb_limits, budget_bytes):
    """Finest scale_mult whose packed frame fits in budget_bytes.

    scale_mult > 1 coarsens (raises the allowed noise).
    """
    lo, hi = 0.05, 256.0
    best = None
    best_body = None
    for _ in range(10):
        mid = math.sqrt(lo * hi)
        idx, steps = perceptual_quantize(X, thr, sfb_limits, scale_mult=mid)
        body = encode_frame_body(steps, run_length_encode(idx))
        if len(body) > budget_bytes:
            lo = mid
        else:
            hi = mid
            best = mid
            best_body = body
    if best_body is None:
        idx, steps = perceptual_quantize(X, thr, sfb_limits, scale_mult=hi)
        best_body = encode_frame_body(steps, run_length_encode(idx))
    return best_body


def encode_wav(input_wav, output_bit, target_kbps=96.0):
    fs, audio = wavfile.read(input_wav)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype != np.int16:
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    audio = audio.astype(np.float64) / 32768.0
    orig_len = int(len(audio))

    M, N = 1024, 2048
    sfb_limits = get_sfb_limits(M)
    num_sfb = len(sfb_limits)
    pad_len = _pad_to_frame_boundary(orig_len, M, N)
    padded = np.pad(audio, (0, pad_len), mode="constant")
    frames = [padded[s : s + N] for s in range(0, len(padded) - N + 1, M)]
    if not frames:
        raise ValueError("no frames")

    frame_bytes_target = target_kbps * 1000.0 * (M / fs) / 8.0
    reservoir_bytes = 0.0
    max_reservoir = frame_bytes_target * 4.0

    freqs = mdct_bin_freqs(M, fs)
    sfb_bark = np.array([np.mean(compute_bark(freqs[s:e])) for (s, e) in sfb_limits])
    ath_sfb = np.array(
        [np.mean(ath_linear_mdct_power(freqs[s:e], M)) for (s, e) in sfb_limits]
    )

    mdct = MDCT_TDAC(N=N, beta=8.0)
    mdct.reset()
    prepared = []
    for frame in frames:
        X = mdct.forward(frame)
        energy = np.array([np.sum(X[s:e] ** 2) for (s, e) in sfb_limits])
        tonality = compute_tonality(X, sfb_limits)
        offset_db = 6.0 * tonality - 6.0 * (1.0 - tonality)
        spread = np.zeros(num_sfb)
        for i in range(num_sfb):
            if energy[i] > 1e-20:
                spread += spread_masker(sfb_bark[i], energy[i], sfb_bark)
        thr = np.maximum(spread * 10.0 ** (offset_db / 10.0), ath_sfb)
        prepared.append((X, thr))

    bodies = []
    for X, thr in prepared:
        budget = frame_bytes_target + reservoir_bytes
        budget = max(budget, 32.0)
        body = _select_scale(X, thr, sfb_limits, budget)
        reservoir_bytes += frame_bytes_target - len(body)
        reservoir_bytes = float(np.clip(reservoir_bytes, -max_reservoir, max_reservoir))
        bodies.append(body)

    with open(output_bit, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIII", fs, len(bodies), M, num_sfb, orig_len))
        for body in bodies:
            f.write(struct.pack("<I", len(body)))
            f.write(body)

    duration = orig_len / fs
    total = os.path.getsize(output_bit)
    print(
        f"Encoded {len(bodies)} frames, target {target_kbps:.1f} kbps, "
        f"file {total * 8 / duration / 1000:.1f} kbps -> {output_bit}"
    )


# =============================================================================
# 6. Decoder
# =============================================================================
def decode_bit(input_bit, output_wav):
    with open(input_bit, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("Not a PCD6 bitstream")
        fs, num_frames, M, num_sfb, orig_len = struct.unpack("<IIIII", f.read(20))
        sfb_limits = get_sfb_limits(M)
        mdct = MDCT_TDAC(N=2 * M, beta=8.0)
        mdct.reset()
        audio = []
        for _ in range(num_frames):
            blen = struct.unpack("<I", f.read(4))[0]
            start = f.tell()
            Xq = decode_frame_body(f, num_sfb, M, sfb_limits)
            used = f.tell() - start
            if used != blen:
                f.seek(start + blen)
            audio.append(mdct.inverse(Xq))
        pcm = np.concatenate(audio + [mdct.prev_imdct])[:orig_len]

    pcm16 = np.clip(pcm * 32768.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(output_wav, fs, pcm16)
    print(f"Decoded {orig_len} samples -> {output_wav}")


# =============================================================================
# 7. Self-test
# =============================================================================
def _snr_db(ref, test):
    n = min(len(ref), len(test))
    ref = ref[:n].astype(np.float64)
    test = test[:n].astype(np.float64)
    return 10.0 * math.log10(
        (np.sum(ref ** 2) + 1e-20) / (np.sum((ref - test) ** 2) + 1e-20)
    )


def _assert_pr():
    mdct = MDCT_TDAC()
    M = mdct.M
    rng = np.random.default_rng(1)
    nframes = 6
    x = rng.standard_normal((nframes + 1) * M)
    outs = []
    mdct.reset()
    for i in range(nframes):
        outs.append(mdct.inverse(mdct.forward(x[i * M : i * M + mdct.N])))
    rec = np.concatenate(outs + [mdct.prev_imdct])
    err = np.max(np.abs(x[M:-M] - rec[M:-M]))
    if err > 1e-8:
        raise AssertionError(f"MDCT PR failed, max error {err}")


def selftest(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    _assert_pr()
    fs = 44100
    t = np.arange(fs * 2) / fs
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    tone *= np.hanning(len(tone)) ** 0.1
    rng = np.random.default_rng(0)
    sig = np.clip(tone + 0.02 * rng.standard_normal(len(t)), -1, 1)

    in_wav = os.path.join(outdir, "selftest_in.wav")
    bit = os.path.join(outdir, "selftest.bit")
    out_wav = os.path.join(outdir, "selftest_out.wav")
    wavfile.write(in_wav, fs, (sig * 32767).astype(np.int16))
    encode_wav(in_wav, bit, target_kbps=96.0)
    decode_bit(bit, out_wav)

    _, ref = wavfile.read(in_wav)
    _, got = wavfile.read(out_wav)
    assert len(ref) == len(got), f"length mismatch: {len(ref)} vs {len(got)}"
    rms_in = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(got.astype(np.float64) ** 2)))
    ratio = rms_out / max(rms_in, 1.0)
    assert 0.9 <= ratio <= 1.1, f"RMS ratio out of range: {ratio:.3f}"
    print(f"in={len(ref)} out={len(got)}")
    print(f"SNR = {_snr_db(ref, got):.2f} dB  |  RMS ratio = {ratio:.3f}")
    in_bytes = os.path.getsize(in_wav) - 44
    bit_bytes = os.path.getsize(bit)
    duration = len(ref) / fs
    print(
        f"PCM {in_bytes} B  bitstream {bit_bytes} B  "
        f"ratio {in_bytes / max(bit_bytes, 1):.2f}x  "
        f"bitrate {bit_bytes * 8 / duration / 1000:.1f} kbps"
    )


def _usage():
    print(
        "Usage:\n"
        "  python perceptual_codec_5_6.py encode input.wav output.bit [target_kbps]\n"
        "  python perceptual_codec_5_6.py decode input.bit output.wav\n"
        "  python perceptual_codec_5_6.py selftest [outdir]"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "encode" and len(sys.argv) >= 4:
        kbps = float(sys.argv[4]) if len(sys.argv) >= 5 else 96.0
        encode_wav(sys.argv[2], sys.argv[3], target_kbps=kbps)
    elif cmd == "decode" and len(sys.argv) == 4:
        decode_bit(sys.argv[2], sys.argv[3])
    elif cmd == "selftest":
        selftest(sys.argv[2] if len(sys.argv) >= 3 else ".")
    else:
        _usage()
        sys.exit(1)
