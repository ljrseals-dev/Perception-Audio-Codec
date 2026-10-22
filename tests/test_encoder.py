import unittest

import numpy as np

from perception_audio_codec import EncoderConfig, PerceptionEncoder


class PerceptionEncoderTests(unittest.TestCase):
    def test_encode_decode_preserves_length_rate_and_reasonable_signal(self):
        rate = 48000
        time = np.arange(rate // 4, dtype=np.float64) / rate
        source = 0.5 * np.sin(2 * np.pi * 997 * time)
        encoder = PerceptionEncoder(EncoderConfig(frame_size=256, band_count=16, quality=0.8))
        result = encoder.encode(source, rate)
        decoded, decoded_rate = encoder.decode(result.bitstream)

        self.assertEqual(decoded_rate, rate)
        self.assertEqual(decoded.size, source.size)
        self.assertEqual(result.input_samples, source.size)
        self.assertEqual(result.frame_count, (source.size + 255) // 256)
        self.assertTrue(np.all(np.isfinite(decoded)))
        error = source - decoded
        snr_db = 10 * np.log10(np.mean(source * source) / np.mean(error * error))
        self.assertGreater(snr_db, 35.0)

    def test_integer_stereo_input_policy_is_explicit(self):
        source = np.array([[16384, -16384], [0, 32767]], dtype=np.int16)
        encoder = PerceptionEncoder(EncoderConfig(frame_size=64, band_count=8, channel_policy="average"))
        result = encoder.encode(source, 44100)
        decoded, _ = encoder.decode(result.bitstream)
        self.assertEqual(result.original_channels, 2)
        self.assertEqual(decoded.size, 2)

    def test_rejects_bad_quality_and_invalid_pcm(self):
        with self.assertRaises(ValueError):
            EncoderConfig(quality=1.5)
        encoder = PerceptionEncoder(EncoderConfig(frame_size=64, band_count=8))
        with self.assertRaises(ValueError):
            encoder.encode(np.array([0.0, np.nan]), 48000)

    def test_rejects_wrong_magic_and_corrupt_payload(self):
        with self.assertRaises(ValueError):
            PerceptionEncoder.decode(b"NOPEpayload")
        valid = PerceptionEncoder(EncoderConfig(frame_size=64, band_count=8)).encode(
            np.ones(65, dtype=np.float64) * 0.1, 48000
        ).bitstream
        with self.assertRaises(ValueError):
            PerceptionEncoder.decode(valid[:-1] + bytes([valid[-1] ^ 0xFF]))

    def test_configuration_respects_binary_header_limits(self):
        with self.assertRaises(ValueError):
            EncoderConfig(frame_size=65536)
        with self.assertRaises(ValueError):
            EncoderConfig(frame_size=512, band_count=256)


if __name__ == "__main__":
    unittest.main()
