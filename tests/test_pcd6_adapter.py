import unittest
from unittest.mock import patch

from perception_audio_codec.pcd6 import PCD6Encoder


class PCD6EncoderAdapterTests(unittest.TestCase):
    def test_rejects_nonpositive_nonfinite_or_excessive_bitrate(self):
        for value in (0, -1, float("nan"), float("inf"), 10001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PCD6Encoder(value)

    def test_accepts_valid_target_bitrate(self):
        self.assertEqual(PCD6Encoder(128).target_kbps, 128.0)

    def test_encode_delegates_to_reference_codec(self):
        encoder = PCD6Encoder(96)
        with patch("perception_codec_5_6.encode_wav") as encode:
            encoder.encode_file("in.wav", "out.pcd6")
        encode.assert_called_once_with("in.wav", "out.pcd6", target_kbps=96.0)

    def test_decode_delegates_to_reference_codec(self):
        with patch("perception_codec_5_6.decode_bit") as decode:
            PCD6Encoder().decode_file("in.pcd6", "out.wav")
        decode.assert_called_once_with("in.pcd6", "out.wav")

    def test_refuses_same_input_and_output_path(self):
        with self.assertRaises(ValueError):
            PCD6Encoder().encode_file("same.wav", "./same.wav")
        with self.assertRaises(ValueError):
            PCD6Encoder().decode_file("same.pcd6", "./same.pcd6")

    def test_rejects_empty_paths(self):
        with self.assertRaises(ValueError):
            PCD6Encoder().encode_file("", "out.pcd6")


if __name__ == "__main__":
    unittest.main()
