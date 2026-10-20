import unittest

import numpy as np

from perception_audio_codec import AudioProcessor, ProcessorConfig


class AudioProcessorTests(unittest.TestCase):
    def test_frame_count_and_final_padding(self):
        processor = AudioProcessor(ProcessorConfig(frame_size=8, hop_size=4))
        audio = processor.prepare(np.arange(10, dtype=np.float64), 48000)
        frames = list(processor.frames(audio))
        self.assertEqual([frame.start_sample for frame in frames], [0, 4, 8])
        self.assertEqual([frame.valid_samples for frame in frames], [8, 6, 2])
        np.testing.assert_array_equal(frames[-1].samples, [8, 9, 0, 0, 0, 0, 0, 0])

    def test_default_stereo_policy_preserves_channel_zero(self):
        data = np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float64)
        audio = AudioProcessor().prepare(data, 44100)
        np.testing.assert_array_equal(audio.samples, [0.25, 0.5])
        self.assertEqual(audio.source_channels, 2)
        self.assertEqual(audio.source_frames, 2)

    def test_average_stereo_policy_is_explicit(self):
        data = np.array([[1.0, -1.0], [0.5, 0.5]])
        audio = AudioProcessor(ProcessorConfig(channel_policy="average")).prepare(data, 48000)
        np.testing.assert_array_equal(audio.samples, [0.0, 0.5])

    def test_integer_pcm_is_scaled_to_float_range(self):
        audio = AudioProcessor().prepare(np.array([-32768, 0, 32767], dtype=np.int16), 48000)
        self.assertAlmostEqual(audio.samples[0], -1.0)
        self.assertAlmostEqual(audio.samples[1], 0.0)
        self.assertGreater(audio.samples[2], 0.99)
        self.assertLessEqual(audio.samples[2], 1.0)

    def test_rejects_invalid_rate_empty_and_nonfinite_audio(self):
        processor = AudioProcessor()
        with self.assertRaises(ValueError):
            processor.prepare(np.ones(8), 0)
        with self.assertRaises(ValueError):
            processor.prepare(np.array([]), 48000)
        with self.assertRaises(ValueError):
            processor.prepare(np.array([0.0, np.nan]), 48000)

    def test_config_rejects_invalid_hop(self):
        with self.assertRaises(ValueError):
            ProcessorConfig(frame_size=8, hop_size=9)

    def test_prepared_samples_do_not_alias_caller_memory(self):
        source = np.array([0.1, 0.2])
        prepared = AudioProcessor().prepare(source, 48000)
        source[0] = 0.9
        self.assertAlmostEqual(prepared.samples[0], 0.1)


if __name__ == "__main__":
    unittest.main()
