import queue
import unittest
from unittest.mock import patch

from perception_audio_codec.pipeline import PCD6Worker


class PCD6WorkerTests(unittest.TestCase):
    def test_encode_job_runs_on_worker_and_completes_future(self):
        with patch("perception_audio_codec.pipeline.PCD6Encoder.encode_file") as encode:
            with PCD6Worker() as worker:
                result = worker.encode_file("in.wav", "out.pcd6")
                self.assertIsNone(result.result(timeout=2))
            encode.assert_called_once_with("in.wav", "out.pcd6")

    def test_worker_propagates_codec_errors_to_future(self):
        with patch("perception_audio_codec.pipeline.PCD6Encoder.decode_file",
                   side_effect=ValueError("bad stream")):
            with PCD6Worker() as worker:
                result = worker.decode_file("bad.pcd6", "out.wav")
                with self.assertRaisesRegex(ValueError, "bad stream"):
                    result.result(timeout=2)

    def test_closed_worker_rejects_new_jobs(self):
        worker = PCD6Worker()
        worker.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.encode_file("in.wav", "out.pcd6")

    def test_queue_capacity_is_validated(self):
        with self.assertRaises(ValueError):
            PCD6Worker(max_pending=0)


if __name__ == "__main__":
    unittest.main()
