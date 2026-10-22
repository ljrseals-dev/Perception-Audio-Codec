# LewSpEw integrated PCD6 worker pipeline

This phase groups the existing PCD6 encoder behind one bounded worker interface.
It keeps the existing PCD6 bitstream implementation authoritative and avoids
introducing another incompatible format.

## Example

```python
from perception_audio_codec.pipeline import PCD6Worker

with PCD6Worker(target_kbps=96, max_pending=8) as pipeline:
    future = pipeline.encode_file("input.wav", "output.pcd6")
    future.result()  # propagates codec errors to the caller
```

The worker serializes file encode/decode/self-test operations on a dedicated
thread. Its bounded queue supplies explicit backpressure, and returned
`concurrent.futures.Future` objects carry completion and exceptions. Closing
the context manager stops new submissions, drains accepted jobs, and joins the
worker.

## Important boundaries

- This is an offline/file worker, not an in-memory PCM streaming encoder.
- Python's `queue.Queue` uses synchronization locks. It is **not lock-free** and
  is not suitable for direct use inside a WASAPI/ASIO real-time callback.
- Do not do file I/O, codec work, waiting on Futures, or queue operations from
  a real-time audio callback.
- The current reference codec is mono-oriented and selects channel zero from
  stereo input.
- A native low-latency device pipeline still needs a native in-memory PCM API,
  preallocated SPSC rings with verified memory ordering, a worker-side codec
  interface, and hardware loopback/latency/stress validation. This commit does
  not claim those parts are implemented.

## Intended topology

```text
WASAPI capture -> native input SPSC ring -> codec worker
               -> native output SPSC ring -> ASIO/DAW host buffers
ASIO/DAW host buffers -> native output SPSC ring -> WASAPI render
```

This topology is an integration target, not functionality provided by this
Python worker. The worker here is a safe organizational layer for offline PCD6
jobs while native real-time components are developed and validated separately.
