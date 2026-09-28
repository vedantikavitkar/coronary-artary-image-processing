## Latency benchmark

| Pipeline | Resolution | Frames | Avg (ms) | Min (ms) | Max (ms) | P95 (ms) | FPS | Budget (ms) | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| realtime | 512x512 | 400 | 16.59 | 11.12 | 33.19 | 21.22 | 60.3 | 36 | PASS |

**realtime — per-stage mean latency (ms)**

| Stage | Mean (ms) |
| --- | --- |
| vessel_enhancement | 4.35 |
| output | 4.23 |
| background_subtraction | 2.34 |
| rib_vessel_cleanup | 1.74 |
| preprocess | 1.34 |
| ingest | 0.95 |

**realtime — 27.3 ms of CPU time per frame summed over all threads**

**Hardware / software configuration**

- cpu: 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz
- logical cores: 8
- ram gb: 7.8
- platform: Windows 10 (AMD64)
- python: 3.11.4
- torch: 2.5.1+cpu
- torch threads: 4
- cuda available: False
- opencv: 4.10.0
- opencv threads: 8
- numpy: 1.26.4

