# Optional Tenstorrent Benchmarking

Tenstorrent support is intentionally optional. The core benchmark suite has no
hard dependency on Tenstorrent hardware, drivers, or Python packages.

Use `financial_volatility.benchmark.is_tenstorrent_available()` to detect
whether a supported runtime is importable. If it is unavailable, Tenstorrent
experiments should be skipped and the standard CPU/CUDA thesis benchmarks should
continue to run normally.

The current project does not include a Tenstorrent model runtime. Future work can
extend `TenstorrentBenchmark` once compatible hardware and toolchains are
available.
