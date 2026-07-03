"""Optional Tenstorrent benchmark integration points."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


def is_tenstorrent_available() -> bool:
    """Return whether a supported Tenstorrent Python runtime is importable."""
    return find_spec("ttnn") is not None or find_spec("tt-metalium") is not None


@dataclass(frozen=True, slots=True)
class TenstorrentBenchmark:
    """Placeholder for optional Tenstorrent experiments."""

    experiment_name: str = "tenstorrent_optional"

    def run(self) -> None:
        """Run an optional Tenstorrent benchmark when the runtime is available."""
        if not is_tenstorrent_available():
            raise RuntimeError(
                "Tenstorrent runtime is not available. Install the optional "
                "Tenstorrent toolchain on supported hardware to run this benchmark."
            )

        raise NotImplementedError(
            "Tenstorrent benchmarking is optional and has no implemented model "
            "runtime in this thesis benchmark codebase yet."
        )


__all__ = ["TenstorrentBenchmark", "is_tenstorrent_available"]
