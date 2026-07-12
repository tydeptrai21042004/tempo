from __future__ import annotations

from pathlib import Path
from typing import Iterable
import urllib.request

import numpy as np

from .core import make_demo_host, make_demo_watermark, save_image


DEFAULT_VIDEO_URLS = (
    "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
    "https://samplelib.com/lib/preview/mp4/sample-5s.mp4",
)


def ensure_demo_image_assets(workdir: str | Path, *, host_size: int = 512, wm_size: int = 64) -> tuple[Path, Path]:
    """Create deterministic 512x512 RGB host and 64x64 binary watermark inside workdir."""
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    host_path = root / f"host_{host_size}x{host_size}.png"
    wm_path = root / f"wm{wm_size}.png"
    if not host_path.exists():
        save_image(host_path, make_demo_host(host_size, host_size))
    if not wm_path.exists():
        save_image(wm_path, make_demo_watermark(wm_size))
    return host_path, wm_path


def _create_synthetic_video(path: Path, *, width: int = 512, height: int = 512, frames: int = 16, fps: float = 12.0, codec: str = "mp4v") -> Path:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to create fallback demo videos. Install opencv-python-headless.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic demo video at {path}")
    base = make_demo_host(width, height)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    for i in range(int(frames)):
        frame = base.astype(np.int16).copy()
        motion = 18 * np.sin(2 * np.pi * (xx * 2.0 + yy * 1.5 + i / max(1, frames)))
        frame[..., 0] = np.clip(frame[..., 0] + motion, 0, 255)
        frame[..., 1] = np.clip(frame[..., 1] + i * 2, 0, 255)
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def ensure_demo_video(
    workdir: str | Path,
    *,
    input_path: str | Path | None = None,
    auto_download: bool = True,
    urls: Iterable[str] = DEFAULT_VIDEO_URLS,
    width: int = 512,
    height: int = 512,
    frames: int = 16,
) -> Path:
    """Return an MP4 path under workdir.

    If input_path is supplied it is returned unchanged. Otherwise, the function
    tries to download a small public MP4 into the folder. If download is not
    available, it creates a deterministic synthetic MP4, so tests work offline.
    """
    if input_path is not None:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    video_path = root / "demo_input.mp4"
    if video_path.exists() and video_path.stat().st_size > 0:
        return video_path

    if auto_download:
        for url in urls:
            try:
                urllib.request.urlretrieve(url, video_path)  # nosec - user-facing demo asset download
                if video_path.exists() and video_path.stat().st_size > 1024:
                    return video_path
            except Exception:
                if video_path.exists():
                    try:
                        video_path.unlink()
                    except OSError:
                        pass
                continue

    return _create_synthetic_video(video_path, width=width, height=height, frames=frames)
