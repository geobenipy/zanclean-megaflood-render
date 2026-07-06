"""
Assembles a PNG frame sequence into an H.264 MP4.

Tries imageio.v2 first; falls back to a direct subprocess ffmpeg call so
the script works regardless of imageio version (v2 vs v3 API differences).
"""

from pathlib import Path
import logging
import re
import subprocess
import tempfile
import os
import time

import imageio.v2 as iio
from tqdm import tqdm


# =============================================================================
# PARAMETERS
# =============================================================================

INPUT_FOLDER = Path(
    r"output\anim"
)

OUTPUT_VIDEO = INPUT_FOLDER / "animation.mp4"

FPS = 30

# H.264 quality: lower = bigger/better (18 = visually lossless, 23 = default, 28 = small)
CRF = 18

# Encoding speed/quality trade-off (slower = better compression at same CRF)
PRESET = "slow"

OVERWRITE = True

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


# =============================================================================
# LOGGING  –  goes to INPUT_FOLDER so it sits next to the frames
# =============================================================================

LOG_FILE = INPUT_FOLDER / "video_generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def natural_sort_key(path: Path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.stem)]


def collect_images(folder: Path):
    imgs = []
    for ext in IMAGE_EXTENSIONS:
        imgs.extend(folder.glob(ext))
    return sorted(imgs, key=natural_sort_key)


def _encode_imageio(images, output, fps, crf, preset):
    """Encode via imageio.v2 (stable legacy API, works with imageio-ffmpeg)."""
    ffmpeg_params = [
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-movflags", "+faststart",
    ]
    writer = iio.get_writer(str(output), fps=fps, codec="libx264",
                            format="ffmpeg", ffmpeg_params=ffmpeg_params)
    first_shape = None
    skipped = 0
    for img_path in tqdm(images, desc="  encoding", unit="frame",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        try:
            frame = iio.imread(str(img_path))
            if first_shape is None:
                first_shape = frame.shape[:2]
            if frame.shape[:2] != first_shape:
                logger.warning(f"size mismatch, skipping {img_path.name}")
                skipped += 1
                continue
            writer.append_data(frame)
        except Exception:
            logger.exception(f"failed reading {img_path.name}")
            skipped += 1
    writer.close()
    return skipped


def _encode_subprocess(images, output, fps, crf, preset):
    """Encode via ffmpeg subprocess (fallback; ffmpeg must be on PATH)."""
    # Write a concat list file so frame ordering/naming doesn't matter
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in images:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-r", str(fps),
        "-i", list_path,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-movflags", "+faststart",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg stderr:\n{result.stderr[-3000:]}")
            raise RuntimeError(f"ffmpeg exited {result.returncode}")
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info(f"  zanclean flood – video assembly")
    logger.info("=" * 60)
    logger.info(f"  input  : {INPUT_FOLDER}")
    logger.info(f"  output : {OUTPUT_VIDEO}")
    logger.info(f"  fps={FPS}  crf={CRF}  preset={PRESET}")

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    images = collect_images(INPUT_FOLDER)
    if not images:
        raise RuntimeError("No images found in input folder.")

    logger.info(f"  frames : {len(images):,}")

    if OUTPUT_VIDEO.exists() and OVERWRITE:
        OUTPUT_VIDEO.unlink()

    # -- Primary: imageio.v2 ------------------------------------------------
    skipped = 0
    try:
        skipped = _encode_imageio(images, OUTPUT_VIDEO, FPS, CRF, PRESET)
        if not OUTPUT_VIDEO.exists():
            raise RuntimeError("imageio produced no output file.")
    except Exception as e1:
        logger.warning(f"imageio encoding failed ({e1!s:.120}) – trying subprocess ffmpeg …")
        try:
            _encode_subprocess(images, OUTPUT_VIDEO, FPS, CRF, PRESET)
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found on PATH.\n"
                "  Install option A: pip install imageio-ffmpeg\n"
                "  Install option B: https://ffmpeg.org/download.html (add to PATH)"
            )
            raise
        except Exception as e2:
            logger.error(f"subprocess encoding also failed: {e2}")
            raise

    dt = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"  done in {dt:.1f} s  |  skipped {skipped} frames")
    logger.info(f"  saved  : {OUTPUT_VIDEO}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
