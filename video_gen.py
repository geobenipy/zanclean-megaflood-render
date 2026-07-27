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

# Where render_waterfall.py's ANIM_DIR wrote the frames (its default is DATA_DIR/anim).
INPUT_FOLDER = Path(
    r"."
    r"\output_overview_old\output_overview\anim"
)

OUTPUT_VIDEO = INPUT_FOLDER / "zanclean_flood.mp4"

FPS = 25                    # must match ANIM_FPS in render_waterfall.py, or the clip's
                            # timing (the hold, the advance) is not what was keyframed

# H.264 quality: lower = bigger/better (18 = visually lossless, 23 = default, 28 = small).
# Keep this near-lossless: journals re-encode on ingest, and compressing twice at 23 turns
# the cascade's fine whitewater into mush.
CRF = 16

# Encoding speed/quality trade-off (slower = better compression at same CRF)
PRESET = "slow"

OVERWRITE = True

# H.264 profile. NOT "baseline": it forbids CABAC and the 8x8 transform, which costs real
# quality at a given size, and it only ever existed for 2010-era mobile decoders.
PROFILE = "high"

# Warn if the sequence has gaps. ANIM_RESUME lets an interrupted render be restarted, and a
# frame that never got written would otherwise be encoded as a silent jump cut.
CHECK_SEQUENCE = True

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


# =============================================================================
# LOGGING  –  goes to INPUT_FOLDER so it sits next to the frames
# =============================================================================

LOG_FILE = INPUT_FOLDER / "video_generation.log"

_handlers = [logging.StreamHandler()]
if INPUT_FOLDER.is_dir():
    # only attach the file handler if the folder is really there - otherwise basicConfig
    # raises at import time and hides the readable "input folder not found" from main()
    _handlers.append(logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=_handlers,
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


def check_sequence(images):
    """Report gaps and empty frames before spending an encode on them.

    Blender names frames frame_0001.png…; a resumed render (ANIM_RESUME) can leave a hole if
    a frame died mid-write or the run was stopped between two frames. Concatenating over the
    hole produces a jump cut that is very easy to miss on a first viewing and impossible to
    explain in a caption.
    """
    nums = []
    for p in images:
        m = re.search(r"(\d+)$", p.stem)
        if m:
            nums.append((int(m.group(1)), p))
    if not nums:
        logger.info("  frames are not numbered - skipping the sequence check")
        return
    nums.sort()
    lo, hi = nums[0][0], nums[-1][0]
    have = {n for n, _ in nums}
    missing = [n for n in range(lo, hi + 1) if n not in have]
    empty = [p.name for _, p in nums if p.stat().st_size == 0]
    if missing:
        logger.warning(f"  MISSING {len(missing)} frame(s) in {lo}..{hi}: "
                       f"{missing[:12]}{' …' if len(missing) > 12 else ''}")
        logger.warning("  re-run the render with the same settings - ANIM_RESUME fills the holes")
    if empty:
        logger.warning(f"  EMPTY (0-byte) frame(s): {empty[:12]}")
    if not missing and not empty:
        logger.info(f"  sequence OK: {lo}..{hi}, no gaps")


def _encode_imageio(images, output, fps, crf, preset):
    """Encode via imageio.v2 (stable legacy API, works with imageio-ffmpeg)."""
    # pixelformat goes through imageio's own argument, NOT ffmpeg_params: imageio always
    # emits a -pix_fmt of its own, so passing a second one just makes ffmpeg warn about
    # duplicate options
    ffmpeg_params = [
        "-preset", preset,
        "-crf", str(crf),
        "-profile:v", PROFILE,
        "-movflags", "+faststart",
    ]
    writer = iio.get_writer(str(output), fps=fps, codec="libx264", pixelformat="yuv420p",
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
        "-profile:v", PROFILE,
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

    logger.info(f"  frames : {len(images):,}  ({len(images) / FPS:.1f} s at {FPS} fps)")
    if CHECK_SEQUENCE:
        check_sequence(images)

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
