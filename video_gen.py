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
    r"\output_final\anim_16x9"
)
# The 3:2 clip's frames are in output_final\anim next door. Keep the two apart: both number
# their frames from 1, so a folder mix-up produces a video that silently changes aspect ratio
# partway through rather than failing.

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

# A SECOND, size-capped file for submission. Nature allows at most 30 MB per supplementary
# video, and a CRF-16 master of a 30 s cascade goes well past that - CRF is a quality target,
# it makes no promise at all about size. So the master above stays near-lossless (it is the
# archival copy, and the one to re-encode from if the journal asks for another format), and
# this writes `<name>_SI.mp4` at a bitrate computed from the clip's own length so it lands
# under the limit. Two-pass, because a one-pass average is only an average: it overshoots on
# exactly the passage that matters here, the plunge at full discharge.
# Set to None to skip the second file.
SI_MAX_MB = 30
SI_SAFETY = 0.93            # aim this fraction of the limit - container overhead, and a file
                            # that lands at 30.4 MB is as rejected as one at 60
SI_MIN_KBPS = 2000          # below this the water breaks up into blocks; if the target falls
                            # under it the clip is too long for the limit, and the honest fix
                            # is a shorter clip, not a worse one. Warned about, not silently
                            # accepted.

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


def _ffmpeg_exe():
    """Path to an ffmpeg binary.

    Plain "ffmpeg" is NOT on PATH on this machine - imageio-ffmpeg ships its own and that is
    what the primary encoder has been using all along. Hard-coding the bare name meant the
    fallback path could only ever fail with FileNotFoundError, i.e. the fallback was not one.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _concat_list(images):
    """A concat-demuxer list file, so frame naming/ordering doesn't matter."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in images:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
        return f.name


def _input_args(images, fps):
    """ffmpeg input arguments for the sequence, plus a temp file to delete afterwards.

    Prefers the image2 demuxer over a concat list. The concat demuxer has no notion of frame
    duration for stills, so it emits garbage timestamps ("PTS -230584300921369 invalid
    dropping", thousands of lines) and ffmpeg has to reconstruct the timing; image2 with
    -framerate gets it right by construction. Blender always writes frame_NNNN.png, so the
    fast path is the normal one - the concat fallback is there for a hand-assembled folder.
    """
    stems = [p.stem for p in images]
    m = [re.match(r"^(.*?)(\d+)$", s) for s in stems]
    if all(m) and len({x.group(1) for x in m}) == 1 and len({len(x.group(2)) for x in m}) == 1:
        nums = [int(x.group(2)) for x in m]
        if nums == list(range(nums[0], nums[0] + len(nums))):
            prefix, width = m[0].group(1), len(m[0].group(2))
            pattern = str(images[0].parent / f"{prefix}%0{width}d{images[0].suffix}")
            return (["-framerate", str(fps), "-start_number", str(nums[0]), "-i", pattern],
                    None)
    list_path = _concat_list(images)
    return ["-f", "concat", "-safe", "0", "-r", str(fps), "-i", list_path], list_path


def _run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg stderr:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"ffmpeg exited {result.returncode}")


def _encode_subprocess(images, output, fps, crf, preset):
    """Encode via ffmpeg subprocess (fallback for the imageio path)."""
    in_args, list_path = _input_args(images, fps)
    cmd = [
        _ffmpeg_exe(), "-y",
        *in_args,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", PROFILE,
        "-movflags", "+faststart",
        str(output),
    ]
    try:
        _run_ffmpeg(cmd)
    finally:
        if list_path:
            try:
                os.unlink(list_path)
            except OSError:
                pass


def _encode_size_capped(images, output, fps, preset, max_mb, safety=SI_SAFETY):
    """Two-pass ABR encode aimed at `max_mb` megabytes. Returns the achieved size in MB.

    Encoded from the PNGs, not transcoded from the master: the master is already lossy, and
    re-compressing its artefacts is strictly worse than compressing the original frames once.
    """
    duration = max(len(images) / float(fps), 1e-6)
    kbps = int((max_mb * safety * 8 * 1000 * 1000) / duration / 1000)
    if kbps < SI_MIN_KBPS:
        logger.warning(f"  {max_mb} MB over {duration:.1f} s leaves only {kbps} kbit/s - "
                       f"below the {SI_MIN_KBPS} kbit/s floor where moving water starts to "
                       f"block up. Shorten the clip rather than ship this.")
    logger.info(f"  SI copy: {duration:.1f} s at {kbps} kbit/s  -> target ~{max_mb * safety:.0f} MB")

    in_args, list_path = _input_args(images, fps)
    # x264's two-pass state goes in a log file next to the output, not in the temp dir, so a
    # failed run leaves it where it can be seen rather than silently in %TEMP%.
    passlog = str(Path(output).with_suffix("")) + "_2pass"
    common = [
        _ffmpeg_exe(), "-y",
        *in_args,
        "-c:v", "libx264",
        "-preset", preset,
        "-b:v", f"{kbps}k",
        "-pix_fmt", "yuv420p",
        "-profile:v", PROFILE,
        "-passlogfile", passlog,
    ]
    try:
        _run_ffmpeg(common + ["-pass", "1", "-an", "-f", "mp4", os.devnull])
        _run_ffmpeg(common + ["-pass", "2", "-movflags", "+faststart", str(output)])
    finally:
        if list_path:
            try:
                os.unlink(list_path)
            except OSError:
                pass
        for suffix in ("-0.log", "-0.log.mbtree", ".log", ".log.mbtree"):
            try:
                os.unlink(passlog + suffix)
            except OSError:
                pass
    return Path(output).stat().st_size / 1e6


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

    master_mb = OUTPUT_VIDEO.stat().st_size / 1e6
    logger.info(f"  master : {master_mb:.1f} MB")

    si_path = None
    if SI_MAX_MB:
        si_path = OUTPUT_VIDEO.with_name(OUTPUT_VIDEO.stem + "_SI.mp4")
        if master_mb <= SI_MAX_MB * SI_SAFETY:
            logger.info(f"  master already fits {SI_MAX_MB} MB - no separate SI copy needed")
            si_path = None
        else:
            si_mb = _encode_size_capped(images, si_path, FPS, PRESET, SI_MAX_MB)
            logger.info(f"  SI     : {si_mb:.1f} MB  ({si_path.name})")
            if si_mb > SI_MAX_MB:
                logger.warning(f"  SI copy is {si_mb:.1f} MB, OVER the {SI_MAX_MB} MB limit - "
                               f"lower SI_SAFETY and re-run")

    dt = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"  done in {dt:.1f} s  |  skipped {skipped} frames")
    logger.info(f"  saved  : {OUTPUT_VIDEO}")
    if si_path:
        logger.info(f"           {si_path}  <- submit this one")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
