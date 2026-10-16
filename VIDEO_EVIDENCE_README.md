# TEMPO: real-video evidence protocol (October 2026)

**This change is experimental orchestration only.** It does not change the embedder,
extractor, orthogonal carrier, QIM quantizer, integer correction, or phase estimator.

## Why

The previous `run_multiwatermark_video.py` generated a deterministic synthetic clip
when `--input` was not provided. This was fine for regression tests but it was not
independent video evidence. The upgraded `scripts/kaggle_pipeline.sh` refuses to
continue unless the *public video download and decode validation* succeeds.

## Automatically downloaded footage

The machine-readable video manifest is `configs/video_evidence_sources.json`. It
lists four public sample videos:

| Name | What it checks | Notes |
|---|---|---|
| `mdn_flower` | Short real temporal footage | Public MDN/CC0 example; used in `quick` |
| `bbb_motion_360` | Low-resolution motion, upsampling risk | Blender Foundation Big Buck Bunny, CC BY 3.0 |
| `sintel_trailer` | Scene cuts and realistic temporal variation | Blender Foundation Sintel, CC BY 3.0 |
| `bbb_native_720p` | Genuine 1280×720 source for 720p latency / JPEG70 | Large clip (~hundreds MB); download may be slow |

Actual availability, encoding and resolution are checked at execution, not assumed.
The 360p clip is **not** called native HD, even if benchmark frames are resized.
Credits apply to film creators; the listed host URLs should be checked for rights
before redistributing their encoded files. Downloaded videos are cached locally
and **are not bundled in the GitHub/source ZIP or results ZIP**.

## One-command full run (Kaggle CPU; Internet ON)

```bash
export VIDEO_FRAMES=180            # max sampled frames / clip; full corpus
export STREAM_FRAMES=180           # per clip/watermark wall-clock stream
export NUMBA_NUM_THREADS=4
bash scripts/kaggle_auto_video.sh full
```

The script:

1. Verifies the code and maintains separate development/test images.
2. Downloads each video over HTTPS, enforces a per-file byte limit, decodes a
   few frames, and records SHA-256, native size/fps and credits in `video_sources/corpus.json`.
   Bad/missing videos cause an **error**, never a synthetic fallback.
3. Normalizes frames for the 4,096-bit, 512×512 framewise baseline protocol using
   aspect-preserving letterboxing and records that transformation.
4. Runs the *two actual watermark files* against **all available common video
   methods** for every clip, storing pre-codec PSNR, per-frame NC/BER,
   running temporal-vote NC, serialized key overhead, and embed/extract
   kernel/algorithm/end-to-end timing.
5. Runs the actual source video through a bounded-queue wall-clock stream at 30 FPS
   with clean/JPEG90/JPEG70/JPEG40, Gaussian noise, blur and 10% random frame drops
   at 512×512, then additionally runs the
   genuinely native-HD clip at 1280×720. It logs FPS, p95 service/latency,
   frame drops, queue saturation, temporal vote NC and deadline misses.
6. Audits payload throughput using **4096 × observed effective output FPS**.
7. Checks evidence coverage and packages all numeric CSV/JSON outputs.

The complete pipeline also runs the existing image attacks, ROC/FPR, integer
rejection, matched-PSNR, confidence intervals, parameter sensitivity and
separate temporal error tests. Video baseline methods remain common framewise
adapters, not necessarily native implementations of their original papers.

## Quick real-video smoke test

```bash
VIDEO_FRAMES=24 STREAM_FRAMES=24 bash scripts/kaggle_pipeline.sh quick
```

`quick` downloads one real MDN clip and tests both watermarks in the video suite.
Quick mode limits stream channels to clean, JPEG70 and JPEG40.
It is NOT a complete submission benchmark; other image benchmark stages may still
run only one watermark in quick mode as documented in the main pipeline.

## Key video evidence outputs

- `video_sources/corpus.json`: exact URLs, checksums, licensing notes, native
  resolution/fps, decoded-frame validation, derived image resolution and frame count.
- `video/video_corpus_evidence_manifest.json`: clip-level completion and scope.
- `video/all_watermarks_video_summary.csv`: video-level NC, BER, PSNR, timing,
  FPS and temporal voting for **each clip × watermark × method**.
- `video/all_clips_per_frame.csv`: frame IDs, NC, bit accuracy and running NC.
- `video/video_corpus_stream_summary.csv`: all actual measured stream conditions.
- `video/video_corpus_coverage.csv`: complete clip × watermark presence.
- `live_stream/<clip>_<watermark>/ALL_STREAM_realtime_trace.csv`: measured per-frame
  wall-clock trace, not just an estimated FPS.
- `throughput_audit/wallclock_throughput_audit.csv`: checked throughput denominator.
- `reviewer_completeness.json`: fails video evidence fields if conditions are absent.

Results and provenance are packaged under the existing report directory; the
raw downloaded MP4s are cached separately to keep results archives small.

## Honest limits

The current pipeline is a **software-decoded real source video replay**, not
capture from a live camera. The source is decoded in advance before the stream
pacer starts; acquisition latency and hardware camera scheduling are not covered.

`precodec_mean_psnr` is PSNR *before* MP4 encoding. It must not be reported as
post-codec reconstructed-video PSNR. Live-stream PSNR is computed before the
attacking channel; decoder NC/BER are measured afterward. Do not conflate the
quantities. The framewise encoder runs `mp4v` unless configured otherwise.

Input resizing/letterboxing is reported, not hidden. Only the native-HD file
provides a real 720p source; artificially upscaled clips do not.

Passing this evidence protocol does not mean the method achieves 30 FPS or
corrects every attack. Negative results and missed deadlines are retained.

## Offline developer tests

No internet is available in some CI/container environments. These tests use an
explicit local test fixture, which `corpus.json` labels as such and **cannot
satisfy the reviewer real-download check**:

```bash
PYTHONPATH="$PWD:$PWD/src" python -m pytest -q tests/test_video_evidence_corpus.py
```

For custom video sources, edit or replace the manifest using
`VIDEO_CORPUS_MANIFEST=/path/to/video_sources.json`. Keep the HTTPS URLs, attribution,
required native height and download caps accurate. Cached file checksums are
recomputed and stored on every run; optionally set `sha256` in the manifest to
pin a precise version and fail on upstream file changes.
