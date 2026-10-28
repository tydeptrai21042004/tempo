# TEMPO evidence expansion — changed-files patch

This patch adds experimental evidence only. No TEMPO watermark embedding or extraction core has been changed. Automated scientific gate enforcement is not part of this patch. Existing legacy validators may remain in the original repository but are not invoked by this launcher.

## Apply

Extract `TEMPO_EVIDENCE_CODE_CHANGES_ONLY.zip` into the root of your `tempo-main` repo, preserving the `scripts/` hierarchy. Install the project's existing requirements (see `requirements-kaggle.txt`). Use Python 3.10+ with OpenCV, Pillow, NumPy, Numba, Matplotlib and scikit-image as required by current code.

## Run on Kaggle

In a Kaggle notebook with the repository cloned and internet enabled for the official external benchmark:

```bash
cd /kaggle/working/tempo-main
export PYTHONPATH="$PWD/src:$PWD"
python scripts/run_tempo_evidence.py --output /kaggle/working/tempo-evidence --quick --only theory,capacity,security
# Full suite, requires the real video and external Kodak BMPs for all stages:
python scripts/run_tempo_evidence.py --output /kaggle/working/tempo-evidence --resume \
    --video /kaggle/input/my-videos/video.mp4 \
    --external-host-dir /kaggle/input/external-images \
    --mm-host-dir /kaggle/working/kodak_bmp_hosts \
    --video-corpus /kaggle/working/tempo-video-corpus/corpus.json
```

Use `--only` to run selected stages independently (e.g. `theory,security,detection,sync`), `--resume` to skip completed stages with unchanged commands/scripts, and `--no-zip` to skip the compact archive. A suite can be **partial** because video inputs, external images, or official Git repositories are unavailable; the stage ledger records that rather than inventing data. A stage failure does not label a scientific conclusion as invalid: it simply records that execution did not finish.

For the original MMWatermark **paper-100** mode, the official upstream and companion repos are cloned from Jielin-Qiu's GitHub repositories, the source is not changed, and the original 20 families (Noise 4, Blur 4, Weather 4, Digital 4, Geometric 4) at severities 1–5 are selected. This is **not** a reproduction of original authors' watermark model weights, datasets or payload settings. The default external integration remains TEMPO-proposal-only; internal fair baselines are run in the `baselines` stage. If the upstream Frost attack/assets cannot be run, the stage reports incomplete coverage rather than substituting another attack. Original-source extra attacks are only available separately (`--protocol source_all` at the low-level runner).

The new `verification` stage adds held-out attack-conditioned known-message decisions and independent direct/synchronization-aware thresholds. `key_schedule` provides sampled schedule-overlap measurements (not cryptographic key-space certification). `learned_native` accepts only independently executed model results and signed provenance; it does not synthesize attack-model results. The detection stage calibrates the known-message threshold on the *development-only* host split; the default split is deterministic by sorted BMP filename and reports source hashes. Its FPR/TPR results are finite-sample descriptive estimates. The security stage intentionally includes wrong-key, wrong-message, unmarked, wrong-host, image transplant, overwrite, two-copy collusion, public-carrier attenuation, JPEG and geometric attacks, **not** certified cryptographic authentication or proof of resistance to adaptive attackers.

The `video_baselines` stage uses the repository's existing method implementations on real video. The optional `video_corpus` stage uses a verified multi-clip JSON manifest for both framewise and causal-stream comparisons. The video-security stage uses actual decoded video frames and reports temporal attacks, majority voting and delays in frame-arrival units. It does not fabricate a source video or claim that video reordering is detectable when the same message is repeated each frame. The original causal-video benchmarks remain available in the repo's existing video scripts; this new stage complements rather than replaces them.

The theory stage numerically tests simultaneous orthogonal carrier update and measures rounding/clipping behavior. It is not a mathematical proof or integer global optimality certificate. The full capacity stage compares 256, 1024, 2304 and 4096 bits (square watermarks), with the **full internal 45 attacked conditions**, 2 watermarks and all held-out hosts; no unseen results are prefilled.

## ZIP results — numerical evidence only

`<output>_PAPER_EVIDENCE.zip` contains CSV, JSON, Markdown, LaTeX, and selected figure PDF files. It does **not** package the raw source videos, attacked/watermarked/extracted image collections, source corpora, model weights, checkpoints, or gigantic logs. The code patch ZIP likewise includes **only changed/added files**, not the entire repository. Every completed stage retains its numerical rows; figures can be regenerated from the measured data.

## Existing experiments available separately

The project already has `scripts/kaggle_pipeline.sh`, `scripts/run_video_corpus_suite.py`, `scripts/run_multiwatermark_video.py`, and other full experiments. This patch avoids altering the existing core pipeline, and does not claim to have executed those expensive external/video tests during patch development.
