# TEMPO: full external Kodak suite, proposal only

This incremental patch is based on the provided `tempo-main (1).zip` (commit `a839bb73...`). It **adds** external-dataset tools; it does not edit the proposed DCT-QIM/orthogonal-pilot embedding or decoder, change its hyperparameters, or claim to reproduce third-party watermarking methods. All 24 Kodak PCD0992 photographs are downloaded automatically; no external results CSV, Kaggle uploaded dataset, or manual watermark upload is needed.

## What “full dataset” means

- Kodak Lossless True Color Image Suite: **exactly 24 PNG files**, `kodim01.png` through `kodim24.png`.
- Native evaluation: original 768x512 / 512x768 images, **no resize or crop**, both existing 64x64 watermarks at 4096 independent bits, frozen selected configuration `configs/frozen_selected_20261021.json`.
- Built-in `basic` attacks: clean, JPEG quality 70, rotation 2.5° → **24×2×3 = 144 cases**.
- Optional built-in `unified109`: clean + **all 109 TEMPO attack conditions** → **24×2×110 = 5280 cases**. This is the TEMPO *local* attack registry, **not** the official third-party MMWatermark functions. Can be a long CPU run; checkpoints every 10 cases, and `--out-dir` can be reused to resume. Do not mix these profiles in one output directory.
- Optional **genuine official MMWatermark-Robustness** provider suite: separate original upstream repositories and unchanged attack functions, proposal-only, all **24 Kodak image identities**, two watermarks and 5 severities. This third-party repository uses a documented center 512x512 crop of each Kodak original for 512x512 compatibility, never pretends to evaluate every Kodak pixel. For the catalog previously observed in the provided Kaggle logs (23 attack functions), expected coverage is **24×2×(1+23×5) = 5568 rows**, subject to upstream catalog and validation; no trials are manufactured to fill gaps.
- Full 24-image downloaded-image integrity checks: PNG stream, native size, pixel color mode, SHA256, unique hashes, required filename coverage. Provenance includes download URLs and hashes; do **not** misrepresent locally computed SHA256 as an externally authenticated publisher checksum.

## Runtime and scientific reports

Algorithm time = time to embed + time for blind extraction after the selected attack. JPEG encoding/decoding, other attacks, PSNR/SSIM/NC/BER, frame ingestion and disk I/O are excluded **from algorithm-only timing**, not silently erased. Results include clean PSNR and SSIM; attacked NC/BER; per-case embedding/extraction time; algorithm-equivalent FPS; recovery path, frozen config hash, watermark sha and host sha. Missing/failed cases produce a non-zero exit; a partial CSV does not pass the completeness gate. Exact clean recovery is reported separately and is not fabricated.

Kodak is **external test data**: it must not be used to tune the proposal hyperparameters. Run your existing development-only sweep on the repository development split, freeze a configuration, then evaluate Kodak once.

## Files added by this patch

- `scripts/download_full_kodak.py`: download/cached 24-image dataset, strict verification.
- `scripts/run_full_kodak_external.py`: native full-dataset proposal benchmark; built-in attack profile switch, checkpoint/resume, completeness validation.
- `scripts/prepare_kodak_official_512.py`: separate 24 deterministic 512x512 crops for official upstream provider, including crop manifest.
- `KAGGLE_KODAK_FULL_EXTERNAL_ONE_CELL.py`: end-to-end Kaggle launch cell (clone/pull after you commit changes, setup, run, archive); settings at top.
- `tests/test_kodak_full_external.py`: offline fixture tests; mock evaluation explicitly **not** real Kodak results.

## Run locally if 24 cached PNGs are already present

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/run_full_kodak_external.py \
  --dataset-dir /path/to/kodak_full_native_24 \
  --watermark-dir data/user_dataset/watermark \
  --config configs/frozen_selected_20261021.json \
  --out-dir /path/to/results/kodak_full_native_proposal \
  --attack-profile basic --offline
```

If Internet access is enabled, omit `--offline` to let it fetch all 24 originals. Use `--attack-profile unified109` for all 110 built-in conditions. Never switch attack profile/config/images/watermarks within an existing output; the run identity check will reject it.

## Optional direct genuine official attack suite, all 24 hosts

```bash
python scripts/prepare_kodak_official_512.py \
  --dataset-dir /path/to/kodak_full_native_24 \
  --out-dir /path/to/kodak_official_512

python scripts/run_mmwatermark_in_official_repo.py \
  --official-repo /path/to/fresh/MMWatermark-Robustness \
  --companion-repo /path/to/fresh/MM_Robustness \
  --host-dir /path/to/kodak_official_512 \
  --watermark-dir data/user_dataset/watermark \
  --config configs/frozen_selected_20261021.json \
  --out-dir /path/to/results/official_full_24 \
  --severities 1 2 3 4 5
```

Upstream checksum/commit, attack catalog, severity, failures and exact trial count remain in its own `official_mmwatermark_manifest.json`. It never requires an external competitor CSV. Do not call official upstream attacks or cropped inputs the same experiment as the native Kodak built-in attacks.

## Primary result files

Native: `kodak_full_native_proposal/kodak_all_cases.csv`, `kodak_summary_by_watermark_attack.csv`, `kodak_external_full_manifest.json`, `kodak_run_identity.json`, `kodak_progress.json`.

Dataset: `kodak_full_native_24/kodak_full_manifest.json`.

Official (optional): `official_mmwatermark_full_24/official_mmwatermark_raw.csv`, `official_mmwatermark_manifest.json`, `official_integration_provenance.json`, and `kodak_official_512_crops_24/kodak_official_crops_manifest.json`.

**Not yet validated on external Kaggle network/CPU:** this environment has no outbound networking, so the real Kodak download, complete 144/5280 evaluation and official 5568-row run must be verified in Kaggle. A local smoke experiment confirmed that the unchanged proposal can embed and exactly decode 4096 bits in a 768x512 host. Offline 24-image mock/validation tests passed.
