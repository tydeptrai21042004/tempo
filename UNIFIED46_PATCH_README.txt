TEMPO — ONE FROZEN INTERNAL ATTACK PROTOCOL, FULL EXTERNAL EXCEPTIONS
==================================================================
Based on the user's uploaded tempo-main(5).zip (2026-10-25).

WHAT IS CHANGED
===============
- Proposed watermark embedding/extraction code: UNCHANGED.
- The ordinary 512x512 host-image and common-baseline protocol uses precisely
  46 conditions = one clean + 45 attacked cases, with a versioned SHA-256 JSON.
- The same attack definitions/parameters are applied to every method. Each of
  the selected 46 entries is byte-for-byte the same definition as in the
  existing configs/attacks/unified_109.json; no simplified substitutions.
- Development search, all-method held-out runs, matched-PSNR comparisons,
  image ablations, diagnostic geometric studies and galleries select attacks
  exclusively from the same frozen internal manifest (some narrowly scoped
  studies sample a named SUBSET from this one source; never another catalog).
- Official external MMWatermark attack functions, severity levels and
  full Kodak 24-image proposal-only suite remain INDEPENDENT and unchanged.
- Paper-native reproduction experiments with original per-paper attack models
  are optional (`TEMPO_RUN_PAPER_NATIVE_SEPARATELY=1`); they cannot be mixed
  into a same-attack primary ranking.
- No external benchmark CSV is necessary for native TEMPO baselines or Kodak;
  third-party model results must be genuine before claiming those comparisons.
- Quality PSNR is MEASURED AND REPORTED, but the wrapper no longer rejects
  the experiment for failing the previously imposed 50 dB threshold.
- 4096-bit watermark (64 x 64) is unchanged for all primary image methods;
  two watermarks, held-out split, and original detector assumptions retained.
- The extra method-specific DCT zeroing, integer-boundary, ROC, and temporal
  diagnostics remain separately labeled analytic/control experiments. They
  are NOT additional conditions in the 46-condition benchmark mean.

CANONICAL 46 CONDITIONS (INCLUDING CLEAN)
=========================================
Compression: jpeg90, jpeg70, jpeg50, jpeg40, jpeg30, webp50,
             jpeg70_subsampling2.
Noise: gaussian_sigma2, gaussian_sigma5, gaussian_variance_0p003,
       gaussian_variance_0p05, speckle_variance_0p001,
       speckle_variance_0p03, saltpepper_0_02, poisson.
Filtering: blur_r1_0, motion_blur7_h, median3, chroma_blur1.
Geometry: resize075, resize050, aspect_resize_075_125,
          crop_restore10, crop_restore25, rotate1, rotate_2p5,
          rotate_7p5, rotate_15p0, translate4_4,
          subpixel_translate_0p5_0p5, scale_geometry_0p9,
          scale_geometry_1p1, affine_shear005, affine_shear_0p1,
          perspective002.
Color and occlusion: brightness090_contrast110, gamma120, grayscale_y,
                     chroma420, cutout20.
Mixed & acquisition: jpeg70_gaussian1, rotate2p5_jpeg40,
                     resize090_jpeg40, screen_capture_light,
                     print_scan_light.
Control: clean.

REPORTS AND FAILURE POLICY
==========================
* reports/validation/reviewer_unified_46_manifest.json: frozen exact operators.
* reports/validation/reviewer_unified_46.json: checksum and category audit.
* reports/validation/common_protocol.json: per-baseline native detector/capacity
  disclosure. A common 4096-bit adapter is NOT automatically an official
  mathematically identical paper reproduction. Do not conceal this distinction.
* reports/validation/internal_coverage.json: fails if any host/watermark/method
  pair in all-method or matched-PSNR output is missing even one of the 46.
* reports/all_methods_all_metrics: six methods (five capacity-eligible common
  adapters plus the proposal), identical attack names and 4096-bit payload.
* reports/matched_psnr: separately calibrated on DEVELOPMENT images; primary
  matched-PSNR eligible rows are the intersection, not all attempted rows.
  The calibration target is predeclared 44.18 dB from the manuscript (NOT
  recalculated from held-out evaluation images).
* reports/statistical_evidence: host-clustered CIs and same-condition paired
  proposal-baseline effects; host is the resampling unit, not attacks.
* Kodak native: existing scripts/run_full_kodak_external.py --attack-profile
  unified109 -- full 24 images native resolution, two watermarks, proposal-only.
  It saves its own CSV and manifest automatically.
* Official MMWatermark: existing upstream independent attack catalog, unchanged.
* Real-time video: stream latency/codec/queue statistics must not be mislabeled
  algorithm-only; video channel performance probes are subsets of canonical
  JPEG channels (clean, jpeg90, jpeg70, jpeg40).

HOW TO APPLY AND RUN
====================
1. Extract the ZIP into the ROOT of tempo GitHub repository, preserving paths.
   This patch is based on tempo-main(5).zip; apply to that same revision only.
2. Push to GitHub main.
3. Set Kaggle notebook: Accelerator=None, Internet=On.
4. Run the repo's KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py via the existing
   single-cell downloader, or run bash scripts/kaggle_pipeline.sh full in the
   Kaggle environment after setup. The manifest audit runs before benchmarks.
5. For external Kodak full 109, additionally run:
   python scripts/run_full_kodak_external.py \
      --dataset-dir /kaggle/working/kodak_full \
      --watermark-dir data/user_dataset/watermark \
      --config configs/frozen_selected_20261021.json \
      --attack-profile unified109 \
      --out-dir /kaggle/working/kodak_proposal_only_109
6. Do not upload outside CSVs: native common baselines and Kodak write all
   their own result CSVs. The OPTIONAL third-party baseline-import stage stays
   "not run" if no genuine external observations are available.

IMPORTANT LIMITATIONS
=====================
- No Kaggle run is claimed completed by this source patch.
- Method-specific targeted DCT suppression and integer boundary probes are
  not replaced with fake generic attacks; those controlled experiments are
  mathematically different questions and remain separate from rankings.
- Source-level protocol consistency is verified, but claims of exact original
  baseline implementation need individual research-paper correspondence.
- A result failing actual 720p end-to-end 30 FPS or p95 deadlines must remain
  a failure in its separate stream test; algorithm-only throughput is a
  distinct, independently reported metric.
- The scientific gate may still flag incomplete manuscript/external evidence;
  this should not be silenced by inventing comparisons or claims.
- The second attached DWT-Hessenberg reviewer file concerns a DIFFERENT
  method; its BSS, Firefly, RGB-channel, and 4x4-block requests are not claims
  about the TEMPO/DCT proposal, though overlapping attack requirements are
  represented here where technically relevant.
