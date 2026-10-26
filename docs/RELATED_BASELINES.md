# Supplied literature and comparative-evidence scope

These six user-provided sources inform TEMPO's baseline selection and evidence presentation. A paper PDF is **not** an executable, independently validated reproduction. All comparisons require a declared payload, image color mode, detector side-information class, attack implementation, PSNR setting and timing platform.

| Supplied source | Described method and scope | How it should be used in TEMPO |
| --- | --- | --- |
| Ernawan et al., *Robust Image Watermarking using Hessenberg Decomposition and SVD with Integer Wavelet Transform*, ICSECS 2023, DOI 10.1109/ICSECS58457.2023.10256345 | IWT + Hessenberg + SVD image-watermarking design | Related transform baseline; verify watermark payload and original-host detector requirements before adapting |
| Ernawan, Sudrajat and Amrullah, *Robust Image Watermarking Based on Lifting Wavelet Transform–Hessenberg Decomposition for Copyright Protection*, 2024, DOI 10.52866/ijcsm.2024.05.03.043 | LWT + HD + SVD; reports multiple watermark sizes and robustness metrics | Recent robust-image baseline candidate; cannot assume 4096 bits or blind decoding |
| Kusuma and Panggabean, *Robust Digital Image Watermarking Using DWT, Hessenberg, and SVD for Copyright Protection*, 2026, DOI 10.71129/ijaci.v2i1.pp41-52 | DWT–HD–SVD on grayscale test images, PSNR, SSIM, NC | Contemporary transform-domain context; grayscale/side-information differences must remain explicit |
| Lakasek and Elhadi, *A Robust Image Watermarking Technique using NSST-HD-SVD*, 2026, DOI 10.71129/ijaci.v2i2.pp115-126 | NSST + directional sub-bands + SVD, explicitly claims blind extraction | The abstract claims blindness, but the extraction algorithm explicitly uses original host singular values and watermark singular vectors. Verify its decoder assumptions before classifying as fully blind. |
| Amrullah and Aminuddin, *Tamper Localization and Content Restoration in Fragile Image Watermarking: A Review*, 2026, DOI 10.71129/ijaci.v2i1.pp62-74 | Fragile authentication, tamper localization, self-recovery, tamper coincidence | Background only: not a robust fixed-message video watermark comparator |
| Desti and Hamdan, *Forgery Detection and Authentication of Medical Images using Adaptive Bit Substitution Watermarking*, 2026, DOI 10.71129/ijaci.v2i2.pp139-151 | Adaptive LSB/bit-plane embedding for medical forgery detection | Background only: different security objective and spatial-domain evaluation |

## Experiment needed for the contemporary-baseline reviewer comment

1. Obtain official authors' code, or implement the equations only with a clearly documented independent adapter and original-paper fidelity limitations.
2. Establish whether the *native* method is blind or nonblind. Preserve native payload; do not retrofit a 4096-bit payload and call it native.
3. Run both selected watermarks, all eligible held-out hosts, common 46 conditions, a payload-capacity and PSNR-eligibility audit, and time embedding and extraction after warm-up.
4. Archive code commit, source DOI, original result CSV SHA-256, input image checksums and method configuration. Use `import_external_baseline_results.py` to validate per-trial records.
5. Keep non-blind/non-matched/native-small-payload methods outside the primary equal-payload matched-PSNR claim. Do not quote summary NC from different attack menus as a head-to-head common-protocol result.

## Executable modern comparison now included

`src/realtime_watermark/baselines/haar_hd_svd_qim_adapter2024.py` implements a **new independent control** (not a cited method reproduction): RGB blockwise 8x8 one-bit allocation, orthonormal Haar LL, Hessenberg reduction, largest singular-value binary center-QIM, and blind recovery from the attacked image using a seed and quantizer step. A 512x512 image carries exactly 4096 bits. The implementation is timed end-to-end and calibrates its step on development images only. It therefore provides a testable modern-transform **method-class control**, with honest PSNR-matched host/attack comparisons and decoded-video evaluation.

Faithful original 2023/2024/2026 full-image singular-value addition uses host or watermark SVD information at detection, and/or a different capacity. It is a different experiment. It cannot be relabeled as this adapter, and its published NC on a different dataset cannot appear in a matched 4096-bit ranking. A strict external import route remains available once author/source code and provenance are obtained.

## Separately measured 2024 paper-structure informed case

The code also includes `scripts/run_recent_svd_paper_structure.py`, which independently implements a Haar lifting/LL → Hessenberg → SVD singular-value addition structure motivated by DOI 10.52866/ijcsm.2024.05.03.043. It uses original host singular values plus watermark singular vectors, so is marked `detector_class=informed`, `paper_exact=False`, and `native_independent_payload_bits=not_identified_from_64_singular_values`. Its outputs are **not** pooled with the 4096-bit blind center-QIM primary comparisons. Enable it only as a transparent secondary result (`TEMPO_RUN_SECONDARY_2024_INFORMED=1`); never relabel its restored logo size as a 4096-bit blind payload.
