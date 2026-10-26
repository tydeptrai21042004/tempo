# Original-paper baseline: 2024 LWT–Hessenberg–SVD

Source: Sudrajat, A.; Ernawan, F.; Amrullah, A. (2024), *Robust Image Watermarking Based on LiftingWavelet Transfrom-Hessenberg Decomposition for Copyright Protection*, Iraqi Journal for Computer Science and Mathematics, 5(3), 695–706. DOI **10.52866/ijcsm.2024.05.03.043**. URL: https://doi.org/10.52866/ijcsm.2024.05.03.043

Source equations: page 700, (10)–(22), verified against the supplied PDF.

| Paper reference | Source operation | Local Python operation |
|---|---|---|
| (10), (18) | R-level Haar DWT, LL plus detail bands | `haar_forward` (R=3 for 512/64), `haar_inverse` |
| (11), (19) | Hessenberg factorization of LL | `scipy.linalg.hessenberg(calc_q=True)` |
| (12), (20) | SVD of Hessenberg matrix | `numpy.linalg.svd` |
| (13) | SVD of watermark logo | `numpy.linalg.svd(logo)` |
| (14) | HS_modified = HS + alpha*Sw | `markedS=S+alpha*Sw` |
| (15) | Inverse SVD | `(U*markedS)@Vt` |
| (16) | Inverse Hessenberg | `P@Hmarked@P.T`; paper's printed `P@WH` omits P.T |
| (17) | Inverse Haar reconstruction | `haar_inverse` |
| (21) | Extract S_W = (S_attacked - S_original)/alpha | `recovered=(S-original_host_spectrum)/alpha` |
| (22) | Inverse SVD with received HU and HV | `(U_received*recovered)@Vt_received` |

## Reproduction qualification

This is an **independent equation-level reimplementation**, not verified reproduction of the unavailable original MATLAB source or its reported PSNR/NC values. The article's R/alpha/normalization decisions are insufficient to guarantee exact numerical duplication. For 512 host with 64 logo, R=3 is a declared dimensionality-compatible choice. The paper's Eq.(16) is algebraically missing P transpose, and Eq.(22) reconstructs with received host's singular vectors, which can fail to recover a visually correct logo: measure it, do not change it. To test alternative 2023 watermark-vector decoding explicitly, pass `--reconstruction paper2023_watermark_vectors`, and never call that 2024 Eq.(22).

**Detector classification**: informed; requires original host singular values. **Information-bearing capacity**: at most 64 modified singular values for 64x64 logo; therefore do not equate reconstructed 4096 logo pixels with 4096 blind payload bits.

**Output**: `paper_2024_equation_trials.csv`, `paper_2024_dev_calibration.csv`, `paper_2024_fidelity.json`. Each trial stores attack, source DOI, PSNR, BER, TEMPO-standard NC=1-2 BER, paper-defined normalized cross-correlation `nc_paper`, detector class, payload caveat and run status. Poor BER, failed extraction, and missed PSNR remain in the table. No scientific gate.

**Original-paper version**: https://ijcsm.researchcommons.org/ijcsm/vol5/iss3/31/
