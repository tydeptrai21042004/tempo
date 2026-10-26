## 2026-10-26 — Source-paper baseline fidelity correction

- Default reviewer pipeline now **runs** an independent equation-level reproduction of Sudrajat, Ernawan & Amrullah (2024), DOI `10.52866/ijcsm.2024.05.03.043`, p. 700, Eqs. (10)-(22). Three-level orthonormal Haar -> LL Hessenberg -> SVD -> additive watermark singular values -> inverse steps; detector uses original-host singular values and paper Eq.(22) reconstruction. Actual test-time NC/BER are logged, including low NC; no scientific gate.
- Removed independent Haar-HD-SVD-QIM as default **published paper** baseline. It remains clearly named as an optional *new independent blind control* (`TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL=1`) and is not attributed to Sudrajat et al.
- Original paper has an algebraic omission in printed Eq.(16); the implementation documents the corrected orthogonal similarity inverse rather than claiming numerical identity with unavailable author MATLAB source.
- Grayscale paper-native host and alpha-only transparent watermark loading are explicit and tested; 4096 extracted image pixels are not 4096 independent blind bits.
- DOI/source/equation fidelity map in `docs/PAPER_2024_FIDELITY.md`; output CSV and JSON store actual run evidence, side-information class and calibration.
- **Honest remaining limitation**: a faithful *contemporary blind equal-4096-bit* original-paper baseline is still needed for Reviewer #5; incompatible informed methods cannot be relabeled blind.

# TEMPO — reviewer evidence code revision (2026-10-26)

- Preserves the manuscript's DCT payload/synchronization carriers, min-norm RGB embedding, 8-bit correction, keyed blind extraction and pilot-gated geometry.
- Adds a **new independent 4096-bit blind** Haar/HD/SVD center-QIM transform-class adapter, with common-condition matched-PSNR calibration only on development hosts and strict held-out pairing. Not an author-paper reproduction.
- Adds a **secondary, informed** LWT/HD/SVD-style singular-value addition experiment whose original-host / original-watermark side information and effective capacity are clearly disclosed. Never mixed into the primary blind ranking.
- Extends real-video runner to include the new blind control when explicitly requested. Uses observed frame decoding and timing; synthetic smoke clips never count as downloaded real-video manuscript evidence.
- Adds actual matched Baboon/JPEG gallery replay that cross-checks each NC with stored CSV before generating PDF figures; manuscript-result figures only by default. No schematic redesign.
- Preserves pre-existing reviewer checks for joint parameters, integer failure denominators, temporal correlation, off-grid geometry, targeted coefficient-zeroing, ROC and false positives, capacity, throughput and official Kodak compatibility.
- Removes retired scientific acceptance gate; retains errors, eligibility, status and missing-evidence diagnostics. Does not claim acceptance when a desired outcome is not reached.
- Cleans retired experiment assets and rewrites README around the article and explicit replication protocol.

**Validation**: 128 unit/protocol tests passed in the development container before packaging. Modern full-parameter smoke study: 1 held-out host, 1 watermark, 45 non-clean conditions, 3 methods, 90 cross-method paired attacks. Two-watermark 12-frame MP4 smoke run also passed. This is not a completed nine-host / two-watermark / full-video campaign.

**Limits:** faithful original 2023/2024/2026 author methods remain outside the primary comparison until verified source/algorithm and side information are supplied. Paper tables must be regenerated from the final full-run CSVs; no numerical claim is backfilled.
