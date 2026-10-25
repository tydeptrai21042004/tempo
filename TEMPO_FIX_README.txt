TEMPO KAGGLE RECHECK — 25 October 2026

Based on TEMPO_FINAL_CORRECTED_SOURCE.zip matching the failed b186c3e6 launcher.

CONFIRMED FAILURE A
Stage 09_frozen_720p_fps included a stray positional argument
'scripts/benchmark_algorithm_only_720p.py'. Remove that string from the
argument list. The separate algorithm-only benchmark remains stage 09b.

CONFIRMED FAILURE B
Official Kodak preflight supported 110/115 upstream attack/severity pairs.
All 24*2*(1+110)=5328 compatible trials completed with zero failed trials.
Five unsupported upstream pairs had NO RESULTS, so the prior strict all-115
coverage assertion rejected a valid compatible-only experiment.

FIX:
- Run official 24-Kodak MMWatermark with --allow-partial, preserving upstream
  attack functions and all 4096-bit watermark bits.
- Require complete coverage of every supported pair, all 24 hosts, both
  watermarks, exact condition labels, no duplicates, no failed trials.
- Write upstream_unsupported_conditions.csv with original error reasons.
- Mark report COMPLETE_SUPPORTED_CATALOG_ONLY when exclusions remain. Never
  call it full original-catalog coverage (5568). A true 5568/5568 benchmark
  requires separately resolving and rerunning the five upstream failures.
- Do not reenable the scientific PSNR/FPS gate or native Kodak 109 suite.
- Internal 46-condition experiments remain on the repository's 12 selected hosts.

RECOVER EXISTING KAGGLE EVIDENCE:
Run TEMPO_RECOVER_EXISTING_KAGGLE_RESULTS.py as ONE Python cell in the SAME
Kaggle notebook session, BEFORE the /kaggle/working directory disappears.
It audits and summarizes the already saved genuine official CSV, reruns only
original stage 09 video (proper CLI), and archives evidence in
TEMPO_RECOVERED_EVIDENCE_<runstamp>.zip. It does NOT rerun the official 5328
trials or the already successful algorithm-only video benchmark.

PERMANENT FRESH RUN:
Extract TEMPO_RECHECK_TWO_ERRORS_PATCH.zip into your GitHub repository root,
preserving relative folder paths; commit and push all changed files.
Then run KAGGLE_TEMPO_FRESH_CORRECTED_ONE_CELL.py inside a new Kaggle CPU
notebook with Internet ON. It checks exact patch markers before execution.

TESTS:
- 29 focused Python tests pass in the sandbox.
- 244/244 source SHA-256 manifest entries match.
- A simulated recovery on complete 24x2 supported rows ran and generated
  the expected summary and recovery archive successfully.
- FULL live Kaggle run NOT re-executed in this environment.
- The user's actual preflight error reasons are in the Kaggle output file
  extra/official_mmwatermark_kodak_24/official_mmwatermark_preflight.json.
  The attached pasted log did not include the five exact reasons.
