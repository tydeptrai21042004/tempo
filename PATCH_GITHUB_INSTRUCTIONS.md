# TEMPO: minimal source overlay onto tempo-main(10).zip

This patch is intended **only** for the exact uploaded source archive
`tempo-main(10).zip`; it is not a full replacement repository.

1. Unzip the delta. Copy contents of the `tempo-main/` folder to the root
   of your local clone, preserving relative paths, and overwrite changed files.
2. Optionally delete the old scientific-gate and historical Kaggle runners
   listed in `DELETE_LEGACY_OPTIONAL.txt`; they are not needed by this runner.
3. Review `git diff`, commit and push to the `main` branch at
   `https://github.com/tydeptrai21042004/tempo`.
4. In Kaggle set Accelerator=None (CPU), Internet=ON and paste the entire
   `KAGGLE_CLONE_ALL_EXPERIMENTS.py` into one Python cell. It clones GitHub.
5. `PROFILE='all'` requests the full internal reviewer experiment plus official
   MMWatermark Kodak; slow stages may reach the Kaggle time limit. To finish
   independently run `PROFILE='internal'` and `PROFILE='official_kodak'`
   in separate sessions and keep their separate run evidence.

## Baseline fidelity and reviewer evidence

- DOI `10.52866/ijcsm.2024.05.03.043`: executable *independent equation-level*
  2024 LWT–HD–SVD reproduction; its decoder requires original-host spectrum
  and its singular-spectrum payload is **not 4096 independent blind bits**.
- The paper-exact original-author numerical code has not been obtained.
- The optional Haar–HD–SVD–QIM control is independently designed; it is **not**
  a claim to reproduce the published 2024/2026 algorithms.
- The primary blind, common-46-condition, matched-PSNR comparison still uses
  independent DCT QIM/ISS/STDM implementations, clearly labeled as adapters.
- No scientific acceptance gate; keep per-stage failures, clean-recovery failures,
  deadline misses, and reviewer completeness diagnostics rather than declaring
  a run complete without evidence.
- Non-result manuscript architecture/geometry panels are never redrawn unless
  explicitly requested; result PDFs are generated only from actual run evidence.
