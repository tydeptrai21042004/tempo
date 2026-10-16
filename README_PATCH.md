# TEMPO recovered watermark validation fix

This patch changes **only experiment I/O and a regression test**. It does not modify the JILP-QIM proposed method or any baseline implementation.

## Cause
`scripts/run_multiwatermark_video.py` loaded extracted `recovered.png` using the input-payload validator. Valid robustness failures (all-black or all-white decoded bits) raised `ValueError: Degenerate watermark payload` and aborted the entire corpus run.

## Fix
- `load_binary_watermark(..., allow_degenerate=True)` explicitly supports **decoded results**. Default remains `False` for actual input watermarks.
- Video evaluation scores constant recovered outputs, rather than discarding them; reports `recovered_is_constant` and `recovered_one_fraction` alongside the existing metrics.
- Added tests for constant-black and constant-white extraction; verified all 117 repository tests passed in the local environment.

## Apply to cloned GitHub repository
Unzip the patch into the Git repository root, preserving directories, overwriting the three files. Or upload the three patched files to matching paths in GitHub and commit. Do not overwrite the repository's original watermark PNG images.

## Kaggle warning
The existing full shell pipeline begins by deleting its output directory, so rerunning it destroys earlier partial results. Either start in a fresh run or explicitly resume at the interrupted video stage without invoking `scripts/kaggle_pipeline.sh` from the beginning. A previous Kaggle session's files persist only if the environment/session is still available.

## Validation
`PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` -> 117 passed (local). Full four-video, 180-frame CPU suite has **not** been re-executed after patch.
