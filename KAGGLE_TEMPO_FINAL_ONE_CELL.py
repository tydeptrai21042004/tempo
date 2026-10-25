# TEMPO 2026-10-25 final execution entrypoint
# Kaggle Settings: Accelerator = None (CPU), Internet = On
# IMPORTANT: Push the corrected TEMPO_FINAL_NO_GATE patch to GitHub main FIRST.
# This cell does not need uploaded images, model result CSVs, or extra notebooks.

import urllib.request
from pathlib import Path

WORK = Path('/kaggle/working')
assert WORK.is_dir(), 'Use Kaggle Notebook with CPU and Internet enabled.'
URL = ('https://raw.githubusercontent.com/tydeptrai21042004/tempo/main/'
       'KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py')
with urllib.request.urlopen(URL, timeout=90) as response:
    source = response.read().decode('utf-8')
required = (
    "'12_official_mmwatermark_kodak_24'",
    "'13_summarize_official_kodak'",
    "'scripts/prepare_kodak_official_512.py'",
    "'IMAGE_ATTACK_PRESET':'reviewer46'",
    "raise SystemExit(run_exit_code)",
)
missing = [word for word in required if word not in source]
if missing or "launch('10_strict_scientific_gate'" in source:
    raise RuntimeError(
        'GitHub main has not been updated with the final Kodak BMP + no-gate '
        'patch. Apply the supplied ZIP, commit it, then rerun this cell. '
        f'Missing markers: {missing}'
    )
# The upstream integration must be the ONLY external Kodak experiment.
if '03_KODAK_ALL24_NATIVE_109' in source:
    raise RuntimeError('Old 109-attack Kodak runner detected; update GitHub main.')
launcher = WORK / 'TEMPO_FINAL_FROM_GITHUB.py'
compile(source, str(launcher), 'exec')
launcher.write_text(source, encoding='utf-8')
print('Verified final launcher on GitHub main. Cloning and running all experiments...')
# The launcher clones GitHub main into a timestamped fresh run directory,
# records the git commit, archives native results and original official data.
exec(compile(source, str(launcher), 'exec'), {'__name__':'__main__', '__file__':str(launcher)})
