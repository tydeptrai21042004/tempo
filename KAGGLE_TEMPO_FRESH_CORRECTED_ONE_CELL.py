# Run ONLY after applying/pushing TEMPO_RECHECK_TWO_ERRORS_PATCH.zip to GitHub main.
# Kaggle: Accelerator=None (CPU), Internet=On.
import urllib.request, sys, hashlib
from pathlib import Path
WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Requires a Kaggle notebook'
url=('https://raw.githubusercontent.com/tydeptrai21042004/tempo/'
     'main/KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py')
with urllib.request.urlopen(url,timeout=90) as resp:
    source=resp.read().decode('utf8')
checks={
 'correct video CLI': "REPO/'scripts/benchmark_frozen_720p.py',\n" in source,
 'wrong extra CLI arg absent': "REPO/'scripts/benchmark_frozen_720p.py','scripts/benchmark_algorithm_only_720p.py'" not in source,
 'official compatible option': "'--allow-partial'" in source,
 'reject missing supported trials': "official.get('complete')" in source,
 'disclose unavailable originals': "'unsupported_conditions':unsupported" in source,
 'no scientific gate': "launch('10_strict_scientific_gate'" not in source,
 'no native Kodak 109 execution': "launch('03_KODAK_ALL24_NATIVE_109'" not in source,
}
missing=[name for name,okay in checks.items() if not okay]
if missing:
    raise RuntimeError('GitHub main is not updated with the corrected patch: '+repr(missing))
file=WORK/'TEMPO_FRESH_PATCHED_FROM_GITHUB.py'
compile(source,str(file),'exec')
file.write_text(source,encoding='utf8')
print('GitHub launcher verified; SHA-256:',hashlib.sha256(source.encode()).hexdigest())
print('Launching complete original internal benchmark plus compatible official Kodak and separate 4096-bit video.')
exec(compile(source,str(file),'exec'),{'__name__':'__main__','__file__':str(file)})
