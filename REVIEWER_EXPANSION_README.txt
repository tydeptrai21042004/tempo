# TEMPO reviewer experimental extension — no proposal-method edits

This patch adds **external experiments only**. It does not change any file under
`src/`, the frozen proposal configs, the original experiments, or the runtime
behavior of the encoder and decoder. Apply the ZIP paths over the repository root.

## Install and run

The existing `scripts/kaggle_setup.sh` installs/verifies the standard dependencies.
From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/run_reviewer_expansion.py --quick --output results/extra_review \
  --only paired,sync,integer,latency,attack
```

Full experiments, with **pre-existing, independent** data sources:

```bash
python scripts/run_reviewer_expansion.py --output results/extra_review \
  --development-dir /path/to/development \
  --external-host-dir /path/to/new_external_hosts \
  --video /path/to/genuine_video.mp4 \
  --only paired,pareto,sync,integer,latency,external,temporal,attack \
  --strict
```

In full mode, the Pareto scan MUST use only the development split. The external
validation stage rejects byte-identical hosts appearing in that split; it
cannot detect near-duplicate crops and should be accompanied by a provenance
report. The video stage never silently substitutes synthetic frames.

To run separate Kaggle sessions in parallel, **use different output paths**:

```bash
# Image/maths session:
python scripts/run_reviewer_expansion.py --output /kaggle/working/extra_image \
 --only paired,pareto,integer,sync,attack \
 --development-dir /kaggle/working/development

# Independent video/latency session:
python scripts/run_reviewer_expansion.py --output /kaggle/working/extra_video \
 --only latency,temporal \
 --video /kaggle/input/my-video/video.mp4
```

The prior full reviewer pipeline still handles multiwatermark matched-PSNR,
ROC, orthogonality, integer rejection, geometric attacks, temporal stress,
galleries, and live-stream scheduling; do **not** count them as newly run just
because this extension was applied.

| Stage | Evidence | Interpretation |
| --- | --- | --- |
| `paired` | Raw conditions and host-cluster paired CI | Original versus refined *configuration*, not matched PSNR |
| `pareto` | Development-only PSNR–NC–FPS frontier | Sequential service throughput excluding JPEG attack time; not live FPS |
| `sync` | Forced recovery versus default and raw decoder | Diagnostic oracle-style counterfactual, not geometric invariance |
| `integer` | Exact enumeration of three integer pixel-channel coordinates | Restricted local study; **not** full integer global optimum |
| `latency` | Alternating 720p JPEG70 wall-clock trials; optional cProfile | End-to-end image/codec/extraction service, not paced stream FPS |
| `external` | Hash-verified unseen real image host experiments | Strict file-level overlap check; requires user-supplied dataset |
| `temporal` | Bit errors/voting on changing real MP4 frames | Correlated video errors; 4096 bits per frame; no ECC |
| `attack` | Payload/pilot attenuation versus measured attack PSNR | Intentional public-subband removal; not certified security |
| `learned` | Native pretrained baseline provenance/CSV import | **Needs independently run official model weights and outputs** |

### Pretrained neural baseline

No neural-model checkpoint is included or invented. To record results from an
independently executed official pretrained model, supply a JSON provenance
file with `model_name`, `paper_url`, `source_commit`, `weights_sha256`,
`native_payload_bits`, `detector_assumptions`, and `evaluation_protocol`.
Supply a CSV containing `host,watermark,attack,payload_bits,psnr_db,nc,ber,embed_ms,extract_ms`.
Then:

```bash
python scripts/reviewer_learned_baseline.py \
 --provenance /path/to/baseline_provenance.json \
 --observations /path/to/baseline_evaluation.csv \
 --out-dir results/extra_review/learned
```

A 32-bit native model must not be presented as a 4096-bit matched-payload
baseline. Native results are clearly labeled, and no missing measurements are
fabricated.

### Reading the evidence ledger

`expansion_ledger.json` lists completed, failed and blocked stages independently.
The runner automatically creates a results ZIP next to the chosen output directory
(unless `--no-zip` is supplied).
A stage requiring missing external data will be `blocked`, **never** marked
complete. The ledger records SHA-256 fingerprints of the untouched method files.
`--strict` fails the command if selected stages are incomplete.

### Scope and limitations

* Performance comparisons measure paired cases, not guaranteed improvements.
* All hardware/OS and standard benchmark caveats still apply.
* `pareto` sweeps an existing exposed quantization step, not a new algorithm.
* `integer` checks an *exact restricted* local neighborhood; the full 192D
  integer optimum requires substantially more expensive mixed-integer search.
* Existing streams provide the authoritative deadline/FPS evidence; a sequential
  stage's `1000 / mean_ms` is not equivalent to real-time 30 fps throughput.
* For manuscript claims, validate the native/modern baseline separately and
  run the original paper's benchmark protocols without silently changing bit counts.
