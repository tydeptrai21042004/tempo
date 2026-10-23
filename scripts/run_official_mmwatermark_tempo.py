#!/usr/bin/env python3
"""Run TEMPO watermarks through real official MMWatermark-Robustness corruptions.

This is an OPTIONAL external provider track, separate from TEMPO's canonical
109 built-in conditions. Adapted from the user-supplied QR_candicate audit
runner; the upstream perturbations themselves are loaded from the verified
MMWatermark-Robustness Git checkout, never reimplemented here.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.util, inspect, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from realtime_watermark.core import load_rgb, load_binary_watermark, bit_accuracy, ber, nc_binary, psnr, save_csv, save_json
from realtime_watermark.registry import embed, extract, normalize_method_id, COMMON_IMAGE_METHODS
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.paper_fidelity import fidelity_status

OFFICIAL_URL = 'https://github.com/Jielin-Qiu/MMWatermark-Robustness.git'
COMPANION_URL = 'https://github.com/Jielin-Qiu/MM_Robustness.git'
IMAGE_EXT = {'.png','.bmp','.jpg','.jpeg','.tif','.tiff','.webp'}
# Only methods from the upstream module can be invoked. Never fall back to locally
# implemented attacks. Names are limited to common benchmark-corruption families.
KNOWN = ('gaussian_noise','shot_noise','impulse_noise','speckle_noise',
         'gaussian_blur','glass_blur','defocus_blur','motion_blur','zoom_blur',
         'fog','frost','snow','spatter','brightness','contrast','saturate',
         'pixelate','jpeg_compression','elastic_transform',
         'scaling','scale','rotation','rotate','shearing','shear',
         'piecewise_affine','piecewise_affine_transform')


def prepare_repo(path: Path, *, clone: bool=True) -> str:
    path=Path(path)
    if not (path/'.git').exists():
        if not clone:
            raise RuntimeError(f'Official checkout absent: {path}')
        if path.exists() and any(path.iterdir()):
            raise RuntimeError('Refusing to clone into nonempty unverified directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--depth','1',OFFICIAL_URL,str(path)],check=True,timeout=180,
                       env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    origin=subprocess.run(['git','-C',str(path),'remote','get-url','origin'],check=True,capture_output=True,text=True).stdout.strip()
    if origin.rstrip('/').removesuffix('.git').lower()!=OFFICIAL_URL.removesuffix('.git').lower():
        raise RuntimeError(f'Not the official repository: {origin}')
    dirty=subprocess.run(['git','-C',str(path),'status','--porcelain'],check=True,capture_output=True,text=True).stdout.strip()
    if dirty: raise RuntimeError('Official checkout contains uncommitted changes; cannot attest to source revision')
    sha=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha.lower()):
        raise RuntimeError('Invalid upstream revision')
    return sha


def prepare_companion(path: Path, *, clone: bool=True) -> str:
    """Keep the upstream dependency separate from the attested MMWatermark checkout."""
    path=Path(path)
    if not (path/'.git').is_dir():
        if not clone:
            raise RuntimeError(f'MM_Robustness checkout absent: {path}')
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f'Refusing a non-empty unverified companion directory: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--depth','1',COMPANION_URL,str(path)],check=True,timeout=180,
                       env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    origin=subprocess.run(['git','-C',str(path),'remote','get-url','origin'],check=True,
                          capture_output=True,text=True).stdout.strip()
    if origin.rstrip('/').removesuffix('.git').lower()!=COMPANION_URL.removesuffix('.git').lower():
        raise RuntimeError(f'Incorrect MM_Robustness origin: {origin}')
    if subprocess.run(['git','-C',str(path),'status','--porcelain'],check=True,
                      capture_output=True,text=True).stdout.strip():
        raise RuntimeError('MM_Robustness checkout has uncommitted changes')
    required=path/'image_perturbation'/'imagenet_c'/'imagenet_c'/'__init__.py'
    if not required.is_file():
        raise RuntimeError(f'Upstream MM_Robustness package missing: {required}')
    return subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,
                          capture_output=True,text=True).stdout.strip()


def adapt_skimage_gaussian(module) -> bool:
    """Translate only the retired skimage multichannel keyword, preserving its meaning."""
    gaussian=getattr(module,'gaussian',None)
    if gaussian is None or not callable(gaussian):
        return False
    if 'multichannel' in inspect.signature(gaussian).parameters:
        return False
    if 'channel_axis' not in inspect.signature(gaussian).parameters:
        return False
    def compatible_gaussian(image,*args,**kwargs):
        if 'multichannel' in kwargs:
            multic=kwargs.pop('multichannel')
            kwargs.setdefault('channel_axis', -1 if multic else None)
        elif 'channel_axis' not in kwargs:
            # Older skimage releases auto-recognized last-axis RGB/RGBA when
            # the official source omitted multichannel (e.g., glass_blur).
            # Set it explicitly with current skimage; keep grayscale 2-D.
            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                kwargs['channel_axis'] = -1
        return gaussian(image,*args,**kwargs)
    module.gaussian=compatible_gaussian
    return True


def find_module(repo: Path) -> Path:
    candidates = [p for p in (repo/'perturbations').rglob('image_perturb.py') if p.is_file()]
    if len(candidates)!=1:
        raise RuntimeError(f'Expected one official image_perturb.py under perturbations, found {len(candidates)}. '
                           'Inspect the upstream checkout; do not substitute local attacks.')
    return candidates[0]


def discover_names(module_path: Path) -> list[str]:
    # Parse before importing; --list-attacks works even when optional dependencies are absent.
    tree=ast.parse(Path(module_path).read_text(encoding='utf8'))
    names=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in KNOWN:
            names.append(node.name)
    return sorted(set(names))


def import_official(path: Path, repo: Path):
    # Import source from the verified checkout; intentionally no "mock attack" substitution.
    spec=importlib.util.spec_from_file_location('mmwatermark_upstream_image_perturb',path)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load official perturbation module')
    module=importlib.util.module_from_spec(spec)
    old=list(sys.path)
    old_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        sys.path[:0]=[str(path.parent),str(repo/'perturbations'),str(repo)]
        spec.loader.exec_module(module)
    finally:
        sys.path[:]=old
        sys.dont_write_bytecode=old_bytecode
    # The official source imports skimage.filters.gaussian into its module
    # namespace and still calls it with multichannel=True. Recent versions
    # accept channel_axis=-1 instead. Adapt the bound function *only in the
    # loaded module*, without editing or dirtying the attested upstream Git.
    module._tempo_skimage_gaussian_compat = adapt_skimage_gaussian(module)
    return module


def call_official(fn, image_bgr: np.ndarray, severity: int, seed: int|None=None, *, companion_repo: Path|None=None) -> np.ndarray:
    """Invoke a compatible upstream function; reject all unsupported signatures/results."""
    sig=inspect.signature(fn)
    params=list(sig.parameters.values())
    if not params or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError('Upstream perturbation requires an unsupported first parameter')
    if len(params)<2 or params[1].name not in {'severity','level'} or params[1].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError('Upstream perturbation does not expose (image, severity/level)')
    if any(p.default is inspect.Parameter.empty for p in params[2:] if p.kind not in
           (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD)):
        raise TypeError('Upstream perturbation requires additional unsupported arguments')
    rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Upstream functions may require uint8 ndarray inputs.
    # PIL conversion breaks 50+ combinations. Pass a fresh RGB uint8 ndarray.
    np_state=np.random.get_state();py_state=random.getstate()
    original_path=list(sys.path)
    original_cwd=Path.cwd()
    original_no_bytecode=sys.dont_write_bytecode
    # numpy>=2 removed np.float_, which the original plasma_fractal still uses.
    add_float_alias=not hasattr(np,'float_')
    try:
        sys.dont_write_bytecode=True
        if add_float_alias:
            np.float_=np.float64
        if companion_repo is not None:
            companion_repo=Path(companion_repo).resolve()
            # Original imported functions resolve MM_Robustness *at call time*.
            sys.path.insert(0,str(companion_repo.parent))
            # Original frost() reads ./MM_Robustness/image_perturbation/frost*.png.
            os.chdir(companion_repo.parent)
        if seed is not None:
            np.random.seed(int(seed)%2**32);random.seed(int(seed))
        result=fn(rgb.copy(),severity)
    finally:
        if add_float_alias:
            del np.float_
        os.chdir(original_cwd)
        sys.path[:]=original_path
        sys.dont_write_bytecode=original_no_bytecode
        np.random.set_state(np_state);random.setstate(py_state)
    if isinstance(result,Image.Image): result=np.asarray(result.convert('RGB'))
    else: result=np.asarray(result)
    if result.shape!=rgb.shape:
        raise ValueError(f'Perturbation changed geometry: {result.shape} != {rgb.shape}')
    if not np.issubdtype(result.dtype,np.number) or not np.all(np.isfinite(result)):
        raise ValueError('Non-numeric or non-finite official perturbation output')
    if result.dtype!=np.uint8:
        if result.size and float(np.min(result))>=0 and float(np.max(result))<=1.0 and result.dtype.kind=='f':
            result=result*255
        result=np.clip(np.rint(result),0,255).astype(np.uint8)
    return cv2.cvtColor(result,cv2.COLOR_RGB2BGR)



def trial_seed(seed, host, watermark, method, attack, severity):
    blob=f'{seed}|{host}|{watermark}|{method}|{attack}|{severity}'.encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], 'big')


def preflight_official(module, names, severities, sample_bgr, host_name,
                       watermark_name, seed=2026, companion_repo=None):
    """Check the upstream (attack, severity) pairs on the actual image shape.

    Unsupported/invalid upstream outputs are *not* silently substituted by
    TEMPO implementations. They are recorded as unavailable with their errors.
    """
    supported, unsupported = [], []
    for name in names:
        fn = getattr(module, name)
        for severity in severities:
            severity = int(severity)
            try:
                call_official(
                    fn, sample_bgr, severity,
                    seed=trial_seed(seed, host_name, watermark_name,
                                    'preflight', name, severity),
                    companion_repo=companion_repo,
                )
                supported.append((name, severity))
            except Exception as exc:
                reason = f'{type(exc).__name__}: {exc}'
                unsupported.append({'attack': name, 'severity': severity,
                                    'reason': reason})
                print(f'UPSTREAM_UNSUPPORTED mmw::{name}::severity{severity}: {reason}',
                      flush=True)
    return supported, unsupported


def main(args):
    if not 1 <= min(args.severities) <= max(args.severities) <= 5:
        raise ValueError('Official severity must be in 1..5')
    selected_methods=[normalize_method_id(s.strip()) for s in args.methods.split(',') if s.strip()]
    if not selected_methods or any(s not in COMMON_IMAGE_METHODS for s in selected_methods):
        raise ValueError('Only registered TEMPO common-image methods are allowed')
    if args.official_repo.resolve().is_relative_to(args.out_dir.resolve()):
        raise ValueError('Official Git checkout must not be under result output folder')
    upstream_sha=prepare_repo(args.official_repo,clone=not args.no_clone)
    if args.expected_sha and upstream_sha != args.expected_sha:
        raise ValueError('Upstream commit differs from expected frozen SHA')
    official_source=find_module(args.official_repo)
    names=discover_names(official_source)
    selected=[x.strip() for x in args.attacks.split(',') if x.strip()] if args.attacks else names
    if not selected:
        raise ValueError('No compatible official perturbation functions discovered')
    missing=set(selected)-set(names)
    if missing:raise ValueError(f'Not declared by official source: {sorted(missing)}')
    companion=None
    if args.companion_repo:
        companion_sha=prepare_companion(args.companion_repo,clone=not args.no_clone)
        companion=args.companion_repo
    else: companion_sha=None
    module=import_official(official_source,args.official_repo)
    hosts=sorted(p for p in args.host_dir.glob('*.bmp') if 'lenna' not in p.name.lower() and 'lena' not in p.name.lower())
    watermarks=sorted(args.watermark_dir.glob('*.png'))
    if args.host_limit:hosts=hosts[:args.host_limit]
    if args.watermark_limit:watermarks=watermarks[:args.watermark_limit]
    if not hosts or not watermarks:raise RuntimeError('Missing host BMPs or watermark PNGs')
    config=JILPQIMConfig.load(args.config)
    # Upstream corruptions are not all compatible with 512x512 images or
    # current Python libraries. Check each condition and preserve diagnostics
    # instead of aborting the entire official-source track at the first error.
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    sample=np.asarray(Image.open(hosts[0]).convert('RGB'),dtype=np.uint8)
    sample_bgr=cv2.cvtColor(sample,cv2.COLOR_RGB2BGR)
    supported, unsupported = preflight_official(
        module, selected, args.severities, sample_bgr,
        hosts[0].name, watermarks[0].name, args.seed, companion)
    preflight = {
        'provider': 'official_MMWatermark_Robustness',
        'official_commit': upstream_sha,
        'official_source_sha256': hashlib.sha256(official_source.read_bytes()).hexdigest(),
        'scikit_image_gaussian_keyword_adapter': bool(module._tempo_skimage_gaussian_compat),
        'image_shape': list(sample.shape),
        'requested_conditions': len(selected)*len(args.severities),
        'supported_conditions': [{'attack': name, 'severity': sev} for name, sev in supported],
        'unsupported_conditions': unsupported,
        'warning': ('Unsupported upstream conditions are excluded, not approximated with local '
                    'TEMPO attacks. Full external coverage requires resolving these incompatibilities.'),
    }
    save_json(out/'official_mmwatermark_preflight.json', preflight)
    if args.strict_official_coverage and unsupported:
        raise RuntimeError(f'{len(unsupported)} official conditions failed the 512x512 preflight; '
                           f'see {out / "official_mmwatermark_preflight.json"}')
    if not supported:
        raise RuntimeError('No official upstream attack conditions passed preflight; '
                           'see official_mmwatermark_preflight.json')
    print(f'Official preflight supported {len(supported)}/'
          f'{len(selected)*len(args.severities)} conditions', flush=True)
    rows=[];failures=[]
    expected_full=len(hosts)*len(watermarks)*len(selected_methods)*(1+len(selected)*len(args.severities))
    expected=len(hosts)*len(watermarks)*len(selected_methods)*(1+len(supported))
    for wm_i,wp in enumerate(watermarks):
        wm=load_binary_watermark(wp,size=64)
        for hp in hosts:
            host=load_rgb(hp)
            if host.shape[:2] != (512,512):
                raise ValueError(f'Host must be 512x512 for the primary fixed-capacity comparison: {hp}')
            for mid in selected_methods:
                if mid == 'proposal':params=config.to_dict()
                else:params=None
                seed=trial_seed(args.seed,hp.name,wp.name,mid,'embed',0)
                t0=time.perf_counter_ns()
                marked,key=embed(mid,host,wm,seed=seed,repeat='auto',method_params=params)
                embed_ms=(time.perf_counter_ns()-t0)/1e6
                clean_psnr=psnr(host,marked)
                conditions=[('clean',0,None)]+[(name,sev,getattr(module,name)) for name,sev in supported]
                for name,severity,fn in conditions:
                    label='clean' if fn is None else f'mmw::{name}::severity{severity}'
                    try:
                        attacked=marked if fn is None else cv2.cvtColor(call_official(
                            fn,cv2.cvtColor(marked,cv2.COLOR_RGB2BGR),severity,
                            seed=trial_seed(args.seed,hp.name,wp.name,mid,name,severity),
                            companion_repo=companion),cv2.COLOR_BGR2RGB)
                        t1=time.perf_counter_ns()
                        recovered=extract(attacked,key)
                        extract_ms=(time.perf_counter_ns()-t1)/1e6
                        rows.append({'provider':'official_MMWatermark_Robustness',
                          'provider_commit':upstream_sha,'method_id':mid,
                          'detector_class':fidelity_status(mid)['blindness_tier'],
                          'host':hp.name,'watermark':wp.name,'attack':label,
                          'severity':severity,'payload_bits':int(wm.size),
                          'clean_embedding_psnr':clean_psnr,'nc_bipolar':nc_binary(wm,recovered),
                          'ber':ber(wm,recovered),'embed_ms':embed_ms,'extract_ms':extract_ms})
                    except Exception as exc:
                        failures.append({'method':mid,'host':hp.name,'watermark':wp.name,
                                         'attack':label,'error':repr(exc)})
                # Write partial evidence after each host/method so a Kaggle
                # timeout does not discard the full external benchmark run.
                if rows:
                    partial_csv=out/'official_mmwatermark_raw.csv.part'
                    save_csv(partial_csv,rows)
                    partial_csv.replace(out/'official_mmwatermark_raw.csv')
                progress={'status':'RUNNING_PARTIAL', 'provider_commit':upstream_sha,
                          'last_host':hp.name,'last_watermark':wp.name,
                          'last_method':mid,'completed_rows':len(rows),
                          'expected_compatible_rows':expected,
                          'expected_full_requested_rows':expected_full,
                          'failure_count':len(failures),
                          'preflight_unsupported_count':len(unsupported)}
                save_json(out/'official_mmwatermark_progress.json',progress)
                print('Official progress:',len(rows),'/',expected,
                      'rows;',len(failures),'failed trials',flush=True)
    save_csv(out/'official_mmwatermark_raw.csv',rows)
    manifest={'provider':'official_MMWatermark_Robustness','official_url':OFFICIAL_URL,
              'official_commit':upstream_sha,'official_source_sha256':hashlib.sha256(official_source.read_bytes()).hexdigest(),
              'companion_commit':companion_sha,'selected_attack_functions':selected,
              'severities':args.severities,'methods':selected_methods,
              'hosts':[x.name for x in hosts],'watermarks':[x.name for x in watermarks],
              'expected_full_requested_rows':expected_full,
              'expected_compatible_rows':expected,
              'expected_rows':expected,'completed_rows':len(rows),'failures':failures,
              'preflight_unsupported_conditions':unsupported,
              'scikit_image_gaussian_keyword_adapter':bool(module._tempo_skimage_gaussian_compat),
              'complete':len(rows)==expected and not failures,
              'full_coverage':len(rows)==expected_full and not failures and not unsupported,
              'scope':'official perturbations, not official model baselines; provider IDs are distinct from canonical attacks'}
    save_json(out/'official_mmwatermark_manifest.json',manifest)
    save_json(out/'official_mmwatermark_progress.json', {
        'status': 'COMPLETED_ALL_REQUESTED' if manifest['full_coverage'] else
                  'COMPLETED_COMPATIBLE_ONLY' if manifest['complete'] else 'FAILED_TRIALS',
        'completed_rows':len(rows), 'expected_compatible_rows':expected,
        'expected_full_requested_rows':expected_full,
        'preflight_unsupported_count':len(unsupported),
        'failure_count':len(failures), 'provider_commit':upstream_sha,
    })
    print('Official external compatible results:',len(rows),'/',expected,
          '(full requested:',expected_full,')')
    if unsupported:
        print(f'External coverage PARTIAL: {len(unsupported)} unsupported upstream conditions. '
              'See official_mmwatermark_preflight.json.',flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} official perturbation/decode trials failed; see manifest')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-repo',type=Path,default=Path('/kaggle/working/MMWatermark-Robustness'))
    p.add_argument('--companion-repo',type=Path,default=None)
    p.add_argument('--no-clone',action='store_true')
    p.add_argument('--expected-sha',default=None)
    p.add_argument('--host-dir',type=Path,default=Path('data/user_dataset/host'))
    p.add_argument('--watermark-dir',type=Path,default=Path('data/user_dataset/watermark'))
    p.add_argument('--config',type=Path,default=Path('configs/frozen_selected_20261021.json'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--methods',default='proposal')
    p.add_argument('--attacks',default='',help='Comma-separated official upstream function names; empty means all discovered')
    p.add_argument('--severities',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--host-limit',type=int,default=0)
    p.add_argument('--watermark-limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--strict-official-coverage',action='store_true',
                   help='Abort if any requested upstream attack/severity fails preflight')
    main(p.parse_args())
