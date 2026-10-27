#!/usr/bin/env python3
"""Diagnose independent Haar-HD-SVD QIM video recovery; do NOT change decodes.

This adapter is NOT the author's 2024 algorithm. The report identifies
bitwise complement anomalies and unmatched embedding PSNR, not a correction.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import nc_binary,load_binary_watermark

METHOD='haar_hd_svd_qim_adapter2024'


def diagnose(path:Path, out:Path) -> dict:
    csvpath=path/'all_watermarks_video_summary.csv'
    if not csvpath.is_file():raise FileNotFoundError(csvpath)
    with csvpath.open(newline='',encoding='utf8') as stream:
        rows=list(csv.DictReader(stream))
    if not rows or not any(r['method_id']==METHOD for r in rows):
        raise ValueError('Independent modern adapter did not execute')
    observed={}
    for row in rows:
        name=row['watermark_name'];method=row['method_id']
        if (name,method) in observed:raise ValueError('Duplicate video method/watermark observations')
        source=Path(row['watermark_path']);decoded=Path(row['recovered_watermark'])
        if not source.is_file() or not decoded.is_file():
            raise FileNotFoundError(f'Original output unavailable: {source} / {decoded}')
        original=load_binary_watermark(source,64)
        extracted=load_binary_watermark(decoded,64,allow_degenerate=True)
        nc=float(nc_binary(original,extracted))
        if abs(nc-float(row['nc']))>1e-5:
            raise RuntimeError(f'Archived NC differs from decoded pixels: {name} / {method}')
        inverted=(255-np.asarray(extracted)).astype(np.uint8)
        comp=float(nc_binary(original,inverted))
        observed[(name,method)]={'watermark':name,'method':method,'raw_nc':nc,
            'complement_nc_diagnostic_only':comp,
            'precodec_mean_psnr_db':float(row['precodec_mean_psnr']),
            'bits_opposite_to_reference_fraction':float(np.mean(np.asarray(original)!=extracted)),
            'detector_type':row.get('detector_type','unknown')}
    results=[]
    for key in sorted({name for name,_ in observed}):
        if (key,METHOD) not in observed or (key,'proposal') not in observed:
            raise ValueError(f'No matched proposal + independent adapter observations for {key}')
        modern=observed[(key,METHOD)];baseline=observed[(key,'proposal')]
        results.append({'watermark':key,'adapter':modern,'proposal':baseline,
            'adapter_nc_near_negative_one':modern['raw_nc']<-.95,
            'different_psnr_operating_points':abs(modern['precodec_mean_psnr_db']-baseline['precodec_mean_psnr_db'])>.75})
    report={'status':'diagnostic_only','adapter_is_paper_exact':False,
        'decoder_outputs_changed':False,'method':METHOD,
        'important':'Complement NC is diagnostic only; NEVER report it as measured recovery or flip bits without mechanism validation.',
        'cases':results}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report=diagnose(a.video_dir,a.out)
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),
                      'near_inversion_count':sum(x['adapter_nc_near_negative_one'] for x in report['cases'])},indent=2))
