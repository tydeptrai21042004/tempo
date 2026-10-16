from pathlib import Path
import importlib.util
import numpy as np


def get_script(name):
    source=Path(__file__).resolve().parents[1] / 'scripts' / name
    spec=importlib.util.spec_from_file_location(name[:-3],source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewer_holdout_disjoint(tmp_path):
    split=get_script('create_reviewer_holdout.py')
    hostdir=tmp_path/'hosts'; hostdir.mkdir()
    for i in range(5):
        (hostdir/f'host{i}.bmp').write_bytes(bytes([i+3]) * 64)
    res=split.split_hosts(hostdir,tmp_path/'out',development_count=2)
    assert res['split_sizes']=={'development':2,'test':3}
    assert len(set(e['sha256'] for e in res['development']) & set(e['sha256'] for e in res['test'])) == 0


def test_matched_psnr_primary_summary_excludes_ineligible_rows():
    mod=get_script('run_multiwatermark_fair_benchmark.py')
    pairs=[{'method_id':'proposal','primary_rank_eligible':True,'evaluation_watermarked_psnr':45.,'clean_bit_accuracy':1},
           {'method_id':'proposal','primary_rank_eligible':False,'evaluation_watermarked_psnr':33.,'clean_bit_accuracy':1}]
    attacks=[{'method_id':'proposal','primary_rank_eligible':True,'phase':'jpeg','nc_bipolar':0.9},
             {'method_id':'proposal','primary_rank_eligible':False,'phase':'jpeg','nc_bipolar':0.1}]
    eligible=mod._aggregate(pairs,attacks,primary_only=True)
    assert eligible[0]['host_watermark_pairs']==1
    assert np.isclose(eligible[0]['mean_evaluation_psnr'],45)
    assert np.isclose(eligible[0]['mean_attack_nc'],0.9)


def test_host_cluster_paired_effect_is_not_attack_pseudoreplication():
    module=get_script('reviewer_statistics.py')
    rows=[]
    for host,offset in [('host_A',0.0),('host_B',0.05)]:
        for attack in ['jpeg','rot']:
            rows.extend([
                {'host':host,'watermark_name':'wm.png','attack':attack,
                 'method_id':'proposal','nc_bipolar':str(.9-offset)},
                {'host':host,'watermark_name':'wm.png','attack':attack,
                 'method_id':'baseline','nc_bipolar':str(.7-offset)},
            ])
    summary,effect=module.compute(rows,seed=2026,nboot=100)
    assert len(summary)==2
    assert len(effect)==1
    assert effect[0]['n_paired_hosts']==2
    assert effect[0]['n_paired_conditions']==4
    assert np.isclose(effect[0]['mean_delta_nc_proposal_minus_baseline'],.2)
