from __future__ import annotations
from pathlib import Path
import argparse, json, math
import pandas as pd

TRAIN_EVENTS = {
    'lower-puna-volcano','palu-tsunami','mexico-earthquake','socal-fire','woolsey-fire',
    'portugal-wildfire','pinery-bushfire','midwest-flooding','moore-tornado','joplin-tornado',
    'hurricane-harvey','hurricane-michael','hurricane-florence'
}
TEST_EVENTS = {
    'nepal-flooding','guatemala-volcano','sunda-tsunami','santa-rosa-wildfire',
    'hurricane-matthew','tuscaloosa-tornado'
}
EXPECTED_TRAIN_COUNTS = {0:212466,1:15518,2:19528,3:17761}


def check(cond, name, detail=''):
    if not cond:
        raise AssertionError(f'FAIL: {name}' + (f' | {detail}' if detail else ''))
    print('PASS:', name + (f' | {detail}' if detail else ''))


def close(a,b,tol=1e-10):
    return abs(float(a)-float(b)) <= tol


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root', default='.')
    args=ap.parse_args()
    r=Path(args.repo_root).resolve()

    # Phase7C frozen patch roles
    p7c=pd.read_csv(r/'manifests/event_split/phase7d_frozen_unique_patch_roles.csv.gz', dtype=str)
    counts=p7c['event_role'].value_counts().to_dict()
    check(len(p7c)==11034,'Phase7C total patch count',str(len(p7c)))
    check(counts=={'train':8202,'test':1920,'val':912},'Phase7C role counts',str(counts))
    check(p7c['pair_key'].is_unique,'Phase7C pair_key unique')
    check(not p7c.duplicated(['event','patch_id']).any(),'Phase7C (event,patch_id) unique')
    dev=set(p7c.loc[p7c.event_role.isin(['train','val']),'event'])
    tst=set(p7c.loc[p7c.event_role.eq('test'),'event'])
    check(dev==TRAIN_EVENTS,'Phase7C development event set exact')
    check(tst==TEST_EVENTS,'Phase7C held-out event set exact')
    check(dev.isdisjoint(tst),'No event leakage development vs TEST')

    # Phase7D-A manifests
    tr=pd.read_csv(r/'manifests/buildings/event_train_buildings.csv.gz',low_memory=False)
    va=pd.read_csv(r/'manifests/buildings/event_val_buildings.csv.gz',low_memory=False)
    dm=pd.read_csv(r/'manifests/wrong_pre/event_val_hard_wrongpre_donor_map.csv.gz',low_memory=False)
    check(len(tr)==265273,'Phase7D-A train rows',str(len(tr)))
    check(len(va)==30735,'Phase7D-A validation rows',str(len(va)))
    check(tr['building_id'].astype(str).is_unique,'Train building IDs unique')
    check(va['building_id'].astype(str).is_unique,'Validation building IDs unique')
    check(not (set(tr.building_id.astype(str)) & set(va.building_id.astype(str))),'No train/val building overlap')
    check(not (set(tr.scene_id.astype(str)) & set(va.scene_id.astype(str))),'No train/val scene overlap')
    check(set(tr.disaster.astype(str))==TRAIN_EVENTS,'Train event set exact')
    check(set(va.disaster.astype(str))==TRAIN_EVENTS,'Validation event set exact')
    cc={int(k):int(v) for k,v in tr.label_id.value_counts().sort_index().to_dict().items()}
    check(cc==EXPECTED_TRAIN_COUNTS,'Event-train class counts exact',str(cc))
    check(len(dm)==30698,'Validation wrong-PRE donor rows',str(len(dm)))
    check(dm.target_building_id.astype(str).is_unique,'Wrong-PRE targets unique')
    check((dm.target_building_id.astype(str)!=dm.donor_building_id.astype(str)).all(),'No self donors')
    by=va.assign(building_id=va.building_id.astype(str)).set_index('building_id')
    target_scene=dm.target_building_id.astype(str).map(by.scene_id.astype(str))
    donor_scene=dm.donor_building_id.astype(str).map(by.scene_id.astype(str))
    check(target_scene.notna().all() and donor_scene.notna().all(),'All wrong-PRE IDs exist in validation manifest')
    check((target_scene.to_numpy()==donor_scene.to_numpy()).all(),'All wrong-PRE donors same-scene')
    cov=len(dm)/len(va)
    check(close(cov,0.9987961607288108,1e-15),'Wrong-PRE validation coverage',f'{cov:.16f}')

    # Phase7D-B evidence
    bidx=json.loads((r/'results/development/phase7d_b_results_index.json').read_text())
    check(bidx['status']=='PHASE7D_B_SEED_COMPLETE','Phase7D-B status')
    check(bidx['seed']==42,'Phase7D-B expert seed is 42')
    check(bidx['epochs_completed']==8,'Phase7D-B completed 8 epochs')
    check(bidx['event_test_used'] is False,'Phase7D-B TEST remained sealed')
    check(bidx['router_trained'] is False,'Phase7D-B did not train router')
    check(bidx['post']['best_epoch']==8 and bidx['siamese']['best_epoch']==8,'Both expert best epochs are 8')

    # Phase7D-C evidence
    cidx=json.loads((r/'results/development/phase7d_c_results_index.json').read_text())
    check(cidx['status']=='PHASE7D_C_S1_COMPLETE_FINAL_MODELS_FROZEN','Phase7D-C final freeze status')
    check(cidx['expert_seed']==42,'Phase7D-C expert seed is 42')
    check(cidx['random_seed_robustness_claim_allowed'] is False,'Random-seed robustness claim forbidden')
    check(cidx['event_test_used'] is False,'Phase7D-C TEST remained sealed')
    check(cidx['neural_best_epoch']==1,'Frozen neural router epoch is 1')
    check(close(cidx['static_alpha'],0.45),'Frozen static alpha is 0.45')
    alpha=json.loads((r/'checkpoints/routers/final_static_alpha.json').read_text())
    check(close(alpha['alpha'],0.45),'Final static-alpha artifact equals 0.45')

    # Phase7D-D final sealed test evidence
    didx=json.loads((r/'results/heldout_events/results_index.json').read_text())
    check(didx['status']=='PHASE7D_D_S1B_SEALED_TEST_COMPLETE_NO_RETUNING','Final TEST status')
    check(didx['protocol_sha256']=='faa268d83edda817e013bc0166fe9ade2ad20f3cda71b1bf08e44b092878c533','Final TEST protocol SHA')
    check(didx['post_test_tuning_permitted'] is False,'Post-test tuning forbidden')
    check(didx['expert_seed']==42 and didx['random_seed_robustness_evaluated'] is False,'Final evidence is single-initialization only')
    check(didx['test_patch_count']==1920,'Final TEST patch count',str(didx['test_patch_count']))
    check(didx['test_building_count']==115349,'Final TEST building count',str(didx['test_building_count']))
    check(set(didx['test_events'])==TEST_EVENTS,'Final TEST event set exact')
    check(didx['clean_effectiveness_pass'] is False,'Pre-specified clean effectiveness criterion failed')
    check(didx['class_safety_pass'] is True,'Class-safety criterion passed')
    check(didx['wrongpre_safety_pass'] is True,'Wrong-PRE safety criterion passed')
    check(didx['gate_suppression_pass'] is True,'Gate-suppression criterion passed')
    check(didx['utility_generalization_pass'] is True,'Utility-generalization criterion passed')
    check(didx['all_precommitted_criteria_pass'] is False,'All-precommitted-criteria flag remains FALSE')

    mt=pd.read_csv(r/'results/heldout_events/test_metrics_clean.csv')
    def metric(method,col='macro_f1'):
        x=mt.loc[mt.method.eq(method),col]
        if len(x)!=1: raise AssertionError(f'Expected one row for {method}, got {len(x)}')
        return float(x.iloc[0])
    post=metric('POST-only')
    neural=metric('Neural Temporal Utility Router')
    check(close(post,0.4565629213,5e-10),'Final POST clean Macro-F1',f'{post:.10f}')
    check(close(neural,0.4596737025,5e-10),'Final neural clean Macro-F1',f'{neural:.10f}')
    check(close(neural-post,didx['primary_clean_neural_minus_post'],5e-10),'Primary neural-minus-POST gain agrees with results index',f'{neural-post:.10f}')

    util=json.loads((r/'results/heldout_events/test_neural_utility.json').read_text())
    check(util['n_clean_test']==115349,'Utility population size')
    check(util['n_correctness_discordant']==14340,'Correctness-discordant utility cases')
    check(close(util['auroc'],0.77283834848973,1e-12),'Final utility AUROC',f"{util['auroc']:.12f}")
    check(close(util['spearman_gate_deltaCE'],0.10360709395401074,1e-12),'Final gate-deltaCE Spearman',f"{util['spearman_gate_deltaCE']:.12f}")

    print('\nCANONICAL EVIDENCE AUDIT: PASS')
    print('The restored Phase7C-7D evidence is internally consistent with the frozen paper record.')


if __name__=='__main__':
    main()
