import pandas as pd
from damageactu.evaluation.bootstrap import paired_scene_bootstrap


def test_scene_bootstrap_runs():
    df = pd.DataFrame({
        "scene_id":["a","a","b","b"],
        "true_label":[0,1,2,3],
    })
    a = [0,0,2,2]
    b = [0,1,2,3]
    out = paired_scene_bootstrap(df,a,b,n_replicates=20,seed=1)
    assert out["n_replicates"] == 20
