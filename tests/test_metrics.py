from damageactu.evaluation.metrics import macro_f1, ordinal_mae


def test_perfect_metrics():
    y = [0,1,2,3]
    assert macro_f1(y,y) == 1.0
    assert ordinal_mae(y,y) == 0.0
