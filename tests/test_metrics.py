import numpy as np

def test_size_missmatch_valid():
    from pkoffee.metrics import check_size_match
    a = np.array([2,4,6,8])
    b = np.array([1,3,5,7])
    check_size_match(a,b)


def test_r2():
    from pkoffee.metrics import compute_r2
    rng=np.random.default_rng(seed=0)
    y_true=rng.normal(size=10)
    assert compute_r2(y_true,y_true) == 1.0
    
