import numpy as np
from mechanism_lab.core import Task, simulate, fit, compare, recover, intervention, MODELS

def test_reproducibility():
    task = Task(trials=40, reversal=20)
    assert simulate(task) == simulate(task)

def test_probabilities_and_lengths():
    for name in MODELS:
        rows = simulate(Task(trials=30, reversal=15), name)
        assert len(rows) == 30
        assert all(0 <= r['p_choice1'] <= 1 for r in rows)

def test_fit_and_comparison():
    rows = simulate(Task(trials=50, reversal=25))
    rankings = compare(rows)
    assert len(rankings) == len(MODELS)
    assert rankings == sorted(rankings, key=lambda x:x['bic'])
    assert np.isfinite(fit('qlearning',rows)['log_likelihood'])

def test_recovery():
    result = recover(repeats=2, task=Task(trials=40, reversal=20))
    assert sum(result['winner_counts'].values()) == 2

def test_intervention():
    result = intervention(Task(trials=40,reversal=20), changed=(.6,5.), repetitions=3)
    assert 0 <= result['intervention']['reward_rate'] <= 1
