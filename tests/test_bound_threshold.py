from experiments.run_bound_threshold import downward_crossings
from pooleval.theory import bound_realized, bound_weighted


def test_downward_crossings():
    assert downward_crossings([1.2, 1.0, .8, 1.1, .9]) == [1, 4]
    assert downward_crossings([.9, .8, .7]) == []
    assert downward_crossings([1.0, .9]) == []
    assert downward_crossings([]) == []


def test_realized_reference_changes_threshold():
    L, c, m, N = .36399601, .5824, 8.6538, 150
    expected = bound_weighted(L, c, m)
    realized = bound_realized(L, c, m, N)
    simultaneous = bound_realized(L, c, m, N, delta=.05 / 100)
    assert expected < 1 < realized < simultaneous
