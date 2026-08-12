import numpy as np

from zoo.collision_formulation_real import collision_statistics
from zoo.new_formulation_real import load_run


def test_source_collision_statistics_are_finite():
    stats = collision_statistics([load_run("spider")[0], load_run("bird")[0]])
    assert np.all((stats["gamma"] > 0) & (stats["gamma"] < 1))
    assert 0 < stats["beta"] < 1
    assert np.all(stats["eligible"] >= stats["hits"])
