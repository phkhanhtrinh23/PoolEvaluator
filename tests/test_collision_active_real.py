import numpy as np

from zoo.collision_active_real import labels_at_budget


def test_judge_verdict_replaces_or_rejects_pseudo_label():
    labels = np.array([1, 2, 3])
    log = [dict(item=0, class_id=2, choice=1),
           dict(item=2, class_id=20_000_001, choice=None)]
    changed, applied = labels_at_budget(labels, log, 2)
    np.testing.assert_array_equal(changed, [2, 2, 20_000_001])
    assert all(row["changed"] for row in applied)
