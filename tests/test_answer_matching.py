"""Matchers for free-text answers.

The embedding and entailment backends are driven by INJECTED scorers here, so the tests
are deterministic and need no model download. The point being pinned down is the
asymmetry: a matcher that merges two different answers manufactures agreement, while one
that splits an answer only loses signal.
"""
import numpy as np
import pytest

from pooleval.answer_matching import (EmbeddingMatcher, EntailmentMatcher,
                                      ExactMatcher, LabelGroundingMatcher,
                                      NormalizeMatcher, normalize_answer)
from pooleval.partitions import partition_quality


# --------------------------------------------------------------------------- #
#  Normalisation                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "dog", "a dog", "A Dog", "It's a dog", "the answer is dog", "The answer is a dog",
    "I think it is a dog.", "This is a dog!", "answer: dog", "  dog  ",
])
def test_frames_articles_and_punctuation_all_peel_to_the_same_string(text):
    assert normalize_answer(text) == "dog"


def test_normalisation_does_not_collapse_distinct_labels():
    assert normalize_answer("dog") != normalize_answer("cat")
    assert normalize_answer("Neural Networks") != normalize_answer("Rule Learning")


def test_normalisation_cannot_reach_alias_variation():
    """The honest limit of string surgery: it unifies frames, never vocabulary. This is
    why the hard surface condition exists in the experiment."""
    assert normalize_answer("3") != normalize_answer("three")
    assert normalize_answer("NN") != normalize_answer("Neural Networks")


# --------------------------------------------------------------------------- #
#  Clustering behaviour                                                        #
# --------------------------------------------------------------------------- #
ANSWERS = ["dog", "a dog", "It's a dog", "cat", "The answer is cat"]
ORACLE = np.array([0, 0, 0, 1, 1])


def test_exact_match_shatters_one_meaning_into_many_classes():
    labels = ExactMatcher().cluster(ANSWERS)
    assert len(set(labels.tolist())) == 5
    q = partition_quality(labels, ORACLE)
    assert q["recall"] == 0.0             # it finds none of the true equivalences
    # It also merges nothing, so precision has no pairs to score. partition_quality
    # reports 0.0 rather than a flattering vacuous 1.0; n_predicted_same is what says
    # the number is undefined rather than bad.
    assert q["n_predicted_same"] == 0 and q["precision"] == 0.0


def test_normalisation_recovers_the_oracle_partition_on_frame_variation():
    labels = NormalizeMatcher().cluster(ANSWERS)
    q = partition_quality(labels, ORACLE)
    assert q["precision"] == 1.0 and q["recall"] == 1.0


def test_a_matcher_that_merges_two_labels_shows_up_as_low_precision():
    """The failure that matters. Recall stays perfect, so only precision catches it."""
    merged = np.zeros(5, dtype=int)
    q = partition_quality(merged, ORACLE)
    assert q["recall"] == 1.0
    assert q["precision"] < 0.5


# --------------------------------------------------------------------------- #
#  Embedding backend                                                           #
# --------------------------------------------------------------------------- #
def _fake_encoder(table):
    def encode(texts):
        return np.array([table[t] for t in texts], dtype=float)
    return encode


def test_embedding_matcher_merges_above_the_threshold_and_splits_below():
    vecs = {"dog": [1.0, 0.0], "a dog": [0.99, 0.141], "cat": [0.0, 1.0]}
    vecs = {k: np.array(v) / np.linalg.norm(v) for k, v in vecs.items()}
    m = EmbeddingMatcher(threshold=0.9, encoder=_fake_encoder(vecs))
    assert m.equivalent("dog", "a dog")
    assert not m.equivalent("dog", "cat")


def test_a_crowded_embedding_space_cannot_be_fixed_by_the_threshold():
    """When a paraphrase and a different class sit at the same cosine, no threshold
    separates them -- it either merges both or splits both. Measured on the real
    encoder: dog/"It's a dog" = 0.666 and dog/cat = 0.661."""
    vecs = {"dog": [1.0, 0.0, 0.0],
            "It's a dog": [0.666, 0.746, 0.0],
            "cat": [0.661, 0.0, 0.750]}
    vecs = {k: np.array(v) / np.linalg.norm(v) for k, v in vecs.items()}
    m_loose = EmbeddingMatcher(threshold=0.65, encoder=_fake_encoder(vecs))
    m_tight = EmbeddingMatcher(threshold=0.70, encoder=_fake_encoder(vecs))
    assert m_loose.equivalent("dog", "It's a dog") and m_loose.equivalent("dog", "cat")
    assert not m_tight.equivalent("dog", "It's a dog") and not m_tight.equivalent("dog", "cat")


# --------------------------------------------------------------------------- #
#  Entailment backend                                                          #
# --------------------------------------------------------------------------- #
def _fake_entailment(scores):
    def score(pairs):
        return [scores.get(p, 0.0) for p in pairs]
    return score


def test_entailment_requires_BOTH_directions():
    """One-way entailment is not equivalence: "a poodle" entails "a dog" while "a dog"
    does not entail "a poodle", and they are different answers."""
    scores = {("poodle", "dog"): 0.95, ("dog", "poodle"): 0.10,
              ("dog", "a dog"): 0.94, ("a dog", "dog"): 0.93}
    m = EntailmentMatcher(threshold=0.5, scorer=_fake_entailment(scores))
    assert not m.equivalent("poodle", "dog")
    assert m.equivalent("dog", "a dog")


def test_entailment_scores_are_memoised_per_ordered_pair():
    calls = []

    def score(pairs):
        calls.extend(pairs)
        return [0.9] * len(pairs)

    m = EntailmentMatcher(threshold=0.5, scorer=score)
    for _ in range(5):
        m.equivalent("a", "b")
    assert len(calls) == 2            # one per direction, then cached


def test_warm_prescore_covers_every_ordered_pair_of_distinct_answers():
    m = EntailmentMatcher(threshold=0.5, scorer=lambda pairs: [0.9] * len(pairs))
    n = m.warm(["a", "b", "c", "a"])
    assert n == 6                     # 3 distinct answers -> 3 * 2 ordered pairs


# --------------------------------------------------------------------------- #
#  Label grounding                                                             #
# --------------------------------------------------------------------------- #
def test_grounding_maps_every_phrasing_onto_the_declared_label():
    m = LabelGroundingMatcher(["dog", "cat"])
    labels = m.cluster(ANSWERS)
    q = partition_quality(labels, ORACLE)
    assert q["precision"] == 1.0 and q["recall"] == 1.0


def test_grounding_cannot_invent_a_class_the_label_set_lacks():
    """Its safety property: the output space is closed, so two labels can never merge."""
    m = LabelGroundingMatcher(["dog", "cat"])
    assert not m.equivalent("dog", "cat")
    assert m.ground("dog") != m.ground("cat")


def test_an_answer_that_grounds_nowhere_keeps_its_own_identity():
    m = LabelGroundingMatcher(["dog", "cat"])
    assert not m.equivalent("a wolverine", "dog")
    assert not m.equivalent("a wolverine", "a badger")
    assert m.ungrounded > 0


def test_grounding_falls_back_to_the_backup_matcher_when_the_surface_misses():
    vecs = {"3": [1.0, 0.0], "three": [0.99, 0.141], "seven": [0.0, 1.0]}
    vecs = {k: np.array(v) / np.linalg.norm(v) for k, v in vecs.items()}
    backup = EmbeddingMatcher(threshold=0.9, encoder=_fake_encoder(vecs))
    m = LabelGroundingMatcher(["three", "seven"], backup=backup)
    assert m.equivalent("3", "three")
    assert not m.equivalent("3", "seven")
