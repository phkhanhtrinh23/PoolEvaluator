"""Matchers that decide when two FREE-TEXT answers mean the same thing.

For a VLM or GLM the answer to "what is in this image?" is a sentence, not a label
index: "dog", "a dog", "It's a dog", "The answer is dog" are one meaning wearing four
surfaces.  Exact string match sees four classes, and since every estimator in this repo
reads agreement through one equality test, that mis-count propagates straight into
``e``, ``gamma`` and the latent posterior.

Each matcher here exposes ``cluster(answers) -> integer labels``, so it drops into
:func:`pooleval.semantic.greedy_meaning_clusters` as the ``equivalent`` predicate and
into the estimator as a partition.  They are deliberately different KINDS of evidence,
and they fail in different directions:

    ExactMatcher        string equality.  The baseline that breaks.
    NormalizeMatcher    strip casing, punctuation, articles and meta-frames, then
                        compare.  Deterministic, no model, no threshold to tune.
    EmbeddingMatcher    sentence-embedding cosine over a threshold.
    EntailmentMatcher   BIDIRECTIONAL entailment, as in Farquhar et al. (Nature 2024):
                        a == b iff a entails b AND b entails a, judged by an NLI model.
    LabelGroundingMatcher
                        map each answer back onto a KNOWN, CLOSED label set and compare
                        the labels.  Only available when such a set exists -- which for
                        image and node classification it does.

Which one is right is an empirical question with a clear asymmetry: a matcher that
MERGES two different answers manufactures agreement and corrupts every downstream
estimate, while one that SPLITS an answer merely loses signal.  Measure with
:func:`pooleval.semantic.partition_quality` before adopting.
"""
import functools
import re

import numpy as np

from .partitions import greedy_meaning_clusters

__all__ = ["ExactMatcher", "NormalizeMatcher", "EmbeddingMatcher",
           "EntailmentMatcher", "LabelGroundingMatcher", "normalize_answer"]


# --------------------------------------------------------------------------- #
#  Deterministic surface normalisation                                         #
# --------------------------------------------------------------------------- #
_META = re.compile(
    r"^\s*(?:i\s+(?:think|believe|would\s+say)\s*(?:it(?:'s|\s+is)?)?"
    r"|the\s+(?:answer|label|class|category|topic)\s+is"
    r"|this\s+(?:is|node\s+is\s+about|paper\s+is\s+about|image\s+shows|shows)"
    r"|it(?:'s|\s+is)"
    r"|that(?:'s|\s+is)"
    r"|looks\s+like"
    r"|probably"
    r"|answer\s*:"
    r"|label\s*:"
    r")\s*", re.I)
_ARTICLE = re.compile(r"^\s*(?:an?|the)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]+")


def normalize_answer(text, rounds=4):
    """Lowercase, drop punctuation, and peel meta-frames and articles until stable.

    Peeling repeats because the frames nest -- "I think it's a dog" is a hedge, then a
    copula, then an article. Four rounds is well past the depth anything real reaches.
    """
    s = str(text or "").strip()
    for _ in range(rounds):
        before = s
        s = _META.sub("", s)
        s = _ARTICLE.sub("", s)
        if s == before:
            break
    s = _PUNCT.sub(" ", s.lower())
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
#  Matchers                                                                    #
# --------------------------------------------------------------------------- #
class _Matcher:
    name = "base"

    def equivalent(self, a, b):
        raise NotImplementedError

    def cluster(self, answers):
        answers = list(answers)
        labels, _ = greedy_meaning_clusters(
            len(answers), lambda i, j: self.equivalent(answers[i], answers[j]))
        return labels


class ExactMatcher(_Matcher):
    name = "exact string"

    def equivalent(self, a, b):
        return a == b


class NormalizeMatcher(_Matcher):
    """Surface stripping, then equality. No model, no threshold, fully reproducible."""

    name = "normalize"

    def equivalent(self, a, b):
        return normalize_answer(a) == normalize_answer(b)


class EmbeddingMatcher(_Matcher):
    """Sentence-embedding cosine over a threshold.

    Included because it is the obvious thing to reach for, and because on SHORT label
    phrasings it does not work: the crowding of a general-purpose embedding space puts
    "dog" and "cat" at roughly the same cosine as "dog" and "It's a dog", so no single
    threshold separates paraphrase from a different class. The experiment measures that
    rather than taking it on faith.
    """

    name = "embedding"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", threshold=0.75,
                 encoder=None):
        self.threshold = float(threshold)
        if encoder is not None:
            self._encode = encoder
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model)
            self._encode = lambda xs: self._model.encode(
                xs, normalize_embeddings=True, show_progress_bar=False)
        self._cache = {}

    def _vec(self, text):
        if text not in self._cache:
            self._cache[text] = np.asarray(self._encode([text])[0], dtype=float)
        return self._cache[text]

    def equivalent(self, a, b):
        return float(self._vec(a) @ self._vec(b)) >= self.threshold


class EntailmentMatcher(_Matcher):
    """Bidirectional entailment -- the clustering rule of Farquhar et al. (2024).

    ``a == b`` iff an NLI model says a entails b AND b entails a. The relation is not
    transitive, which is why the caller clusters greedily against representatives
    rather than taking connected components; see
    :func:`pooleval.semantic.greedy_meaning_clusters`.

    Scores are memoised on the ordered string pair. The answer vocabulary of a pool is
    small -- a handful of labels times a handful of phrasings -- so the whole run costs
    a few thousand NLI calls no matter how many items there are.
    """

    name = "entailment"

    def __init__(self, model="cross-encoder/nli-deberta-v3-xsmall", threshold=0.5,
                 scorer=None, batch_size=64):
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        if scorer is not None:
            self._score = scorer
        else:
            import warnings
            from transformers import pipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = pipeline("text-classification", model=model, top_k=None)
            self._score = functools.partial(_entail_scores, pipe)
        self._cache = {}

    def _entails(self, a, b):
        key = (a, b)
        if key not in self._cache:
            self._cache[key] = float(self._score([(a, b)])[0])
        return self._cache[key]

    def warm(self, answers):
        """Pre-score every ordered pair of the DISTINCT answers, in batches."""
        uniq = sorted(set(answers))
        pairs = [(a, b) for a in uniq for b in uniq
                 if a != b and (a, b) not in self._cache]
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            for pair, score in zip(chunk, self._score(chunk)):
                self._cache[pair] = float(score)
        return len(pairs)

    def equivalent(self, a, b):
        if a == b:
            return True
        return (self._entails(a, b) >= self.threshold
                and self._entails(b, a) >= self.threshold)


def _entail_scores(pipe, pairs):
    out = pipe([{"text": a, "text_pair": b} for a, b in pairs])
    scores = []
    for row in out:
        d = {x["label"].lower(): x["score"] for x in row}
        scores.append(d.get("entailment", 0.0))
    return scores


class LabelGroundingMatcher(_Matcher):
    """Map each answer onto a KNOWN closed label set, then compare the labels.

    When the task's answer space is closed -- ten digits, five paper topics -- the
    cheapest correct move is not to cluster free text at all but to ground it back onto
    the label set the task actually has. That is what constrained decoding does at
    generation time, and this is its post-hoc equivalent. It cannot invent a class the
    label set lacks, so it cannot merge two labels the way a coarse embedder can.

    Grounding is by normalised containment first (exact, then substring), falling back
    to ``backup`` (an embedding matcher, say) when the surface matches nothing. An
    answer that grounds nowhere keeps an identity of its own rather than being forced
    into the nearest label.

    ``aliases`` maps a label to the other surfaces it is known to appear as -- "3" and
    "the digit 3" for ``three``, "NN" for ``Neural Networks``.  This is the part a
    practitioner actually has: a closed label set almost always arrives with the
    vocabulary its classes are named in, and declaring it costs one dictionary.  It is
    what lets grounding handle the variation that defeats both string surgery (which
    cannot unify "3" with "three") and bidirectional entailment (which cannot either,
    because "3" does not ENTAIL "numeral 3" even though both name the same class).
    """

    name = "label grounding"

    def __init__(self, labels, backup=None, aliases=None):
        self.labels = [str(x) for x in labels]
        self.backup = backup
        aliases = aliases or {}
        # Every declared surface of label i, longest first, so "the digit 3" is tried
        # before "3" and a longer name never loses to a shorter one it contains.
        self._alias = []
        for lab in self.labels:
            forms = {normalize_answer(x) for x in ([lab] + list(aliases.get(lab, [])))}
            self._alias.append(sorted((f for f in forms if f), key=len, reverse=True))
        self._cache = {}
        self.ungrounded = 0

    def ground(self, text):
        if text in self._cache:
            return self._cache[text]
        s = normalize_answer(text)
        hit = None
        for i, forms in enumerate(self._alias):
            if s in forms:
                hit = i
                break
        if hit is None:
            matches = {i for i, forms in enumerate(self._alias)
                       for f in forms if re.search(rf"\b{re.escape(f)}\b", s)}
            if len(matches) == 1:
                hit = matches.pop()
        if hit is None and self.backup is not None:
            for i, lab in enumerate(self.labels):
                if self.backup.equivalent(text, lab):
                    hit = i
                    break
        if hit is None:
            self.ungrounded += 1
            hit = ("ungrounded", s)
        self._cache[text] = hit
        return hit

    def equivalent(self, a, b):
        return self.ground(a) == self.ground(b)
