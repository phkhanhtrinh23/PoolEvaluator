"""RealJudge -- a label-free gpt-5-mini verifier for the real Spider zoo.

Implements the judge protocol ActivePoolEval expects (`query(run, obs, i) -> class`).
For a selected item it shows gpt-5-mini the natural-language question, the DB schema,
and each DISTINCT executed result the pool produced (one representative SQL + a small
row preview per result-equivalence class), then asks which result correctly answers
the question -- or NONE, when no candidate is right. It is label-free: the gold query
and gold result are never shown. The chosen result's equivalence class (an id in the
same space the estimator observes) is returned; NONE returns a fresh id absent from
the pool -- the candidate-coverage fix, executed on real data.

The gold labels in the PoolRun are used ONLY to (a) map the judge's chosen result to
its class id for bookkeeping and (b) score the final ranking; they never enter the
prompt or the judge's decision.
"""
import json
import re

import numpy as np

from .clients import chat
from .execute import run_query, result_key


_TBL = re.compile(r"\b(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z_][\w]*)", re.I)


def _referenced_tables(sqls):
    names = set()
    for s in sqls:
        for m in _TBL.finditer(s or ""):
            names.add(m.group(1).lower())
    return names


def _compact_schema(schema, sqls, max_chars):
    """One line per table `table(col type, ...)`, restricted to the tables the
    candidate SQLs actually reference (all tables if none match), so the judge sees
    the relevant schema without the whole database blowing the context budget."""
    items = schema.get("schema_items", []) if isinstance(schema, dict) else []
    refs = _referenced_tables(sqls)
    keep = [t for t in items if t.get("table_name", "").lower() in refs] or items
    lines = []
    for t in keep:
        cols = t.get("column_names", [])
        types = t.get("column_types", [""] * len(cols))
        cs = ", ".join(f"{c} {types[j] if j < len(types) else ''}".strip()
                       for j, c in enumerate(cols))
        lines.append(f"{t.get('table_name','?')}({cs})")
    return "\n".join(lines)[:max_chars]


def _preview(db_path, sql, timeout, max_rows=5):
    ok, payload, err = run_query(db_path, sql, timeout)
    if not ok:
        return f"[execution error: {err}]"
    rows, arity = payload
    if not rows:
        return "[empty result]"
    head = rows[:max_rows]
    body = "\n".join(str(tuple(r)) for r in head)
    more = f"\n... (+{len(rows) - max_rows} more rows)" if len(rows) > max_rows else ""
    return f"{len(rows)} row(s), {arity} col(s):\n{body}{more}"


class RealJudge:
    def __init__(self, items, member_names, preds_dev, true_class,
                 model="gpt-5-mini", timeout=5.0, max_tokens=2048, verbose=False,
                 max_schema_chars=2500, max_candidates=6, synthesize=False):
        # items[i] must align with PoolRun column i (same order build_poolrun used)
        self.items = items
        self.names = member_names
        self.preds = preds_dev
        self.tc = true_class
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.max_schema_chars = max_schema_chars
        self.max_candidates = max_candidates
        self.synthesize = synthesize
        self.calls = 0
        self._fresh = 20_000_000
        self.log = []

    def _distinct_classes(self, obs, i):
        """Up to max_candidates representatives (class_id, model_name, sql), one per
        distinct observed result class at item i, ordered by model support (so a
        prompt-length cap drops only the least-supported classes), errors excluded."""
        support, rep = {}, {}
        for m, name in enumerate(self.names):
            c = int(obs[m, i])
            if c < 0:                       # execution error -> no agreement, skip
                continue
            support[c] = support.get(c, 0) + 1
            rep.setdefault(c, (name, self.preds[name].get(self.items[i]["id"], "")))
        top = sorted(support, key=lambda c: -support[c])[:self.max_candidates]
        return {c: rep[c] for c in top}

    def _majority_class(self, obs, i):
        col = obs[:, i]
        vals, counts = np.unique(col, return_counts=True)
        return int(vals[np.argmax(counts)])

    def query(self, run, obs, i):
        self.calls += 1
        item = self.items[i]
        reps = self._distinct_classes(obs, i)
        if not reps:                        # everything errored -> nothing to certify
            self._fresh += 1
            return self._fresh
        classes = list(reps.keys())
        sqls = [reps[c][1] for c in classes]
        cand_txt = []
        for j, c in enumerate(classes):
            name, sql = reps[c]
            prev = _preview(item["db_path"], sql, self.timeout, max_rows=3)[:300]
            cand_txt.append(f"Candidate {j}:\nSQL: {sql[:400]}\nResult: {prev}")
        schema = _compact_schema(item.get("schema", {}), sqls, self.max_schema_chars)
        if self.synthesize:
            tail = ("\n\nReply with ONLY a JSON object. If one candidate's RESULT is "
                    "correct: {\"choice\": <candidate number>}. If NONE is correct, "
                    "WRITE a correct query yourself: {\"sql\": \"SELECT ...\"}.")
        else:
            tail = ("\n\nReply with ONLY a JSON object: {\"choice\": <candidate number>} "
                    "or {\"choice\": \"none\"}.")
        prompt = (
            "You are a strict Text-to-SQL grader. Given a database schema, a question, "
            "and several candidate SQL queries WITH their executed results, decide "
            "which candidate's RESULT correctly answers the question.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {item['question']}\n\n"
            + "\n\n".join(cand_txt) + tail
        )
        try:
            content = chat("openai", self.model, [{"role": "user", "content": prompt}],
                           max_tokens=self.max_tokens)
        except Exception as e:  # noqa  -- a failed judge call must not abort the run
            if self.verbose:
                print(f"  [judge] item {i}: API error {repr(e)[:70]} -> majority (no-op)")
            self.log.append(dict(item=int(i), error=repr(e)[:80],
                                class_id=self._majority_class(obs, i)))
            return self._majority_class(obs, i)   # safe no-op pin

        kind, val = self._parse(content, len(classes), self.synthesize)
        if kind == "choice":
            picked, note = classes[val], f"choice={val}"
        elif kind == "sql":                       # synthesis: map written SQL to a class
            picked, note = self._map_sql(item, val, classes, reps), "sql"
        else:                                     # "none" / unparseable
            self._fresh += 1
            picked, note = self._fresh, "none"
        if self.verbose:
            print(f"  [judge] item {i}: {len(classes)} classes -> {note} -> class {picked}")
        self.log.append(dict(item=int(i), n_classes=len(classes), verdict=note,
                            class_id=int(picked)))
        return picked

    def _map_sql(self, item, sql, classes, reps):
        """Execute the judge's written SQL and map its result to the class space: the
        class of a candidate with the same result, else a fresh class (a correct answer
        no model produced -- the candidate-coverage fix, label-free)."""
        jk, _ = result_key(item["db_path"], sql, self.timeout)
        if jk is None:
            self._fresh += 1
            return self._fresh
        for c in classes:
            ck, _ = result_key(item["db_path"], reps[c][1], self.timeout)
            if ck is not None and ck == jk:
                return c
        self._fresh += 1
        return self._fresh

    @staticmethod
    def _parse(content, n, synthesize=False):
        """Return (kind, value): ("choice", int) | ("sql", str) | ("none", None)."""
        if not content:
            return ("none", None)
        obj = {}
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:  # noqa
                obj = {}
        if synthesize and isinstance(obj.get("sql"), str) and obj["sql"].strip():
            return ("sql", obj["sql"].strip())
        v = obj.get("choice")
        if v is None:                                   # fall back to a regex probe
            mm = re.search(r"\"choice\"\s*:\s*\"?(\d+|none)\"?", content, re.I)
            v = mm.group(1) if mm else None
        if v is None or (isinstance(v, str) and v.strip().lower() == "none"):
            return ("none", None)
        try:
            k = int(v)
        except (TypeError, ValueError):
            return ("none", None)
        return ("choice", k) if 0 <= k < n else ("none", None)
