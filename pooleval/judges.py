"""Budgeted GPT-5.4/GPT-5.5/Claude Opus 4.5 judge ensemble for Text2SQL."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class JudgeItem:
    question: str
    schema: str
    candidates: tuple[str, ...]
    evidence: str = ""


class Judge(Protocol):
    name: str

    def choose(self, item: JudgeItem) -> int:
        """Return a zero-based candidate index, or ``-1`` when none is correct."""


def _prompt(item: JudgeItem) -> str:
    candidates = "\n\n".join(f"Candidate {i}:\n{value}" for i, value in enumerate(item.candidates))
    evidence = f"\nEvidence: {item.evidence}" if item.evidence else ""
    return (
        "Choose the candidate whose executed answer correctly answers the database question. "
        "Choose none (-1) when no candidate is correct. Do not reward SQL style; judge semantics.\n\n"
        f"Schema:\n{item.schema}{evidence}\n\nQuestion: {item.question}\n\n{candidates}"
    )


def _parse(text: str, count: int) -> int:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        value = int(json.loads(match.group(0))["choice"]) if match else int((text or "").strip())
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return -1
    return value if -1 <= value < count else -1


class OpenAIJudge:
    def __init__(self, model: str, client: Any = None, reasoning_effort: str = "low"):
        self.model = model
        self.name = model
        self.reasoning_effort = reasoning_effort
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("install judge dependencies: pip install -e '.[judges]'") from exc
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = client

    def choose(self, item: JudgeItem) -> int:
        schema = {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": len(item.candidates) - 1,
                }
            },
            "required": ["choice"],
            "additionalProperties": False,
        }
        response = self.client.responses.create(
            model=self.model,
            instructions="You are a strict, independent model-output judge.",
            input=_prompt(item),
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pool_judgment",
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
            store=False,
        )
        return _parse(response.output_text, len(item.candidates))


class AnthropicJudge:
    def __init__(self, model: str = "claude-opus-4-5", client: Any = None, effort: str = "low"):
        self.model = model
        self.name = model
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("install judge dependencies: pip install -e '.[judges]'") from exc
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client

    def choose(self, item: JudgeItem) -> int:
        schema = {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": len(item.candidates) - 1,
                }
            },
            "required": ["choice"],
            "additionalProperties": False,
        }
        message = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system="You are a strict, independent model-output judge. Return JSON only.",
            messages=[
                {
                    "role": "user",
                    "content": _prompt(item) + '\n\nReturn exactly {"choice": INDEX}; use -1 for none.',
                }
            ],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        return _parse(text, len(item.candidates))


class JudgeEnsemble:
    def __init__(self, judges: Iterable[Judge]):
        self.judges = list(judges)
        if not self.judges:
            raise ValueError("judge ensemble cannot be empty")
        self.last_votes: dict[str, int] = {}

    def choose(self, item: Any, support: Sequence[float] | None = None) -> int:
        """Majority vote; ties go to the tied choice with the largest accuracy-weighted support.

        ``support[k]`` is the summed estimated accuracy of the pool models that produced
        candidate ``k``.  NONE (-1) is backed by no model, so its support is zero.  Any
        remaining tie keeps the configured judge order.
        """
        votes: list[int] = []
        self.last_votes = {}
        for judge in self.judges:
            vote = int(judge.choose(item))
            votes.append(vote)
            self.last_votes[judge.name] = vote
        counts = {vote: votes.count(vote) for vote in set(votes)}
        best = max(counts.values())
        winners = [vote for vote, count in counts.items() if count == best]
        if len(winners) == 1:
            return winners[0]

        def weight(vote: int) -> float:
            if vote < 0 or support is None or vote >= len(support):
                return 0.0
            return float(support[vote])

        top = max(weight(vote) for vote in winners)
        return next(vote for vote in votes if vote in winners and weight(vote) == top)


_PROVIDERS = {"openai": ("OPENAI_API_KEY", OpenAIJudge), "anthropic": ("ANTHROPIC_API_KEY", AnthropicJudge)}
DEFAULT_MEMBERS = (
    {"provider": "openai", "model": "gpt-5.4"},
    {"provider": "openai", "model": "gpt-5.5"},
    {"provider": "anthropic", "model": "claude-opus-4-5"},
)


def paper_judges(
    allow_partial: bool = True, members: Iterable[Mapping[str, str]] | None = None
) -> JudgeEnsemble:
    """Construct the configured API ensemble, skipping unavailable providers only if allowed."""
    judges: list[Judge] = []
    missing: list[str] = []
    for member in members or DEFAULT_MEMBERS:
        provider, model = member["provider"], member["model"]
        if provider not in _PROVIDERS:
            raise ValueError(f"unknown judge provider {provider!r}")
        key, cls = _PROVIDERS[provider]
        if os.environ.get(key):
            judges.append(cls(model))
        else:
            missing.append(f"{model} ({key})")
    if missing and not allow_partial:
        raise RuntimeError("missing credentials for: " + ", ".join(missing))
    if missing:
        print("[judges] unavailable: " + ", ".join(missing))
    return JudgeEnsemble(judges)
