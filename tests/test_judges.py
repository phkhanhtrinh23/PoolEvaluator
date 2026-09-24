from types import SimpleNamespace

from pooleval.judges import AnthropicJudge, JudgeEnsemble, JudgeItem, OpenAIJudge


class FixedJudge:
    def __init__(self, name, vote):
        self.name = name
        self.vote = vote

    def choose(self, item):
        return self.vote


def test_ensemble_majority_vote():
    item = JudgeItem("question", "schema", ("a", "b"))
    assert JudgeEnsemble([FixedJudge("a", 1), FixedJudge("b", 1), FixedJudge("c", 0)]).choose(item) == 1
    assert JudgeEnsemble([FixedJudge("a", -1), FixedJudge("b", -1), FixedJudge("c", 0)]).choose(item) == -1


def test_ensemble_tie_goes_to_largest_accuracy_weighted_support():
    item = JudgeItem("question", "schema", ("a", "b", "c"))
    three_way = [FixedJudge("a", 0), FixedJudge("b", 2), FixedJudge("c", -1)]
    assert JudgeEnsemble(three_way).choose(item, support=[0.4, 0.9, 1.3]) == 2
    assert JudgeEnsemble(three_way).choose(item, support=[1.5, 0.9, 1.3]) == 0
    # NONE is backed by no model, so any supported candidate wins a tie against it.
    assert JudgeEnsemble([FixedJudge("a", -1), FixedJudge("b", 1)]).choose(item, support=[0.2, 0.1, 0.0]) == 1
    # Equal support keeps the configured judge order.
    assert JudgeEnsemble([FixedJudge("a", 2), FixedJudge("b", 0)]).choose(item, support=[0.5, 0.0, 0.5]) == 2


def test_openai_judge_uses_responses_structured_output():
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"choice": 0}')

    client = SimpleNamespace(responses=Responses())
    judge = OpenAIJudge("gpt-5.5", client=client)
    assert judge.choose(JudgeItem("q", "s", ("answer",))) == 0
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["store"] is False
    assert calls[0]["text"]["format"]["type"] == "json_schema"


def test_anthropic_judge_uses_messages_structured_output():
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"choice": -1}')])

    client = SimpleNamespace(messages=Messages())
    judge = AnthropicJudge(client=client)
    assert judge.choose(JudgeItem("q", "s", ("answer",))) == -1
    assert calls[0]["model"] == "claude-opus-4-5"
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_paper_judges_reads_configured_members(monkeypatch):
    from pooleval import judges

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(judges, "_PROVIDERS", {**judges._PROVIDERS, "anthropic": ("ANTHROPIC_API_KEY", lambda model: FixedJudge(model, 0))})
    ensemble = judges.paper_judges(members=[{"provider": "anthropic", "model": "claude-opus-4-5"}])
    assert [judge.name for judge in ensemble.judges] == ["claude-opus-4-5"]
    assert "claude-opus-4-5" in [member["model"] for member in judges.DEFAULT_MEMBERS]
