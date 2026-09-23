from types import SimpleNamespace

from pooleval.judges import AnthropicJudge, JudgeEnsemble, JudgeItem, OpenAIJudge


class FixedJudge:
    def __init__(self, name, vote):
        self.name = name
        self.vote = vote

    def choose(self, item):
        return self.vote


def test_ensemble_majority_and_safe_tie():
    item = JudgeItem("question", "schema", ("a", "b"))
    assert JudgeEnsemble([FixedJudge("a", 1), FixedJudge("b", 1), FixedJudge("c", 0)]).choose(item) == 1
    assert JudgeEnsemble([FixedJudge("a", 1), FixedJudge("b", -1)]).choose(item) == -1


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
    assert calls[0]["model"] == "claude-opus-5-5"
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
