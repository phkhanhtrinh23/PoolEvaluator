"""Prompt builders. Prompt STYLE is what makes within-group members near-clones
(same base, different prompt) vs the same base -- the paper's provenance structure."""
from .data_spider import schema_to_prompt

_SYS = ("You are an expert data analyst. Translate the question into a single "
        "SQLite SQL query. Output ONLY the SQL, no explanation, no markdown fences.")

_FEWSHOT = """Example:
Schema:
CREATE TABLE singer ( singer_id int, name text, age int );
Question: How many singers are older than 30?
SQL: SELECT count(*) FROM singer WHERE age > 30

"""


def build_messages(item, style="schema"):
    schema = schema_to_prompt(item["schema"], with_values=(style != "minimal"))
    q = item["question"]
    if style == "minimal":
        user = f"Schema:\n{schema}\n\nQuestion: {q}\nSQL:"
    elif style == "fewshot":
        user = f"{_FEWSHOT}Now answer:\nSchema:\n{schema}\n\nQuestion: {q}\nSQL:"
    else:  # schema-rich (default)
        user = (f"Given the following SQLite database schema:\n{schema}\n\n"
                f"Write a SQL query that answers: {q}\nSQL:")
    return [{"role": "system", "content": _SYS},
            {"role": "user", "content": user}]
