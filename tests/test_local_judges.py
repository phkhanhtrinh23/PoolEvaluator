from pooleval.local_judges import NodeJudgeItem, gofa_graph, image_prompt, ImageJudgeItem, parse_option


def test_gofa_graph_matches_official_json_layout():
    graph = gofa_graph(NodeJudgeItem("target paper", ("neighbor a", "neighbor b"), ("Databases", "Networking")))
    prompt_id = 3
    assert graph["question"] == [prompt_id] and graph["complete"] == []
    assert graph["node"]["0"] == "target paper"
    assert "[NODEID.AA]" in graph["node"][str(prompt_id)]
    assert "C. None of the above" in graph["node"][str(prompt_id)]
    to_prompt = [e for e in graph["edge"] if e["target"] == prompt_id]
    assert len(to_prompt) == 3


def test_parse_option_maps_letters_names_and_none():
    candidates = ["Databases", "Networking", "Computer Vision"]
    assert parse_option("B. Networking", candidates) == 1
    assert parse_option("The answer is A", candidates) == 0
    assert parse_option("I think B", candidates) == 1
    assert parse_option("D. None of the above", candidates) == -1
    assert parse_option("It is about networking protocols", candidates) == 1
    assert parse_option("Databases or Networking", candidates) == -1


def test_image_prompt_lists_candidates_and_none():
    text = image_prompt(ImageJudgeItem(None, ("cat", "dog")))
    assert "0: cat" in text and "1: dog" in text and "-1" in text
