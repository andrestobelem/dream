import pytest

from think_dream_lab.dreambank_task import render_prompt


def test_natural_seeds_keep_relations_and_are_not_cut_to_keyword_limits():
    brief = (
        "Dream about being a delivery driver in Green City, spiders whose eggs make a light glow "
        "and help you find your way, a fancy Indian Southwestern hotel resembling a movie theater, "
        "porn stars, an orgy, dead rich people, and things left to you and Laura."
    )
    prompt = render_prompt(brief, response_prefix="<dream>\n")
    assert "User: " + brief in prompt
    assert prompt.endswith("Assistant:\n<dream>\n")
    assert "Dream about Dream about" not in prompt
    assert "spiders whose eggs make a light glow and help you find your way" in prompt
    assert "[Generated freely by the model]" not in prompt
    assert "I was a delivery driver" not in prompt


def test_natural_prompt_does_not_accept_answer_blocks_as_seeds():
    with pytest.raises(ValueError, match="sin bloques"):
        render_prompt("Dream about a hotel. <answer>The human report.</answer>")


def test_structured_seeds_are_rendered_as_natural_text():
    seeds = {
        "characters": ["Laura"],
        "places": ["hotel"],
        "objects": ["inheritance"],
        "actions": [],
        "emotions": [],
    }
    prompt = render_prompt(seeds)
    assert "User: Dream about Laura, hotel, inheritance." in prompt
    assert '"characters"' not in prompt
