"""Entradas literales: nunca se aplica un chat template."""

CONDITIONS = {
    "A": ("think", "think"),
    "B": ("think", "dream"),
    "C": ("dream", "think"),
    "D": ("dream", "dream"),
}
TEMPLATE = (
    "A conversation between User and Assistant. The User asks a problem. "
    "The Assistant first {verb}s about the problem, writing this process inside "
    "<{tag}>...</{tag}>, and then gives the final answer inside <answer>...</answer>.\n"
    "User: {problem}\nAssistant:"
)
NEUTRAL = (
    "Solve the following problem. You may write working before the final answer. "
    "Give the final integer answer inside <answer>...</answer>.\n"
    "Problem: {problem}\nAnswer:"
)


def render(problem: str, condition: str) -> str:
    if condition == "neutral":
        return NEUTRAL.format(problem=problem)
    if condition.startswith("ordinary_"):
        verb, _ = CONDITIONS[condition.removeprefix("ordinary_")]
        return TEMPLATE.format(verb=verb, tag="work", problem=problem)
    verb, tag = CONDITIONS[condition]
    return TEMPLATE.format(verb=verb, tag=tag, problem=problem)


def process_tag(condition: str) -> str:
    if condition == "neutral":
        return "neutral"
    if condition.startswith("ordinary_"):
        return "work"
    return CONDITIONS[condition][1]
