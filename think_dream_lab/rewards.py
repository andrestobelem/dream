"""Verificadores independientes de la librería de entrenamiento."""

import re
from dataclasses import asdict, dataclass

INTEGER = re.compile(r"\s*([+-]?[0-9]+)\s*", flags=re.ASCII)
ANSWER = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


@dataclass(frozen=True)
class Reward:
    accuracy: int
    format: int
    extracted: int | None

    @property
    def total(self) -> int:
        return self.accuracy + self.format

    def to_dict(self) -> dict:
        return {**asdict(self), "total": self.total}


def extract_answer(text: str) -> int | None:
    # Contar aperturas y cierres evita aceptar un bloque dentro de etiquetas repetidas.
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    blocks = ANSWER.findall(text)
    if len(blocks) != 1 or (match := INTEGER.fullmatch(blocks[0])) is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def valid_format(text: str, tag: str) -> bool:
    if tag == "neutral":
        # Métrica neutral independiente: trabajo libre seguido por una única respuesta.
        return (
            extract_answer(text) is not None
            and re.fullmatch(r".*<answer>\s*[+-]?[0-9]+\s*</answer>\s*", text, re.DOTALL)
            is not None
        )
    pattern = (
        rf"\s*<{re.escape(tag)}>(?P<work>[^<>]+)</{re.escape(tag)}>"
        r"\s*<answer>(?P<answer>[^<>]*)</answer>\s*"
    )
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    # '<' o '>' comunes son válidos en el trabajo; se rechazan etiquetas reales.
    if match is None:
        pattern = (
            rf"\s*<{re.escape(tag)}>(?P<work>.*?)</{re.escape(tag)}>"
            r"\s*<answer>(?P<answer>[^<>]*)</answer>\s*"
        )
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        if match and re.search(r"</?[A-Za-z][^>]*>", match.group("work")):
            return False
    return match is not None and bool(match.group("work").strip())


def score(text: str, expected: int, tag: str) -> Reward:
    extracted = extract_answer(text)
    return Reward(
        int(extracted is not None and extracted == expected),
        int(valid_format(text, tag)),
        extracted,
    )
