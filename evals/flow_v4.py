"""Revisar evaluaciones guardadas de v4, incluidas las dimensiones no puntuadas."""

from pathlib import Path

from dream_evals.flows import dream_flow

SPEC = dream_flow(Path(__file__).resolve().parents[1] / "results/dream-v4")
