"""Ejecutar desde la raíz: uv run --project evals flow run evals/flow.py"""

from pathlib import Path

from dream_evals.flows import dream_flow

SPEC = dream_flow(Path(__file__).resolve().parents[1] / "results")
