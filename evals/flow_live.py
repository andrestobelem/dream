"""Un lote nuevo de validación local, sin entrenamiento ni calibración automática."""

from pathlib import Path

from dream_evals.flows import live_dream_flow

ROOT = Path(__file__).resolve().parents[1]
SPEC = live_dream_flow(ROOT / "results/dream-v4-qwen", ROOT / "configs/dream-v4.yaml", start_index=1)
