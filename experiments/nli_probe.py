"""Contraste diagnóstico de NLI; no modifica la recompensa ni habilita entrenamiento.

Se ejecuta en un entorno opcional separado, con dependencias registradas en la salida.
Entrada: JSON con casos {name, candidate, requirements: [{id, text, expected}]}.
"""

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

MODEL = "cross-encoder/nli-deberta-v3-small"
REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
MAPPING = {"contradiction": "contradicted", "entailment": "preserved", "neutral": "omitted"}


def fingerprint(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    cases = json.loads(payload)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True
    ).cpu().eval()
    model.requires_grad_(False)
    mapping = {int(k): MAPPING[v] for k, v in model.config.id2label.items()}
    if set(mapping.values()) != set(MAPPING.values()) or len(mapping) != 3:
        raise ValueError("Mapeo de etiquetas inesperado.")
    before = fingerprint(model)
    results = []
    first_inputs = first_logits = None
    with torch.inference_mode():
        for case in cases:
            items = []
            for requirement in case["requirements"]:
                # Dirección NLI: relato como premisa, requisito como hipótesis.
                inputs = tokenizer(
                    case["candidate"], requirement["text"], return_tensors="pt"
                )
                if inputs["input_ids"].shape[-1] > 512:
                    raise ValueError("El diagnóstico no trunca silenciosamente el relato.")
                logits = model(**inputs).logits[0]
                probabilities = logits.softmax(dim=-1).tolist()
                ordered = sorted(range(3), key=lambda i: probabilities[i], reverse=True)
                gap = probabilities[ordered[0]] - probabilities[ordered[1]]
                predicted = mapping[ordered[0]]
                items.append(
                    {
                        **requirement,
                        "prediction": predicted,
                        "probabilities": {mapping[i]: p for i, p in enumerate(probabilities)},
                        "margin": gap,
                        "status": predicted if gap >= 0.15 else "abstain",
                        "matches_expected": predicted == requirement["expected"],
                    }
                )
                if first_inputs is None:
                    first_inputs, first_logits = inputs, logits.clone()
            results.append({"name": case["name"], "items": items})
            print(case["name"], [(i["id"], i["status"]) for i in items], flush=True)
        repeated = model(**first_inputs).logits[0]
    report = {
        "model": MODEL,
        "revision": REVISION,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {p: version(p) for p in ["torch", "transformers", "tokenizers"]},
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device": "cpu",
        "threads": 2,
        "id2status": mapping,
        "results": results,
        "cases_argmax_correct": sum(all(i["matches_expected"] for i in r["items"]) for r in results),
        "cases_decided_correct": sum(
            all(i["status"] == i["expected"] for i in r["items"]) for r in results
        ),
        "n_cases": len(results),
        "repeat_max_logit_delta": (first_logits - repeated).abs().max().item(),
        "weights_unchanged": before == fingerprint(model),
        "generated_tokens": 0,
        "seconds": time.perf_counter() - started,
        "calibrated": False,
        "training_enabled": False,
        "limitation": "Contraste sintético de diagnóstico. Margen 0,15 provisional, no calibrado; no valida sueños ni sustituye revisión humana.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
