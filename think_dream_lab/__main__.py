"""CLI pública: doctor, smoke, experiment y report."""

import argparse
import os
import signal
import sys
from pathlib import Path

from .config import load_config


def parser():
    root = argparse.ArgumentParser(
        prog="python -m think_dream_lab",
        description="Dream: generación y evaluación local de relatos de sueños",
    )
    commands = root.add_subparsers(
        dest="command", required=True, metavar="{doctor,corpus,judge,smoke,experiment,report}"
    )
    for command in [
        "doctor",
        "corpus",
        "judge",
        "anchor-check",
        "anchor-holdout",
        "closed-check",
        "base-check",
        "smoke",
        "experiment",
        "_worker",
    ]:
        cmd = commands.add_parser(command, **({"help": command} if command != "_worker" else {}))
        cmd.add_argument("--config", default="configs/dream-v4.yaml")
        cmd.add_argument("--results", type=Path, default=Path("results/dream-v4"))
        if command in ["smoke", "experiment", "_worker"]:
            cmd.add_argument("--resume", action="store_true")
        if command == "smoke":
            cmd.add_argument(
                "--verify-continuity",
                action="store_true",
                help="Repetir desde el paso 2 y comparar el checkpoint final real",
            )
        if command in ["experiment", "_worker"]:
            cmd.add_argument("--steps", type=int, required=command == "experiment")
            cmd.add_argument("--condition", choices=["A", "B", "C", "D"])
            cmd.add_argument("--seed", type=int)
        if command == "_worker":
            cmd.add_argument(
                "action",
                choices=[
                    "doctor",
                    "corpus",
                    "judge",
                    "judge-check",
                    "anchor-check",
                    "anchor-holdout",
                    "closed-check",
                    "base-check",
                    "smoke",
                    "experiment",
                    "verify",
                ],
            )
    cmd = commands.add_parser("report", help="Generar tablas, gráficos e informe")
    cmd.add_argument("--results", type=Path, default=Path("results"))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "report":
        if (args.results / "dream").exists():
            from .dream_report import report
        else:
            from .report import report

        report(args.results.resolve())
        return 0
    config = load_config(args.config)
    if args.command in {"smoke", "experiment"} and config["model"]["id"] == "Qwen/Qwen3-1.7B-Base":
        raise ValueError(
            "Configuración histórica: el nuevo generador debe ser SmolLM2 base sin posentrenamiento de razonamiento."
        )
    if (
        args.command in {"smoke", "experiment"}
        and config["judge"].get("rubric_version") == "dream-v4"
    ):
        raise ValueError(
            "dream-v4 está en implementación y calibración; todavía no habilita entrenamiento."
        )
    root = args.results.resolve()
    if args.command == "_worker":
        if config.get("task") == "dream":
            from .dream_workflow import worker
        else:
            from .workflow import worker

        def stop(_signal, _frame):
            raise KeyboardInterrupt(
                "Interrumpido por supervisor; se conserva el último checkpoint completo."
            )

        signal.signal(signal.SIGTERM, stop)
        if args.action == "judge-check":
            from .judge_checks import verify_reference

            verify_reference(config, root)
            return 0
        if args.action == "verify":
            if config.get("task") == "dream":
                from .dream_verification import verify_continuity
            else:
                from .verification import verify_continuity

            verify_continuity(config, root)
            return 0
        worker(
            args.action,
            config,
            root,
            resume=args.resume,
            condition=args.condition,
            seed=args.seed,
            steps=args.steps,
        )
        return 0
    if args.command == "experiment":
        if args.steps < 1:
            raise ValueError("--steps debe ser positivo y acordado según el benchmark.")
        if args.seed is not None and args.seed not in config["seeds"]:
            raise ValueError("La semilla debe pertenecer a las tres semillas configuradas.")
        if (args.condition is None) != (args.seed is None):
            raise ValueError("Para una sola corrida especificá --condition y --seed juntos.")
    from .supervisor import supervise

    common = ["--config", str(Path(args.config).resolve()), "--results", str(root)]
    if args.command == "experiment":
        targets = (
            [(args.condition, args.seed)]
            if args.condition
            else [(c, seed) for seed in config["seeds"] for c in config["conditions"]]
        )
        code = 0
        for condition, seed in targets:
            extra = ["--steps", str(args.steps), "--condition", condition, "--seed", str(seed)]
            if args.resume:
                extra.append("--resume")
            code = supervise(config, root, ["experiment", *common, *extra])
            if code:
                break
    else:
        extra = ["--resume"] if getattr(args, "resume", False) else []
        code = supervise(config, root, [args.command, *common, *extra])
        if not code and args.command == "smoke" and args.verify_continuity:
            from .storage import read_json

            proof_root = root / "dream" if config.get("task") == "dream" else root
            proof_path = proof_root / "smoke" / "continuity-validation.json"
            if proof_path.exists() and read_json(proof_path)["status"] == "passed":
                print("La prueba de continuidad real ya pasó y sus artefactos se conservan.")
            else:
                code = supervise(config, root, ["verify", *common])
    if config.get("task") == "dream":
        from .dream_report import report
    else:
        from .report import report

    report(root)
    return code


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FloatingPointError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
