"""Lote de preferencias completo para revisar localmente antes de entrenar."""

import html
from pathlib import Path

from experiments.preferences_v21 import ROOT, locked_write
from think_dream_lab.holdout_evaluation import sha256
from think_dream_lab.storage import digest, read_json, read_jsonl


def render(root):
    p = read_json(root / "protocol.json")
    pack = read_json(root / "preferences/training-pack.json")
    notes = read_json(root / "preferences/annotations.json")
    frozen = read_json(root / "preferences/review-frozen.json")
    assert digest(notes) == frozen["annotations_hash"]
    assert digest(pack["pairs"]) == pack["pairs_hash"]
    source = read_json(Path(p["source_run"]))
    origin = Path(source["training_validation"]["checkpoint"])
    assert all(sha256(origin / n) == v for n, v in read_json(root / "source-files.json").items())
    samples = read_jsonl(root / "train-pool/samples.jsonl")
    key = {r["review_id"]: r for r in read_json(root / "preferences/key.json")}
    by_id = {r["sample_id"]: r for r in samples}
    rows = {r["problem_id"]: r for r in p["train_rows"]}
    escape = html.escape
    body = [
        "<h1>Dream · lote revisado desde el checkpoint 32</h1>",
        f"<p><strong>Preparación terminada. {pack['eligible_pairs']} pares claros de 24 pedidos; no se inició entrenamiento.</strong></p>",
        "<p>Cuatro candidatos por pedido, 96 generaciones del 32. Revisión del asistente; no validación humana independiente. Se conserva el texto completo muestreado, sin recortar ni reescribir. Referencia humana completa fuera del prompt.</p>",
        f"<p>Viabilidad del lote: {'cumple' if pack['ready'] else 'no cumple'}. Preferido más largo: {pack['length_directions'].get('longer', 0)}; más corto: {pack['length_directions'].get('shorter', 0)}; igual: {pack['length_directions'].get('equal', 0)}. No se invirtieron etiquetas para alcanzar cuotas.</p>",
        '<nav><a href="#training">Qué se entrenaría</a> · <a href="#criteria">Criterio de mejora</a> · <a href="#pairs">Pares revisados</a> · <a href="#evaluation">Pedidos de evaluación</a></nav>',
        '<h2 id="training">Qué se entrenaría</h2>',
        f"<p>Qwen3-1.7B-Base con los adaptadores exactos del checkpoint 32. Una pasada DPO por {pack['eligible_pairs']} pares; {pack['eligible_pairs']} actualizaciones. Referencia fija 32, optimizador nuevo. Learning rate 0,000005; beta 0,1; LoRA heredado. Se entrena toda la continuación dream + answer, excluyendo prompt y apertura prellenada. Sin penalizaciones automáticas por copia o longitud.</p>",
        "<p>La selección compara calidad entre los cuatro candidatos, sin elegir primero por similitud de longitud. Los empates quedan fuera. El límite operativo de secuencia es 1.536 tokens; no se recorta una trayectoria para hacerla entrar.</p>",
        '<h2 id="criteria">Criterio fijado antes de entrenar</h2>',
        "<p>24 pedidos nuevos, cuatro muestras por modelo y pedido: 96 comparaciones contra el 32. Misma plantilla, generación y semillas a ambos lados. La calidad se revisará con procedencia oculta y notas congeladas antes de abrir la clave. La variedad también tendrá orden aleatorio, sin alternancia deducible.</p>",
        "<ul><li>Ventaja neta de calidad de al menos 0,15 (gana +1, pierde −1, empate o duda 0), y límite inferior del intervalo exploratorio del 95% mayor que cero. Se remuestrean pedidos completos, no muestras independientes.</li><li>Formato válido en al menos 95/96 y sin bajar del 32.</li><li>Sin aumentar los relatos con omisiones o contradicciones importantes de los motivos pedidos.</li><li>En variedad: al menos tantas preferencias por el nuevo modelo como por el 32, y sin aumentar los grupos con menos de dos episodios utilizables.</li><li>Conservar al menos 16/18 hechos completos y 18/18 formatos en las mismas pruebas locales de dependencia.</li></ul>",
        "<p>Todos los requisitos deben cumplirse. Si falla alguno, falta revisión o el resultado es incierto, se conserva el 32. No se ajustan umbrales ni se prolonga la corrida mirando los resultados. Es desarrollo revisado por el mismo asistente, no prueba poblacional ni humana independiente. Los 200 relatos de prueba siguen sin usarse.</p>",
        '<h2 id="pairs">Los 24 pedidos y su revisión</h2>',
    ]
    for note in notes["ratings"]:
        rid = note["review_id"]
        k = key[rid]
        row = rows[k["problem_id"]]
        clear = note["preference"] == "clear"
        status = (
            f"Entrenamiento: {note['chosen']} preferido a {note['rejected']}"
            if clear
            else "Fuera: empate o duda"
        )
        body += [
            f'<section id="{rid}"><h3>{rid} · {escape(status)}</h3>',
            f"<p><b>{escape(row['seeds'])}</b></p><p>Relato final orientativo: {row['target_words']} palabras. Confianza: {escape(note['confidence'])}.</p>",
            f"<p>{escape(note['note'])}</p>",
            f"<details><summary>Ver prompt exacto</summary><pre>{escape(row['prompt'])}</pre></details>",
        ]
        for side in "ABCD":
            r = by_id[k[side + "_id"]]
            label = (
                " · preferido"
                if clear and side == note["chosen"]
                else " · alternativa rechazada"
                if clear and side == note["rejected"]
                else ""
            )
            body += [
                f"<details><summary>Candidato {side}{label} · {len(r['dream'].split())} palabras en dream</summary>",
                f"<p>{escape(note['candidate_notes'][side])}</p><h4>dream</h4><pre>{escape(r['dream'])}</pre>",
                f"<h4>answer</h4><pre>{escape(r['answer'])}</pre><p>Formato: {'válido' if r['format_valid'] else 'inválido'}.</p></details>",
            ]
        body.append("</section>")
    body += [
        '<h2 id="evaluation">Los 24 pedidos nuevos de evaluación</h2><p>Escritos antes de generar el lote; no derivados de relatos reservados. Relato final orientativo: 100 palabras. No entran al entrenamiento y no tienen un único relato humano esperado.</p><ol>'
    ]
    body += [f"<li>{escape(r['seeds'])}</li>" for r in p["dev_rows"]]
    body += [
        "</ol><h2>Trazabilidad</h2>",
        f"<p>Huella del lote: <code>{pack['pairs_hash']}</code>. Base y cuatro archivos del origen intactos. Se validaron las congelaciones y los tokens originales. Textos completos sólo en este informe local.</p>",
        "<footer>DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Código MIT.</footer>",
    ]
    style = "body{max-width:1000px;margin:40px auto;padding:0 20px;background:#f6f4ed;color:#213331;font:18px/1.6 system-ui}h1,h2{line-height:1.2}section{padding:18px 0;border-top:1px solid #bac7bc}details{margin:12px 0;padding:10px;background:#eaf0e6}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 system-ui}a{color:#175849}code{overflow-wrap:anywhere}nav{padding:16px 0}li{margin-bottom:10px}"
    (root / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · lote previo al entrenamiento</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "</html>"
    )
    audit = {
        "protocol_hash": digest(p),
        "pairs_hash": pack["pairs_hash"],
        "annotations_hash": frozen["annotations_hash"],
        "source_files_unchanged": True,
        "generated_candidates": len(samples),
        "valid_candidates": sum(r["format_valid"] for r in samples),
        "truncated_candidates": sum(r["truncated"] for r in samples),
        "selected_pairs": len(pack["pairs"]),
        "ready": pack["ready"],
        "training_updates": 0,
        "test_reports_used": 0,
    }
    locked_write(root / "preparation-audit.json", audit)
    locked_write(Path("docs/preferencias-v21-auditoria.json"), audit)


if __name__ == "__main__":
    render(ROOT)
