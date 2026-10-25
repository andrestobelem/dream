"""Vista local de notas y resultados; no publica textos del corpus."""

import argparse
import html
from pathlib import Path

from experiments.preferences_v20 import ROOT
from think_dream_lab.storage import read_json, read_jsonl


def render(root):
    result = read_json(root / "report/summary.json")
    quality = result["quality"]
    body = [
        "<h1>Dream · resultado del piloto de preferencias</h1>",
        "<p><strong>El piloto empeoró en esta revisión. Se mantiene el checkpoint 32 como referencia.</strong></p>",
        f"<p>Una pasada de 14 actualizaciones sobre preferencias claras de 24 pares. En 24 comparaciones nuevas: 32 preferido {quality['counts'].get('before', 0)}, piloto preferido {quality['counts'].get('after', 0)}, empates {quality['counts'].get('tie', 0)}. Revisión del asistente, no humana independiente.</p>",
        "<p>Ambos modelos conservaron formato en 24/24. Los bloques que sólo repiten la consigna pasaron de 1 a 5. La revisión de variedad fue exploratoria y no cegada: dos descensos, cinco empates y cinco indeterminados. No se acreditó preservar variedad.</p>",
        "<p>Las pruebas locales de dependencia conservaron el hecho completo en 16/18 antes y 17/18 después. Es sensibilidad a detalles, no fidelidad completa. Base y origen intactos; los 200 relatos de prueba no se usaron.</p>",
        '<nav><a href="#quality">Calidad</a> · <a href="#variety">Variedad</a> · <a href="#dependency">Dependencia</a> · <a href="index.html">Lote entrenado</a></nav>',
        '<h2 id="quality">Calidad: identidades reveladas después de congelar notas</h2>',
    ]
    escape = html.escape
    folder = root / "evaluation"
    notes = {r["review_id"]: r for r in read_json(folder / "quality-annotations.json")["ratings"]}
    key = {r["review_id"]: r for r in read_json(folder / "quality-key.json")}
    phase_names = {"before": "Checkpoint 32", "after": "Piloto DPO"}
    for row in read_json(folder / "quality-packet.json"):
        rid = row["review_id"]
        n = notes[rid]
        k = key[rid]
        chosen = n["preference"]
        pref = (
            phase_names[k[chosen + "_phase"]]
            if chosen in {"A", "B"}
            else {"tie": "Empate", "no_se": "Indeterminado"}[chosen]
        )
        body += [
            f'<section><h3>{rid}</h3><p>{escape(row["request"])}</p><p>Preferencia: <b>{pref}</b>. Confianza: {escape(n["confidence"])}.</p><p>{escape(n["note"])}</p><div class="columns">'
        ]
        for side in ["A", "B"]:
            body += [
                f"<article><h4>{phase_names[k[side + '_phase']]}</h4><details><summary>Leer dream</summary><p>{escape(row[side])}</p></details></article>"
            ]
        body += ["</div></section>"]
    body += [
        '<h2 id="variety">Variedad exploratoria</h2><p>El orden de grupos era deducible desde el código escrito por el mismo revisor. No se presenta como evaluación cegada ni independiente. Sólo dos muestras por modelo y pedido.</p>'
    ]
    notes = {r["review_id"]: r for r in read_json(folder / "variety-annotations.json")["ratings"]}
    key = {r["review_id"]: r for r in read_json(folder / "variety-key.json")}
    for row in read_json(folder / "variety-packet.json"):
        rid = row["review_id"]
        n = notes[rid]
        k = key[rid]
        body += [
            f'<section><h3>{rid}</h3><p>{escape(row["request"])}</p><p>{escape(n["note"])}</p><div class="columns">'
        ]
        for side in ["A", "B"]:
            body += [f"<article><h4>{phase_names[k[side + '_phase']]}</h4>"]
            for i, text in enumerate(row[side]):
                body += [
                    f"<details><summary>Muestra {i + 1}</summary><p>{escape(text)}</p></details>"
                ]
            body += ["</article>"]
        body += ["</div></section>"]
    body += [
        '<h2 id="dependency">Dependencia local: antes y después</h2><p>La respuesta se anotó sin mostrar condición. Los hechos son artificiales y agregados al dream. Un detalle conservado no garantiza relatar el resto de las escenas.</p>'
    ]
    old = {
        r["task_id"]: r for r in read_jsonl(Path("results/dream-v19/interventions/answers.jsonl"))
    }
    new = {r["task_id"]: r for r in read_jsonl(root / "dependency/answers.jsonl")}
    for item in result["dependency"]["after"]["rows"]:
        tid = item["task_id"]
        body += [
            f'<section><h3>{item["case_id"]} · {item["condition"]} · semilla {item["seed"]}</h3><p>Estado anotado después: {item["state"]}. Coincide completamente: {"sí" if item["match"] else "no"}.</p><div class="columns">'
        ]
        for phase, record in [("before", old[tid]), ("after", new[tid])]:
            body += [
                f"<article><h4>{phase_names[phase]}</h4><details><summary>Leer answer</summary><p>{escape(record['answer'])}</p></details></article>"
            ]
        body += ["</div></section>"]
    style = "body{max-width:1100px;margin:36px auto;padding:0 24px;background:#faf9f5;color:#202622;font:17px/1.6 system-ui}h1{line-height:1.15}section{border-top:1px solid #ccd1cb;margin-top:28px;padding-top:8px}p{white-space:pre-wrap}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{padding:18px;background:white;border:1px solid #e2e5dd;border-radius:10px;min-width:0}summary,a{color:#325844;cursor:pointer}@media(max-width:800px){.columns{grid-template-columns:1fr}}"
    (root / "report.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · resultado de preferencias</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "<footer>DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Textos completos sólo en este artefacto local.</footer></html>"
    )
    index = (root / "index.html").read_text()
    if "report.html" not in index:
        index = index.replace(
            "<h1>",
            '<p><strong>Piloto completado: resultado desfavorable. <a href="report.html">Ver comparación y auditoría</a>.</strong></p><h1>',
            1,
        )
    (root / "index.html").write_text(index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    render(args.root)
