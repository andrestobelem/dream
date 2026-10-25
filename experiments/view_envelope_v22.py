"""Vista local que distingue cierre escrito por programa y fin efectivo del contenido."""

import html

from experiments.envelope_v22 import ROOT
from think_dream_lab.storage import read_json, read_jsonl


def render(root):
    result = read_json(root / "report/summary.json")
    q = result["quality"]
    names = {
        "before": "Checkpoint 32 + envoltura",
        "after": "Candidato v21 + envoltura",
        "tie": "Empate",
        "no_se": "Indeterminado",
    }
    heading = (
        "Se elige el candidato v21 con la nueva envoltura"
        if result["promotion"]["promote"]
        else "Se conserva el checkpoint 32; no se acreditó cumplir todos los criterios"
    )
    escape = html.escape
    body = [
        "<h1>Dream · cierre controlado sin entrenar</h1>",
        f"<p><strong>{heading}.</strong></p>",
        f"<p>96 comparaciones sobre 24 pedidos nuevos: candidato gana {q['counts'].get('after', 0)}, 32 gana {q['counts'].get('before', 0)}, empates {q['counts'].get('tie', 0)}, dudas {q['counts'].get('no_se', 0)}. Ventaja neta {q['net_mean']:.3f}, intervalo exploratorio por pedido [{q['interval_95'][0]:.3f}, {q['interval_95'][1]:.3f}]. Revisión del asistente, no humana independiente.</p>",
        "<p><b>Las etiquetas las escribe el programa.</b> El contenido lo genera el modelo en dos etapas. Se separan etiquetas, turnos nuevos y EOS; no se reescribe la prosa. Cuerpo vacío o etapa que agota el límite cuentan como fallo incluso si la envoltura es válida. No hay nuevos pesos ni reintentos.</p>",
        "<p>El cierre pasó en esta prueba: 96/96 muestras por sistema y 18/18 regresiones de dependencia, sin límites agotados. La calidad del candidato no alcanzó la magnitud ni la certeza exigidas. El control de motivos heredado mide literalidad, no si una transformación es oníricamente admisible; se conserva para no cambiar esta prueba después de verla.</p>",
        '<nav><a href="#checks">Criterios</a> · <a href="#quality">96 pares</a> · <a href="#variety">Variedad</a> · <a href="#dependency">36 regresiones</a></nav>',
        '<h2 id="checks">Criterios fijados y resultados</h2><ul>',
    ]
    translations = {
        "quality_effect": "Ventaja de calidad ≥0,15",
        "quality_interval": "Intervalo de calidad por encima de cero",
        "format": "Finalización y formato sin retroceso",
        "motifs": "Sin aumentar pérdida importante de motivos",
        "variety_preference": "Preferencia de variedad sin retroceso",
        "usable_groups": "Sin aumentar grupos sin dos episodios utilizables",
        "dependency": "Al menos 16/18 hechos completos",
        "dependency_format": "18/18 respuestas de dependencia completas",
        "all_reviews_frozen": "Notas completas y congeladas",
    }
    body += [
        f"<li>{translations[k]}: <b>{'cumple' if v else 'no cumple'}</b></li>"
        for k, v in result["promotion"]["checks"].items()
    ]
    body += ["</ul><table><tr><th>Medida</th><th>32</th><th>Candidato</th></tr>"]
    for k, title in [
        ("syntax_valid", "Envoltura sintácticamente válida /96"),
        ("format_valid", "Contenido finalizado y formato /96"),
        ("truncated", "Muestras con límite agotado"),
        ("motif_major", "Pérdidas importantes de motivos"),
        ("dream_words_mean", "Palabras en dream, media"),
        ("answer_words_mean", "Palabras en answer, media"),
        ("dependency_full", "Hechos completos /18"),
        ("dependency_format", "Dependencia finalizada y formato /18"),
    ]:
        body += [
            f"<tr><td>{title}</td><td>{result['before'][k]:.2f}</td><td>{result['after'][k]:.2f}</td></tr>"
        ]
    body += [
        "</table><h2>Cómo terminó cada etapa</h2><p>Estos cierres originales son diagnóstico del modelo; no se confunden con las etiquetas que el programa siempre proporciona.</p>"
    ]
    for p, stages in result["controller"].items():
        body += [f"<h3>{names[p]}</h3>"]
        for stage, metrics in stages.items():
            body += [
                f"<p>{stage}: {escape(str(metrics['stop_reasons']))}. Delimitador esperado emitido por el modelo: {metrics['model_emitted_expected_delimiter']}/96.</p>"
            ]
    folder = root / "evaluation"
    body += ['<h2 id="quality">Calidad y motivos, notas congeladas antes de revelar modelos</h2>']
    for kind in ["quality", "variety"]:
        notes = {
            r["review_id"]: r for r in read_json(folder / f"{kind}-annotations.json")["ratings"]
        }
        keys = {r["review_id"]: r for r in read_json(folder / f"{kind}-key.json")}
        if kind == "variety":
            body += ['<h2 id="variety">Variedad entre cuatro muestras por modelo</h2>']
        for r in read_json(folder / f"{kind}-packet.json"):
            rid = r["review_id"]
            n, k = notes[rid], keys[rid]
            pref = k[n["preference"] + "_phase"] if n["preference"] in "AB" else n["preference"]
            body += [
                f'<section id="{rid}"><h3>{rid} · {names[pref]}</h3><p>{escape(r["request"])}</p><p>{escape(n["note"])}</p>'
            ]
            for s in "AB":
                texts = [r[s]] if kind == "quality" else r[s]
                body += [f"<details><summary>{names[k[s + '_phase']]}</summary>"]
                body += [f"<pre>{escape(t)}</pre><hr>" for t in texts]
                if kind == "quality":
                    m = n["motifs"][s]
                    body += [
                        f"<p>Motivos: {'pérdida importante' if m['major'] else 'sin pérdida importante'}. {escape(m['note'])}</p>"
                    ]
                else:
                    body += [f"<p>Episodios utilizables: {n['usable_episodes'][s]}/4.</p>"]
                body += ["</details>"]
            body += ["</section>"]
    body += [
        '<h2 id="dependency">Dependencia conocida: ambos sistemas regenerados</h2><p>Se reutilizaron 35 etiquetas congeladas porque el texto coincide exactamente y se revisó una respuesta diferente. Las claves nuevas de modelo y condición no se consultaron antes de congelar, pero los textos previos pueden reconocerse: no se reclama revisión independiente cegada. La respuesta con cola «Write the dream account in» se conserva tal cual; la envoltura evita turnos adicionales, no repara esa prosa.</p>'
    ]
    keys = {r["review_id"]: r for r in read_json(root / "dependency/key.json")}
    labels = {r["review_id"]: r for r in read_json(root / "dependency/annotations.json")["labels"]}
    raw = {
        p: {r["task_id"]: r for r in read_jsonl(root / "dependency" / p / "samples.jsonl")}
        for p in ["before", "after"]
    }
    for r in read_json(root / "dependency/packet.json"):
        rid = r["review_id"]
        k = keys[rid]
        item = raw[k["phase"]][k["task_id"]]
        body += [
            f'<section id="{rid}"><h3>{rid} · {names[k["phase"]]}</h3><p>Hecho: {escape(labels[rid]["state"])}. Finalización: {item["format_valid"]}; parada: {escape(item["stages"]["answer"]["stop_reason"])}.</p><details><summary>Leer contenido</summary><pre>{escape(item["answer"])}</pre></details></section>'
        ]
    body += (
        ["<h2>Límites</h2><ul>"]
        + [f"<li>{escape(t)}</li>" for t in result["limitations"]]
        + [
            "</ul><footer>Los 200 relatos de prueba no se usaron. DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Textos completos sólo en este informe local.</footer>"
        ]
    )
    style = "body{max-width:1000px;margin:40px auto;padding:0 20px;background:#f6f4ed;color:#213331;font:18px/1.6 system-ui}h1,h2{line-height:1.2}section{padding:18px 0;border-top:1px solid #bac7bc}details{margin:12px 0;padding:10px;background:#eaf0e6}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 system-ui}a{color:#175849}table{border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #bac7bc}"
    (root / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · envoltura v22</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "</html>"
    )


if __name__ == "__main__":
    render(ROOT)
