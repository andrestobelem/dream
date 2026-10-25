"""Informe local del único punto final del ensayo v21."""

import html

from experiments.preferences_v21 import ROOT
from think_dream_lab.storage import read_json


def render(root):
    r = read_json(root / "report/summary.json")
    q = r["quality"]
    escape = html.escape
    label = {
        "before": "Checkpoint 32",
        "after": "Piloto v21",
        "tie": "Empate",
        "no_se": "Indeterminado",
    }
    keep = not r["promotion"]["promote"]
    heading = (
        "Se conserva el checkpoint 32"
        if keep
        else "El piloto supera los criterios fijados en esta evaluación de desarrollo"
    )
    body = [
        "<h1>Dream · resultados v21</h1>",
        f"<p><strong>{heading}.</strong></p>",
        f"<p>Una pasada de 19 actualizaciones desde el 32. En 96 comparaciones nuevas: 32 gana {q['counts'].get('before', 0)}, piloto gana {q['counts'].get('after', 0)}, empates {q['counts'].get('tie', 0)}, dudas {q['counts'].get('no_se', 0)}. Revisión del asistente, no humana independiente.</p>",
        f"<p>Ventaja neta del nuevo: {q['net_mean']:.3f}; intervalo exploratorio por pedido [{q['interval_95'][0]:.3f}, {q['interval_95'][1]:.3f}]. Umbral fijado: ≥0,15 y límite inferior &gt;0. Los 200 relatos de prueba no se usaron.</p>",
        '<nav><a href="#gates">Criterios</a> · <a href="#quality">96 comparaciones</a> · <a href="#variety">Variedad</a> · <a href="index.html">Lote entrenado</a></nav>',
        '<h2 id="gates">Criterios aplicados sin cambiar umbrales</h2><ul>',
    ]
    translations = {
        "quality_effect": "Magnitud de mejora de calidad",
        "quality_interval": "Intervalo de calidad por encima de cero",
        "format": "Formato conservado",
        "motifs": "Sin aumentar omisiones o contradicciones importantes",
        "variety_preference": "Preferencias de variedad sin retroceso",
        "usable_groups": "Grupos de escenas utilizables sin retroceso",
        "dependency": "Al menos 16/18 hechos completos",
        "dependency_format": "Formato 18/18 en dependencia",
        "all_reviews_frozen": "Revisiones completas y congeladas",
    }
    body += [
        f"<li>{translations[k]}: <b>{'cumple' if v else 'no cumple'}</b></li>"
        for k, v in r["promotion"]["checks"].items()
    ]
    body += ["</ul><table><tr><th>Medida</th><th>32</th><th>v21</th></tr>"]
    for k, name in [
        ("format_valid", "Formato /96"),
        ("motif_major", "Muestras con pérdida importante de motivos"),
        ("prompt_only_echoes", "Sólo consigna"),
        ("dream_words_mean", "Palabras dream, media"),
        ("answer_words_mean", "Palabras answer, media"),
        ("groups_fewer_than_two_episodes", "Grupos con menos de dos episodios utilizables"),
    ]:
        body += [
            f"<tr><td>{name}</td><td>{r['before'][k]:.2f}</td><td>{r['after'][k]:.2f}</td></tr>"
        ]
    body += [
        f"</table><p>Dependencia local: {r['dependency']['before']['full']}/18 → {r['dependency']['after']['full']}/18 hechos completos. Pruebas artificiales, no fidelidad completa.</p>",
        '<h2 id="quality">Calidad y motivos</h2><p>Notas congeladas antes de revelar procedencia. Calidad y cumplimiento son ejes separados; una consigna repetida puede conservar palabras y carecer de escena.</p>',
    ]
    folder = root / "evaluation"
    notes = {n["review_id"]: n for n in read_json(folder / "quality-annotations.json")["ratings"]}
    key = {n["review_id"]: n for n in read_json(folder / "quality-key.json")}
    for row in read_json(folder / "quality-packet.json"):
        rid = row["review_id"]
        n, k = notes[rid], key[rid]
        pref = k[n["preference"] + "_phase"] if n["preference"] in "AB" else n["preference"]
        body += [
            f'<section id="{rid}"><h3>{rid} · {label[pref]}</h3><p>{escape(row["request"])}</p><p>{escape(n["note"])}</p>'
        ]
        for s in "AB":
            m = n["motifs"][s]
            body += [
                f"<details><summary>{label[k[s + '_phase']]}</summary><pre>{escape(row[s])}</pre><p>Motivos: {'pérdida importante' if m['major'] else 'sin pérdida importante'}. {escape(m['note'])}</p></details>"
            ]
        body.append("</section>")
    body += [
        '<h2 id="variety">Variedad entre cuatro muestras por modelo</h2><p>Orden privado aleatorio; clave sin consultar antes de congelar las notas. Diferencias de episodio y relaciones, no un score léxico. Las dudas se conservan.</p>'
    ]
    notes = {n["review_id"]: n for n in read_json(folder / "variety-annotations.json")["ratings"]}
    key = {n["review_id"]: n for n in read_json(folder / "variety-key.json")}
    for row in read_json(folder / "variety-packet.json"):
        rid = row["review_id"]
        n, k = notes[rid], key[rid]
        pref = k[n["preference"] + "_phase"] if n["preference"] in "AB" else n["preference"]
        body += [
            f"<section><h3>{rid} · {label[pref]}</h3><p>{escape(row['request'])}</p><p>{escape(n['note'])}</p>"
        ]
        for s in "AB":
            body += [
                f"<details><summary>{label[k[s + '_phase']]} · {n['usable_episodes'][s]} episodios utilizables</summary>"
            ]
            body += [f"<pre>{escape(t)}</pre><hr>" for t in row[s]]
            body += ["</details>"]
        body += ["</section>"]
    body += (
        ["<h2>Límites</h2><ul>"]
        + [f"<li>{escape(t)}</li>" for t in r["limitations"]]
        + [
            "</ul><footer>DreamBank · CC BY-NC-SA 4.0 · dreambank.net. Textos completos sólo en este informe local.</footer>"
        ]
    )
    style = "body{max-width:1000px;margin:40px auto;padding:0 20px;background:#f6f4ed;color:#213331;font:18px/1.6 system-ui}h1,h2{line-height:1.2}section{padding:18px 0;border-top:1px solid #bac7bc}details{margin:12px 0;padding:10px;background:#eaf0e6}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 system-ui}a{color:#175849}table{border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #bac7bc}"
    (root / "report.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dream · resultado v21</title><style>'
        + style
        + "</style>"
        + "".join(body)
        + "</html>"
    )
    index = (root / "index.html").read_text()
    if "report.html" not in index:
        index = index.replace(
            "<h1>",
            f'<p><strong>Ensayo completado. <a href="report.html">{heading}: ver resultados</a>.</strong></p><h1>',
            1,
        )
    index = index.replace(
        "Preparación terminada. 19 pares claros de 24 pedidos; no se inició entrenamiento.",
        "Lote histórico entrenado una sola vez: 19 pares claros de 24 pedidos; ver el resultado arriba.",
    )
    (root / "index.html").write_text(index)


if __name__ == "__main__":
    render(ROOT)
