# Continuación del checkpoint 16 al 32

Se fijaron otros 16 pasos antes de iniciar la continuación. Se conserva el lote de ocho
relatos, sus semillas cortas y longitudes, la recompensa pareada, el encoder y todos los
hiperparámetros. Los cuatro relatos del piloto, los 24 de la evaluación ampliada y los
200 de prueba quedan fuera de las actualizaciones.

Se restaura el checkpoint 16 completo: adaptadores, momentos del optimizador, contadores
y estados aleatorios de Python, NumPy y MLX. Se conserva la identidad científica original,
que incluye implementación, modelo, lote y recompensa. Cada segmento tiene además una
identidad de procedencia con origen, SHA-256 y paso de destino. Los archivos históricos
no se sobrescriben. La nueva salida se guarda en `results/dream-v8/`.

El perfil `configs/dream-v8.yaml` conserva los parámetros científicos originales, incluido
el horizonte histórico del piloto; el destino absoluto 32 se pasa de manera explícita
con `--target-step`. Sólo cambia la ventana operativa de supervisión a dos horas: el
usuario levantó el límite anterior de tiempo. Los consumos históricos se conservan aparte.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v8.yaml --results results/dream-v8 \
  --source-run results/dream-v6/dream/paired/run.json --target-step 32
```

Si se interrumpe, el mismo comando restaura el último checkpoint completo del segmento.
Al terminar, se verifica que la base permanezca intacta y que la recarga reproduzca los
adaptadores guardados. Se evalúa el paso 32, sin elegir el mejor paso con validación.

## Resultado observado

Se completaron los pasos 17–32: 64 rollouts nuevos y 18.805 tokens de entrenamiento.
Los nuevos pasos consumieron 314,31 segundos de cómputo de entrenamiento, además del
arranque, la evaluación y las verificaciones. La base y el checkpoint 16 quedaron intactos;
los adaptadores del paso 32 cambiaron y su recarga fue exacta.

| Medida | Base | Paso 16 | Paso 32 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 66/72 | 70/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6330 · n=66 | 0,6453 · n=70 |
| Score incluyendo fallos de formato | 0,5857 | 0,7485 | 0,7998 |
| Truncamientos | 11 | 2 | 0 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 37,9% | 36,4% |

**Paso 16 → 32:** en los mismos 64 pares válidos, coseno **0,6320 → 0,6398**, cambio
medio **+0,0078**. Dieciséis medias por relato mejoraron y ocho empeoraron.
La media con igual peso por relato/persona fue +0,0099 y su intervalo exploratorio del
95% fue **[-0,0100; +0,0284]**. Este último no es el intervalo de la media de todos los
pares: cambia la ponderación. La señal semántica es pequeña y su incertidumbre incluye
empeoramiento; no se considera evidencia sólida de mejora general.

**Base → 32:** 51 pares válidos, coseno 0,6213 → 0,6428, cambio +0,0215.
Diecisiete medias por relato mejoraron y seis empeoraron; uno no tuvo pares válidos.
La media por relato/persona fue +0,0272, con intervalo [-0,0072; +0,0661].
La mejora de formato es más clara que la de cercanía.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde
vectores guardados, sin nuevas inferencias. Confirmó referencias asociadas correctas,
ningún relato ni persona reservada en las actualizaciones, pesos intactos durante la
evaluación y cero actualizaciones en ella. Pasaron las 142 pruebas.

[Datos numéricos](resultados-v8.json). Informe, gráfico y auditoría local:
`results/dream-v8/dream/report/`. El checkpoint está en
`results/dream-v8/dream/paired/checkpoints/step-000032/`. No se entrenó más allá del paso 32.

## Evaluación posterior

Se reutiliza exactamente el lote congelado de 24 relatos de seis grupos, con tres
muestras por relato, semillas de muestreo por posición y las mismas longitudes.
Las respuestas anteriores de base y paso 16 se reutilizan sin regenerarlas.
El paso 32 genera otras 72 respuestas. La referencia humana completa sigue fuera del prompt.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v8.yaml --results results/dream-v8
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v8 --previous results/dream-v7-eval
```

Las comparaciones pareadas usan sólo posiciones válidas en ambos modelos y muestran
cuántas quedan. El score con cero por formato inválido se informa por separado. El
intervalo por persona promedia primero dentro de cada relato y luego dentro de cada
grupo; su estimador difiere de la media de todos los pares válidos.

Estos 24 relatos ya informaron la decisión de continuar: siguen reservados de las
actualizaciones, pero constituyen validación de desarrollo y no una prueba ciega nueva.
La cercanía se mide con el mismo MiniLM del entrenamiento; no valida calidad onírica.
Los 200 relatos de prueba permanecen sin utilizar.

Fuente DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT. Referencias,
generaciones completas y pesos permanecen fuera de Git.

## Preparación del lote reutilizado

Los comandos de evaluación requieren la copia local del lote v7, conservando íntegramente
sus filas y prompts. No se vuelve a ejecutar el extractor ni se seleccionan otros relatos.
Esta preparación sólo cambia el vínculo al entrenamiento evaluado en los metadatos:

```python
from pathlib import Path
from think_dream_lab.config import load_config
from think_dream_lab.storage import digest, read_json, write_json

config = load_config("configs/dream-eval-v8.yaml")
previous = Path("results/dream-v7-eval/dream/holdout")
target = Path("results/dream-v8/dream/holdout")
selection = read_json(previous / "selection.json")
selection["config"] = config["heldout_evaluation"]
pack = read_json(previous / "evaluation-pack.json")
pack["selection_hash"] = digest(selection)
pack["reused_cohort_from"] = str(previous / "evaluation-pack.json")
write_json(target / "selection.json", selection)
write_json(target / "evaluation-pack.json", pack)
```

Se preparó antes de generar con el paso 32 y el hash de filas siguió siendo
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
