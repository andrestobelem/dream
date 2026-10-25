# Dream

Un experimento local para pedirle a Qwen3-1.7B-Base que **sueñe dentro de `<dream>` y
cuente lo soñado dentro de `<answer>`**, aprendiendo de relatos humanos de DreamBank.
Sólo se trabaja con Dream, sin comparación con think ni problemas de aritmética.

El generador recibe semillas naturales breves y la longitud aproximada del relato esperado.
La referencia humana completa queda fuera del prompt. El contenido de dream es libre: no
se evalúa su significado ni se exige fidelidad detallada entre bloques.

En el ensayo actual, la recompensa depende de la similitud de answer con **su referencia
humana asociada**: `(1 + coseno) / 2`, usando MiniLM congelado. GRPO asigna esa recompensa a
todos los tokens muestreados, incluidos los del bloque de sueño. No hay penalización por
coincidencias textuales ni por diferencias de longitud. Un formato inválido recibe cero.

## Resultado actual

Se continuó desde el checkpoint 48 hasta el **64**, con el mismo lote de ocho relatos,
restaurando optimizador y aleatoriedad. Los **24 relatos reservados de seis personas**
quedaron fuera del entrenamiento y se evaluaron con tres generaciones por modelo y relato.

| Medida | Base | Paso 48 | Paso 64 |
| --- | ---: | ---: | ---: |
| Coseno, sólo respuestas válidas | 0,6219 (n=52) | 0,6537 (n=69) | 0,6460 (n=67) |
| Formato válido | 52/72 | 69/72 | 67/72 |
| Score incluyendo fallos de formato | 0,5857 | 0,7924 | 0,7659 |
| Truncamientos | 11 | 1 | 4 |

En los **66 pares válidos del paso 48 y el 64**, el coseno pasó de 0,6584 a 0,6469:
cambio de **−0,0115**. Catorce medias por relato mejoraron y diez empeoraron, con pérdidas
mayores que las ganancias. Esta ronda retrocedió en promedio; la incertidumbre por persona
incluye cero y no demuestra un deterioro general. Base y checkpoint 48 quedaron intactos.
[Continuación y límites](docs/continuacion-v10.md), [datos numéricos](docs/resultados-v10.json).

Las [rondas hasta el paso 48](docs/continuacion-v9.md),
[paso 32](docs/continuacion-v8.md), la [evaluación del paso 16](docs/evaluacion-ampliada-v7.md)
y el [piloto con cuatro relatos](docs/resultado-entrenamiento-v6.md) se conservan aparte.
Los 24 casos siguen siendo validación de desarrollo; los 200 de prueba permanecen sin usar.

## Datos y ejecución

La importación real de DreamBank conserva 1.000 relatos de entrenamiento, 100 de validación
y 200 de prueba, deduplicados y separados en 43 grupos. El entrenamiento usó ocho relatos
y el piloto evaluó cuatro. La evaluación ampliada usa otros 24 relatos de validación;
los 200 de prueba no se utilizaron. El registro de
alias y hogares está en [configs/dreambank-groups.json](configs/dreambank-groups.json).

El piloto original usa [configs/dream-v6.yaml](configs/dream-v6.yaml); los comandos para
continuarlo están en [continuación v10](docs/continuacion-v10.md). Con el lote
local aprobado y las dependencias instaladas:

```sh
uv sync --extra embeddings
.venv/bin/python -m think_dream_lab experiment --config configs/dream-v6.yaml --results results/dream-v6 --steps 16 --resume
.venv/bin/python -m think_dream_lab report --results results/dream-v6
.venv/bin/pytest -q
```

La corrida completada se reutiliza. Para reanudar en una máquina afectada por el error de
interactividad de Metal, se usaron variables limitadas al proceso; ver el documento de
resultados. No se reinicia ningún contador ni se eliminan intentos fallidos.

El lote local, las referencias y los pesos permanecen fuera de Git. La preparación
revisable está en `results/dream-v6/preparation-short-seeds/`; incluye los prompts,
referencias separadas, longitudes y hashes. Un clon del repositorio necesita esos datos
locales antes de ejecutar este ensayo.

## Método e historia

- [Plan y estado](docs/plan.md).
- [Continuación hasta el paso 64](docs/continuacion-v10.md).
- [Continuación hasta el paso 48](docs/continuacion-v9.md).
- [Continuación hasta el paso 32](docs/continuacion-v8.md).
- [Evaluación ampliada sin entrenar](docs/evaluacion-ampliada-v7.md), [datos numéricos](docs/resultados-v7.json).
- [Preparación del lote corto](docs/preparacion-natural-v6.md).
- [Entrenamiento pareado](docs/entrenamiento-pareado-v6.md).
- [Resultados numéricos](docs/resultados-v6.json).
- [Prueba inicial de la plantilla natural](docs/plantilla-natural.md).
- [Piloto anterior y continuidad](docs/validacion-dreambank.md).

La recompensa anterior al vecino más cercano del banco queda documentada en
[recompensa por embeddings](docs/recompensa-embeddings.md). DreamViews y su juez se
archivaron en `archive/dreamviews-judge-20261025`. La aritmética, las condiciones A/B/C
y los experimentos con otros generadores son antecedentes.

Fuente principal: [DreamBank](https://dreambank.net/). Código **MIT**; datos **CC BY-NC-SA 4.0**,
con atribución a dreambank.net. La licencia del código no reemplaza la de los datos.
El [site del proyecto](https://del-think-al-drean.andrestobelem.chatgpt.site/proyecto-dream)
presenta el método y sus resultados. La posible memorización de un corpus público sigue
siendo una limitación de interpretación.
