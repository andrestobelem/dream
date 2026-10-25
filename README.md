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

Se continuó desde el checkpoint 112 hasta el **128**, con el mismo lote de ocho relatos,
restaurando optimizador y aleatoriedad. Los **24 relatos reservados de seis personas**
quedaron fuera del entrenamiento y se evaluaron con tres generaciones por modelo y relato.
La propuesta de ampliar a 128 relatos sigue pendiente; este segmento conserva los ocho originales.

| Medida | Base | Paso 112 | Paso 128 |
| --- | ---: | ---: | ---: |
| Coseno, sólo respuestas válidas | 0,6219 (n=52) | 0,6413 (n=68) | 0,6425 (n=67) |
| Formato válido | 52/72 | 68/72 | 67/72 |
| Score incluyendo fallos de formato | 0,5857 | 0,7750 | 0,7642 |
| Truncamientos | 11 | 2 | 4 |

En los **64 pares válidos del paso 112 y el 128**, el coseno pasó de 0,6394 a 0,6473:
ganancia observada de **+0,0079**. Trece medias por relato mejoraron y once empeoraron.
La incertidumbre por persona incluye cero: mejora pequeña, todavía no concluyente.
El resultado general es mixto: formato y truncamientos retrocedieron, mientras que mejoró
el ajuste de longitud. Base y checkpoint 112 quedaron intactos.
[Continuación y límites](docs/continuacion-v14.md), [datos numéricos](docs/resultados-v14.json).

Las rondas anteriores, desde el [paso 112](docs/continuacion-v13.md) al
[piloto con cuatro relatos](docs/resultado-entrenamiento-v6.md), se conservan aparte en
el [plan y su historia](docs/plan.md). Los 24 casos siguen siendo validación de desarrollo;
los 200 de prueba permanecen sin usar.

## Datos y ejecución

La importación real de DreamBank conserva 1.000 relatos de entrenamiento, 100 de validación
y 200 de prueba, deduplicados y separados en 43 grupos. El entrenamiento usó ocho relatos
y el piloto evaluó cuatro. La evaluación ampliada usa otros 24 relatos de validación;
los 200 de prueba no se utilizaron. El registro de
alias y hogares está en [configs/dreambank-groups.json](configs/dreambank-groups.json).

El piloto original usa [configs/dream-v6.yaml](configs/dream-v6.yaml); los comandos para
continuarlo están en [continuación v14](docs/continuacion-v14.md). Con el lote
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
- [Continuación hasta el paso 128](docs/continuacion-v14.md).
- [Continuación hasta el paso 112](docs/continuacion-v13.md).
- [Continuación hasta el paso 96](docs/continuacion-v12.md).
- [Continuación hasta el paso 80](docs/continuacion-v11.md).
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
