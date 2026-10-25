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

## Último experimento completado

Se completó una pasada de **128 actualizaciones nuevas desde el checkpoint 32 hasta el 160**,
una por cada relato aprobado y cuatro candidatos por actualización. Ambas evaluaciones
usaron 2.048 tokens y los mismos 24 relatos reservados de seis personas.
Un juez Phi-4 independiente evaluó ambas fases mezcladas, sin participar en la recompensa.

| Medida | Antes · paso 32 | Después · paso 160 |
| --- | ---: | ---: |
| MiniLM, mismos 67 pares válidos | 0,6435 | 0,6485 |
| Juez de cercanía, mismos pares, 0–4 | 2,2537 | 2,0896 |
| Formato válido | 70/72 | 69/72 |
| Score incluyendo fallos de formato | 0,7998 | 0,7916 |

**No se observó una mejora convincente.** MiniLM subió apenas y el juez bajó; ambos
intervalos exploratorios incluyen cero. El error de longitud también aumentó: 0,3643 → 0,7443.
El paso 32 sigue como referencia práctica y el 160 se conserva como resultado experimental.
Los reservados quedaron fuera del entrenamiento y los 200 de prueba no se evaluaron.
Auditoría completa aprobada y 166 pruebas exitosas.
[Método, resultados y límites](docs/experimento-ampliado-v17.md),
[datos numéricos](docs/resultados-v17.json), [auditoría](docs/auditoria-v17.json).

## Preparación y diagnóstico previos

El [diagnóstico de longitud y el lote ampliado](docs/preparacion-v16.md) están completos.
En una muestra de seis relatos elegida para incluir los truncamientos del paso 144,
aumentar de 1.024 a 2.048 tokens recuperó los cuatro formatos incompletos: 14/18 → 18/18.
Los prefijos coincidieron exactamente; las respuestas ya cerradas no cambiaron.
Es un diagnóstico de fallos, no evidencia de mejor cercanía ni un ranking general de modelos.

El lote contiene los ocho relatos anteriores más 120 nuevos, balanceados entre 30 personas
de entrenamiento. Todos los reservados quedan fuera. Las 128 semillas fueron revisadas
contra sus fuentes; se registraron 92 correcciones y tres recortes trazables de referencia.
El lote local, sus prompts separados y la vista `results/dream-v16/preparation/revision.html`
quedaron aprobados para esta pasada.

## Ensayo anterior completado

Se continuó desde el checkpoint 128 hasta el **144**, con el mismo lote de ocho relatos,
restaurando optimizador y aleatoriedad. Los **24 relatos reservados de seis personas**
quedaron fuera del entrenamiento y se evaluaron con tres generaciones por modelo y relato.
Este ensayo conservó los ocho relatos originales; el lote ampliado se entrenó después en la bifurcación 32 → 160.

| Medida | Base | Paso 128 | Paso 144 |
| --- | ---: | ---: | ---: |
| Coseno, sólo respuestas válidas | 0,6219 (n=52) | 0,6425 (n=67) | 0,6406 (n=68) |
| Formato válido | 52/72 | 67/72 | 68/72 |
| Score incluyendo fallos de formato | 0,5857 | 0,7642 | 0,7747 |
| Truncamientos | 11 | 4 | 4 |

En los **64 pares válidos del paso 128 y el 144**, el coseno pasó de 0,6436 a 0,6398:
cambio de **−0,0038**. Doce medias por relato mejoraron y doce empeoraron. La cercanía
quedó prácticamente estable, con leve baja observada y sin mejora concluyente. El formato
mejoró ligeramente, pero aumentó el error de longitud. Base y checkpoint 128 quedaron intactos.
[Continuación y límites](docs/continuacion-v15.md), [datos numéricos](docs/resultados-v15.json).

Las rondas anteriores, desde el [paso 128](docs/continuacion-v14.md) al
[piloto con cuatro relatos](docs/resultado-entrenamiento-v6.md), se conservan aparte en
el [plan y su historia](docs/plan.md). Los 24 casos siguen siendo validación de desarrollo;
los 200 de prueba permanecen sin usar.

## Datos y ejecución

La importación real de DreamBank conserva 1.000 relatos de entrenamiento, 100 de validación
y 200 de prueba, deduplicados y separados en 43 grupos. Los primeros entrenamientos usaron ocho relatos
y el piloto evaluó cuatro; la última pasada usó 128 relatos de 30 personas.
La evaluación ampliada usa otros 24 relatos de validación;
los 200 de prueba no se utilizaron. El registro de
alias y hogares está en [configs/dreambank-groups.json](configs/dreambank-groups.json).

El piloto original usa [configs/dream-v6.yaml](configs/dream-v6.yaml); los comandos para
continuarlo están en [continuación v15](docs/continuacion-v15.md). Con el lote
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
del lote corto está en `results/dream-v6/preparation-short-seeds/`; incluye los prompts,
referencias separadas, longitudes y hashes. Un clon del repositorio necesita esos datos
locales antes de ejecutar esos ensayos. El lote ampliado aprobado está en
`results/dream-v16/preparation/`; sus comandos de entrenamiento y evaluación están en
el [experimento ampliado](docs/experimento-ampliado-v17.md).

## Método e historia

- [Plan y estado](docs/plan.md).
- [Experimento ampliado: una pasada por 128 relatos](docs/experimento-ampliado-v17.md).
- [Preparación del lote de 128 y diagnóstico de longitud](docs/preparacion-v16.md).
- [Continuación hasta el paso 144](docs/continuacion-v15.md).
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
