# Dream

Entrenar un modelo para **imaginar una escena dentro de `<dream>` y después contar ese sueño dentro de `<answer>`**, aprendiendo de relatos humanos mediante un juez local. La aritmética quedó como [validación técnica anterior](docs/validacion-tecnica.md); no es el experimento del proyecto.

El [plan actualizado](docs/plan.md) distingue implementación y validación pendiente.
El protocolo actual en desarrollo es [dream-v4](docs/rubrica-v4.md): generador base sin
posentrenamiento de razonamiento y juez de etiquetas cerradas. Su entrenamiento está
deshabilitado hasta calibrar el evaluador. El generador vuelve a ser el Qwen3 base del primer piloto.

## Diseño definido

El recorrido activo es únicamente **D: Soñar + `<dream>`**, seguido del relato en `<answer>`. Las variantes A/B/C y neutral quedan como antecedentes.

- **Corpus principal:** [DreamViews, versión 1](https://zenodo.org/records/19161757), descrito en [Mallett, 2026](https://arxiv.org/abs/2603.26992). Conservamos las etiquetas originales de sus autores: lúcido, no lúcido, pesadilla, fragmento, memorable y falso despertar.
- **Entrada:** requisitos visibles derivados de los relatos, guardados para generar y evaluar sueños. El evaluador de cumplimiento recibe los mismos requisitos, sin detalles ocultos del relato reservado.
- **Generador actual:** [Qwen3-1.7B-Base](docs/modelo-base.md), revisión del primer piloto, sin plantilla de chat aplicada ni modo de razonamiento. Sólo se usa Dream; la asimetría con el token think queda fuera del diseño actual. Falta el nuevo piloto con el evaluador v4.
- **Juez:** `mlx-community/Qwen3-8B-4bit`, revisión fijada, congelado y temperatura cero. Toda la ejecución es local, sin APIs pagas ni envío de relatos.
- **Piloto histórico:** baseline A/B/C/D y neutral; cinco actualizaciones de D con Qwen3, evaluación y recarga. La comparación de doce corridas quedó fuera del alcance actual.

Un sueño humano puede ser cotidiano y coherente: **absurdo no equivale a onírico**. El score mide semejanza textual; no demuestra experiencias subjetivas del modelo ni autenticidad humana.

## Corpus y categorías

La selección conserva relatos en inglés de 80–350 palabras con alguna categoría explícita de sueño. Excluye entradas sólo sin categorizar o de notas laterales. Verifica el checksum de Zenodo, deduplica texto normalizado, limita a diez relatos por autor y separa train/validación/test **por autor**. Congela 1.000/100/200 ejemplos, distribuidos entre las categorías disponibles sin inventar etiquetas.

También registra indicadores débiles de familia, animales, escuela, hogar, exteriores, vuelo, amenaza, miedo, alegría y transporte. Se inspiran en los dominios del [sistema Hall/Van de Castle](https://dreams.ucsc.edu/Coding/), pero son búsquedas por palabras: **no son anotaciones humanas H/VdC** ni interpretaciones psicológicas.

[DreamBank](https://dreambank.net/) queda identificado como alternativa y futuro contraste externo. Sólo recurriremos a escritores surrealistas si el corpus no alcanza después de los filtros. No mezclamos automáticamente fuentes ni descargamos obras de Bretón.

Estas categorías describen contenido y tipo de sueño; no traen un score universal de
oniricidad. La rúbrica del proyecto es una definición operativa que hay que validar.

La [rúbrica completa en español](docs/rubrica.md) fija los criterios de cada score y cómo
hacer la revisión ciega.

## Juez y recompensa históricos

Esta sección describe `dream-v2`. La nueva [clasificación cerrada](docs/rubrica-v4.md)
reemplaza las notas libres de cumplimiento y fidelidad por probabilidades de clases,
agregación fija y abstención. Su primera comprobación técnica pasó, pero sólo coincidió
con cinco de ocho controles semánticos sin ejemplos. La variante con ejemplos pasó
siete de ocho controles conocidos y tres de cuatro nuevos; todavía no está habilitada para entrenar.

El juez hace tres pasadas independientes:

1. Ve sólo el relato final y puntúa **semejanza con un relato de sueño** de 0 a 4. Desconoce condición, fase de entrenamiento, referencia y procedencia humana/sintética.
2. Ve únicamente referencia y relato, y puntúa **alineación semántica con la referencia**, de 0 a 4. Permite paráfrasis y variaciones; la referencia es un ejemplo válido, no la única continuación posible.
3. Ve únicamente escena previa y relato final, y puntúa **fidelidad de la narración a esa escena**, de 0 a 4. La referencia no participa de esta evaluación.

Normalizamos las dimensiones a [0,1]:

```text
calidad = 0.50 × onírico + 0.30 × alineación + 0.20 × fidelidad
reward  = 0.90 × calidad + 0.10 × formato
```

Guardamos componentes separados, prompts, respuesta original del juez, tiempos y caché. El formato exige una escena y un relato no vacíos, en un único bloque de cada tipo y sin texto exterior. Un relato final válido puede recibir calidad aunque falle el formato completo; sin bloque final, la calidad es cero.

El prompt neutral pide directamente el relato en `<answer>`. Al no tener escena previa, usa sólo onírico y alineación, renormalizados. Sus dimensiones comunes pueden compararse; su reward total no se presenta como equivalente al de dos bloques.

Antes de RL, el juez debe distinguir tres relatos humanos de tres controles: prosa instructiva, palabras oníricas desordenadas y una petición de score alto. La mediana humana debe superar en al menos un punto a la negativa y la manipulación debe recibir como máximo 1/4. Si falla, **RL queda bloqueado**. Los textos se presentan como JSON escapado y datos no confiables.

Estos controles básicos no validan científicamente al juez. La revisión humana ciega y un juez de otra familia quedan pendientes antes de interpretar una comparación larga. Compartir familia Qwen puede introducir sesgos. Al ser público el corpus, tampoco se puede excluir memorización previa.

## Ejecutar en Apple Silicon

```bash
uv sync --locked
uv run python -m think_dream_lab doctor
uv run python -m think_dream_lab corpus
uv run python -m think_dream_lab evaluate --condition D --limit 1
uv run python -m think_dream_lab closed-check
uv run python -m think_dream_lab report --results results/dream-v4-qwen
```

Configuración predeterminada de la CLI: `configs/dream-v4.yaml`; resultados en
`results/dream-v4-qwen`. `configs/dream.yaml` conserva el protocolo histórico y
`configs/dream-v4-smollm2.yaml` conserva la prueba con SmolLM2. El corpus ocupa unos 110 MB. Los pesos
se guardan en la caché local de Hugging Face. `.venv` usa Python 3.11 y `uv.lock` fija las
dependencias sin modificar el Python del sistema.

El piloto histórico está en `results/dream/`; los diagnósticos nuevos están en
`results/dream-v4/dream/` para SmolLM2 y `results/dream-v4-qwen/dream/` para el retorno a Qwen. Se conservan datos y hashes, auditorías, probabilidades y contexto.

```text
results/dream/report/informe.md
results/dream/report/examples.json
```

El historial original tiene un presupuesto de 45 minutos en `results/budget.json`.
El nuevo perfil Qwen v4 tiene 450 segundos acumulados en su propio `budget.json`; las
ejecuciones anteriores y sus consumos se conservan. Se registran RSS, presión y swap;
la ejecución se interrumpe ante presión crítica o crecimiento de swap superior a 2 GiB.
Un límite conserva el último checkpoint completo y deja la prueba incompleta.

```bash
uv run python -m think_dream_lab smoke --resume
```

Cada actualización guarda adaptadores, optimizador, PRNG y cursor de datos. Un error del juez interrumpe la corrida y se registra; no se convierte en un castigo al generador. Las llamadas al juez preservan el PRNG del generador y sus pesos permanecen congelados.

Los comandos de entrenamiento están bloqueados para v4 mientras se completa su
integración y calibración. Después se deberá fijar explícitamente el presupuesto y los pasos:

```bash
uv run python -m think_dream_lab experiment --config configs/dream-v4.yaml --steps PASOS
```

La configuración actual ejecuta sólo D=dream/dream. Cada entrenamiento parte del checkpoint base con adaptadores nuevos. Las variantes A/B/C y neutral pertenecen al diseño histórico; el flujo actual no las agrega automáticamente.

Sin escena previa, la fidelidad se fuerza a cero en código aunque el juez diga otra cosa;
el juicio original se conserva para auditarlo. El informe separa cobertura de formato y
score de los relatos extraíbles para no confundir aprender tags con producir mejores sueños.

## Revisión humana

El informe prepara `results/dream/report/revision-ciega.csv`, que mezcla relatos generados
y humanos sin mostrar su origen, condición o score. Las columnas de puntuación humana y
notas están vacías. `clave-revision.json` permite contrastarlas después; no hay que abrir
la clave antes de puntuar. Esta revisión está preparada, no realizada.

Regenerar informes conserva el paquete original y sus anotaciones. Los paquetes nuevos
se escriben con otro nombre para no borrar ni cambiar la revisión en curso.

Por autorización del usuario, el asistente completó una revisión separada de esos seis
relatos y de 29 pares de requisitos. Las copias `revision-asistente.csv` distinguen la
procedencia de las notas; no se rellenaron columnas humanas con puntuaciones del asistente.
Se usarán como referencias de desarrollo para el piloto exploratorio. Los detalles y límites
están en la [rúbrica actual](docs/rubrica-v4.md).

La generación de sueños termina en `</answer>`, EOS o el límite de tokens, con el mismo
criterio en todas las condiciones. Los tokens muestreados del cierre se conservan para GRPO.
La detección del cierre es textual y admite que BPE lo fusione con puntuación; no recorta
ni recodifica la salida muestreada. La comprobación actual de Qwen en D produjo 10/10
respuestas con formato completo, con pesos intactos y sin llamadas al juez.

## Inspect AI e Inspect Flow

Las tareas y resultados del entrenamiento también se pueden organizar con
[Inspect AI + Inspect Flow](evals/README.md). La integración opcional usa un entorno separado
y reproduce las salidas y scores existentes, sin nuevas inferencias:

```bash
uv sync --project evals --locked
uv run --project evals flow run evals/flow.py --dry-run
uv run --project evals flow run evals/flow.py
```

MLX/GRPO actualiza los adaptadores. Inspect define muestras, solvers y scorers; Flow organiza
condiciones, fases y corridas, y reutiliza logs mediante snapshots verificables.

## Verificación y licencias

```bash
uv run pytest -q
uv run ruff check think_dream_lab tests
```

Se usa GRPO on-policy de `mlx-lm-lora`, fijado a un commit, con una actualización por rollout y referencia congelada para KL. Se comprueban contexto causal, máscaras, primer token, gradientes, pesos base y recuperación. El generador base produce los tags completos sin chat template.

Código: **MIT**. DreamViews: **CC BY-NC 4.0**, según los [metadatos archivados](https://zenodo.org/api/records/19161757). DreamBank declara CC BY-NC-SA 4.0. Las licencias del corpus no se sustituyen por MIT; datos, relatos y pesos quedan fuera de Git y no se publican automáticamente.

Motivación original: [Del think al dream](https://del-think-al-drean.andrestobelem.chatgpt.site/).
