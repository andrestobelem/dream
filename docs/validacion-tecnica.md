# Validación técnica anterior

Documento histórico del diagnóstico aritmético. No define el experimento actual de Dream.
La configuración correspondiente es `configs/pilot.yaml`; para relatos de sueños se usa
`configs/dream.yaml` y el [README actual](../README.md).

**Estado del proyecto:** la aritmética se usó únicamente para comprobar el backend y la
recuperación de checkpoints. No es el experimento de investigación que queremos llevar
adelante. El tipo de problemas vinculados con sueños y su evaluación está por definirse.
Los comandos y resultados descritos debajo corresponden a esa validación técnica;
no se ejecutaron las doce corridas comparativas.

Experimento local reproducible para comparar **think** y **dream** durante una adaptación
con LoRA y aprendizaje por refuerzo. La pregunta es si la instrucción cambia las respuestas
iniciales y el aprendizaje, manteniendo la recompensa por acierto y formato.

Contexto, conversación y prompt original:
[Del think al dream](https://del-think-al-drean.andrestobelem.chatgpt.site/).
Inspirado en [DeepSeek-R1](https://arxiv.org/abs/2501.12948), con antecedentes sobre
[plantillas durante RL](https://arxiv.org/abs/2503.20783) e
[instrucciones aprendidas](https://arxiv.org/abs/2610.06851).

## Instalación

Requiere Apple Silicon, Metal, espacio para el checkpoint y [uv](https://docs.astral.sh/uv/).
Desarrollado y probado en M2 Max con 64 GiB y macOS 27.0.1.

```bash
uv sync --locked
```

Esto crea `.venv` con Python 3.11 y dependencias fijadas en `uv.lock`. No modifica el
Python del sistema. El checkpoint es **Qwen/Qwen3-1.7B-Base**, sin cuantizar, con sus
pesos originales. La primera ejecución descarga aproximadamente 3,4 GB desde Hugging Face.
`doctor` resuelve la revisión una sola vez y la fija en `results/model-lock.json`;
para reproducir una corrida en otra carpeta, copiá ese lock o especificá su revisión
en `model.revision` antes de empezar.

## Tres comandos para trabajar

```bash
uv run python -m think_dream_lab doctor
uv run python -m think_dream_lab smoke
uv run python -m think_dream_lab experiment --config configs/pilot.yaml --steps PASOS_ACORDADOS
```

El tercer comando requiere reemplazar `PASOS_ACORDADOS` por un entero elegido después de
revisar el benchmark del smoke. Ejecuta doce corridas independientes y secuenciales:
A/B/C/D para las semillas 11, 22 y 33. **El smoke no las inicia.**

Para generar o regenerar el informe:

```bash
uv run python -m think_dream_lab report --results results/
```

Todos los comandos aceptan `--results`; los que ejecutan modelo también aceptan `--config`.
No se abre un servidor ni se usa un servicio de seguimiento externo. Sólo se descargan
dependencias y el modelo público; los prompts, respuestas y resultados se guardan localmente.

### Qué hace el smoke

1. Ejecuta las pruebas y el diagnóstico del entorno, sin registrar seriales ni UUID.
2. Audita tokenizer, embeddings, revisión del modelo y hashes de los pesos originales.
3. Calibra dificultad en validación: empieza en `medium`; si todas las condiciones fallan
   o todas aciertan, prueba una única alternativa predefinida, `easy` o `hard`. Sólo cambia
   si obtiene un acierto no extremo. No consulta test. Registra cada candidato y la elección.
4. Congela 2.000 problemas de entrenamiento, 200 de validación y 500 de test. Evalúa ocho
   problemas comunes de validación por condición, un prompt neutral y el diagnóstico de marcadores.
5. Entrena A durante cinco actualizaciones con cuatro candidatos por pregunta.
6. Comprueba cambio de adaptadores, pesos base congelados y recarga del checkpoint. Evalúa
   después con A y con el mismo prompt neutral. Guarda respuestas crudas y produce el informe.

Cinco pasos no tienen por qué mejorar el acierto. Si todos los grupos tienen recompensa
constante, puede no haber cambio de adaptadores: el informe lo declara y bloquea el piloto
largo hasta resolver la falta de señal. No se altera la recompensa para forzar una actualización.

### Presupuesto y memoria

El límite inicial es **45 minutos acumulados** para los procesos supervisados de `doctor`,
`smoke` y `experiment`, incluyendo descargas, carga, evaluación y entrenamiento. El registro
`results/budget.json` persiste entre comandos; reanudar no reinicia el tiempo. Se reserva
un margen de hasta diez segundos para interrumpir y finalizar el proceso. La última
actualización incompleta se descarta; se conserva el último checkpoint completo.

El supervisor registra RSS, memoria disponible, presión y swap cada cinco segundos.
Interrumpe ante presión crítica o crecimiento de swap superior a 2 GiB. El pico MLX se
registra por separado: RSS y memoria MLX no son la misma métrica. Las reducciones de longitud
o los cambios de configuración requieren una nueva corrida y repetir las mediciones afectadas.

Después de acordar el piloto largo, ampliá explícitamente `runtime.budget_seconds` según
la estimación. El presupuesto es acumulado, por lo que debe incluir lo ya consumido.
Cambiar sólo límites operativos permite reanudar; cambiar datos, código del protocolo,
prompts, modelo o hiperparámetros invalida la identidad del checkpoint.

Si el supervisor termina abruptamente antes de cerrar su registro, conserva una reserva
conservadora del presupuesto completo. No borres el ledger para presentar el tiempo como
si no se hubiera consumido; inspeccioná el incidente y ajustá el presupuesto explícitamente.

## Controles experimentales

| Condición | Verbo | Delimitador |
|---|---|---|
| A | think | `<think>…</think>` |
| B | think | `<dream>…</dream>` |
| C | dream | `<think>…</think>` |
| D | dream | `<dream>…</dream>` |

Plantilla literal, en texto plano:

```text
A conversation between User and Assistant. The User asks a problem. The Assistant first {verb}s about the problem, writing this process inside <{tag}>...</{tag}>, and then gives the final answer inside <answer>...</answer>.
User: {problem}
Assistant:
```

El modelo genera la respuesta completa, incluidos los tags. No se aplica chat template,
ni se agregan ejemplos ni se antepone razonamiento. Se guardan los IDs exactos de entrada y salida.
No se agregan tokens nuevos. Los metadatos del checkpoint contienen `<think>` y `</think>`
como tokens agregados, marcados como no especiales; `dream` se segmenta de otra manera.
La auditoría registra esa diferencia y verifica que las filas de embeddings existen y son
finitas. Eso **no demuestra cuánto se entrenó cada embedding**.

El diagnóstico adicional compara las instrucciones think/dream con un marcador común
`<work>`, manteniéndolo fuera de la comparación principal de doce entrenamientos. No aísla
perfectamente toda asociación semántica: se interpreta como diagnóstico exploratorio.

El prompt neutral común pide resolver el problema y dar el entero dentro de `<answer>`,
permitiendo trabajo previo libre. No contiene think ni dream. Su cumplimiento de formato
se define por una única respuesta entera al final y se reporta por separado.

### Datos y evaluación

Los problemas usan árboles de suma, resta y multiplicación, con respuestas enteras.
La forma canónica ordena operandos conmutativos para evitar duplicados por intercambio.
Los splits se generan conjuntamente sin reutilizar árboles canónicos; no hay paráfrasis
de enunciados compartidas. La mitad de test reserva árboles con tres operaciones y dos
subexpresiones, frente a las dos operaciones anidadas de entrenamiento.

El generador verifica cada respuesta por dos recorridos aritméticos, sin `eval` y sin
ejecutar texto generado. Se conservan árboles, IDs, dificultad y familia. Esta deduplicación
no pretende resolver equivalencia algebraica general entre todas las expresiones posibles.

Se usa una muestra por problema (`pass@1`), con semillas de evaluación derivadas del ID
y emparejadas entre condiciones. Evaluar preserva el estado aleatorio del entrenamiento.
El experimento largo mide el test completo antes y después con prompt propio y neutral.
Reporta cambios desde cada baseline, diferencias emparejadas y efectos de instrucción,
tag e interacción por semilla. Tres semillas dan una estimación limitada de la variabilidad.

### Recompensa

`r_total = r_acierto + r_formato`, con componentes binarios de peso uno.

- Acierto: exactamente un bloque `<answer>` que contenga un entero ASCII con signo opcional.
  Se compara el número completo con la respuesta conocida; no se buscan substrings.
- Formato: exactamente un bloque de proceso no vacío con el tag de la condición, seguido
  por un único bloque de respuesta. Se rechazan etiquetas extra, cruzadas, repetidas y texto
  exterior salvo espacios. Que el contenido de answer sea un entero pertenece al componente
  de acierto, por lo que formato y acierto se computan independientemente.

No se premian longitud, creatividad, “wait”, estilo onírico ni autoevaluaciones. No hay un LLM juez.

## Integración de GRPO y checkpoints

Se usa el backend existente
[`mlx-lm-lora@4fa3c514`](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/tree/4fa3c5148b17a5833b3074e4e58c6299baec1018)
mediante sus funciones de generación, normalización de ventajas, pérdida y microbatches.
El objetivo no se reimplementa. El wrapper agrega control del orden de datos, clipping global
de gradientes, validaciones, logs y persistencia completa.

Es **GRPO on-policy con una actualización por rollout**, con normalización por secuencia.
El cociente usa `logp - stop_gradient(logp)`: tiene valor uno y conserva el gradiente.
El clipping no entra en acción en ese paso único. La referencia congelada se usa para KL,
no como política anterior. El backend normaliza ventajas por grupo con desvío poblacional
y estabilizador `1e-4`; la estimación KL limita su exponente y reporta saturación. No se
presenta esta variante como una réplica exacta del entrenamiento de DeepSeek.

Configuración inicial: LoRA rango 8, escala MLX 16, todas las capas `q_proj` y `v_proj`,
dropout cero; AdamW `1e-5`, sin weight decay, clipping global 1; beta KL 0,1; grupo 4;
512 tokens como máximo; temperatura 0,8, top-p 0,95, sin top-k ni min-p adicional.
Los valores completos están en `configs/pilot.yaml`.

La generación y el scoring usan modo eval y dropout cero. Se conservan los tokens muestreados
sin decode/encode para calcular la pérdida. El scoring recibe el prompt completo, incluye
el primer token generado y el EOS muestreado, y excluye padding y tokens del prompt.
El texto enviado al reward omite EOS. No hay un stop artificial en `</answer>`, para poder
detectar texto o bloques adicionales; los truncamientos quedan registrados.

Cada actualización guarda un directorio de checkpoint mediante renombrado atómico, seguido
de un puntero `latest.json`. Incluye adaptadores, estado AdamW, PRNG de Python/NumPy/MLX,
paso/cursor del orden determinista, métricas y rollouts. Los logs se recuperan del checkpoint
si la ejecución se interrumpe después de guardarlo y antes de volcarlos.

```bash
uv run python -m think_dream_lab smoke --resume
uv run python -m think_dream_lab experiment --config configs/pilot.yaml --steps PASOS_ACORDADOS --resume
```

Para una sola corrida del piloto se requieren ambos selectores:
`--condition D --seed 22`. Ninguna condición continúa desde los adaptadores de otra.
Las corridas completas se omiten al reanudar la secuencia.

La continuidad de pesos, optimizador y aleatoriedad se prueba con un modelo pequeño y el
objetivo real del backend. No se promete identidad bit a bit entre versiones, equipos o
implementaciones de kernels. La recuperación del modelo grande se valida mediante recarga
de adaptadores; una prueba completa de interrupción y continuación del modelo grande es
un control adicional, no un resultado implícito de la prueba pequeña. Para ejecutarlo:

```bash
uv run python -m think_dream_lab smoke --resume --verify-continuity
```

Esta comprobación carga el checkpoint del paso 2 en un proceso nuevo y repite los pasos 3–5
en un directorio de diagnóstico separado. Compara adaptadores, estado del optimizador y PRNG
con la corrida original, y verifica logits finitos después de recargar. Son tres actualizaciones
adicionales de diagnóstico que consumen el mismo presupuesto acumulado; no cuentan como otra
condición experimental. Sus artefactos no se sobrescriben al repetir el comando.

## Archivos y pruebas

`results/` contiene los locks y auditorías, datos congelados, respuestas crudas, checkpoints,
telemetría, errores y el informe. `results/report/informe.md` resume la ejecución;
`evaluations.csv`, `training.csv`, `deltas.csv`, `paired-effects.csv`, `estimate.json`
y `curvas.png` se generan cuando hay datos suficientes. Los pesos quedan en la caché local
de Hugging Face. Los artefactos pesados y resultados están excluidos de Git.

```bash
uv run pytest -q
uv run ruff check think_dream_lab tests
uv run ruff format --check think_dream_lab tests
```

Las pruebas cubren recompensas adversariales, prompts factoriales, datasets, contexto causal,
primer token, EOS, padding, ventajas constantes, gradiente on-policy, pesos congelados,
continuidad de checkpoints, recuperación de logs, presupuesto efectivo y concurrencia.

## Alcance de las conclusiones

La apariencia del texto no demuestra procesos internos ni experiencias subjetivas.
Igualar resultados en este piloto no demuestra equivalencia general. Una ventaja de dream
tampoco demuestra un sueño interno. Las conclusiones corresponden a este checkpoint,
estas tareas, LoRA, este backend y el presupuesto medido.

El código del proyecto usa licencia MIT. Las dependencias y pesos conservan sus propias
licencias; el backend integrado y Qwen declaran Apache-2.0.
