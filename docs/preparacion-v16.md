# Diagnóstico de longitud y lote ampliado, sin entrenar

Se completó el siguiente paso autorizado: diagnóstico breve de 1.024 frente a 2.048 tokens
y preparación revisable de 128 relatos. **No se ejecutó entrenamiento.** Los checkpoints
32 y 144 quedaron intactos, y los 200 relatos de prueba no se evaluaron.

## Qué mostró el límite de generación

Se fijaron seis relatos, uno por cada grupo reservado. Dentro de cada grupo se priorizó
la cantidad de truncamientos observados con 1.024 tokens en los pasos 32/144; los empates
se resolvieron por longitud de la referencia descendente y luego identificador. Esta regla
incluyó los tres relatos donde estaban los cuatro truncamientos del paso 144.
La selección se congeló antes de generar a 2.048 tokens.

Es una muestra enriquecida por fallos: sirve para diagnosticar el límite, no para estimar
la calidad general ni seleccionar el mejor checkpoint.

| Checkpoint | Límite | Formato válido | Truncamientos | Score integral |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1.024 | 17/18 | 0 | 0,7980 |
| 32 | 2.048 | 17/18 | 0 | 0,7980 |
| 144 | 1.024 | 14/18 | 4 | 0,6681 |
| 144 | 2.048 | 18/18 | 0 | 0,8543 |

Se reutilizaron las 36 respuestas anteriores de 1.024 y se generaron 36 de 2.048. En los
36 pares los tokens anteriores se conservaron exactamente como prefijo. Las 32 secuencias
ya cerradas no cambiaron, incluido un formato inválido del paso 32 que no estaba truncado.
Los cuatro casos truncados del 144 completaron el formato a los 1.117, 1.234, 1.226 y
1.844 tokens. El score sube porque se pudieron puntuar respuestas antes incompletas;
no hubo cambio de pesos ni mejora aprendida de cercanía.

Antes de otro entrenamiento conviene fijar 2.048 como candidato de límite y evaluar el
checkpoint elegido al inicio y al final con exactamente ese mismo límite. El diagnóstico
no demuestra que el 144 sea mejor que el 32 en el conjunto completo.
[Datos del diagnóstico](resultados-longitud-v16.json).

## Lote listo para revisar

Se seleccionaron 128 relatos de la partición original de entrenamiento: los ocho anteriores
más 120 nuevos. Las 30 personas de entrenamiento aportan cuatro o cinco relatos: 22 grupos
de cuatro y ocho de cinco. Las cuotas y los relatos se eligen por un hash con semilla fija,
sin consultar scores. Se excluyen todas las personas e identificadores de validación y prueba.

Qwen3-8B congelado extrajo las semillas nuevas, sin actualizar pesos y sin sustituir ningún
relato. El asistente comparó las 128 parejas de fuente y semilla en ocho lotes de dieciséis,
y corrigió 92 semillas, especialmente atribuciones de acciones y hechos pensados o temidos
que no ocurrieron. Esta revisión no es una anotación humana independiente.
Las ocho semillas aprobadas del piloto, sus prompts, referencias y longitudes se conservaron
exactamente. Las semillas finales tienen entre 17 y 30 palabras; los tamaños objetivo van
de 40 a 344 palabras y se aplican sólo a answer.

Se documentaron tres normalizaciones mediante posiciones exactas dentro del original:

- Retirar un preámbulo de diario anterior al sueño.
- Retirar una entrevista posterior al despertar.
- Conservar únicamente los episodios no sexuales de un relato que contenía un segmento
  sexual intrafamiliar explícito. Ese segmento no forma parte de la referencia ni de la semilla.

Los originales permanecen en la selección local sin modificar. El lote conserva los 128
identificadores y las cuotas; no se reemplazaron casos después de conocer scores.
Cada referencia preparada guarda el hash original, sus segmentos y el hash resultante.

Hash de selección original:
`b01c79e44d581790a44c003339d6e16809daeace461778711875f8897f1d1002`.
Hash final de las filas de entrenamiento y validación:
`d910fdda3577dfd3b0188face94a99ad6ac0cff09a92ecd85ed9f8a338e2453e`.
[Auditoría resumida](lote-v16-resumen.json).

## Archivos locales

- `results/dream-v16/preparation/revision.html`: 128 fichas con semillas, tamaños, prompts
  y referencias esperadas desplegables. Vista local: `http://127.0.0.1:7577/preparation/revision.html`.
- `training-pack.json`: 128 de entrenamiento y los 24 de validación congelados, separados.
- `prompts-only.jsonl`: datos del generador; no incluye el campo reference.
- `selection.json`: selección original, cuotas, procedencia y fuentes completas.
- `extraction-progress.json` y `dream/holdout/seed-cache/`: extracciones originales.
- `review-batch*.json` y `review-coverage.json`: cambios y cobertura de la revisión.
- `audit.json`: comprobaciones de integridad y separación.
- `results/dream-v16/report/diagnostico.md`: informe de longitud por muestra.

Los enlaces de archivos se refieren al directorio de preparación salvo indicación contraria.
Estas referencias, prompts completos y pesos permanecen fuera de Git. El repositorio
publica código, configuraciones, métricas y hashes. DreamBank: CC BY-NC-SA 4.0,
atribución a dreambank.net; código MIT.

## Reproducción y límites de autorización

```sh
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-length32-v16.yaml --results results/dream-v16/length-step32
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-length144-v16.yaml --results results/dream-v16/length-step144
.venv/bin/python -m think_dream_lab.length_report --results results/dream-v16
.venv/bin/python -m think_dream_lab expanded-prepare \
  --config configs/dream-expanded-v16.yaml --results results/dream-v16/preparation
```

La preparación de los perfiles y la revisión explícita son requisitos locales de esos
comandos. Se conservan snapshots del código y configuración y los contadores de recursos;
se utilizaron las variables de Metal limitadas al proceso documentadas en las rondas previas.

El perfil de preparación rechaza comandos de entrenamiento. Finalizar el lote requiere
la extracción completa y un registro de revisión que cubra los 128 identificadores.
La referencia humana completa no entra en el prompt; dream sigue libre y se mantiene la
misma fórmula de recompensa. Las 149 pruebas pasaron. El entrenamiento de un recorrido
por el lote y el juez independiente continúan pendientes; no se crearon optimizadores ni
nuevos checkpoints en esta etapa.
