# Continuación del checkpoint 96 al 112

Se fijaron otros 16 pasos antes de iniciar: pasos 97–112. Se conserva el lote de ocho
relatos, las semillas y longitudes, el encoder, la recompensa pareada y los hiperparámetros.
Se restaura el checkpoint 96 completo, con adaptadores, optimizador y estados aleatorios.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones,
con tres muestras por relato. Base y paso 96 reutilizan sus respuestas originales.
Se evalúa el paso 112, sin seleccionar un checkpoint intermedio mirando validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v13.yaml --results results/dream-v13 \
  --source-run results/dream-v12/dream/paired/run.json --target-step 112
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v13.yaml --results results/dream-v13
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v13 --previous results/dream-v12
```

El perfil conserva el horizonte histórico del piloto para mantener la identidad científica;
`--target-step 112` establece el destino absoluto de este segmento. La ventana de supervisión
sigue siendo de dos horas y los consumos históricos se conservan. El usuario había levantado
el límite total de tiempo.

Antes de generar con el paso 112 se reutilizó el lote local v12 en v13, siguiendo la preparación
de [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), usando
`previous = results/dream-v12/dream/holdout`, destino v13 y `configs/dream-eval-v13.yaml`.
Sólo cambia el vínculo de procedencia. Las filas mantienen el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se extrajeron ni reescribieron semillas.

## Resultado observado

Se completaron los pasos 97–112: 64 rollouts nuevos, 32.201 tokens de entrenamiento y
517,39 segundos adicionales medidos en los pasos, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 96 quedaron intactos. La recarga del 112 fue exacta.

| Medida | Base | Paso 96 | Paso 112 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 68/72 | 68/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6476 · n=68 | 0,6413 · n=68 |
| Score incluyendo fallos de formato | 0,5857 | 0,7780 | 0,7750 |
| Truncamientos | 11 | 4 | 2 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 44,3% | 63,8% |

**Paso 96 → 112:** en los mismos **64 pares válidos**, coseno **0,6611 → 0,6381**,
cambio medio **−0,0230**. Nueve medias por relato mejoraron y catorce empeoraron; un relato
no tuvo posiciones válidas en ambos modelos. La media con igual peso por relato/persona
fue −0,0244, con intervalo exploratorio del 95% **[−0,0564; +0,0026]**. Este intervalo
corresponde a la media por relato/persona, no a la media de todos los pares válidos.

Esta ronda no mejoró la cercanía respecto del paso 96: retrocedió en promedio y el score
integral también bajó. El formato quedó igual y los truncamientos disminuyeron, mientras
que el error de longitud aumentó. El intervalo incluye cero, por lo que no se afirma un
deterioro general demostrado. Se conserva el checkpoint 96 y no se presenta el 112 como superior.
Aunque ambos modelos tienen 68 respuestas válidas, no son las mismas 68 posiciones; por
eso la comparación semántica directa usa los 64 pares comunes.

**Base → 112:** 51 pares válidos, coseno 0,6210 → 0,6357, cambio +0,0147. Catorce medias
por relato mejoraron y ocho empeoraron; dos no tuvieron pares válidos. La media por
relato/persona fue +0,0271, con intervalo [−0,0173; +0,0689]. Esto no altera el resultado
de la comparación directa con el paso 96. Los pares válidos difieren entre rondas y no
deben restarse sus deltas como si fueran las mismas posiciones.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Comprobó las referencias asociadas, la separación
por relato y persona, la integridad del checkpoint fuente y cero actualizaciones durante
la evaluación. No se modificó el código de entrenamiento ni de evaluación en este segmento.

[Datos numéricos](resultados-v13.json). Informe, gráfico y auditoría local:
`results/dream-v13/dream/report/`. Checkpoint final:
`results/dream-v13/dream/paired/checkpoints/step-000112/`. Se terminó en el paso 112.
La propuesta recibida del panel lateral para el siguiente experimento quedó registrada
en [el plan](plan.md#propuesta-recibida-del-panel-lateral-siguiente-experimento), sin ejecutar.

## Interpretación

El coseno pareado usa sólo posiciones válidas en ambos modelos e informa su cantidad.
El score que incluye cero por formato inválido se presenta aparte. Los intervalos
exploratorios remuestrean seis grupos completos y su media da igual peso por relato y
persona; no es el mismo estimador que la media de todos los pares válidos.

Los 24 relatos siguen fuera de las actualizaciones, pero ya informaron decisiones de
desarrollo; no son una nueva prueba ciega. MiniLM es el encoder usado en la recompensa,
y la cercanía no equivale a calidad onírica. Los 200 relatos de prueba siguen sin usar.

Referencias, generaciones y pesos permanecen fuera de Git. Fuente DreamBank, CC BY-NC-SA 4.0,
atribución a dreambank.net. Código MIT.
