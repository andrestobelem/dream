# Continuación del checkpoint 80 al 96

Se fijaron otros 16 pasos antes de iniciar: pasos 81–96. Se conserva el lote de ocho
relatos, las semillas y longitudes, el encoder, la recompensa pareada y los hiperparámetros.
Se restaura el checkpoint 80 completo, con adaptadores, optimizador y estados aleatorios.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones,
con tres muestras por relato. Base y paso 80 reutilizan sus respuestas originales.
Se evalúa el paso 96, sin seleccionar un checkpoint intermedio mirando validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v12.yaml --results results/dream-v12 \
  --source-run results/dream-v11/dream/paired/run.json --target-step 96
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v12.yaml --results results/dream-v12
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v12 --previous results/dream-v11
```

El perfil conserva el horizonte histórico del piloto para mantener la identidad científica;
`--target-step 96` establece el destino absoluto de este segmento. La ventana de supervisión
sigue siendo de dos horas y los consumos históricos se conservan. El usuario había levantado
el límite total de tiempo.

Antes de generar con el paso 96 se reutilizó el lote local v11 en v12, siguiendo la preparación
de [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), usando
`previous = results/dream-v11/dream/holdout`, destino v12 y `configs/dream-eval-v12.yaml`.
Sólo cambia el vínculo de procedencia. Las filas mantienen el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se extrajeron ni reescribieron semillas.

## Resultado observado

Se completaron los pasos 81–96: 64 rollouts nuevos, 26.215 tokens de entrenamiento y
426,29 segundos adicionales medidos en los pasos, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 80 quedaron intactos. La recarga del 96 fue exacta.

| Medida | Base | Paso 80 | Paso 96 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 69/72 | 68/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6376 · n=69 | 0,6476 · n=68 |
| Score incluyendo fallos de formato | 0,5857 | 0,7847 | 0,7780 |
| Truncamientos | 11 | 3 | 4 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 56,0% | 44,3% |

**Paso 80 → 96:** en los mismos **66 pares válidos**, coseno **0,6350 → 0,6469**,
cambio medio **+0,0119**. Doce medias por relato mejoraron y doce empeoraron.
La media con igual peso por relato/persona fue +0,0177, con intervalo exploratorio del
95% **[−0,0019; +0,0380]**. Este intervalo corresponde a la media por relato/persona,
no a la media de todos los pares válidos.

La cercanía aumentó un poco en promedio, pero su incertidumbre incluye cero y no se
considera una mejora concluyente respecto del paso 80. El resultado general es mixto:
formato, score integral y truncamientos retrocedieron ligeramente, mientras que el
error de longitud mejoró. Se conserva el checkpoint 80 y no se presenta el 96 como
superior de forma general.

**Base → 96:** 51 pares válidos, coseno 0,6202 → 0,6476, cambio +0,0275. Dieciséis
medias por relato mejoraron y siete empeoraron; uno no tuvo pares válidos. La media por
relato/persona fue +0,0259, con intervalo [−0,0176; +0,0624]. Los pares válidos difieren
de los de otras rondas y no deben restarse sus deltas como si fueran las mismas posiciones.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Comprobó las referencias asociadas, la separación
por relato y persona, la integridad del checkpoint fuente y cero actualizaciones durante
la evaluación. No se modificó el código de entrenamiento ni de evaluación en este segmento.

[Datos numéricos](resultados-v12.json). Informe, gráfico y auditoría local:
`results/dream-v12/dream/report/`. Checkpoint final:
`results/dream-v12/dream/paired/checkpoints/step-000096/`. Se terminó en el paso 96.

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
