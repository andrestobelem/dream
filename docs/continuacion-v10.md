# Continuación del checkpoint 48 al 64

Se fijaron otros 16 pasos antes de iniciar: pasos 49–64. Se mantiene el lote de ocho
relatos, sus semillas y longitudes, el encoder, la recompensa pareada y los hiperparámetros.
Se restaura el checkpoint 48 completo, con adaptadores, optimizador y estados aleatorios.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones,
con tres muestras por relato. Base y paso 48 reutilizan sus respuestas originales.
Se evalúa el paso 64, sin seleccionar un checkpoint intermedio mirando validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v10.yaml --results results/dream-v10 \
  --source-run results/dream-v9/dream/paired/run.json --target-step 64
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v10.yaml --results results/dream-v10
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v10 --previous results/dream-v9
```

El perfil conserva el horizonte histórico del piloto para mantener la identidad científica;
`--target-step 64` establece el destino absoluto de este segmento. La ventana de supervisión
sigue siendo de dos horas y los consumos históricos se conservan. El usuario había levantado
el límite total de tiempo.

Antes de generar con el paso 64 se reutilizó el lote local v9 en v10, siguiendo la preparación
de [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), usando
`previous = results/dream-v9/dream/holdout`, destino v10 y `configs/dream-eval-v10.yaml`.
Sólo cambia el vínculo de procedencia. Las filas mantienen el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se extrajeron ni reescribieron semillas.

## Resultado observado

Se completaron los pasos 49–64: 64 rollouts nuevos, 16.541 tokens de entrenamiento y
283,93 segundos adicionales medidos en los pasos, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 48 quedaron intactos. La recarga del 64 fue exacta.

| Medida | Base | Paso 48 | Paso 64 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 69/72 | 67/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6537 · n=69 | 0,6460 · n=67 |
| Score incluyendo fallos de formato | 0,5857 | 0,7924 | 0,7659 |
| Truncamientos | 11 | 1 | 4 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 39,9% | 50,2% |

**Paso 48 → 64:** en los mismos **66 pares válidos**, coseno **0,6584 → 0,6469**,
cambio medio **−0,0115**. Catorce medias por relato mejoraron y diez empeoraron; las
pérdidas de estos últimos fueron mayores que las ganancias de los otros. La media con
igual peso por relato/persona fue −0,0108, con intervalo exploratorio del 95%
**[−0,0251; +0,0030]**. Este intervalo corresponde a la media por relato/persona y no a
la media de todos los pares válidos.

Esta ronda no mostró mejora respecto del paso 48: bajaron la cercanía media observada,
el formato y el score integral, y aumentaron los truncamientos y el error de longitud.
El intervalo incluye cero; no se afirma un deterioro general demostrado. Se conserva el
checkpoint 48 y no se presenta el 64 como superior.

**Base → 64:** 50 pares válidos, coseno 0,6206 → 0,6506, cambio +0,0300. Catorce medias
por relato mejoraron y ocho empeoraron; dos no tuvieron pares válidos. La media por
relato/persona fue +0,0406, con intervalo exploratorio [+0,0066; +0,0804]. Esto no revierte
el resultado de la comparación directa con el paso 48. Los pares válidos difieren de los
de otras rondas y no deben restarse sus deltas como si fueran las mismas posiciones.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Comprobó las referencias asociadas, la separación
por relato y persona, la integridad del checkpoint fuente y cero actualizaciones durante
la evaluación. No se modificó el código de entrenamiento ni de evaluación en este segmento.

[Datos numéricos](resultados-v10.json). Informe, gráfico y auditoría local:
`results/dream-v10/dream/report/`. Checkpoint final:
`results/dream-v10/dream/paired/checkpoints/step-000064/`. Se terminó en el paso 64.

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
