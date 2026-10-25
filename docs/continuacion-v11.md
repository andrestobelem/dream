# Continuación del checkpoint 64 al 80

Se fijaron otros 16 pasos antes de iniciar: pasos 65–80. Se conserva el lote de ocho
relatos, las semillas y longitudes, el encoder, la recompensa pareada y los hiperparámetros.
Se restaura el checkpoint 64 completo, con adaptadores, optimizador y estados aleatorios.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones,
con tres muestras por relato. Base y paso 64 reutilizan sus respuestas originales.
Se evalúa el paso 80, sin seleccionar un checkpoint intermedio mirando validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v11.yaml --results results/dream-v11 \
  --source-run results/dream-v10/dream/paired/run.json --target-step 80
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v11.yaml --results results/dream-v11
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v11 --previous results/dream-v10
```

El perfil conserva el horizonte histórico del piloto para mantener la identidad científica;
`--target-step 80` establece el destino absoluto de este segmento. La ventana de supervisión
sigue siendo de dos horas y los consumos históricos se conservan. El usuario había levantado
el límite total de tiempo.

Antes de generar con el paso 80 se reutilizó el lote local v10 en v11, siguiendo la preparación
de [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), usando
`previous = results/dream-v10/dream/holdout`, destino v11 y `configs/dream-eval-v11.yaml`.
Sólo cambia el vínculo de procedencia. Las filas mantienen el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se extrajeron ni reescribieron semillas.

## Resultado observado

Se completaron los pasos 65–80: 64 rollouts nuevos, 22.965 tokens de entrenamiento y
401,93 segundos adicionales medidos en los pasos, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 64 quedaron intactos. La recarga del 80 fue exacta.

| Medida | Base | Paso 64 | Paso 80 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 67/72 | 69/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6460 · n=67 | 0,6376 · n=69 |
| Score incluyendo fallos de formato | 0,5857 | 0,7659 | 0,7847 |
| Truncamientos | 11 | 4 | 3 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 50,2% | 56,0% |

**Paso 64 → 80:** en los mismos **64 pares válidos**, coseno **0,6483 → 0,6469**,
cambio medio **−0,0013**. Once medias por relato mejoraron y doce empeoraron; un relato
no tuvo posiciones válidas en ambos modelos. La media con igual peso por relato/persona
fue −0,0013, con intervalo exploratorio del 95% **[−0,0178; +0,0145]**. Aunque ambas medias
redondean al mismo valor, sus ponderaciones difieren y ese intervalo corresponde a la
media por relato/persona. Las diferencias se calculan antes de redondear.

La cercanía quedó prácticamente igual, sin evidencia concluyente de mejora respecto del
paso 64. El formato y el score integral mejoraron, y hubo un truncamiento menos, mientras
que el error de longitud aumentó. El resultado es mixto; el aumento del score integral
no representa una mejora de cercanía equivalente. Se conserva el checkpoint 64 y no se
presenta el 80 como superior en cercanía.

**Base → 80:** 50 pares válidos, coseno 0,6213 → 0,6415, cambio +0,0202. Trece medias
por relato mejoraron y diez empeoraron; uno no tuvo pares válidos. La media por relato/persona
fue +0,0149, con intervalo [−0,0363; +0,0633]. Esto no altera el resultado de la comparación
directa con el paso 64. Los pares válidos difieren de los de otras rondas y no deben restarse
sus deltas como si fueran las mismas posiciones.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Comprobó las referencias asociadas, la separación
por relato y persona, la integridad del checkpoint fuente y cero actualizaciones durante
la evaluación. No se modificó el código de entrenamiento ni de evaluación en este segmento.

[Datos numéricos](resultados-v11.json). Informe, gráfico y auditoría local:
`results/dream-v11/dream/report/`. Checkpoint final:
`results/dream-v11/dream/paired/checkpoints/step-000080/`. Se terminó en el paso 80.

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
