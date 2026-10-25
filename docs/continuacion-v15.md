# Continuación del checkpoint 128 al 144

Se fijaron otros 16 pasos antes de iniciar: pasos 129–144. Se conserva el lote de ocho
relatos, las semillas y longitudes, el encoder, la recompensa pareada y los hiperparámetros.
Se restaura el checkpoint 128 completo, con adaptadores, optimizador y estados aleatorios.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones,
con tres muestras por relato. Base y paso 128 reutilizan sus respuestas originales.
Se evalúa el paso 144, sin seleccionar un checkpoint intermedio mirando validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v15.yaml --results results/dream-v15 \
  --source-run results/dream-v14/dream/paired/run.json --target-step 144
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v15.yaml --results results/dream-v15
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v15 --previous results/dream-v14
```

El perfil conserva el horizonte histórico del piloto para mantener la identidad científica;
`--target-step 144` establece el destino absoluto de este segmento. La ventana de supervisión
sigue siendo de dos horas y los consumos históricos se conservan. El usuario había levantado
el límite total de tiempo.

Antes de generar con el paso 144 se reutilizó el lote local v14 en v15, siguiendo la preparación
de [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), usando
`previous = results/dream-v14/dream/holdout`, destino v15 y `configs/dream-eval-v15.yaml`.
Sólo cambia el vínculo de procedencia. Las filas mantienen el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se extrajeron ni reescribieron semillas.

## Resultado observado

Se completaron los pasos 129–144: 64 rollouts nuevos, 27.344 tokens de entrenamiento y
445,02 segundos adicionales medidos en los pasos, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 128 quedaron intactos. La recarga del 144 fue exacta.

| Medida | Base | Paso 128 | Paso 144 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 67/72 | 68/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6425 · n=67 | 0,6406 · n=68 |
| Score incluyendo fallos de formato | 0,5857 | 0,7642 | 0,7747 |
| Truncamientos | 11 | 4 | 4 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 38,7% | 41,9% |

**Paso 128 → 144:** en los mismos **64 pares válidos**, coseno **0,6436 → 0,6398**,
cambio medio **−0,0038**. Doce medias por relato mejoraron y doce empeoraron.
La media con igual peso por relato/persona fue −0,0024, con intervalo exploratorio del
95% **[−0,0207; +0,0157]**. Este intervalo corresponde a la media por relato/persona,
no a la media de todos los pares válidos.

La cercanía quedó prácticamente estable, con una leve baja observada y sin mejora
concluyente respecto del paso 128. El formato y el score integral mejoraron ligeramente,
los truncamientos quedaron iguales y el error de longitud aumentó. El aumento del score
integral no representa una ganancia de cercanía: cambió la cobertura de formato.
Se conserva el checkpoint 128 y no se presenta el 144 como superior en cercanía.

**Base → 144:** 51 pares válidos, coseno 0,6225 → 0,6367, cambio +0,0142. Quince medias
por relato mejoraron y siete empeoraron; dos no tuvieron pares válidos. La media por
relato/persona fue +0,0205, con intervalo [−0,0264; +0,0648]. Los pares válidos difieren
entre rondas y no deben restarse sus deltas como si fueran las mismas posiciones.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Comprobó las referencias asociadas, la separación
por relato y persona, la integridad del checkpoint fuente y cero actualizaciones durante
la evaluación. No se modificó el código de entrenamiento ni de evaluación en este segmento.

[Datos numéricos](resultados-v15.json). Informe, gráfico y auditoría local:
`results/dream-v15/dream/report/`. Checkpoint final:
`results/dream-v15/dream/paired/checkpoints/step-000144/`. Se terminó en el paso 144.
El lote sigue siendo de ocho relatos; la propuesta de ampliar a 128 relatos permanece
registrada en [el plan](plan.md), sin ejecutar.

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
