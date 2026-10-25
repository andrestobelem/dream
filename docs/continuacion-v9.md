# Continuación del checkpoint 32 al 48

Se fijaron otros 16 pasos antes de ejecutar: pasos 33–48. Se conservan las mismas ocho
referencias de entrenamiento, semillas, longitudes, recompensa pareada e hiperparámetros.
Se restaura el checkpoint 32 completo, con adaptadores, optimizador y aleatoriedad.
La base y los checkpoints anteriores permanecen intactos.

Los cuatro relatos del piloto, los 24 de validación ampliada y los 200 de prueba quedan
fuera de las actualizaciones. La evaluación posterior usa las mismas 72 posiciones
(tres por relato) y compara base, paso 32 y paso 48. Las respuestas anteriores se reutilizan;
no se elige el mejor checkpoint mirando la validación.

```sh
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab continue-training \
  --config configs/dream-v9.yaml --results results/dream-v9 \
  --source-run results/dream-v8/dream/paired/run.json --target-step 48
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=10 MLX_MAX_MB_PER_BUFFER=20 \
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-v9.yaml --results results/dream-v9
.venv/bin/python -m think_dream_lab.continuation_report \
  --results results/dream-v9 --previous results/dream-v8
```

El perfil de entrenamiento mantiene el horizonte histórico del piloto para conservar
su identidad científica; `--target-step 48` fija el destino absoluto de este segmento.
La ventana de supervisión sigue siendo de dos horas y los consumos anteriores se conservan;
el usuario había levantado el límite total de tiempo.

Antes de generar con el paso 48 se copió el lote local v8 a v9, siguiendo la preparación
documentada en [continuación v8](continuacion-v8.md#preparación-del-lote-reutilizado), con
`previous = results/dream-v8/dream/holdout`, destino v9 y `configs/dream-eval-v9.yaml`.
Sólo cambia el vínculo de procedencia en los metadatos. Las filas conservan el hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.
No se volvió a ejecutar el extractor ni se reescribieron semillas.

## Resultado observado

Se completaron los pasos 33–48: 64 rollouts nuevos, 17.947 tokens de entrenamiento y
297,23 segundos adicionales de cómputo de entrenamiento, además del arranque y la evaluación.
Los 16 grupos tuvieron recompensas variables y gradientes no nulos. Los adaptadores
cambiaron; la base y el checkpoint 32 quedaron intactos y la recarga del 48 fue exacta.

| Medida | Base | Paso 32 | Paso 48 |
| --- | ---: | ---: | ---: |
| Formato válido | 52/72 | 70/72 | 69/72 |
| Coseno, sólo respuestas válidas | 0,6219 · n=52 | 0,6453 · n=70 | 0,6537 · n=69 |
| Score incluyendo fallos de formato | 0,5857 | 0,7998 | 0,7924 |
| Truncamientos | 11 | 0 | 1 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 36,4% | 39,9% |

**Paso 32 → 48:** en los mismos **67 pares válidos**, coseno **0,6446 → 0,6523**,
cambio medio **+0,0078**. Doce medias por relato mejoraron y doce empeoraron. La media con
igual peso por relato/persona fue +0,0078, con intervalo exploratorio 95%
**[-0,0132; +0,0317]**. Aunque ambas medias redondean al mismo valor, usan ponderaciones
distintas y ese intervalo corresponde a la media por relato/persona.

El resultado es mixto: la cercanía aumentó apenas en promedio, mientras que formato,
score integral, truncamientos y error de longitud empeoraron ligeramente. La incertidumbre
incluye deterioro; no se considera una mejora concluyente respecto del paso 32.

**Base → 48:** 51 pares válidos, coseno 0,6225 → 0,6443, cambio +0,0218. Trece medias
por relato mejoraron y nueve empeoraron; dos no tuvieron pares válidos. La media por
relato/persona fue +0,0085, con intervalo [-0,0475; +0,0622]. Estos pares no son exactamente
los mismos de la comparación base → 32; no deben restarse sus deltas como si lo fueran.

La auditoría recalculó los 216 scores de evaluación y los 64 de entrenamiento desde los
vectores guardados, sin nuevas inferencias. Confirmó las referencias asociadas, la separación
por relato y persona y cero actualizaciones durante la evaluación. Las 142 pruebas pasaron.
El generador de informes se generalizó a los pasos indicados por la procedencia y reprodujo
exactamente los resultados v8 en una comprobación de regresión, sin sobrescribirlos.

[Datos numéricos](resultados-v9.json). Informe, gráfico y auditoría local:
`results/dream-v9/dream/report/`. Checkpoint final:
`results/dream-v9/dream/paired/checkpoints/step-000048/`. Se terminó en el paso 48.

## Interpretación

Los cosenos se comparan en las posiciones válidas de ambos modelos, indicando su cantidad;
el score que incluye cero por formato inválido se informa aparte. Los intervalos
exploratorios remuestrean seis personas completas. Su media da igual peso por relato y
persona, por lo que no es el mismo estimador que la media de todos los pares válidos.

Los 24 relatos siguen fuera del entrenamiento, pero ya informaron decisiones de desarrollo.
No constituyen una nueva prueba ciega. MiniLM es el mismo encoder usado en la recompensa;
la cercanía no equivale a calidad onírica. Los 200 relatos de prueba permanecen sin usar.

Referencias, generaciones y pesos quedan fuera de Git. Fuente DreamBank, CC BY-NC-SA 4.0,
atribución a dreambank.net. Código MIT.
