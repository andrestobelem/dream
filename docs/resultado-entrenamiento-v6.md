# Resultado del entrenamiento con semillas cortas

25 de octubre de 2026. En esta prueba pequeña aumentó la cercanía al relato humano esperado y mejoró el formato. No demuestra eficacia generalizada.

Se completaron los 16 pasos aprobados sobre ocho relatos de entrenamiento, con cuatro candidatos por paso. Los cuatro relatos de validación pertenecen a otras personas y tuvieron tres generaciones por fase, con los mismos prompts y semillas de muestreo.

| Medida | Antes | Después |
| --- | ---: | ---: |
| Coseno medio al relato humano, respuestas válidas | 0.5327 (n=10) | 0.5986 (n=12) |
| Formato válido | 10/12 | 12/12 |
| Recompensa media, incluyendo cero por formato inválido | 0.6386 | 0.7993 |
| Error relativo medio de longitud, respuestas válidas | 60.6% | 49.3% |
| Truncamientos | 1 | 0 |

En los 10 pares con formato válido en ambas fases, el coseno aumentó 0.0388 en promedio. Esto separa parte de la mejora semántica del aumento de cobertura de formato. Son las mismas consignas y posiciones de muestra; no se presume una trayectoria aleatoria idéntica al cambiar la política.

## Por relato reservado

| Caso | Válidas antes/después | Coseno antes/después |
| --- | --- | --- |
| Green City | 3/3 → 3/3 | 0.3894 → 0.5218 |
| Cumpleaños en Tailandia | 3/3 → 3/3 | 0.5814 → 0.5532 |
| Escuela y tubos | 3/3 → 3/3 | 0.5851 → 0.5939 |
| Castillo e inundación | 1/3 → 3/3 | 0.6591 → 0.7253 |

Tres medias por relato aumentaron y una bajó. En el caso del castillo también cambió la cantidad de respuestas válidas, de una a tres.

El ejemplo de Green City, candidato 1 elegido por posición, pasó de una interpretación simbólica larga a un relato más directo en primera persona. La versión posterior todavía omite elementos como Laura y la herencia; el aumento de similitud no implica reconstrucción completa de la referencia.

## Verificación y recursos

Los 16 pasos tuvieron gradientes no nulos y los 16 grupos tuvieron recompensas variables. Los adaptadores cambiaron, los pesos base quedaron intactos y la recarga reprodujo exactamente el checkpoint final.

Tiempo supervisado total: **794.45 segundos** (13 min 14.5 s), por debajo de los 900 autorizados. Incluye tres procesos fallidos por Metal, el entrenamiento y ambas evaluaciones. Los consumos anteriores se conservaron aparte.

Los errores de Metal ocurrieron antes del primer checkpoint y después del paso 7. Reducir los bloques enviados al GPU no resolvió por sí solo el fallo reproducible. La reanudación final usó límites de 10 operaciones/20 MB por bloque y `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`, sólo en el proceso. La recomendación del controlador procede de la [incidencia MLX #3267](https://github.com/ml-explore/mlx/issues/3267); las variables de bloques están en la [documentación de MLX](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/environment_variables.rst). No se cambiaron el lote, la recompensa ni los hiperparámetros para sortear los fallos.

## Artefactos locales

- Comparación completa con generaciones antes/después: `results/dream-v6/dream/report/comparacion.md`.
- Datos numéricos y auditoría: `comparison.json` y `final-audit.json` en ese directorio.
- Checkpoint: `results/dream-v6/dream/paired/checkpoints/step-000016/`.
- Código y configuración inmutables: `results/dream-v6/source-snapshots/` y `config-snapshots/`.

El lote, las referencias originales y los pesos permanecen fuera de Git. Este documento publica métricas agregadas y límites. Fuente DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT.

## Límites

Cuatro relatos reservados y tres generaciones por caso. Comparación descriptiva; no demuestra eficacia generalizada.

Los relatos de validación se usaron para esta comprobación de desarrollo, no para actualizar pesos. Los 200 relatos de prueba permanecen sin usar. La longitud sólo se indicó como orientación y todavía presenta desvíos importantes. La similitud semántica no valida oniricidad ni experiencia subjetiva.
