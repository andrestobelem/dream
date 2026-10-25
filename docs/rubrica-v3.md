# Evaluación por elementos de Dream

`dream-v3` se configura en `configs/dream-v3.yaml`. La configuración anterior `configs/dream.yaml` conserva `dream-v2`; sus resultados no se reinterpretan con la nueva recompensa.

## Contrato visible

El juez extrae entre tres y seis requisitos breves de cada relato humano: participantes, escenario, acontecimientos, relaciones o emociones explícitas. Cada requisito conserva una cita literal de procedencia en el registro local. Esa cita se verifica en código, pero su presencia por sí sola no prueba que sostenga semánticamente el requisito: la revisión humana de la extracción sigue siendo necesaria.

El generador recibe los mismos requisitos que se evalúan, con sus identificadores y tipos. El evaluador de cumplimiento recibe únicamente esos requisitos y el relato final. No recibe la referencia reservada, la condición ni la escena previa. Las citas de procedencia quedan fuera de la consigna. No se puntúan requisitos ocultos ni las categorías del corpus como si fueran hechos solicitados.

## Cumplimiento por requisito

Cada requisito se evalúa en una llamada independiente al juez congelado:

| Estado | Criterio | Evidencia |
|---|---|---|
| Preservado | El relato expresa el requisito o una paráfrasis equivalente. | Fragmento literal del relato. |
| Omitido | No aparece o es ambiguo. | Cadena vacía. |
| Contradicho | El relato afirma algo explícitamente incompatible. | Fragmento literal que expresa la incompatibilidad. |

Los detalles nuevos compatibles no reducen la nota. Una emoción no mencionada se considera omitida; una emoción negada explícitamente puede estar contradicha. Las palabras sueltas no bastan para establecer acontecimientos o relaciones.

El código exige exactamente los identificadores solicitados, sin duplicados, estados válidos y citas presentes en el relato. Un error del juez interrumpe el proceso, sin convertirlo en recompensa cero para el generador.

Ante un error de esquema o cita se permite una única llamada de reparación, conservando ambas respuestas y sin indicar el veredicto esperado. Si vuelve a fallar, se interrumpe. La verificación inicial detectó citas compuestas o inexistentes que esa reparación no resolvió; el juez todavía no está habilitado para entrenar con esta recompensa.

Con `P` requisitos preservados, `C` contradichos y `N` requisitos totales:

```text
cumplimiento = máximo(0, (P − 0.5 × C) / N)
calidad = 0.50 × oniricidad + 0.30 × cumplimiento + 0.20 × fidelidad
recompensa = 0.90 × calidad + 0.10 × formato
```

Los requisitos pesan igual. La penalización adicional por contradicción es una decisión inicial del protocolo, pendiente de contraste humano. Oniricidad y fidelidad conservan sus evaluaciones independientes de `dream-v2`. La alineación global con el relato humano se sigue registrando como diagnóstico y ya no modifica la recompensa.

La condición neutral omite fidelidad y renormaliza los pesos de oniricidad y cumplimiento. Los informes separan cumplimiento de relatos evaluables y tasa de respuestas válidas. Si no hay relato extraíble, no se consulta al juez de cumplimiento y se registra cero con todos los requisitos omitidos; ese caso se distingue de una evaluación semántica realizada.

## Controles y ejecución

```sh
.venv/bin/python -m think_dream_lab anchor-check \
  --config configs/dream-v3.yaml --results results/dream-v3
```

El comando usa el supervisor y un presupuesto acumulado de diez minutos en ese directorio. No entrena pesos. Prueba identidad, paráfrasis, adiciones compatibles, omisiones, negación emocional, cambios de participante, relatos ajenos y vocabulario sin relato. Luego verifica extracción y cumplimiento contra tres referencias del conjunto de validación.

Los fallos bloquean el inicio de RL con el juez nuevo. Estos controles de desarrollo no constituyen una prueba independiente de generalización. No se deben ajustar los controles para ocultar fallos: se conservan las llamadas originales y los veredictos para inspección.

Cada llamada adicional por requisito aumenta el costo respecto de `dream-v2`. Se necesita un benchmark nuevo antes de estimar el tiempo de las doce corridas. El cambio de código modifica la identidad científica del protocolo: los checkpoints históricos quedan conservados, pero no se reanudan bajo esta implementación.

Antes de interpretar resultados extensos quedan pendientes la revisión humana ciega, el contraste con otra familia de juez, la estabilidad del formato y un piloto nuevo. La existencia de la fórmula y los controles automáticos no equivale a haber validado la medida.
