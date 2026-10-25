# Juez de etiquetas cerradas

`dream-v4` incorpora el generador base SmolLM2 y un evaluador cerrado para cumplimiento y fidelidad. Su implementación e integración todavía están en curso; el entrenamiento permanece deshabilitado.

## Clasificación sin generación

El módulo `closed_judge.py` calcula directamente los logits del siguiente token del modelo congelado. Obtiene probabilidades de tres etiquetas de un solo token y las asocia con preservado, omitido y contradicho. No muestrea tokens, no genera JSON de puntuación ni explicaciones libres.

La auditoría comprueba las etiquetas como continuaciones del contexto exacto, sin cambios en el prefijo y con IDs distintos. En el Qwen3-8B fijado se observaron A=32, B=33 y C=34. Tres rotaciones del mapeo entre letras y clases permiten promediar una preferencia sistemática por una letra. Esto reduce un posible sesgo de salida, sin garantizar que el juicio sea correcto.

Se conserva tanto la distribución condicional entre A/B/C como su masa bajo el vocabulario completo. Una distribución aparentemente segura entre etiquetas no basta si el modelo casi no les asigna probabilidad total.

## Evidencia y abstención

El texto se divide en fragmentos contiguos identificados, con offsets verificables. Se evalúa cada requisito contra cada fragmento. La evidencia se recupera del texto original; el modelo no puede inventar su redacción.

Las reglas de agregación son fijas en Python. Una etiqueta demasiado próxima a la segunda o con masa insuficiente produce una abstención. También se abstiene ante fragmentos con juicios incompatibles o inciertos. Sin contenido, se registra la ausencia conocida; una abstención del juez es otro caso.

Si algún requisito queda sin decidir, el score global es `None`, con cobertura y score condicional reportados por separado. Ese `None` no equivale a cero ni puede pasar directamente a GRPO. Falta integrar el manejo de abstenciones en generación, evaluación, entrenamiento e Inspect antes de habilitar el nuevo circuito.

Los umbrales actuales de margen 0,15 y masa 0,05 son provisionales. Requieren calibración con pares revisados por humanos, validación independiente y un criterio de cobertura fijado antes de entrenar. La revisión de oniricidad de seis relatos no reemplaza las anotaciones necesarias para calibrar estas tres clases.

## Qué se verificó

La primera prueba real de clasificación cerrada coincidió con cuatro de ocho situaciones sintéticas; la última con cinco de ocho. La repetición de un forward sin usar caché produjo los mismos logits de etiquetas en este equipo y mantuvo los pesos del juez sin cambios. Se generaron cero tokens para la clasificación.

La repetibilidad técnica no demuestra corrección semántica: hubo errores ante omisiones, cambios de roles, palabras sueltas e instrucciones al evaluador. El circuito no está validado para entrenar. Los resultados están en `results/dream-v4/dream/closed-check.json`; cada forward queda trazado en `closed-calls.jsonl` y las repeticiones en `closed-repeats.jsonl`.

El carácter onírico permanece separado, con revisión humana pendiente. No se presentan las evaluaciones automáticas como estados mentales, autenticidad humana o una escala clínica.

## Próximos pasos

1. Mejorar la corrección del clasificador y contrastar un modelo especializado en inferencia textual, sin ajustar los resultados para aprobar los controles iniciales.
2. Preparar pares para anotación humana y calibrar umbrales, sesgos de etiquetas y cobertura; congelar el protocolo antes de la comparación.
3. Integrar cumplimiento y fidelidad con reglas de abstención que nunca castiguen al generador por incertidumbre del juez.
4. Repetir el piloto con SmolLM2 y verificar reanudación antes de ampliar el entrenamiento.
