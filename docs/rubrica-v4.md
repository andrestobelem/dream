# Juez de etiquetas cerradas

`dream-v4` incorpora el generador base SmolLM2 y un evaluador cerrado para cumplimiento y fidelidad. La integración está implementada; falta validar el evaluador y el entrenamiento permanece deshabilitado. El único recorrido activo es Soñar + `<dream>`, seguido de `<answer>`.

## Clasificación sin generación

El módulo `closed_judge.py` calcula directamente los logits del siguiente token del modelo congelado. Obtiene probabilidades de tres etiquetas de un solo token y las asocia con preservado, omitido y contradicho. No muestrea tokens, no genera JSON de puntuación ni explicaciones libres.

La auditoría comprueba las etiquetas como continuaciones del contexto exacto, sin cambios en el prefijo y con IDs distintos. En el Qwen3-8B fijado se observaron A=32, B=33 y C=34. Tres rotaciones del mapeo entre letras y clases permiten promediar una preferencia sistemática por una letra. Esto reduce un posible sesgo de salida, sin garantizar que el juicio sea correcto.

Se conserva tanto la distribución condicional entre A/B/C como su masa bajo el vocabulario completo. Una distribución aparentemente segura entre etiquetas no basta si el modelo casi no les asigna probabilidad total.

## Evidencia y abstención

El texto se divide en fragmentos contiguos identificados, con offsets verificables. Se evalúa cada requisito contra cada fragmento. La evidencia se recupera del texto original; el modelo no puede inventar su redacción.

Las reglas de agregación son fijas en Python. Una etiqueta demasiado próxima a la segunda o con masa insuficiente produce una abstención. También se abstiene ante fragmentos con juicios incompatibles o inciertos. Sin contenido, se registra la ausencia conocida; una abstención del juez es otro caso.

Si algún requisito queda sin decidir, el score global es `None`, con cobertura y score condicional reportados por separado. Ese `None` no equivale a cero ni puede pasar directamente a GRPO. El circuito conserva la abstención en los informes y en Inspect, y detiene el grupo completo antes de calcular ventajas o actualizar adaptadores. No elimina candidatos inciertos ni les imputa una recompensa.

Los umbrales actuales de margen 0,15 y masa 0,05 son provisionales. Requieren calibración con pares revisados por humanos, validación independiente y un criterio de cobertura fijado antes de entrenar. La revisión de oniricidad de seis relatos no reemplaza las anotaciones necesarias para calibrar estas tres clases.

## Qué se verificó

La primera prueba real de clasificación cerrada coincidió con cuatro de ocho situaciones sintéticas; la última con cinco de ocho. La repetición de un forward sin usar caché produjo los mismos logits de etiquetas en este equipo y mantuvo los pesos del juez sin cambios. Se generaron cero tokens para la clasificación.

La repetibilidad técnica no demuestra corrección semántica: hubo errores ante omisiones, cambios de roles, palabras sueltas e instrucciones al evaluador. El circuito no está validado para entrenar. Los resultados están en `results/dream-v4/dream/closed-check.json`; cada forward queda trazado en `closed-calls.jsonl` y las repeticiones en `closed-repeats.jsonl`.

El carácter onírico permanece separado, con revisión humana pendiente. No se presentan las evaluaciones automáticas como estados mentales, autenticidad humana o una escala clínica.

## Contraste con un clasificador especializado

El 25 de octubre se probó `cross-encoder/nli-deberta-v3-small`, revisión `fa2804872c3b4bd748f38c0185cc85775361e735`. Su [ficha oficial](https://huggingface.co/cross-encoder/nli-deberta-v3-small) describe una clasificación de pares en contradicción, implicación y neutralidad, entrenada con SNLI y MultiNLI. El relato se usó como premisa y el requisito como hipótesis; las clases se mapearon a contradicho, preservado y omitido.

El resultado fue **26 de 32 pares y cinco de ocho situaciones completas**. Corrigió la emoción ausente, pero falló en dos paráfrasis, tres juicios de la lista de palabras y un juicio de la instrucción al evaluador. También obtuvo cinco de ocho situaciones al aplicar el margen provisional de 0,15. No se adoptó para la recompensa.

Se ejecutó en CPU, float32, dos hilos y sin gradientes, con PyTorch 2.10.0 y Transformers 5.20.2 en un entorno local separado. La repetición del primer forward dio diferencia máxima de logits cero y los pesos permanecieron intactos. No generó texto. El supervisor del diagnóstico registró 38,18 segundos en el presupuesto acumulado de v4; no se reinició el presupuesto.

El código está en [experiments/nli_probe.py](../experiments/nli_probe.py). Los datos de entrada, probabilidades por par, hashes y versiones están en `results/dream-v4/dream/nli-probe/`. Este contraste usa controles sintéticos ya inspeccionados durante desarrollo; no es una validación independiente ni una calibración humana. La correspondencia entre NLI y narración de experiencias tiene límites: una lista de palabras puede activar implicación sin constituir un sueño relatado.

## Próximos pasos

1. Mejorar la corrección del clasificador y verificar los cambios con ejemplos independientes. El primer contraste especializado no resolvió los errores.
2. Preparar pares para anotación humana y calibrar umbrales, sesgos de etiquetas y cobertura; congelar el protocolo antes de la comparación.
3. Revisar la integración de cumplimiento y fidelidad con los pares anotados; las reglas de abstención ya están implementadas.
4. Repetir el piloto con SmolLM2 y verificar reanudación antes de ampliar el entrenamiento.
