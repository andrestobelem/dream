# Juez de etiquetas cerradas

`dream-v4` usa actualmente Qwen3-1.7B-Base, el generador del primer piloto, y un evaluador cerrado para cumplimiento y fidelidad. Los diagnósticos intermedios con SmolLM2 se conservan separados. La integración está implementada; falta calibrar el evaluador y el entrenamiento permanece deshabilitado. El único recorrido activo es Soñar + `<dream>`, seguido de `<answer>`.

## Clasificación sin generación

El módulo `closed_judge.py` calcula directamente los logits del siguiente token del modelo congelado. Obtiene probabilidades de tres etiquetas de un solo token y las asocia con preservado, omitido y contradicho. No muestrea tokens, no genera JSON de puntuación ni explicaciones libres.

La auditoría comprueba las etiquetas como continuaciones del contexto exacto, sin cambios en el prefijo y con IDs distintos. En el Qwen3-8B fijado se observaron A=32, B=33 y C=34. Tres rotaciones del mapeo entre letras y clases permiten promediar una preferencia sistemática por una letra. Esto reduce un posible sesgo de salida, sin garantizar que el juicio sea correcto.

La variante actual del prompt, `closed-fewshot-v1`, incluye seis ejemplos sintéticos breves: dos por clase. Sus respuestas rotan junto con el mapeo de letras. No hay ejemplos tomados del corpus ni de los controles nuevos dentro del prompt. Tanto los mensajes completos como los IDs de contexto y el hash de implementación se registran en cada forward.

Se conserva tanto la distribución condicional entre A/B/C como su masa bajo el vocabulario completo. Una distribución aparentemente segura entre etiquetas no basta si el modelo casi no les asigna probabilidad total.

## Evidencia y abstención

El texto se divide en fragmentos contiguos identificados, con offsets verificables. Se evalúa cada requisito contra cada fragmento. La evidencia se recupera del texto original; el modelo no puede inventar su redacción.

Las reglas de agregación son fijas en Python. Una etiqueta demasiado próxima a la segunda o con masa insuficiente produce una abstención. También se abstiene ante fragmentos con juicios incompatibles o inciertos. Sin contenido, se registra la ausencia conocida; una abstención del juez es otro caso.

Si algún requisito queda sin decidir, el score global es `None`, con cobertura y score condicional reportados por separado. Ese `None` no equivale a cero ni puede pasar directamente a GRPO. El circuito conserva la abstención en los informes y en Inspect, y detiene el grupo completo antes de calcular ventajas o actualizar adaptadores. No elimina candidatos inciertos ni les imputa una recompensa.

Los umbrales actuales de margen 0,15 y masa 0,05 son provisionales. Para el próximo piloto exploratorio se usarán, por autorización del usuario, referencias anotadas por el asistente para la calibración de desarrollo, con procedencia explícita y criterios de cobertura fijados antes de entrenar. La validación humana independiente sigue siendo una etapa diferente. La revisión de oniricidad de seis relatos no reemplaza las anotaciones necesarias para calibrar estas tres clases.

## Qué se verificó

Las pruebas sin demostraciones coincidieron primero con cuatro y después con cinco de ocho situaciones sintéticas. Con seis ejemplos breves en el prompt, Qwen3 pasó **7/8 controles conocidos y 3/4 controles nuevos**: diez de doce situaciones completas. Los nuevos controles se fijaron en el snapshot antes de ejecutar la prueba. Son controles de transferencia sintéticos; no constituyen una evaluación humana independiente.

La variante con ejemplos corrigió la omisión de la emoción y las listas de palabras en los controles conocidos. Sigue equivocándose ante dos requisitos de la orden al evaluador y confunde con omisión una contradicción de participantes en el control nuevo. El cambio de participantes obtuvo 0,933 de probabilidad para la clase incorrecta: un margen alto tampoco garantiza un juicio correcto.

La repetición de un forward sin usar caché produjo los mismos logits de etiquetas en este equipo y mantuvo los pesos del juez sin cambios. Se generaron cero tokens para la clasificación. La prueba supervisada consumió 121,92 segundos y el estado de entrenamiento continúa deshabilitado.

La repetibilidad técnica no demuestra corrección semántica: hubo errores ante omisiones, cambios de roles, palabras sueltas e instrucciones al evaluador. El circuito no está validado para entrenar. Los resultados están en `results/dream-v4/dream/closed-check.json`; cada forward queda trazado en `closed-calls.jsonl` y las repeticiones en `closed-repeats.jsonl`.

El carácter onírico permanece separado, con revisión humana pendiente. No se presentan las evaluaciones automáticas como estados mentales, autenticidad humana o una escala clínica.

## Pares preparados para revisión humana

El paquete `results/dream-v4/dream/review-pairs/7c803087be40/revision.csv` contiene 29 pares de requisito y relato, agrupados en 14 fuentes. Incluye los ocho pares de los controles nuevos, primera y última oración de nueve escenas válidas producidas por SmolLM2 y tres requisitos de la evaluación real de cumplimiento. Se preparó con las respuestas vacías; la revisión posterior del asistente se conserva separada, sin presentar etiquetas sintéticas como anotaciones de personas.

Se completa `etiqueta_humana` con `preservado`, `omitido`, `contradicho` o `no_se`, y se pueden agregar notas. No muestra predicciones del juez. La clave de procedencia se mantiene aparte; no debe abrirse antes de revisar. El script [prepare_pair_review.py](../experiments/prepare_pair_review.py) verifica el texto inmutable y conserva las anotaciones al repetir la exportación.

Este paquete permite revisar discrepancias y preparar una calibración posterior. Todavía no es un conjunto de calibración validado. Debe separarse por fuente antes de ajustar y evaluar umbrales; una revisión pequeña tampoco prueba precisión general. La revisión de oniricidad de seis relatos sigue siendo un archivo distinto.

### Revisión del asistente completada

El usuario autorizó que el asistente hiciera la revisión con su criterio. Se conservaron las dos etiquetas ya cargadas y se completaron 27 adicionales. La copia `revision-asistente.csv` tiene una columna `origen_revision` y el resultado conjunto es: 11 preservados, 15 omitidos, dos contradichos y uno ambiguo (`no_se`). Los originales no se alteraron ni se llenaron columnas humanas con notas automáticas.

Cada nota juzga la proposición completa: si una oración reúne varios hechos y sólo conserva algunos, se marca omitida con una explicación de la conservación parcial. Se detectó una formulación ambigua que atribuye devolver un bolso a una estación y a un desconocido; quedó `no_se`. Esto muestra un límite de usar oraciones completas como requisitos de fidelidad y debe revisarse antes de interpretar ese score.

Se restauró únicamente en la copia una comilla adicional al final de un relato, contrastándolo con el control original congelado. El manifiesto `revision-asistente.json` registra esa corrección, las procedencias y hashes de entrada y salida. No se cambió la etiqueta preexistente de esa fila.

Los seis relatos de oniricidad se puntuaron aparte, antes de abrir su clave de origen, en `results/dream/report/revision-asistente.csv`. Las notas fueron 4, 2, 3, 2, 3 y 4 en el orden del paquete. Sólo cuatro tenían score histórico del juez: tres coincidieron exactamente y uno difirió en un punto, con error absoluto medio 0,25 sobre una escala de 0–4. Es un diagnóstico de cuatro salidas anteriores, no una validación de `dream-v4`.

## Contraste con un clasificador especializado

El 25 de octubre se probó `cross-encoder/nli-deberta-v3-small`, revisión `fa2804872c3b4bd748f38c0185cc85775361e735`. Su [ficha oficial](https://huggingface.co/cross-encoder/nli-deberta-v3-small) describe una clasificación de pares en contradicción, implicación y neutralidad, entrenada con SNLI y MultiNLI. El relato se usó como premisa y el requisito como hipótesis; las clases se mapearon a contradicho, preservado y omitido.

El resultado fue **26 de 32 pares y cinco de ocho situaciones completas**. Corrigió la emoción ausente, pero falló en dos paráfrasis, tres juicios de la lista de palabras y un juicio de la instrucción al evaluador. También obtuvo cinco de ocho situaciones al aplicar el margen provisional de 0,15. No se adoptó para la recompensa.

Se ejecutó en CPU, float32, dos hilos y sin gradientes, con PyTorch 2.10.0 y Transformers 5.20.2 en un entorno local separado. La repetición del primer forward dio diferencia máxima de logits cero y los pesos permanecieron intactos. No generó texto. El supervisor del diagnóstico registró 38,18 segundos en el presupuesto acumulado de v4; no se reinició el presupuesto.

El código está en [experiments/nli_probe.py](../experiments/nli_probe.py). Los datos de entrada, probabilidades por par, hashes y versiones están en `results/dream-v4/dream/nli-probe/`. Este contraste usa controles sintéticos ya inspeccionados durante desarrollo; no es una validación independiente ni una calibración humana. La correspondencia entre NLI y narración de experiencias tiene límites: una lista de palabras puede activar implicación sin constituir un sueño relatado.

## Contraste de etiquetas cerradas con Phi

Se repitieron las doce situaciones, con el mismo prompt y ejemplos, usando `mlx-community/Phi-4-mini-instruct-4bit`, revisión `ac1c269cb4222a4e136a3d09edad301056c1f36a`, ya disponible localmente. Este contraste no generó texto ni utilizó las respuestas JSON del ensayo anterior.

El resultado fue **cero de doce situaciones completas**, con 14 de 40 pares correctamente decididos, 17 abstenciones y nueve decisiones incorrectas. La repetición de logits tuvo diferencia cero y los pesos permanecieron intactos. Se comprobaron las tres etiquetas como continuaciones de un token. Esto tampoco lo habilita como juez de entrenamiento.

Los resultados se guardaron separados en `results/dream-v4-phi/dream/closed-check.json`. Los 43,17 segundos de ejecución se cargaron al presupuesto Phi existente en `results/dream-v3-phi/budget.json`; no se creó un presupuesto nuevo ni se descargó otro modelo. Durante ese contraste el generador seleccionado era SmolLM2; posteriormente se volvió a Qwen3-1.7B-Base. El juez configurado por defecto continúa siendo Qwen3-8B, pendiente de calibración.

## Próximos pasos

1. Mejorar la corrección del clasificador y verificar los cambios con ejemplos independientes. El primer contraste especializado no resolvió los errores.
2. Usar las etiquetas asistidas para calibrar umbrales y medir cobertura, separando por fuente y conservando `no_se` como incertidumbre; congelar el protocolo antes de entrenar.
3. Revisar la integración de cumplimiento y fidelidad con los pares anotados; las reglas de abstención ya están implementadas.
4. Repetir el piloto con SmolLM2 y verificar reanudación antes de ampliar el entrenamiento.
