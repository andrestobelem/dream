# Modelo base del generador

El generador seleccionado el 25 de octubre de 2026 es **HuggingFaceTB/SmolLM2-1.7B**, revisión `effd688a12921b4cc83e3312b6feb579f70f9c71`. Es el checkpoint base preentrenado; no se usa SmolLM2-Instruct ni un modelo posentrenado para producir razonamiento.

La [ficha oficial](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) distingue el modelo base de la variante Instruct, que recibió SFT y DPO. La elección conserva un tamaño cercano al piloto inicial y usa una arquitectura Llama soportada por MLX.

«Sin razonamiento» se aplica aquí al modo y al posentrenamiento: no afirmamos que el modelo carezca de capacidad de resolver problemas. Los datos de preentrenamiento incluyen matemática y programación. La generación del experimento usa texto plano, sin plantilla de chat ni prefijo de pensamiento añadido.

La auditoría del tokenizer fijado a esa revisión comprobó:

| Propiedad | Resultado |
|---|---|
| Plantilla de chat | Ausente |
| `<think>` como token especial | No |
| `<dream>` como token especial | No |
| IDs de `<think>` | `[44, 17400, 46]` |
| IDs de `<dream>` | `[44, 43363, 46]` |
| Asimetría literal según la auditoría del proyecto | No |

El diagnóstico está en `results/dream-v4/dream/tokenizer-audit.json`. Se comprobaron carga y generación con los pesos de SmolLM2 sin modificarlos y sin llamar al juez. La plantilla conversacional inicial falló el formato en las cinco condiciones; un ejemplo común de formato produjo bloques válidos en las cinco, con una única consigna por condición. Esto no prueba estabilidad ni calidad. Falta comprobar entrenamiento y recarga de adaptadores.

La escena del ejemplo es inventada y se comparte entre las cuatro condiciones factoriales: sólo cambian instrucción y delimitadores. La condición neutral recibe el ejemplo del relato final. Se registran los prompts completos; el modelo sigue generando todos los tags, sin prefijos de bloque añadidos a la salida.

Después de acotar el proyecto a Soñar + `<dream>`, se amplió el diagnóstico a diez consignas sintéticas distintas, sólo en D. Se fijó antes de ejecutarlo el criterio de diez respuestas con ambos bloques completos. El resultado fue **9/10**: en la cuarta respuesta faltó `</dream>`, aunque apareció el relato final. Por lo tanto, el diagnóstico no aprobó ese criterio. No hubo truncamiento de esa respuesta, llamadas al juez ni actualizaciones de entrenamiento; los pesos base permanecieron intactos.

Los artefactos están en `results/dream-v4/dream/diagnostics/base-check/b12c7138fb3a/`, con prompts, respuestas y semillas. El diagnóstico consume 25,66 segundos del presupuesto registrado de v4. Son controles de formato, no relatos del corpus ni una evaluación de calidad onírica. La configuración actual de `base-check` tampoco agrega una condición neutral.

Los resultados anteriores con Qwen3-1.7B-Base se conservan como antecedentes. Que ese checkpoint se llame Base no significa que haya recibido el posentrenamiento de Qwen3-Instruct; el cambio también evita el tratamiento especial de etiquetas de pensamiento de aquella familia. Sus adaptadores no se transfieren a SmolLM2.

El juez es un modelo independiente del generador. Las pruebas de razonamiento explícito efectuadas durante la depuración correspondían al juez. La nueva clasificación cerrada usa sus logits directamente y no genera bloques de razonamiento.
