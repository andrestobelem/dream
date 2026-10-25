# Modelo base del generador

El generador actual vuelve a ser **Qwen/Qwen3-1.7B-Base**, revisión `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, el mismo checkpoint del primer piloto. Su [ficha oficial](https://huggingface.co/Qwen/Qwen3-1.7B-Base) identifica la etapa de entrenamiento como preentrenamiento. Se usa como modelo base, con texto plano y sin activar un modo de razonamiento ni aplicar su plantilla de chat.

La decisión responde al alcance actual: sólo Soñar + `<dream>`, seguido del relato en `<answer>`. Ya no se comparan los delimitadores think/dream. La auditoría de Qwen muestra `<think>` como token añadido de ID 151667 y `<dream>` como tres tokens ordinarios `[27, 56191, 29]`; esa asimetría no requiere cambiar de modelo para este recorrido único.

El piloto anterior completó cinco actualizaciones GRPO con adaptadores modificados, pesos base y juez intactos, y logits finitos al recargar. Es evidencia técnica sobre este modelo, no una prueba de mejora de sueños con el protocolo actual. El nuevo piloto v4 usará una evaluación calibrada y adaptadores propios de esa corrida.

`configs/dream-v4.yaml` selecciona Qwen y sólo D. Los nuevos resultados van a `results/dream-v4-qwen/`, separados de los diagnósticos SmolLM2 de `results/dream-v4/`. El límite del nuevo directorio es 450 segundos; junto al consumo ya registrado, el máximo de esta sesión permanece por debajo de 45 minutos.

La comprobación actual de diez consignas distintas produjo **10/10 respuestas con ambos bloques completos**, sin juez ni actualizaciones, y con pesos base intactos. Los artefactos están en `results/dream-v4-qwen/dream/diagnostics/base-check/4b5c72d6a56c/`. Es un diagnóstico de generación y formato, no de aprendizaje ni calidad onírica.

La primera ejecución de este retorno había dado 7/10 porque algunas respuestas continuaban después de `</answer>`: el cierre tenía otra tokenización al fusionarse con puntuación. Se corrigió el detector para detener cada secuencia en el primer cierre textual, conservando íntegros los tokens muestreados, incluido el token del cierre y cualquier contenido fusionado con él. No se recodifica ni se recorta la respuesta después de generarla. Se mantiene el prefill compartido entre candidatos del grupo y no se altera el EOS global del tokenizer.

## Antecedente con SmolLM2

Durante una etapa intermedia se probó **HuggingFaceTB/SmolLM2-1.7B**, revisión `effd688a12921b4cc83e3312b6feb579f70f9c71`. Su configuración se conserva en `configs/dream-v4-smollm2.yaml`. Los resultados de esa etapa se describen a continuación.

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

La prueba con SmolLM2 no reutilizó adaptadores de Qwen3. El retorno a Qwen tampoco mezcla adaptadores o resultados de ambos modelos.

El juez es un modelo independiente del generador. Las pruebas de razonamiento explícito efectuadas durante la depuración correspondían al juez. La nueva clasificación cerrada usa sus logits directamente y no genera bloques de razonamiento.
