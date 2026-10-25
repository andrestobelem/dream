# Recompensa exploratoria por embeddings

El generador sólo recibe semillas. Produce `<dream>…</dream><answer>…</answer>`.
Se puntúa el relato de `<answer>`; el contenido de `<dream>` no se compara con él ni con
una referencia. GRPO asigna la recompensa a toda la secuencia muestreada, incluidos los
tokens del bloque de sueño. No se recodifican ni recortan los tokens para entrenar.

## Fórmula congelada para el primer piloto

Para el vector normalizado del relato `a` y los vectores normalizados de los 1.000 relatos
de entrenamiento `d_i`:

```
coseno = máximo_i(a · d_i)
recompensa = (1 + coseno) / 2
```

Es una cercanía al vecino más próximo del **banco de entrenamiento**, no una exigencia de
reconstruir el relato que originó la semilla. Elegimos el máximo para permitir distintas
regiones del corpus y dar 1 a una coincidencia exacta. El promedio de los cinco vecinos
se conserva como diagnóstico; no interviene en la recompensa.

Una similitud alta es favorable, también cuando hay coincidencias textuales. No hay
penalización por copia, descuento por similitud alta ni juez de fidelidad del proceso.
La posible memorización previa de un corpus público limita la interpretación del resultado.

Si no se pueden extraer los dos bloques no vacíos, en el orden y formato prescritos,
la recompensa es 0 por formato. No se ejecuta el encoder para ese candidato ni se asigna
un juicio semántico a su sueño. Esto introduce una recompensa inicialmente dispersa:
el formato debe aprenderse o aparecer en las muestras para obtener diferencias útiles.

## Encoder y textos largos

Usamos [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
revisión `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, congelado, en CPU, con cuatro hilos.
La tarjeta oficial lo describe como encoder de similitud semántica de frases y párrafos,
con 384 dimensiones. No es un detector de sueños.

El procedimiento del proyecto:

1. Tokenizar con todos los fragmentos desbordados, sin superposición, hasta 256 tokens por
   fragmento incluyendo los tokens especiales. No descartar el final del relato.
2. Promediar los vectores contextuales de tokens usando la máscara de atención, incluidos
   los tokens especiales, y normalizar L2 cada fragmento.
3. Promediar los fragmentos ponderando por su cantidad de tokens de contenido; normalizar
   L2 el resultado. Un texto corto usa el mismo pooling de la tarjeta oficial.

Los vectores se guardan en caché con hashes del texto, configuración y contenido. El banco
se identifica por encoder, IDs de referencia y vectores. Un diagnóstico de otro banco
no habilita el piloto actual. Las versiones exactas de dependencias están en `uv.lock`.

## Comprobaciones ejecutadas

`embedding-check` construyó el banco completo y contrastó doce casos por grupo:

| Entrada | Coseno máximo medio | Recompensa media | Media de los cinco cosenos, diagnóstico |
| --- | ---: | ---: | ---: |
| Referencias exactas de entrenamiento | 1,000 | 1,000 | 0,614 |
| Relatos humanos de validación | 0,553 | 0,777 | 0,527 |
| Listas de palabras de referencias de entrenamiento | 0,585 | 0,793 | 0,441 |

Los grupos tienen distintos referentes: la tabla no prueba una preferencia por listas
frente a relatos humanos. Sí muestra que una lista temática puede alcanzar un score alto.
No se añadieron penalizaciones ni se cambió la fórmula para forzar una separación.

Se verificó que las referencias exactas reciban el máximo, que el banco admita sólo
entrenamiento y que sus IDs no intersecten validación ni prueba. Las pruebas de código
comprueban además que cambiar el contenido de `<dream>` conservando `<answer>` deja
intacta la recompensa, que se usan todos los fragmentos y que la caché conserva los vectores.
Estas comprobaciones no son una validación científica de oniricidad.

## Riqueza de las semillas: decisión abierta

El límite inicial de tres a ocho frases y 24 palabras es un punto de partida operativo,
no un requisito definitivo. Semillas demasiado escasas permiten muchos sueños distintos.
También pueden perder relaciones o emociones que ayuden a acercarse al relato de origen.

Hay dos preguntas distintas: cercanía a alguna referencia del banco y cercanía al relato
que originó la semilla. La primera es la recompensa de este piloto. Registrar la segunda
como diagnóstico separado permitiría estudiar la riqueza de las semillas sin modificar
la recompensa ni dar acceso al relato original al generador.

Comparar semillas breves, intermedias y ricas para los mismos relatos es una propuesta
pendiente, no un experimento ya aprobado o ejecutado. Se deben conservar personas y
referencias reservadas, fijar las demás condiciones y observar cercanía, variedad y libertad
de generación. Aumentar información no debe entregar el relato completo ni una trama resuelta.

## Ejecución y límites actuales

```sh
uv sync --extra embeddings
.venv/bin/python -m think_dream_lab embedding-check
.venv/bin/python -m think_dream_lab base-check --limit 3
.venv/bin/python -m think_dream_lab smoke
```

La comprobación del prompt mínimo con envoltura de conversación produjo un formato válido
en tres intentos; uno de los otros se truncó. Los pesos base quedaron intactos. El prompt
anterior sin esa envoltura no respetó el formato en un intento. Se conservaron ambos ensayos.

El primer intento de piloto completó su evaluación inicial (un formato válido de dos) y
se interrumpió por tiempo al preparar LoRA, antes de actualizar pesos. No creó un checkpoint.
Faltan las actualizaciones reales, recarga, evaluación posterior y comprobación de continuidad.
No se presenta este intento como evidencia de aprendizaje. No se reinician los contadores
para disponer de más tiempo: cualquier ampliación requiere autorización del usuario.
