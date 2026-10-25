# Dream genera escenas y condiciona answer; el juez de calidad sigue sin validar

25 de octubre de 2026. Diagnóstico completado, **sin entrenamiento**. Se revisó el contenido
de dream, se probó un juez de calidad y se generaron 36 continuaciones del checkpoint 32.
La referencia humana completa no entró en el generador. Los 200 relatos de prueba siguen sin usarse.

## Qué cambia respecto de v17/v18

La cercanía a un relato humano particular queda como diagnóstico histórico, no como medida
de calidad onírica. Se examinan escenas y experiencias, aceptando sueños cotidianos,
fragmentarios o poco emotivos. No se premia rareza, prosa ornamental, longitud, moraleja
ni una trama cerrada. No se incorporaron penalizaciones por copia o longitud.

Dream sigue libre. Las intervenciones examinan dependencia textual; la fidelidad entre
bloques no se convirtió en requisito de recompensa ni se cambiaron pérdidas o máscaras
por segmentos. No se eligió una nueva recompensa y no hubo nuevas actualizaciones de pesos.

## 1. Revisión exploratoria de los bloques dream

Se seleccionó por hash un par válido del checkpoint 32 y del 160 de cada uno de los seis
grupos de desarrollo, excluyendo las once referencias leídas en v18. La selección no usó
la magnitud ni la dirección de los scores. Está condicionada a formato válido y no representa
todas las salidas, incluidas las fallidas. Son seis pedidos, doce bloques de modelo.

Se añadieron seis contrastes: cuatro relatos humanos frente a lista, instrucciones, análisis
o repetición; un texto humano duplicado, que debe empatar; y un relato cotidiano frente a
una escena surrealista construida, sin preferencia correcta prefijada. Los relatos humanos
de calibración provienen de entrenamiento. Los textos no tienen longitudes exactamente
iguales; la longitud es una limitación pendiente de esta calibración pequeña.

El asistente leyó sólo dream y los motivos, sin answer ni scores. Las procedencias de los
modelos se ocultaron hasta congelar las notas; ocupaban A tres veces y B tres veces.
El mismo asistente construyó los controles y conocía resultados agregados anteriores:
**no es revisión humana independiente ni ceguera garantizada**. Las notas y sus hashes
se conservaron antes de abrir las claves.

Resultado de la revisión del asistente:

| Medida | Resultado |
| --- | ---: |
| Bloques clasificados como escena | 11/12 |
| Bloque que mezcla escena y resumen de motivos | 1/12 |
| Preferencias por el checkpoint 32 | 3/6 |
| Preferencias por el checkpoint 160 | 1/6 |
| Empates | 2/6 |

En esta muestra los modelos **ya generan contenido escénico dentro de dream**; no se limitan
a planificar cómo escribirlo. No se encontró una preferencia clara por el 160. Esto es un
diagnóstico de seis pares por un asistente, no una estimación de calidad general.

## 2. El juez de calidad no pasó la calibración

Se fijó una rúbrica nueva y se usó Phi-4-mini-instruct-4bit congelado, revisión
`ac1c269cb4222a4e136a3d09edad301056c1f36a`, temperatura cero, 512 tokens de salida.
Cada par se presentó en ambos órdenes: 24 evaluaciones. El juez recibió sólo los dos textos
y motivos; no recibió procedencias, notas del asistente, answer ni referencias como objetivos
de cercanía. Clasificaba modo y preferencia, con empate y abstención `no_se` permitidos.

- Sólo **12/24** salidas cumplieron el esquema completo. Los errores fueron valores de
  modo o elección fuera del vocabulario fijado, no fallas de carga del modelo. Algunos son
  diferencias de mayúsculas; otros incluyen categorías o incluso motivos completos en el campo.
- Acertó **3/10 presentaciones de los cinco controles con respuesta prefijada**: dos corresponden
  al mismo control de identidad en ambos órdenes y una al contraste con repetición. No son
  diez controles independientes. Hubo además abstenciones, un empate incorrecto y errores de esquema.
- La puerta exploratoria fijada antes del juez exigía al menos 9/10 e identidad correcta en
  ambos órdenes. **No pasó**. Aun pasarla no habría validado calidad onírica general.
- Como diagnóstico posterior, se leyó sólo la elección cuando era A, B o empate, incluso
  si fallaban los campos de modo. Entre ocho pares con elecciones interpretables en ambos
  órdenes, **cuatro cambiaron de preferencia al invertir los lados**. No se repararon los
  resultados originales ni aumentó por eso la cobertura de esquema válido. No se incluyeron
  abstenciones en esa comparación de estabilidad.

Por eso no alcanza con relajar el JSON ni trasladar estas preferencias directamente a RL.
Los errores del evaluador se conservaron separados: no se convirtieron en recompensa cero
del actor. Sus pesos quedaron intactos. Los controles y las preferencias del asistente son
material de desarrollo, no una verdad externa de referencia.

## 3. Intervenciones: 36 answers nuevos, sin actualizar pesos

Se fijaron tres pedidos y sus dreams originales del checkpoint 32, dos semillas de muestreo
(17 y 29), seis condiciones y hasta 1.024 tokens nuevos por respuesta. Se conservaron
temperatura 0,8 y top-p 0,95. Los detalles fueron agregados artificialmente, ausentes de las
semillas originales:

- Ubicación: una llave de cobre debajo de una silla o dentro de un armario.
- Actor: la hermana o el narrador esconde una moneda de plata.
- Desenlace: un barco de papel se hunde antes de llegar o llega sin hundirse.

Cada generación reconstruyó su contexto desde cero. El texto del dream original antes
de su cierre se conservó literalmente; la apertura de answer se normalizó y todos los
prefijos se recodificaron como texto. No se reutilizaron estados ocultos del dream anterior
ni se recortaron prefijos largos. Se guardaron entradas, tokens, límites, semillas y salidas.
La creación de cachés nuevas se comprobó en el código instalado y en la documentación
de [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/_autodocs/api-reference/generate.md).

Las respuestas se mezclaron y se anotó qué hecho afirmaban **sin mostrar condición ni
semilla aleatoria**. Las notas se congelaron antes de abrir esa clave. Se evaluó la afirmación
del evento, no necesariamente el último estado del objeto: colocar una llave bajo una silla
y después usarla sigue afirmando el evento de colocación. No llegar a la costa por sí solo
no se contó como hundirse. Esta revisión también es del asistente, no humana independiente.

| Condición | Detalle mencionado | Contenido esperado completo |
| --- | ---: | ---: |
| Dream original, sin detalle agregado | 0/6 | Ausencia del detalle: 6/6 |
| Dream con detalle agregado | 6/6 | 5/6 |
| Dream con detalle cambiado | 5/6 | 5/6 |
| Dream con el detalle parafraseado | 6/6 | 6/6 |
| Dream de otro pedido, con su detalle | 4/6 | 4/6 |
| Generación directa, sin dream | 0/6 | Ausencia del detalle: 6/6 |

En **4/6 pares de la misma escena y semilla** se conservaron completos tanto el detalle
original como su modificación. Uno falló por omitir la llave cambiada y otro por decir que
el barco no llegó sin afirmar que se hundió. No se reemplazaron esos resultados.

Esto muestra que answer **sí es sensible a contenido de dream**, incluidos detalles
ausentes del pedido. La sensibilidad fue imperfecta. En dos paráfrasis del barco la respuesta
conservó ese episodio pero perdió las otras escenas. Seguir el detalle no demuestra fidelidad
al sueño completo. Parte de la respuesta reutiliza texto literalmente; no se castiga la copia,
pero eso tampoco demuestra una mejora de calidad o una ventaja de dos etapas.

Formato: **35/36** respuestas válidas. La excepción fue una salida repetitiva de etiquetas
y emojis de la condición directa, truncada al límite. Las 36 respuestas permanecen en el
informe, sin regenerar la fallida ni tratarla como un error del juez.

## Costo y límites

Las continuaciones usaron 5.639 tokens nuevos y 80,72 segundos de generación medidos dentro
del worker; no incluyen carga del modelo, revisión ni costo previo de producir los dreams
guardados. Los prefijos medios fueron de 292 tokens en original y 82,7 en directa; el resto
de condiciones rondó 305. La condición directa también cambió la instrucción para omitir
el bloque dream. **No es un control de costo idéntico.**

No se demostró que generar dream mejore calidad, diversidad o eficiencia frente a respuesta
directa. Tampoco se evaluó un mecanismo equivalente al sueño biológico. Son tres escenas,
tres tipos de detalle y dos semillas, no 36 personas independientes. No se calcularon
intervalos de eficacia con esta muestra. Los seis grupos ya son desarrollo y los 200 de
prueba quedaron fuera de selección, generación y puntuación.

## Decisión

No se entrenó de nuevo, no se cambió el reparto de crédito y no se eligió una nueva recompensa.
Los dos pasos que ya funcionan son generar escenas en dream y trasladar parte de su contenido
a answer. El cuello de botella observado sigue siendo una evaluación de calidad confiable.

El siguiente trabajo propuesto es preparar preferencias de calidad revisadas sobre material
de **entrenamiento**, separando uso de motivos, escenas vividas y dudas. Esas preferencias
podrían habilitar un piloto pequeño sin depender de este juez online. Primero se revisaría
ese lote y su protocolo; no se ejecutó aquí un nuevo entrenamiento. No usar los ejemplos
de desarrollo recién revisados para actualizar el modelo.

## Artefactos y reproducción

- [Resultados agregados](diagnostico-v19-resultados.json).
- [Protocolo y hashes congelados](diagnostico-v19-protocolo.json).
- [Auditoría](diagnostico-v19-auditoria.json).
- Vista completa local: `results/dream-v19/index.html`.
- Textos y notas: `results/dream-v19/quality-packet.json`, `quality-annotations.json` y
  `interventions/review-annotations.json`. Permanecen fuera de Git.

```sh
.venv/bin/python -m experiments.dream_quality_v19
.venv/bin/python -m experiments.dream_interventions_v19 prepare
.venv/bin/python -m experiments.dream_interventions_v19 run
.venv/bin/python -m experiments.judge_dream_quality_v19
.venv/bin/python -m experiments.report_dream_v19
```

Las notas del asistente son insumos reales, no se fabrican al reproducir los comandos.
El informe rechaza textos o anotaciones cuyos hashes no coincidan con sus versiones congeladas.
La corrida real utilizó una copia congelada del script de intervenciones. Se guardó además
una copia del código base y se comprobó que coincidía con el workspace usado por el worker.
Se conservan bajo `results/dream-v19/interventions/` junto a los tokens exactos.
La suite completa aprobó **172 pruebas**; el informe audita las notas congeladas, 36 entradas,
checkpoint, prefijos originales y código utilizado antes de declarar la corrida válida.

DreamBank: CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT. Pesos, referencias humanas
y generaciones completas permanecen locales.
