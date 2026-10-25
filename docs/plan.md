# Plan de implementación de Dream

Estado al 25 de octubre de 2026: circuito implementado, continuación hasta el paso 80 y evaluación completadas.

## Alcance vigente

Pedirle a Qwen3-1.7B-Base que sueñe dentro de `<dream>` y cuente lo soñado dentro de
`<answer>`. Sólo D, sin comparación con think ni aritmética. El contenido del sueño queda
libre. No se evalúa su significado ni la fidelidad detallada del relato respecto de él.

La consigna usa la plantilla de conversación propuesta por el usuario, semillas naturales
cortas y el tamaño aproximado del relato esperado. La referencia humana completa se
mantiene fuera del prompt. La apertura de dream pertenece a la entrada; el modelo genera
el contenido, los cierres y el relato final, sin un ejemplo resuelto.

El objetivo de la corrida aprobada es acercar answer a **su relato humano asociado** mediante
embeddings. Esa recompensa alcanza todos los tokens muestreados. No hay castigo automático
por similitud alta o copia, ni una recompensa adicional por longitud. Formato inválido: cero.

## Implementación comprobada

- [x] Repositorio público `dream`, código MIT y configuración de Python.
- [x] DreamBank como corpus principal; archivo fijado por SHA-256, importación sin ejecutar HTML.
- [x] Registro de 96 series, alias y hogares; exclusión de colecciones colectivas sin identidad
  individual resuelta y de idiomas fuera del primer piloto.
- [x] Deduplicación y particiones de 1.000/100/200 relatos en 30/6/7 grupos separados.
- [x] Generador original del piloto, revisión `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`,
  con texto plano y sin activar modo de razonamiento ni cargar adaptadores previos.
- [x] Lote corto revisado antes de ejecutar: ocho referencias de entrenamiento y cuatro
  reservadas, con semillas de 17–25 palabras y longitud esperada calculada por relato.
  El asistente preparó las semillas; el usuario aprobó el lote corto y 900 segundos adicionales.
- [x] Verificación del hash del lote, integridad de referencias, separación por persona y ausencia
  del relato humano en el prompt. Lote: `0560294db2bec293522d737580aba78edf63313851138e362841a93626038450`.
- [x] Encoder MiniLM congelado y versionado, pooling y tratamiento explícito de todos los
  fragmentos largos; caché auditable.
- [x] Recompensa pareada de answer, con referencias internas separadas de los datos del actor.
  La validación se utiliza para evaluar y no para actualizar pesos.
- [x] GRPO + LoRA con los tokens muestreados íntegros, checkpoints de cada actualización,
  restauración de optimizador y estados aleatorios, límites efectivos de tiempo y memoria.
- [x] Dieciséis pasos reales completados, cuatro candidatos por paso: 64 rollouts de entrenamiento.
- [x] Evaluación de doce generaciones por fase, mismos prompts y semillas por consigna.
- [x] Adaptadores modificados, pesos base intactos y recarga idéntica del checkpoint final.
- [x] Auditoría final: lote, dieciséis pasos, referencias correctas, scores recalculados desde
  vectores guardados, separación de validación y presupuesto dentro del autorizado.
- [x] 133 pruebas aprobadas; informe local y resultados agregados preparados para el repositorio.
- [x] Site publicado con método y resultados de esta corrida: versión 13, despliegue exitoso.

## Resultado del piloto de cuatro relatos

| Medida | Antes | Después |
| --- | ---: | ---: |
| Coseno medio al relato humano, respuestas válidas | 0,533 · n=10 | 0,599 · n=12 |
| Formato válido | 10/12 | 12/12 |
| Recompensa media incluyendo fallos de formato | 0,639 | 0,799 |

En los diez pares válidos en ambas fases, el coseno aumentó 0,039 en promedio. En Green City
pasó de 0,389 a 0,522. Tres medias por relato subieron y una bajó. Es una señal favorable en
cuatro referencias de desarrollo; no demuestra eficacia generalizada. La longitud sigue
presentando desvíos importantes. Los 200 relatos de prueba no se utilizaron.

[Resultados y límites](resultado-entrenamiento-v6.md), [datos numéricos](resultados-v6.json) y
[método de la corrida](entrenamiento-pareado-v6.md).

## Evaluación ampliada completada

- [x] Checkpoint 16 congelado frente al modelo base, sin seguir entrenando.
- [x] Otros 24 relatos de validación, cuatro de cada uno de los seis grupos reservados,
  excluyendo los cinco relatos de validación que ya habían sido generados.
- [x] Semillas de 19–25 palabras revisadas antes de evaluar ambos modelos, misma plantilla
  y longitud esperada. Tres generaciones por relato y modelo, con semilla por muestra.
- [x] 72 pares comprobados; 144 scores recalculados desde caché, pesos intactos y cero
  actualizaciones. Suite completa: 139 pruebas aprobadas.
- [x] Informe por relato, gráfico e intervalos exploratorios por persona.

Formato válido: 52/72 → 66/72. Coseno de los 48 pares válidos: cambio medio +0,0102.
La mejora principal corresponde al formato; la cercanía subió poco y de manera heterogénea.
[Resultados nuevos y límites](evaluacion-ampliada-v7.md), [datos numéricos](resultados-v7.json).
El site v13 conserva el piloto anterior; esta evaluación queda documentada en el repositorio.

## Continuación completada: paso 16 → 32

- [x] Restauración del checkpoint 16 completo, incluidos optimizador y estados aleatorios.
- [x] Dieciséis pasos nuevos con el mismo lote de ocho relatos e hiperparámetros; 64 rollouts.
- [x] Personas reservadas fuera de las actualizaciones; base y checkpoint fuente intactos.
- [x] Recarga exacta del checkpoint 32 y evaluación congelada de las mismas 72 posiciones.
- [x] Comparación con las salidas originales de base y paso 16, sin regenerarlas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché; 142 pruebas aprobadas.

Paso 16 → 32: formato 66/72 → 70/72; truncamientos 2 → 0. En 64 pares válidos,
coseno 0,6320 → 0,6398 (+0,0078). La media con igual peso por relato/persona es +0,0099,
con intervalo exploratorio [-0,0100; +0,0284]. La señal de cercanía es pequeña e incierta.
Los 24 relatos son validación de desarrollo ya observada, fuera del entrenamiento;
los 200 de prueba permanecen sin usar. Se terminó en el paso 32.
[Resultados y método](continuacion-v8.md), [datos numéricos](resultados-v8.json).

## Continuación completada: paso 32 → 48

- [x] Restauración del checkpoint 32 completo; 16 pasos y 64 rollouts adicionales con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 48 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 32, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché; 142 pruebas aprobadas.
- [x] Informe generalizado a la procedencia de cada segmento; regresión exacta contra v8.

En 67 pares válidos, coseno 0,6446 → 0,6523 (+0,0078); doce medias por relato suben y doce
bajan. La media por relato/persona tiene intervalo [-0,0132; +0,0317]. Formato 70/72 → 69/72,
truncamientos 0 → 1. Resultado mixto, sin mejora concluyente respecto del paso 32.
Se terminó en el paso 48. Los 24 relatos siguen fuera de las actualizaciones y los 200 de
prueba permanecen sin usar. [Método y resultados](continuacion-v9.md), [datos numéricos](resultados-v9.json).

## Continuación completada: paso 48 → 64

- [x] Restauración completa del checkpoint 48; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 64 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 48, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 66 pares válidos, coseno 0,6584 → 0,6469 (−0,0115). Catorce medias por relato suben y diez
bajan, con pérdidas mayores. La media por relato/persona tiene intervalo [−0,0251; +0,0030].
Formato 69/72 → 67/72; truncamientos 1 → 4. Esta ronda retrocedió en promedio y no muestra
mejora respecto del paso 48; la incertidumbre incluye cero. Se terminó en el paso 64 y se
conserva el 48. Los reservados siguen fuera de las actualizaciones y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v10.md), [datos numéricos](resultados-v10.json).

## Continuación completada: paso 64 → 80

- [x] Restauración completa del checkpoint 64; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 80 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 64, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 64 pares válidos, coseno 0,6483 → 0,6469 (−0,0013 antes de redondear). Once medias por
relato suben y doce bajan; uno no tiene pares válidos. El intervalo por relato/persona es
[−0,0178; +0,0145]. Cercanía prácticamente sin cambio, sin mejora concluyente. Formato
67/72 → 69/72; truncamientos 4 → 3; el error de longitud aumentó. Se terminó en el paso 80
y se conserva el 64. Los reservados siguen fuera de las actualizaciones y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v11.md), [datos numéricos](resultados-v11.json).

## Recursos y recuperación del piloto inicial

Se consumieron 794,45 de los 900 segundos adicionales autorizados, incluidos tres procesos
fallidos por Metal. La corrida se recuperó desde el checkpoint 7 sin cambiar lote, objetivo,
semillas ni hiperparámetros. Un ajuste limitado al proceso del controlador de GPU permitió
terminar. Se conservaron los presupuestos históricos y todos los intentos fallidos.

Las copias inmutables del código y la configuración, el lote congelado, las evaluaciones,
los tokens y los checkpoints se guardan en `results/dream-v6/`. No se publican referencias
originales ni pesos automáticamente. DreamBank declara CC BY-NC-SA 4.0, atribución a
[dreambank.net](https://dreambank.net/); MIT se aplica al código.

## Investigación posterior

La comparación de distintos niveles de riqueza de semillas sigue siendo una propuesta,
no un experimento ejecutado. También quedan abiertas la generalización a más personas,
la estabilidad del formato, el control de longitud y la distinción entre cercanía semántica
y calidad onírica. Estas preguntas no se consideran resueltas por completar el circuito.

Una coincidencia alta es favorable; la posible memorización previa del corpus público
limita su interpretación. Los resultados no prueban experiencia subjetiva del modelo.

## Historia conservada

La extracción local de palabras clave, el score al vecino más próximo del banco y los
pilotos anteriores quedan documentados por separado. El piloto anterior de tres pasos
verificó recarga y continuidad exacta desde su checkpoint 2, sin demostrar mejora en sus
dos muestras reservadas. No se trasladan sus resultados a la nueva plantilla.

DreamViews y su juez de cumplimiento/fidelidad están archivados en
`archive/dreamviews-judge-20261025`. Las condiciones A/B/C, otros generadores y aritmética
son antecedentes. Inspect se conserva como integración opcional; no ejecutó el entrenamiento.
No hicieron falta textos surrealistas porque DreamBank aportó suficientes relatos.
