# Plan de implementación de Dream

Estado al 25 de octubre de 2026: circuito implementado y ensayo aprobado completado.

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
- [ ] Publicación del site con los resultados de esta corrida: en preparación.

## Resultado observado

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

## Recursos y recuperación

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
