# Plan de implementación de Dream

Estado al 25 de octubre de 2026: v21 completado, con mejora de calidad en la revisión del
asistente (30 preferencias frente a 11 y 55 empates), pero fallo de formato en dependencia
(17/18). Se conserva el 32 conforme al criterio previo; el candidato v21 queda guardado.

## Ensayo v21: completado, sin promover por formato

- [x] Auditar el filtro de longitud del pool v20 sin relabelar las notas históricas.
- [x] Seleccionar 24 pedidos nuevos de 24 personas de entrenamiento, separados de reservados.
- [x] Generar 96 candidatos desde el 32 y revisar cuatro alternativas por pedido.
- [x] Congelar 19 preferencias claras; cinco empates fuera, once elegidos más largos y ocho más cortos.
- [x] Fijar 24 pedidos nuevos de evaluación con cuatro muestras por modelo y criterio de promoción previo.
- [x] Verificar trayectorias íntegras, origen intacto y pruebas de preparación y criterio.
- [x] Entrenar una sola pasada de 19 actualizaciones desde el 32, después de mostrar el lote.
- [x] Ejecutar y revisar las 96 comparaciones nuevas, variedad, motivos y dependencia.
- [x] Aplicar todos los criterios sin cambiar umbrales: calidad pasa, dependencia falla formato; keep32.

[Lote, método, límites, criterio fijo y resultados](preferencias-v21.md). No se añadieron
pasos después de evaluar. Los 200 relatos de prueba siguen sin usarse.

## Piloto de preferencias v20: completado

- [x] Veinticuatro pedidos de entrenamiento de 24 personas, con 108 candidatos y selección por longitud comparable.
- [x] Veinticuatro pares revisados: 14 preferencias claras congeladas y 10 empates fuera del entrenamiento.
- [x] Una pasada DPO de 14 actualizaciones desde el 32, optimizador nuevo y referencia estacionaria en caché.
- [x] Base y cuatro archivos del origen intactos; adaptadores cambiados y recarga exacta.
- [x] Evaluación antes/después de doce pedidos nuevos, dos semillas por pedido y modelo, sin usarlos para actualizar pesos.
- [x] Notas de calidad congeladas antes de revelar procedencias: 32 preferido 12 veces, piloto 4, ocho empates.
- [x] Variedad exploratoria: dos preferencias por el 32, cinco empates y cinco indeterminados. Orden deducible, sin reclamar cegado.
- [x] Dependencia posterior: 17/18 hechos completos frente a 16/18; prefijos y semillas iguales a v19.
- [x] Replay de la referencia y diagnóstico de log-ratios: dream sí recibió señal; no se cambiaron pérdidas por bloque.
- [x] Auditoría y validación: 178 pruebas aprobadas; una prueba opcional de Torch omitida, sin efecto en DPO.

La pérdida entrenada bajó de 0,6931 a 0,2500, pero las repeticiones de consigna aumentaron
de 1/24 a 5/24. El piloto no cumplió mejorar calidad sin perder variedad y no se prolongó.
No se evaluaron los 200 relatos de prueba. [Método, resultados y límites](preferencias-v20.md).

## Antecedentes inmediatos

Revisión posterior completada: los cuatro aciertos de v20 se concentran en tres pedidos;
dos también presentan regresiones en su otra muestra. [Aprendizajes y siguiente ensayo
propuesto](aprendizajes-v20.md). Próximo paso propuesto: auditar la selección de contrastes
antes de preparar otro lote. No se inició entrenamiento adicional ni se abrió la prueba final.

Experimento ampliado completado, desde el checkpoint 32
hasta el 160. Los pasos 33–160 recorrieron una sola vez los 128 relatos aprobados.
La comparación no mostró una mejora convincente; el 32 sigue como referencia práctica.

Diagnóstico posterior v19 completado: revisión de calidad de dream, calibración de juez
y 36 continuaciones con intervenciones textuales. Sin entrenamiento; el evaluador no pasó
la calibración y no se eligió una recompensa nueva.

## Escenas y dependencia textual: diagnóstico completado

- [x] Revisar seis pares de bloques dream, uno por grupo y fuera de las referencias de v18.
- [x] Conservar notas congeladas antes de revelar las procedencias; revisión del asistente.
- [x] Evaluar seis contrastes de calidad y los seis pares de modelo en ambos órdenes.
- [x] Fijar tres pedidos, seis condiciones y dos semillas; generar 36 answers con el checkpoint 32 congelado.
- [x] Comprobar cambios de ubicación, actor y desenlace, paráfrasis, intercambio de dream y generación directa.
- [x] Congelar las etiquetas de los hechos antes de revelar las condiciones; conservar la salida fallida.
- [x] Auditar pesos intactos, adaptadores exactos, contextos nuevos y ausencia de referencias completas en el prompt.
- [x] Preparar preferencias revisadas y ejecutar el piloto fijado en v20; el resultado fue desfavorable y se conserva el 32.

Los doce bloques de modelo ya presentan contenido escénico. Preferencias: 32 en tres pares,
160 en uno, dos empates. En las intervenciones, detalle agregado 5/6, cambiado 5/6,
paráfrasis 6/6; ambos lados del cambio correctos en 4/6. Esto prueba sensibilidad textual
limitada, no mejora de calidad frente a generación directa. El juez sólo completó 12/24
evaluaciones y acertó 3/10 presentaciones de controles. [Resultados y límites](diagnostico-dream-v19.md).
La fidelidad entre bloques se mantuvo como diagnóstico, sin transformarla en requisito de recompensa.

## Experimento ampliado aprobado

- [x] Lote revisado de 128 relatos aprobado por el usuario, sin personas reservadas.
- [x] Evaluación inicial del paso 32 a 2.048 tokens: 70/72 formatos válidos, coseno 0,6453.
- [x] Juez independiente Phi-4 fijado antes de entrenar y seis controles superados.
- [x] Restauración exacta de adaptadores, optimizador y aleatoriedad; origen intacto.
- [x] Completar 128 actualizaciones nuevas, una visita por relato y 512 rollouts usados.
- [x] Evaluar el paso 160 con los mismos 24 reservados, prompts, semillas y límite.
- [x] Puntuar 139 respuestas válidas de ambas fases mezcladas con el juez ciego, sólo sobre answer.
- [x] Auditar 512 scores de entrenamiento y 144 de evaluación; 166 pruebas aprobadas.
- [x] Comparar ambas medidas y documentar resultados, límites y recuperación del lector del juez.

En los mismos 67 pares válidos, MiniLM pasó de 0,6435 a 0,6485 (+0,0050) y el juez
de 2,2537 a 2,0896 sobre 4 (−0,1642). Ambos intervalos por persona incluyen cero.
Formato: 70/72 → 69/72; error de longitud: 0,3643 → 0,7443. No se siguió entrenando
después de la pasada fijada. Los 200 relatos de prueba siguen sin usarse.
[Diseño, resultados y comandos](experimento-ampliado-v17.md), [datos](resultados-v17.json).

## Revisión posterior del evaluador: completada

- [x] Seleccionar ocho discrepancias MiniLM↑/Phi-4↓ y cuatro contrastes, dos pares por grupo.
- [x] Revisar 24 respuestas de 11 referencias sin procedencia ni scores visibles durante la lectura.
- [x] Congelar las doce anotaciones antes de revelar la clave; protocolo y hash en `4b771ea`.
- [x] Comparar notas y preferencias del asistente con Phi-4, exponiendo dudas y selección diagnóstica.
- [x] Conservar las referencias completas y notas detalladas localmente; publicar sólo informe y métricas.
- [x] Comprobar después de congelar las notas qué hechos centrales estaban incluidos en las semillas.
- [ ] Separar hechos disponibles en las semillas breves de detalles ocultos de referencia; probar un verificador con evidencia y abstención sobre otros pares de desarrollo.
- [ ] Comprobarlo antes de cambiar recompensa o iniciar otro entrenamiento desde el 32.

En las ocho discrepancias, la nota global del asistente quedó igual en seis y bajó en dos;
las preferencias finas se repartieron 4–4. Sobre los doce pares, seis preferencias fueron
por el 32, cuatro por el 160 y dos empates. También aparecieron desacuerdos importantes
con Phi-4. Son juicios del asistente, no verdad de referencia ni validación humana independiente.
No se entrenó ni se usaron los 200 relatos de prueba. [Revisión y decisión](revision-v18.md).

## Alcance vigente

Pedirle a Qwen3-1.7B-Base que sueñe dentro de `<dream>` y cuente lo soñado dentro de
`<answer>`. Sólo D, sin comparación con think ni aritmética. El contenido del sueño queda
libre. No se exige que reproduzca una referencia humana ni se impone fidelidad detallada
entre bloques como recompensa. V19 examina escenas y dependencia textual como diagnóstico.

La consigna usa la plantilla de conversación propuesta por el usuario, semillas naturales
cortas y el tamaño aproximado del relato esperado. La referencia humana completa se
mantiene fuera del prompt. La apertura de dream pertenece a la entrada; el modelo genera
el contenido, los cierres y el relato final, sin un ejemplo resuelto.

El objetivo de la corrida GRPO v17 fue acercar answer a **su relato humano asociado** mediante
embeddings. Esa recompensa alcanza todos los tokens muestreados. No hay castigo automático
por similitud alta o copia, ni una recompensa adicional por longitud. Formato inválido: cero.

El diagnóstico v19 examinó calidad de las escenas y dependencia textual sin entrenar.
El piloto v20 usó preferencias revisadas de calidad, sin cercanía a una referencia particular
ni fidelidad entre bloques como nueva recompensa. Se mantuvo crédito a toda la continuación,
con un objetivo DPO nuevo y optimizador reiniciado. Su resultado fue desfavorable.

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

## Continuación completada: paso 80 → 96

- [x] Restauración completa del checkpoint 80; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 96 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 80, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 66 pares válidos, coseno 0,6350 → 0,6469 (+0,0119). Doce medias por relato suben y doce
bajan. El intervalo por relato/persona es [−0,0019; +0,0380]: ganancia pequeña, no concluyente.
Formato 69/72 → 68/72; truncamientos 3 → 4; el error de longitud mejoró. Resultado general
mixto. Se terminó en el paso 96 y se conserva el 80. Los reservados siguen fuera de las
actualizaciones y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v12.md), [datos numéricos](resultados-v12.json).

## Continuación completada: paso 96 → 112

- [x] Restauración completa del checkpoint 96; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 112 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 96, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 64 pares válidos, coseno 0,6611 → 0,6381 (−0,0230). Nueve medias por relato suben y
catorce bajan; uno no tiene pares válidos. El intervalo por relato/persona es [−0,0564; +0,0026].
Esta ronda retrocedió en promedio, sin demostrar deterioro general. Formato 68/72 → 68/72;
truncamientos 4 → 2; el error de longitud aumentó. Se terminó en el paso 112 y se conserva el 96.
Los reservados siguen fuera de las actualizaciones y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v13.md), [datos numéricos](resultados-v13.json).
La propuesta recibida para ampliar el diseño queda pendiente, sin ejecutar en este segmento.

## Continuación completada: paso 112 → 128

- [x] Restauración completa del checkpoint 112; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 128 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 112, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 64 pares válidos, coseno 0,6394 → 0,6473 (+0,0079). Trece medias por relato suben y once
bajan. El intervalo por relato/persona es [−0,0281; +0,0445]: ganancia pequeña, no concluyente.
Formato 68/72 → 67/72; truncamientos 2 → 4; el error de longitud mejoró. Resultado general
mixto. Se terminó en el paso 128 y se conserva el 112. El lote sigue siendo de ocho relatos;
no se ejecutó la propuesta de 128 relatos. Los reservados siguen fuera de las actualizaciones
y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v14.md), [datos numéricos](resultados-v14.json).

## Continuación completada: paso 128 → 144

- [x] Restauración completa del checkpoint 128; 16 pasos y 64 rollouts nuevos con el lote original.
- [x] Personas y relatos reservados fuera de las actualizaciones; base y origen intactos.
- [x] Recarga exacta del checkpoint 144 y 72 respuestas nuevas de evaluación congelada.
- [x] Comparación con las salidas originales de base y paso 128, mismos prompts y semillas.
- [x] Auditoría de 216 scores de evaluación y 64 de entrenamiento desde caché, sin cambios de código.

En 64 pares válidos, coseno 0,6436 → 0,6398 (−0,0038). Doce medias por relato suben y doce
bajan. El intervalo por relato/persona es [−0,0207; +0,0157]: cercanía prácticamente estable,
sin mejora concluyente. Formato 67/72 → 68/72; truncamientos 4 → 4; el error de longitud
aumentó. Se terminó en el paso 144 y se conserva el 128. El lote sigue siendo de ocho relatos;
no se ejecutó la propuesta de 128 relatos. Los reservados siguen fuera de las actualizaciones
y los 200 de prueba no se usaron.
[Método y resultados](continuacion-v15.md), [datos numéricos](resultados-v15.json).

## Diagnóstico y preparación completados, sin entrenar

- [x] Diagnóstico en seis relatos que incluyen los cuatro truncamientos previos del paso 144,
  comparando checkpoints 32/144 y límites 1.024/2.048 con prompts y semillas idénticos.
- [x] 36 salidas largas nuevas, 36 anteriores reutilizadas; prefijos idénticos en todos los pares.
- [x] Los cuatro truncamientos del 144 se completaron a 2.048 tokens: formato 14/18 → 18/18.
  En el 32 no hubo cambios (17/18). Muestra enriquecida por fallos, sin ranking general.
- [x] 128 relatos de entrenamiento: ocho retenidos + 120 nuevos, balanceados en 30 personas,
  cuatro o cinco por persona; todos los reservados excluidos por relato y por persona.
- [x] Extracción completa y revisión por el asistente de las 128 semillas, con 92 correcciones.
- [x] Tres referencias con recortes explícitos y hashes originales conservados; los ocho
  ejemplos aprobados se mantienen exactamente. Vista revisable y archivo de prompts sin referencia.
- [x] Auditorías de integridad y modelos congelados; 149 pruebas aprobadas; cero actualizaciones.
- [x] Fijar el límite y la evaluación inicial completa del checkpoint elegido antes de entrenar el lote (v17).
- [x] Fijar el juez independiente y sus controles, separado de oniricidad (v17).
- [x] Completar el recorrido del lote nuevo en v17; ver resultados al inicio de este plan.

[Diagnóstico, lote y límites](preparacion-v16.md), [datos de longitud](resultados-longitud-v16.json),
[auditoría del lote](lote-v16-resumen.json). Los 24 relatos siguen siendo desarrollo y los
200 de prueba se reservan para el protocolo final. No se asumió que el checkpoint más reciente sea mejor.

## Recursos y recuperación del piloto inicial

Se consumieron 794,45 de los 900 segundos adicionales autorizados, incluidos tres procesos
fallidos por Metal. La corrida se recuperó desde el checkpoint 7 sin cambiar lote, objetivo,
semillas ni hiperparámetros. Un ajuste limitado al proceso del controlador de GPU permitió
terminar. Se conservaron los presupuestos históricos y todos los intentos fallidos.

Las copias inmutables del código y la configuración, el lote congelado, las evaluaciones,
los tokens y los checkpoints se guardan en `results/dream-v6/`. No se publican referencias
originales ni pesos automáticamente. DreamBank declara CC BY-NC-SA 4.0, atribución a
[dreambank.net](https://dreambank.net/); MIT se aplica al código.

## Propuesta recibida del panel lateral y ejecución

Propuesta recibida durante la corrida 96 → 112. Primero se completaron el diagnóstico
de longitud y la preparación del lote (v16). Después de su aprobación se completó el
experimento ampliado (v17), con el diseño y resultados al inicio de este plan:

- Comparación breve sin entrenar de 1.024 frente a 2.048 tokens, con checkpoints y consignas
  idénticos, para distinguir truncamientos de problemas de generación.
- Ampliar a 128 relatos balanceados entre personas de la partición de entrenamiento,
  excluyendo reservados tanto por relato como por persona. Conservar plantilla, semillas
  cortas, longitud orientativa, recompensa e hiperparámetros para aislar el cambio de lote.
- Fijar antes de ejecutar un recorrido completo por ese lote y evaluar al inicio y al final,
  sin decidir la duración mirando fluctuaciones cada 16 pasos.
- Mantener MiniLM por continuidad y sumar un juez independiente y ciego de cercanía de
  answer a su referencia: personajes, sucesos y relaciones. Dream continúa libre; calidad
  onírica se puntúa aparte. No agregar penalización por copia ni cambiar a la vez la recompensa.
- Conservar los 24 relatos como validación de desarrollo y los 200 de prueba para una
  evaluación final con modelo y protocolo fijados. Comparar checkpoints en pares válidos
  comunes e informar formato y score integral por separado; el más reciente no se asume mejor.

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
