# Revisión asistida de doce pares: qué mide la cercanía

**Completada. MiniLM y Phi-4 muestran límites distintos; esta revisión no justifica reemplazar directamente la recompensa por el juez actual.** No hubo nuevo entrenamiento ni nuevas generaciones del modelo. Los 200 relatos de prueba siguen sin usarse.

## Cómo se hizo

Se eligieron los ocho pares válidos de v17 donde MiniLM subía y la cercanía global de Phi-4 bajaba. Otros cuatro pares, seleccionados por hash, completaron dos pares por cada uno de los seis grupos de desarrollo. Son 12 pares, 24 respuestas y 11 referencias distintas: una referencia aparece con dos candidatos. Esta muestra está seleccionada por discrepancias; no estima su frecuencia ni la calidad general.

El asistente leyó sólo la referencia y las respuestas A/B, sin identificadores originales, semillas, dream, checkpoint ni scores. Cada checkpoint ocupó A seis veces. Se anotaron personajes/objetos/escenario, sucesos, relaciones y cercanía global con la misma rúbrica de 0–4 de v17. Se guardaron hechos centrales, observaciones, preferencia y confianza. La preferencia fina puede distinguir dos respuestas que reciben la misma nota global entera.

Las 12 anotaciones quedaron congeladas antes de abrir la clave. El commit `4b771ea` registra su hash y el protocolo antes de revelar procedencias. No se cambiaron notas después. El asistente ya conocía el resultado agregado y el criterio de selección, y podía inferir procedencia por el estilo o la longitud: es procedencia oculta durante la lectura, no validación humana independiente ni ceguera garantizada.

## Resultado de los ocho pares discrepantes

- Con mi criterio, el checkpoint 160 recibió una nota global menor en **2/8**, la misma nota en **6/8** y una mayor en **0/8**. Phi-4 había marcado descenso en los ocho por definición de la selección.
- Las preferencias finas quedaron **4 por el 32 y 4 por el 160**. Tres de las cuatro preferencias por el 160 fueron de confianza baja y una media; las del 32 fueron dos altas y dos medias. No corresponde tratar todos esos votos como certezas equivalentes.
- En estos ocho casos, una subida de MiniLM no bastó para asignar una mejora global de un punto. Eso no significa que no hubiera mejoras parciales: por ejemplo, en R01 preferí el 160 y valoré mejor sus relaciones, aunque mantuve la nota global.

En los 12 pares completos, preferí el 32 seis veces, el 160 cuatro y declaré dos empates. La nota global quedó igual en nueve y bajó en tres. Las tres preferencias de confianza alta fueron por el 32. Son observaciones diagnósticas sobre ejemplos elegidos; no prueban superioridad general ni permiten estimar un efecto del entrenamiento.

## Comparación caso por caso

Antes: checkpoint 32. Después: checkpoint 160. Sólo se puntúa answer.

| Caso | Selección | Cambio MiniLM | Phi-4 antes → después | Asistente antes → después | Preferencia | Confianza |
| --- | --- | ---: | ---: | ---: | --- | --- |
| R01 | Discrepancia | +0.1239 | 3 → 2 | 2 → 2 | 160 | Media |
| R02 | Discrepancia | +0.0656 | 2 → 1 | 3 → 2 | 32 | Alta |
| R03 | Discrepancia | +0.0646 | 3 → 2 | 2 → 2 | 32 | Media |
| R04 | Contraste | +0.0922 | 1 → 1 | 2 → 2 | 32 | Media |
| R05 | Discrepancia | +0.0341 | 3 → 1 | 1 → 1 | 160 | Baja |
| R06 | Discrepancia | +0.1545 | 2 → 1 | 2 → 1 | 32 | Alta |
| R07 | Contraste | +0.1147 | 3 → 3 | 1 → 1 | Empate | Media |
| R08 | Discrepancia | +0.0204 | 3 → 1 | 2 → 2 | 32 | Media |
| R09 | Discrepancia | +0.2318 | 3 → 1 | 1 → 1 | 160 | Baja |
| R10 | Contraste | -0.1533 | 1 → 1 | 1 → 1 | Empate | Media |
| R11 | Contraste | -0.0360 | 3 → 3 | 2 → 1 | 32 | Alta |
| R12 | Discrepancia | +0.0123 | 2 → 1 | 2 → 2 | 160 | Baja |

## Errores observados y desacuerdos con el juez

- **Agencia y reacción invertidas (R02).** Una respuesta conserva que los surfistas habituales no quedan impresionados con el narrador; la otra cambia a que él no queda impresionado con ellos. Ambas comparten el escenario y las palabras principales.
- **Temas sin acontecimientos (R03 y R09).** Mirar un objeto no equivale a moverlo con la mirada. Hablar de viajar al espacio o desear hacerlo no equivale a ir como sustituto en una misión.
- **Causalidad reemplazada (R05).** La referencia cuenta un intento de incriminar al narrador; una respuesta lo convierte en ayudar a un joven con problemas migratorios. Phi-4 dio 3/4 a esa respuesta y yo 1/4. La caída posterior de 3 a 1 no fue para mí una caída de dos puntos: ambas respuestas fallaban gravemente.
- **Comparación convertida en identidad (R04).** Una boca parecida a una flor pasa a ser una criatura floral; también cambia una abuela ajena por una propia.
- **Errores que el juez no distingue (R07 y R11).** Dio 3/4 a ambas versiones de cada par, pese a cambios importantes de agencia o a sustituir la reconstrucción de fotos familiares por un encuentro con alguien vivo. En R11 sí preferí claramente el 32.

Estos ejemplos sostienen una conclusión más limitada que “el juez tiene razón y MiniLM no”: MiniLM puede subir con errores narrativos, y Phi-4 también puede sobrevalorar una respuesta o perder una diferencia importante.

En cercanía global coincidí exactamente con Phi-4 en **9/24 respuestas**; diferí en dos puntos en **5/24**, siempre con Phi-4 por encima de mi valoración. No es una tasa de precisión: mis notas también son juicios de un asistente, la muestra fue seleccionada y las dos respuestas de cada par no son independientes. Los seis controles sintéticos aprobados en v17 no bastaron para asegurar acuerdo en estas narraciones reales.

## Comprobación posterior de las semillas

Después de congelar las notas y revelar la clave, se leyeron las semillas originales.
Esta comprobación es posterior y no modificó ninguna calificación.

- En R03 la semilla **sí pide mover un objeto con la mirada**, pero ambas respuestas omiten ese hecho. En R11 también se entregan la inexistencia del hermano y su muerte en moto. Hay pérdidas de información disponible, no sólo datos imposibles de adivinar.
- En R04 la semilla dice que hay una abuela cercana, pero omite que no es una abuela propia. En R07 omite el giro final de volver para atacar al líder. Esos detalles de la referencia no estaban especificados en la entrada.
- En R08 la semilla acentúa el lujo de las cajas; las respuestas amplifican ese lujo frente a una referencia que también describe un hotel barato. La compresión de la referencia puede influir en la desviación.

Una semilla breve admite muchos relatos posibles. Podemos medir su cercanía al relato
humano esperado, pero no atribuir cada diferencia a desobediencia ni asumir que un nuevo
score recuperará información que no se entregó. Conviene distinguir hechos centrales
disponibles en la semilla de detalles ocultos de la referencia. La referencia completa
permanece fuera del prompt y las semillas siguen siendo breves.

## Decisión y siguiente ensayo propuesto

Se conserva el checkpoint 32 como referencia práctica y el 160 como resultado experimental. No se cambió la recompensa ni se siguió entrenando: el requisito de contar con un evaluador mejor comprobado todavía no se cumplió. Los resultados numéricos originales de v17 se mantienen; se matiza su interpretación, especialmente la confianza en Phi-4.

El siguiente candidato debería verificar hechos y relaciones antes de convertirse en recompensa:

1. Extraer y congelar entre tres y seis hechos centrales de cada referencia **sin ver las respuestas**: actor, acción, destinatario, resultado, modalidad y orden cuando correspondan. Anotar cuáles están efectivamente presentes en la semilla breve. Evitar listas de sustantivos.
2. Para cada hecho, marcar preservado, omitido, contradicho o no_se, con un fragmento de evidencia de answer. Distinguir querer, imaginar, escuchar y hacer; no inferir una relación porque sus palabras aparezcan juntas.
3. Informar cobertura, contradicciones y abstenciones por separado. No fijar todavía una mezcla numérica arbitraria ni reutilizar sin comprobar el antiguo juez por requisitos de v4.
4. Probar primero sobre respuestas existentes y otros pares de desarrollo separados de estos 12 ejemplos de diseño. Incluir paráfrasis, copia fiel, cambios de actor/resultado/modalidad y versiones más largas con los mismos hechos; no castigar copia ni longitud por sí mismas.
5. Sólo si ese verificador distingue mejor los casos y expone sus dudas, ensayar entrenamiento breve desde el 32. Dream continúa libre y no entra en esta evaluación. Los 200 relatos de prueba siguen reservados.

**Ese nuevo verificador y el entrenamiento posterior están propuestos, no ejecutados.** Esta etapa completó la revisión y produjo evidencia para elegir el siguiente evaluador.

## Archivos y reproducción

- [Resumen numérico agregado](revision-v18-resumen.json).
- [Protocolo y hashes congelados](revision-v18-protocolo.json).
- Paquete y notas locales: `results/dream-v18/review/packet.json` y `annotations.json`.
- Vista local con textos, notas y procedencias reveladas: `results/dream-v18/review/resultados.html`.
- Clave y resultados originales permanecen separados del paquete de lectura. Referencias, respuestas completas y notas detalladas no se publican en Git.

```sh
.venv/bin/python -m experiments.review_v17_pairs
.venv/bin/python -m experiments.summarize_v17_review
```

El primer comando reproduce la selección y rechaza cambios sobre un paquete existente. No fabrica anotaciones. El segundo exige las 12 notas congeladas y verifica sus hashes antes de abrir la clave; una modificación invalida la revisión.

Validación: **169 pruebas aprobadas**. Las nuevas pruebas comprueban selección equilibrada,
ausencia de procedencia y scores en el paquete, preservación de notas al regenerarlo y
rechazo de cambios en textos o anotaciones antes de abrir la clave. La vista local sirve
12 casos y 24 respuestas; los hashes públicos coinciden con las notas congeladas.

DreamBank: CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT. Revisión del asistente, no validación humana independiente.
