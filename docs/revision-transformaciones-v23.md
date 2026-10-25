# Volver al contenido de los sueños

Revisión descriptiva completada el 25 de octubre de 2026: 18 casos reales de v22, ocho
controles construidos y cuatro presentaciones repetidas. **Sin entrenamiento, nuevos
muestreos de los checkpoints ni cambio de modelo elegido.** Los ocho controles fueron
escritos por el asistente. [Ver textos y notas completos](http://127.0.0.1:62879/index.html).

La pregunta fue qué distingue una experiencia que transforma materiales de una que sólo
repite el pedido, lo comenta o pierde su vínculo. La relación de answer con dream se revisó
aparte. Se fijó una [rúbrica provisional](rubrica-transformaciones-v23.md) antes de revisar.

## Qué mostró esta muestra

Se eligieron por hash doce casos con marca literal importante y seis sin esa marca, cada
uno de un pedido diferente. Se excluyeron los tres casos destacados previamente en la
orientación conceptual. No se balancearon modelos ni se compararon sus resultados.

Entre los doce marcados, **siete conservan material reconocible y cinco material parcial**.
El tratamiento fue transformado en uno, mixto en ocho e indeterminado en tres. La marca
literal no distingue esas situaciones; tampoco los siete reconocibles son automáticamente
buenos sueños: uno es comentario simbólico sin experiencia narrada.

Esto no significa que siete fueran errores de anotación: las notas anteriores describían
literalidad y permanecen intactas. Tampoco estima frecuencias generales, porque se eligió
deliberadamente una muestra enriquecida en esas marcas y se releen textos ya observados.

## Ejemplos concretos

- **S01:** el narrador está envuelto en el paquete con frutas. Paquete, limones, vecino que
  no oye y dirección cambiante siguen presentes. Es una transformación reconocible de la
  entrega, no una pérdida completa de materiales. Answer la conserva.
- **S05:** la bicicleta colgada pasa a pertenecer al mecánico y las voces de motores se
  describen como últimos suspiros. Se reconoce el material y cambia su pertenencia o función.
  No se necesita atribuirle un deseo inconsciente.
- **S06:** la cortina se vuelve objeto con el que bailar; la chaqueta sólo está colgada y
  falta quien la use. Hay elaboración y también material parcial: permitir transformaciones
  no obliga a justificar toda ausencia.
- **S09 y S25:** patio universitario y carretera son fragmentos quietos válidos. No necesitan
  volverse más extraños ni desarrollar una trama para servir como sueños.
- **S10:** explica lo que podrían simbolizar ático, plumas y abuela, pero no desarrolla una
  vivencia. Conservar nombres y hablar de símbolos no basta.

Los ejemplos proceden de generaciones del modelo; no son relatos humanos esperados ni
pruebas de mecanismos internos. Los textos completos se mantienen locales.

## También importa cómo se cuenta después

Sobre los 18 casos reales, doce relatos conservaron la experiencia, tres resumieron con
omisiones o cambios señalados y tres quedaron indeterminados con esta taxonomía.
Dos límites aparecieron con claridad:

- S20 repite en un encabezado refugio y piso de la semilla, aunque dream describía ascenso
  exterior y estrellas a través de nieve. Conserva después casi todo dream: no es sustitución
  total, pero añade elementos para parecerse al pedido.
- S22 vuelve privacidad a silencio y poste exterior a pantalla de lámpara. Recupera la
  semilla, pero modifica lo que el modelo soñó. La etiqueta de «normalizar hacia lo cotidiano»
  no describe bien esa corrección selectiva hacia el prompt; por eso quedó como no sé.

Los controles construidos hacen visible otra distinción. El mismo dream puede presentar
biblioteca/cocina, hermano/maestro y páginas/escalera: un answer conserva esas coexistencias;
otro las vuelve biblioteca ordinaria, sólo hermano y devolución exitosa. El segundo borra
la experiencia aunque sea más fácil de entender. Otro control cuenta una playa en vez del
sueño del libro: eso sí es sustitución completa.

**Los controles son ilustraciones escritas y revisadas por el mismo asistente**, no ejemplos
independientes que certifiquen una tasa de acierto. El caso de sueño ajeno también importa:
una escena del lago rojo puede ser buen sueño y tener buen relato, pero no tener relación
textual defendible con libro, hermano y puerta. No se inventó una conexión para salvarlo.

## Consistencia e integridad

Las cuatro parejas repetidas conservaron las cinco etiquetas. Reconocí los textos al leerlos:
ese acuerdo consigo mismo puede estar inflado y no es fiabilidad entre revisores ni una
validación independiente. Fuente, marca previa y tipo de control no se mostraron en las
fichas; aun así, había visto v22 y escrito los controles. El cegado es limitado.

Se congelaron treinta anotaciones antes de abrir la clave nueva. Cada una incluye evidencia
textual cuya coincidencia exacta se comprobó. La auditoría verifica que packet y notas
congeladas de v22 no cambiaron. No se recalcularon victorias ni criterios de promoción.

[Protocolo](transformaciones-v23-protocolo.json), [resultados descriptivos](transformaciones-v23-resultados.json),
[auditoría](transformaciones-v23-auditoria.json). El 32 sigue como referencia. Los 200 relatos
de prueba siguen sin usarse. Validación: 200 pruebas aprobadas y una omitida por Torch
opcional ausente; las citas, congelaciones y huellas de ambos adaptadores se verificaron.

## Consecuencia para los próximos pares

No elegir por mayor literalidad o extrañeza. Comparar experiencia reconocible y vínculo
defendible con los materiales; permitir escenas literales y transformadas. Revisar también
la continuación answer completa para no favorecer relatos que borran el sueño o lo corrigen
hacia la semilla. Las dudas quedan fuera de entrenamiento, sin sumar etiquetas como si
fueran una recompensa validada.

La propuesta siguiente separa preservación global de las intervenciones del relato, para
tratar los añadidos sin forzarlos a «omisión» o «sustitución»:
[guía para los próximos pares](criterio-proximos-pares-oniricos.md).
