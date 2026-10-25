# Qué enseñan los cuatro aciertos del piloto v20

Revisión posterior al resultado, 25 de octubre de 2026. No es una nueva evaluación
cegada: conozco los modelos y selecciono los casos porque habían ganado. Se mantienen
intactas las notas originales, los empates y las métricas. No se generaron textos nuevos,
no se entrenó y no se abrieron los 200 relatos de prueba.

## Hallazgo principal

Los aciertos sugieren **conservar una perspectiva situada y hacer perceptibles las acciones
o los cambios**, evitando sustituirlos por afirmaciones genéricas de emoción o significado.
Es una hipótesis para mejorar los contrastes de entrenamiento, no evidencia de una habilidad
estable adquirida. Los cuatro aciertos pertenecen a sólo tres pedidos; dos de esos pedidos
también tienen una regresión en la otra muestra.

## Los cuatro casos

Las cantidades son palabras separadas por espacios en dream, sin contar answer. Las
descripciones siguientes parafrasean las generaciones; no son referencias humanas.

| Caso | Qué cambia respecto del 32 | Palabras antes → después | Qué aprender y qué no concluir |
| --- | --- | ---: | --- |
| E06: sala de espera, reloj y padre pintando | El hombre anónimo vuelve a ser el padre; la pared parece alejarse hasta dejar sólo su figura. Hay una transformación percibida frente a descripciones de ansiedad y un cierre genérico de paz. | 126 → 93 | Mantener una relación personal y representar el cambio puede ayudar. No se demuestra mejor cumplimiento de todo el pedido: el padre sostiene el pincel, pero no se narra claramente que pinte. Quedan expresiones genéricas de anticipación. |
| E08: el mismo pedido, otra muestra | Desaparecen varios párrafos que reiteraban orgullo, importancia y expectativas. Se conserva la observación del padre y aparecen preguntas de las personas que esperan. | 364 → 115 | Reducir reiteración sin perder la escena ayudó aquí. El texto nuevo todavía es convencional; no prueba una calidad onírica excepcional ni justifica premiar brevedad. |
| E19: tren, boleto y hermano dormido | Aparecen banco de madera, manta sobre la cabeza, boleto en el suelo, llegada a una estación y cambios entre viaje, casa y despertar. | 49 → 104 | La experiencia se vuelve específica, incluso con confusión temporal. No hay que exigir continuidad realista ni una trama resuelta. Añadir objetos por sí solo no garantiza mejorar. |
| E24: balcón, vecinos y valija | El narrador mira hacia abajo, ve un picnic, se acerca a la valija y escucha campanas; antes predominaban bienvenida y metáforas de pensamientos. | 55 → 88 | Las relaciones entre observación y acción aportan experiencia. La ventaja era moderada y discutible: la valija pasa de vacía a contenido desconocido, de modo que esta preferencia no acredita mayor fidelidad a los motivos. |

La nota original dio confianza alta sólo a E08; las otras tres fueron medias. Dos aciertos
son más breves y dos más largos. Por tanto, no usaría longitud, extrañeza, emoción ni cantidad
de adjetivos como sustitutos de calidad.

## Contrastar los aciertos con lo que falló

- **La sala de espera es la única consigna que gana en ambas muestras** (E06 y E08).
  Son dos realizaciones de un pedido, no dos pedidos independientes.
- **El tren gana en E19 pero pierde en E14**: allí el piloto devuelve sólo la consigna.
  El balcón gana en E24 pero pierde en E11: allí reemplaza el episodio por misterio,
  anticipación y un viaje anunciado. No hay mejora consistente por tema.
- En los otros nueve pedidos hay diez derrotas y ocho empates, sin victorias del piloto.
  Esta lectura de los aciertos no cambia el balance global de 12 derrotas, cuatro victorias
  y ocho empates.
- Cuatro derrotas son consigna sin escena (E03, E14, E16 y E17). E13 es un empate porque
  ambos modelos repiten el pedido. Incluso excluyendo a posteriori las cuatro derrotas
  por copia, quedan **ocho derrotas frente a cuatro victorias y ocho empates**. La regresión
  no se agota en esa falla de formato semántico; esa exclusión no reemplaza la métrica primaria.
- E02 pierde interacción corporal con la silla y la relación del pan con el ausente;
  E05 pierde manipulación del libro y desplazamientos; E18 cambia interacción con el perro
  por ambiente mágico y comentario emocional. Es el reverso de lo útil en los cuatro aciertos.
- E20 y E22 son controles importantes entre los empates: agregar un intento de contacto,
  paisaje o personajes no bastó para producir una ventaja clara. Un fragmento breve y
  cotidiano también puede ser un sueño válido. No convertiría esta lectura en contar verbos.

## Qué revisar del lote de preferencias

Las notas de los 14 pares entrenados ya valoraban acciones y percepciones situadas. No
faltaba por completo ese criterio. Entre los elegidos, diez dreams eran más largos que su
alternativa y cuatro más cortos, aun con la comparación de longitud acotada a una razón
de 1,5. Eso no explica causalmente la reducción de longitud posterior: los cuatro aciertos
también tienen ambas direcciones y la pérdida entrenaba dream y answer completos.

El lote se armó buscando el par de longitudes más próximas entre cuatro u ocho candidatos,
antes de revisar calidad. Ese filtro puede dejar fuera contrastes informativos, como una
escena breve frente a reiteración larga o escena frente a consigna. No contamos cuántos
pares así quedaron descartados; corresponde auditar el pool antes de atribuirle la regresión.

La pérdida bajó y los 14 márgenes de entrenamiento mejoraron también dentro de dream.
La señal llegó al bloque; estas observaciones no identifican si fallaron la selección,
la transferencia de preferencias, la intensidad de actualización u otra combinación.
No hay fundamento aquí para cambiar simultáneamente objetivo, modelo, plantilla y lote.

## Próximo ensayo propuesto, todavía sin ejecutar

**Primero preparar y auditar mejores contrastes desde el checkpoint 32.** No continuar
desde el piloto fallido ni reutilizar estos cuatro aciertos como ejemplos de entrenamiento.

1. Auditar los 108 candidatos ya guardados para medir qué contrastes dejó fuera la selección
   por longitud. Es diagnóstico sobre entrenamiento; no modifica las etiquetas congeladas.
2. Preparar 24 pedidos nuevos de la partición de entrenamiento, sin repetir los 24 anteriores
   ni cruzar personas de validación/prueba. Generar cuatro candidatos del 32 por pedido.
   Revisar calidad antes de elegir el contraste, dejando visibles las diferencias de longitud
   para controlarlas, sin convertir la longitud en el criterio de elección.
3. Buscar contrastes reales entre experiencia situada y comentario genérico, entre escena y
   consigna, y entre escena conservada y reiteración. Incluir fragmentos cotidianos, quietud,
   cambios de perspectiva y discontinuidad temporal. No fabricar una versión mala ni forzar
   una preferencia cuando las dos sean válidas. Si faltan contrastes claros, informarlo antes
   de entrenar; no rellenar con empates.
4. Documentar cuántas preferencias favorecen al texto más largo o más corto y los intervalos
   de longitud cubiertos. Buscar representación de ambas direcciones, sin invertir etiquetas
   para alcanzar una cuota. Revisar motivos centrales por separado: la mejora aparente de E24
   muestra que calidad y cumplimiento pueden divergir. No agregar penalización por copia ni
   exigir que dream imite la referencia humana.
5. Congelar lote, notas y protocolo antes de actualizar. Para estudiar la selección de datos,
   conservar inicialmente DPO, plantilla e hiperparámetros de v20 y una sola pasada desde el
   32, con hasta 24 pares claros. Informar la diferencia de cantidad de pares y actualizaciones:
   sería un ensayo exploratorio del nuevo lote, no una comparación causal que aísle cada factor.
   Una prueba de menor intensidad de actualización quedaría para un experimento separado.
6. Preparar antes de entrenar 24 pedidos de desarrollo nuevos con cuatro muestras por modelo
   y pedido. Los doce de v20 quedan como diagnóstico ya observado, no como nueva confirmación.
   Congelar preferencias sin mostrar modelo ni un orden alternado deducible. Informar calidad
   general, cumplimiento de motivos y variedad por separado; comparar por pedido, sin tratar
   cuatro muestras del mismo pedido como cuatro observaciones independientes.

El resultado buscado es **aumentar la frecuencia de experiencias de sueño reconocibles en
pedidos nuevos**, manteniendo variedad y motivos, y no sólo obtener algunos textos mejores
o bajar la pérdida. Un resultado pequeño o incierto conserva al 32; no se decide prolongar
la corrida mirando ejemplos favorables. Antes de ejecutar se debe fijar el criterio numérico
de promoción y el número efectivo de pares, usando la factibilidad del lote y sin consultar
la evaluación nueva. La revisión del mismo asistente seguirá sin ser validación humana independiente.

## Trazabilidad

Se leyeron los 24 pares completos de `results/dream-v20/evaluation/quality-packet.json`,
la clave `quality-key.json` y las notas originales `quality-annotations.json`, además de
los pares y notas de `results/dream-v20/preferences/`. No se cambiaron esos archivos ni
sus congelaciones. Esta revisión es posterior y exploratoria; no genera métricas nuevas
de evaluación ni evidencia de experiencia subjetiva del modelo.

[Método y resultado original](preferencias-v20.md).
