# V22: comprobar el cierre sin entrenar más

Ensayo completado, 25 de octubre de 2026. Se compararon el checkpoint 32 y el candidato
v21 de 19 actualizaciones, con **la misma envoltura de inferencia**. Formato y contenido
finalizado: **96/96 muestras y 18/18 regresiones por sistema**, sin agotar límites.
La ventaja de calidad del candidato no quedó confirmada: 20 victorias frente a 14 y
62 empates; se conserva el 32. No se actualizaron pesos. Los 200 relatos de prueba siguen
sin usarse. [Informe local con todas las notas](http://127.0.0.1:62878/index.html).

## Cambio propuesto

El modelo genera libremente el contenido de dream y después el contenido de answer.
El programa proporciona las etiquetas que rodean ambos bloques y construye el contexto
de la segunda etapa a partir del dream generado. Se detiene cada etapa al encontrar una
etiqueta estructural, un nuevo turno User/Assistant/Human al principio de línea o EOS.
Si una etiqueta está fusionada con puntuación en un token, se conserva el contenido
decodificado anterior a la frontera y se guarda el token entero original en el registro.

**El cierre es trabajo del programa, no una habilidad aprendida por el modelo.** Se
conservan el texto bruto, los IDs muestreados, la razón de parada y todo sufijo separado.
Se recodifica el contexto canónico de answer; este procedimiento es sólo de inferencia,
no reutiliza esas trayectorias como si fueran las continuaciones originales de DPO.
No se reescribe el contenido ni se inventa un final narrativo.

Cada etapa tiene un máximo de 1.024 tokens: hasta 2.048 tokens de contenido por muestra.
Un cuerpo vacío o una etapa que alcanza el límite sin frontera **cuentan como fallo**,
aunque las etiquetas añadidas por el programa sean sintácticamente válidas. No hay reintentos.
Así se separan sintaxis por construcción y finalización efectiva de las dos etapas.

## Prueba fijada antes de mirar los nuevos textos

- Veinticuatro pedidos nuevos escritos por el asistente, cuatro muestras por modelo:
  96 pares, 192 muestras. Ninguno proviene de relatos reservados.
- Misma plantilla y tamaño orientativo de cien palabras para answer, temperatura 0,8,
  top-p 0,95. Semillas 74242, 74343, 74444, 74545 para dream y semilla +1 para answer.
- Misma envoltura en ambos checkpoints. Etiquetas de calidad, motivos y variedad con
  procedencia oculta mediante claves privadas, congeladas antes de consultarlas.
- Las 18 condiciones de dependencia conocidas se vuelven a generar **para ambos modelos**
  con la envoltura y los prefijos y semillas originales. Son regresión conocida, no evidencia
  nueva independiente. Se mantienen límite 1.024, hechos completos y formato separados.
- Se conservan los umbrales de v21: ventaja de calidad ≥0,15 y límite inferior exploratorio
  >0; finalización/formato en al menos 95/96 y sin bajar del 32; motivos y variedad sin
  retroceso según los controles fijados; hechos completos de dependencia ≥16/18 y formato
  18/18. El formato efectivo exige cuerpos no vacíos y sin agotamiento de presupuesto.
- Se informa por separado cuántos cierres del delimitador esperado produjo el modelo,
  cuántas salidas terminaron por EOS u otra frontera y cuántas agotaron el presupuesto.
  Un éxito de la envoltura no se presenta como una mejora del cierre aprendido.

[Protocolo completo](envoltura-v22-protocolo.json). Rúbrica de calidad, motivos, variedad
y hechos: [la misma de v21](preferencias-v21-rubrica.md). Criterio numérico sin cambios:
`experiments/criteria_preferences_v21.py`.

## Diagnóstico del fallo conocido

Se reprodujo la separación sobre la salida P11 ya guardada de v21, sin regenerarla ni
modificar pesos. La primera frontera no es User: antes aparece una etiqueta answer dentro
de una instrucción repetida. La envoltura detiene ahí, conservando el hecho de la moneda
y evitando los turnos nuevos; también queda una cola textual «Write the dream account in».
No se borró esa prosa para fingir un relato perfecto. Es diagnóstico del caso conocido,
no la evaluación nueva ni evidencia suficiente para elegir el candidato.

## Límites

La comparación evalúa dos sistemas con la nueva generación por etapas. Cambian los pedidos,
las semillas, el contexto de answer y el mecanismo de cierre respecto de v21; no permite
atribuir una diferencia entre rondas exclusivamente a la envoltura. Tampoco mide fidelidad
a un único relato humano esperado, porque estos pedidos nuevos no tienen esa referencia.

El mismo asistente revisa los textos; no es validación humana independiente. La envoltura
detiene marcadores reservados también si aparecen en una cita dentro de una narración:
se registrarán las fronteras observadas. La intención es controlar la estructura del mensaje,
manteniendo libre el contenido de sueño, no demostrar experiencia subjetiva.

## Resultado del protocolo fijado

| Medida | 32 + envoltura | Candidato v21 + envoltura |
| --- | ---: | ---: |
| Calidad, preferencias a favor /96 | 14 | 20 |
| Empates de calidad compartidos | 62 | 62 |
| Envoltura válida /96 | 96 | 96 |
| Contenido finalizado y formato /96 | 96 | 96 |
| Muestras con límite agotado | 0 | 0 |
| Desvío literal importante de motivos /96 | 42 | 49 |
| Preferencias de variedad /24 | 3 | 11 |
| Empates de variedad compartidos | 10 | 10 |
| Grupos con menos de dos episodios utilizables | 0 | 0 |
| Hechos completos de dependencia /18 | 16 | 17 |
| Contenido finalizado y formato de dependencia /18 | 18 | 18 |
| Palabras en dream, media | 83,2 | 110,4 |
| Palabras en answer, media | 67,9 | 80,3 |

Ventaja neta de calidad del candidato **+0,0625**, intervalo exploratorio por pedido
**[−0,0625; +0,1875]**. No alcanza ≥0,15 ni el límite inferior >0. El resultado es una
ventaja incierta en este desarrollo, no prueba de igualdad ni de deterioro causado por
la envoltura. Los pedidos y semillas son nuevos respecto de v21 y la generación cambió;
no se aísla causalmente el efecto del controlador entre rondas.

### Qué hizo el programa y qué emitió el modelo

En desarrollo, ambos modelos emitieron el cierre esperado de dream en 95/96 casos. El
32 terminó la otra etapa dream por EOS; el candidato avanzó directamente a una etiqueta
answer. El programa completó la estructura en ambos. Los dos emitieron el cierre esperado
de answer en 96/96. Por tanto, el formato observado no se obtiene sólo de un cierre
aprendido perfecto; el sistema se encarga de la envoltura.

En dependencia, el 32 emitió cierre de answer en 18/18 y el candidato en 17/18. La respuesta
restante terminó al aparecer otra etiqueta answer dentro de una instrucción repetida,
antes de inventar turnos nuevos. Conserva la moneda escondida por el narrador y una cola
textual «Write the dream account in». Esa cola se muestra sin limpiar: controlar etiquetas
y turnos no garantiza que toda la prosa sea un relato impecable.

Se reutilizaron **35 etiquetas congeladas por coincidencia textual exacta** y se revisó
la respuesta distinta. No se afirma que sean 36 juicios nuevos independientes: son
regresiones conocidas, los textos previos pueden reconocerse y las nuevas claves no se
abrieron antes de congelar. La única respuesta distinta se anotó como narrador, sin borrar
la cola. Se comprobaron prefijos y semillas iguales a ambos lados.

### Decisión y orientación conceptual recibida

La selección formal es **keep32**: el candidato no pasa los dos requisitos de calidad y
el control literal de motivos. Aunque formato, dependencia y variedad sí pasan, no se
promueve otro checkpoint ni se activa el controlador nuevo por defecto en la selección
guardada. La envoltura queda implementada y probada para uso experimental, sin tocar
los generadores históricos ni los pesos.

Durante la revisión llegó la orientación de permitir condensaciones y transformaciones
sin exigir literalidad. Se registró en [transformaciones oníricas](transformaciones-oniricas.md),
sin modificar notas anteriores, umbrales o ganador. Los 42 y 49 desvíos del control viejo
**no son automáticamente sueños de baja calidad**: esa medida no distingue elaboración
admisible de ausencia de material. Incluso sin usar ese control, el criterio primario
de calidad de esta ronda no alcanza lo fijado; no se recalculó una promoción alternativa.

[Resultados agregados](envoltura-v22-resultados.json), [auditoría](envoltura-v22-auditoria.json).
Se comprobaron huellas de ambos adaptadores, archivos del origen y parámetros invariantes
durante generación. **Cero actualizaciones, cero reintentos, cero relatos de prueba usados.**
Validación: 197 pruebas aprobadas y una omitida por Torch opcional ausente; Ruff y la auditoría
real de muestras, claves, congelaciones y criterio aprobaron. Los textos completos quedan locales.
