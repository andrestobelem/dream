# V22: comprobar el cierre sin entrenar más

Ensayo en ejecución, 25 de octubre de 2026. Se comparan el checkpoint 32 y el candidato
v21 de 19 actualizaciones, con **la misma envoltura de inferencia**. No se actualizan
pesos. Los 200 relatos de prueba siguen sin usarse.

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
