# Preparación del entrenamiento natural con longitud objetivo

Estado: **lote corto aprobado y entrenamiento completado**. [Resultados](resultado-entrenamiento-v6.md). Solicitud del usuario: mostrar los datos
antes de entrenar e indicar el tamaño del relato esperado en palabras.

El lote local revisable está en
`results/dream-v6/preparation-short-seeds/revision-antes-de-entrenar.md` y su versión estructurada en
`training-pack.json`, en ese mismo directorio. Incluye las doce consignas exactas, prompts
completos, referencias humanas separadas, procedencia y longitudes.

El usuario autorizó 15 minutos adicionales y después pidió detener la ejecución para acortar
las semillas. Después aprobó el lote corto; el ensayo completó sus 16 pasos y la evaluación
antes/después dentro de los 900 segundos autorizados.

Las doce semillas revisadas tienen entre 17 y 25 palabras. Criterio orientativo: unas 20–30
palabras, cinco a siete anclas distintivas y una o dos relaciones importantes, sin reconstruir
toda la secuencia. La indicación numérica de longitud del relato se agrega por separado.

Propuesta:

- Partir de Qwen3-1.7B-Base original, sin los adaptadores del ensayo anterior.
- Ocho relatos de entrenamiento, de ocho grupos; cuatro de validación, de otros cuatro grupos.
- Dieciséis pasos GRPO, cuatro candidatos por grupo, LoRA rango 8 y tasa 1e-5.
- Tres generaciones por caso de validación antes y después, con condiciones y semillas fijas.
- Hasta 1.024 tokens generados por candidato, como límite de recursos.
- Recompensa propuesta: `(1 + coseno(answer, referencia humana asociada)) / 2`.
  Es un cambio respecto de la recompensa anterior al vecino del banco: el objetivo pedido
  ahora es mejorar la cercanía al relato esperado de cada consigna.
- Puntuar sólo answer; asignar la recompensa a todos los tokens muestreados. Dream libre.
  Formato inválido: cero. Sin penalizaciones automáticas por copia ni por longitud.
- Añadir `Write the dream account in <answer> in approximately N words.` con N calculado
  a partir del relato humano. El número es orientativo, no un criterio de identidad literal.
- Registrar coseno al origen, proporción de formatos válidos, recompensa total y longitud.
  No comparar sólo promedios de candidatos válidos si cambia la tasa de fallos de formato.

Las longitudes de entrenamiento son 82, 125, 76, 176, 106, 155, 195 y 43 palabras.
Las de validación son 71, 241, 90 y 125. Se cuentan palabras separadas por espacios en el
texto normalizado del corpus. El caso de Green City queda en validación (71 palabras).
El conjunto final de 200 relatos no se usa para entrenar ni para elegir este ensayo.

Las consignas ricas fueron redactadas por el asistente a partir de las referencias, salvo
la de Green City aportada por el usuario. No se atribuyen al extractor local anterior.
La referencia humana completa se conserva para revisión y evaluación; no se envía al generador.

Hash del lote preparado:
`0560294db2bec293522d737580aba78edf63313851138e362841a93626038450`.

La propuesta de semillas largas, con hash
`f080397c25cdc4a84ae37c348bfeb1969c64db8c221f679d3ac81c29fbef6697`, se conserva
en su directorio original; no se sobreescribió.

El paquete contempla los 900 segundos adicionales autorizados. No se consumió presupuesto
de inferencia al acortar estas semillas; fueron revisadas por el asistente. Se conservarán los consumos y
resultados anteriores. La recompensa pareada se conectó a este lote y se ejecutó con `configs/dream-v6.yaml`.
El comando histórico de smoke sigue siendo otro protocolo; la corrida pareada usa experiment.
