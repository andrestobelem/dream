# Preparación del entrenamiento natural con longitud objetivo

Estado: **preparado para revisión, no ejecutado**. Solicitud del usuario: mostrar los datos
antes de entrenar e indicar el tamaño del relato esperado en palabras.

El lote local revisable está en
`results/dream-v6/preparation/revision-antes-de-entrenar.md` y su versión estructurada en
`training-pack.json`, en ese mismo directorio. Incluye las doce consignas exactas, prompts
completos, referencias humanas separadas, procedencia y longitudes.

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
`f080397c25cdc4a84ae37c348bfeb1969c64db8c221f679d3ac81c29fbef6697`.

El paquete propone hasta 900 segundos adicionales de ejecución supervisada; ese presupuesto
no está aprobado ni se consumió durante la preparación. Se conservarán los consumos y
resultados anteriores. La recompensa pareada y el nuevo flujo deben conectarse a este lote
antes de ejecutar; el comando histórico de smoke todavía usa el vecino del banco y no es
el comando de este ensayo. No se ejecutará como si ya implementara la nueva recompensa.
