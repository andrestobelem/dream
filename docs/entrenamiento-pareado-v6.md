# Ensayo pareado con semillas cortas y longitud indicada

El usuario aprobó el lote de semillas cortas y 15 minutos adicionales. La ejecución utiliza
`configs/dream-v6.yaml` y conserva el presupuesto y los resultados históricos en sus directorios.

## Datos y objetivo

- Ocho referencias de entrenamiento y cuatro de validación de personas separadas.
- Semillas naturales de 17–25 palabras, revisadas contra la fuente; el relato completo queda
  reservado para calcular recompensa y evaluar. Las semillas fueron preparadas por el asistente.
- Longitud aproximada indicada sólo para answer: 82, 125, 76, 176, 106, 155, 195 y 43 palabras
  en entrenamiento; 71, 241, 90 y 125 en validación.
- Lote aprobado: `0560294db2bec293522d737580aba78edf63313851138e362841a93626038450`.
- Recompensa: `(1 + coseno(answer, referencia humana asociada)) / 2`, usando MiniLM congelado.
  El vecino más cercano del banco ya no determina esta recompensa.
- Se puntúa sólo answer; GRPO asigna la recompensa a todos los tokens muestreados, incluido
  el contenido libre de dream. Formato inválido: cero. Sin penalización por copia ni por longitud.

## Ejecución

Qwen3-1.7B-Base original, revisión fijada, sin adaptadores anteriores. LoRA rango 8, escala 16,
proyecciones q/v; GRPO con 16 pasos, cuatro candidatos por grupo, tasa 1e-5 y beta KL 0,1.
Temperatura 0,8, top-p 0,95, hasta 1.024 tokens generados. Se conserva cada actualización completa.

```sh
.venv/bin/python -m think_dream_lab experiment \
  --config configs/dream-v6.yaml --results results/dream-v6 --steps 16
```

Un ensayo incompleto se retoma con `--resume`, conservando la identidad y el contador.
Las evaluaciones ya terminadas se reutilizan; el modelo base se vuelve a comprobar antes
actualizar pesos. Las referencias humanas nunca forman parte de los prompts del actor.

El primer proceso completó la evaluación inicial y falló por un error de Metal antes del
primer checkpoint. Consumió 165,14 segundos, incluidos en los 900 autorizados. Se preservó su
manifiesto de error. El reintento utiliza un proceso GPU nuevo, la misma configuración de
entrenamiento y las mismas evaluaciones iniciales. No se reinició el presupuesto.

## Comparación

Se generan tres muestras por cada referencia reservada antes y después: doce por fase.
La semilla se fija por consigna. Los textos pueden divergir y consumir distintos trayectos
aleatorios al detenerse; no se presume que una posición de muestra sea idéntica al cambiar pesos.

El informe separa:

1. Coseno al relato humano entre las respuestas válidas, con su denominador.
2. Proporción de formatos válidos y truncamientos.
3. Recompensa conjunta de todos los candidatos, incluyendo cero por formato inválido.
4. Comparación de las posiciones válidas en ambas fases y resultados por consigna.
5. Longitud producida y desvío relativo respecto del tamaño indicado.

Una mejora de formato puede aumentar la recompensa sin mejorar el coseno de las respuestas
válidas. Cuatro relatos reservados no bastan para establecer eficacia generalizada. Los 200
relatos de prueba permanecen fuera de este ensayo de desarrollo.

Artefactos locales: `results/dream-v6/dream/paired/`, `results/dream-v6/budget.json` y
`results/dream-v6/dream/report/`. Las referencias originales y los pesos no se publican en Git.
Fuente DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT.
