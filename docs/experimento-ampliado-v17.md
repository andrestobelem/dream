# Experimento ampliado: una pasada por 128 relatos

**Completado. Esta pasada no mostró una mejora convincente de cercanía.** En 67 pares
válidos comunes, MiniLM subió 0,0050 y el juez independiente bajó 0,1642 puntos sobre 4.
Ambos intervalos exploratorios incluyen cero. El formato y la longitud empeoraron.
El checkpoint 32 sigue como referencia práctica; el 160 se conserva como resultado del ensayo.

La [revisión posterior de doce pares](revision-v18.md) matizó la confianza en Phi-4:
el asistente no corroboró todos sus descensos y encontró valoraciones altas sobre errores
importantes de agencia. Los números de este experimento se mantienen; sus scores no se
tratan como verdad de referencia. La revisión posterior no entrenó ni cambió la recompensa.

Diseño fijado antes de entrenar, a partir del lote revisado y aprobado por el usuario.
Origen: checkpoint 32; destino: paso 160, **128 actualizaciones nuevas**. Los 128 relatos
se visitan exactamente una vez, con cuatro candidatos por actualización: 512 rollouts.
No se evalúan checkpoints intermedios para decidir la duración.

## Datos y configuración

Se usa el lote de ocho relatos anteriores más 120 nuevos, balanceado entre las 30 personas
de entrenamiento. Sus prompts, referencias revisadas y tres recortes trazables no se cambian.
Hash aprobado de filas de entrenamiento/validación:
`d910fdda3577dfd3b0188face94a99ad6ac0cff09a92ecd85ed9f8a338e2453e`.
Todos los reservados quedan fuera de las actualizaciones por relato y por persona.
Los 24 relatos de validación conservan los mismos prompts y los 200 de prueba no se evalúan.

El límite se fija en **2.048 tokens** antes de ambas evaluaciones y del entrenamiento,
tras el diagnóstico v16. Se mantienen temperatura 0,8, top-p 0,95, LoRA rango 8/escala 16,
capas q/v, dropout cero, learning rate 1e-5, beta KL 0,1, grupo de cuatro y recompensa
pareada `(1 + coseno(answer, referencia)) / 2`, con cero por formato inválido. No se agregan
penalizaciones por copia ni por longitud. Dream sigue libre y no se puntúa su contenido.

Se restaura el checkpoint 32 completo, incluidos momentos del optimizador y aleatoriedad.
Una bifurcación bajo identidad nueva registra el cambio de lote y límite. La copia inicial
se guarda en `dream/paired/bootstrap/step-000032`, fuera de los logs de nuevas actualizaciones.
La identidad del origen y los hashes de sus cuatro archivos se conservan y verifican.
La comprobación real confirmó adaptadores, optimizador y estados aleatorios idénticos.

El orden usa la misma semilla 11 y el índice global del paso. Aunque comienza desplazado
32 posiciones dentro de la permutación de 128, el recorrido de los pasos 33–160 cubre
cada relato exactamente una vez. Los contadores de tokens y tiempo se conservan acumulados;
el informe separa los incrementos de esta pasada.

## Evaluación inicial fijada

Checkpoint 32, mismos 24 relatos de desarrollo, tres generaciones por relato, 2.048 tokens:
70/72 formatos válidos, cero truncamientos, coseno medio válido 0,6453 y score integral
0,7998. Las 72 secuencias coincidieron exactamente con la evaluación anterior de 1.024,
porque ninguna había sido truncada. No se utiliza esa coincidencia para omitir la nueva evaluación.

## Juez independiente y ciego

Se fija `mlx-community/Phi-4-mini-instruct-4bit`, revisión
`ac1c269cb4222a4e136a3d09edad301056c1f36a`, temperatura cero y hasta 256 tokens de evaluación.
Es una familia distinta del generador Qwen3-1.7B-Base y del extractor de semillas Qwen3-8B;
no es MiniLM. Se puntúan personajes, sucesos, relaciones y cercanía global, cada uno de 0 a 4.
El juez recibe únicamente la referencia humana y el contenido de answer, sin dream, semillas,
modelo, paso o fase. Las fases se mezclan mediante identificadores ciegos; la clave de origen
se conserva aparte y no se envía al juez. La recompensa GRPO no usa estas notas.

La rúbrica y seis controles sintéticos se fijaron antes de entrenar: identidad, paráfrasis,
texto no relacionado, inversión de actores, inversión del resultado e instrucción incrustada.
El criterio predefinido exige al menos cinco aciertos, incluyendo identidad e inyección.
El juez obtuvo **6/6** y sus pesos quedaron intactos. No se calibró con los 24 reservados.
El hash de rúbrica queda en la configuración y se rechazan cambios posteriores.

Es una evaluación textual automatizada, no validación humana independiente. La calidad
onírica queda separada y no forma parte de esta rúbrica. Los formatos inválidos se informan
por separado y no reciben una nota semántica inventada.

Desviación de procesamiento registrada después de comenzar la evaluación: una salida
contenía los cuatro puntajes y la explicación completos, seguidos de una copia adicional
del candidato que quedó truncada. Se agregó una normalización que acepta únicamente
los cinco campos requeridos completos al comienzo del objeto y conserva sus valores
literalmente. Rechaza campos faltantes, duplicados, ambiguos o truncados; no inventa
puntajes ni explicaciones. No cambió modelo, entrada, temperatura, rúbrica ni límite
de 256 tokens. Las salidas crudas y los errores se conservan; el informe cuenta los casos
normalizados por fase. Esta tolerancia del lector no fue parte del diseño previo.

## Comandos y recuperación

```sh
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-before-v17.yaml --results results/dream-v17/before
.venv/bin/python -m think_dream_lab reference-controls \
  --config configs/dream-expanded-v17.yaml --results results/dream-v17
.venv/bin/python -m think_dream_lab expanded-experiment \
  --config configs/dream-expanded-v17.yaml --results results/dream-v17
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint \
  --config configs/dream-eval-after-v17.yaml --results results/dream-v17/after
.venv/bin/python -m think_dream_lab reference-judge \
  --config configs/dream-expanded-v17.yaml --results results/dream-v17
.venv/bin/python -m think_dream_lab.expanded_report \
  --config configs/dream-expanded-v17.yaml --results results/dream-v17
```

Se mantienen las variables de Metal limitadas al proceso usadas en las corridas anteriores:
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`, `MLX_MAX_OPS_PER_BUFFER=10`, `MLX_MAX_MB_PER_BUFFER=20`.
Cada actualización guarda un checkpoint transaccional. El mismo comando de entrenamiento
reanuda desde el último completo, sin reiniciar el contador ni el optimizador. Los intentos
fallidos y los snapshots se conservan. La ventana operativa reanudable es de seis horas;
el usuario había levantado el límite de tiempo total.

Durante esta pasada, el control de memoria interrumpió el intento siguiente al paso 81.
Se reanudó desde ese checkpoint, restaurando optimizador y aleatoriedad; el paso 82 se
completó sin cambiar parámetros ni datos. Los 512 rollouts previstos corresponden a las
128 actualizaciones completas. Un intento interrumpido puede generar candidatos que se
descartan y se regeneran al reanudar; no cuenta como actualización adicional.

Los perfiles y el lote deben prepararse localmente antes de ejecutar esos comandos. El cache
de embeddings del entrenamiento se copia a la evaluación final y después se incorpora al
cache de auditoría; los vectores anteriores se preservan. Referencias y salidas completas,
claves de procedencia ciega y pesos permanecen fuera de Git.

## Resultados finales

Se completaron los pasos 33–160: 128 actualizaciones, una por cada relato aprobado,
con 512 candidatos usados en actualizaciones completas. Las 30 personas de entrenamiento
quedaron separadas de los reservados. Los 200 relatos de prueba no se evaluaron.

| Medida | Checkpoint 32 · antes | Checkpoint 160 · después |
| --- | ---: | ---: |
| MiniLM, mismos 67 pares válidos | 0,6435 | 0,6485 |
| Juez independiente, mismos 67 pares, 0–4 | 2,2537 | 2,0896 |
| Formato válido | 70/72 | 69/72 |
| Coseno, todas las respuestas válidas de cada fase | 0,6453 · n=70 | 0,6520 · n=69 |
| Score incluyendo cero por formato inválido | 0,7998 | 0,7916 |
| Truncamientos | 0 | 1 |
| Error relativo de longitud, sólo válidas | 0,3643 | 0,7443 |
| Palabras en answer, media válida | 102,9 | 208,2 |

El cambio de coseno en pares comunes fue **+0,0050**. Con igual peso por relato/persona,
la media fue +0,0046 y el intervalo exploratorio del 95% [-0,0082; +0,0182]. Catorce medias
por relato subieron y diez bajaron. El cambio del juez en pares comunes fue **−0,1642**;
la media por relato/persona fue −0,1597, intervalo [-0,3611; +0,0417]. Siete medias por relato
subieron, once bajaron y seis quedaron iguales. Son seis grupos y 20.000 réplicas por persona,
no 144 observaciones humanas independientes.

Las dimensiones del juez también bajaron en pares comunes: personajes/objetos/escenario
2,2388 → 2,0746; sucesos 2,2537 → 2,0597; relaciones 2,2388 → 2,0597. Esta discrepancia
con la pequeña subida de MiniLM impide tomar el coseno aislado como evidencia suficiente
de mejor correspondencia narrativa. Tampoco demuestra un deterioro general concluyente:
la muestra es de desarrollo, pequeña y los intervalos incluyen cero.

La recompensa de entrenamiento se conservó; estas notas del juez no ajustaron pesos.
El mayor largo de las respuestas tampoco recibió castigo, según la regla acordada.
Los 40 casos normalizados (19 antes, 21 después) conservaron exactamente los cinco campos
completos emitidos por el juez. Los demás casos se procesaron con el JSON original.

La auditoría recalculó desde vectores guardados los **512 scores de entrenamiento y 144
de evaluación**, sin nuevas inferencias del encoder. Verificó los pasos únicos, gradientes
finitos y efectivos, restauración inicial y recarga final exactas, pesos base y cuatro archivos
del origen intactos, mismos prompts/tokens/semillas y ausencia de reservados en actualizaciones.
La suite completa aprobó **166 pruebas**. Se entrenaron 168.954 tokens adicionales; las
actualizaciones completas consumieron 3.008,11 segundos, y ambos intentos de entrenamiento
sumaron 3.105,65 segundos incluyendo interrupción y recarga.

[Resultados numéricos](resultados-v17.json), [auditoría agregada](auditoria-v17.json).
Informe completo local: `results/dream-v17/dream/report/comparacion.md`; gráfico PNG/SVG
y detalle de los 24 relatos en la misma carpeta. Las referencias originales permanecen fuera de Git.

## Límites del diseño

Los 24 relatos ya informaron decisiones previas y siguen siendo validación de desarrollo.
Los intervalos por seis personas son exploratorios y no tratan las 144 generaciones como
personas independientes. Se informa tanto la comparación de pares válidos comunes como
formato y score integral. El juez y MiniLM son medidas distintas, ninguna es experiencia subjetiva.

Frente a los pilotos anteriores cambiaron el lote y el límite de generación. La comparación
antes/después controla el límite de 2.048, pero no demuestra por sí sola que una diferencia
frente a otros pilotos se deba exclusivamente a variedad de datos. La posible memorización
previa del corpus público sigue abierta. DreamBank: CC BY-NC-SA 4.0, dreambank.net. Código MIT.
