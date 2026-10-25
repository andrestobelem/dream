# Experimento ampliado: una pasada por 128 relatos

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

Los perfiles y el lote deben prepararse localmente antes de ejecutar esos comandos. El cache
de embeddings del entrenamiento se copia a la evaluación final y después se incorpora al
cache de auditoría; los vectores anteriores se preservan. Referencias y salidas completas,
claves de procedencia ciega y pesos permanecen fuera de Git.

## Límites del diseño

Los 24 relatos ya informaron decisiones previas y siguen siendo validación de desarrollo.
Los intervalos por seis personas son exploratorios y no tratan las 144 generaciones como
personas independientes. Se informa tanto la comparación de pares válidos comunes como
formato y score integral. El juez y MiniLM son medidas distintas, ninguna es experiencia subjetiva.

Frente a los pilotos anteriores cambiaron el lote y el límite de generación. La comparación
antes/después controla el límite de 2.048, pero no demuestra por sí sola que una diferencia
frente a otros pilotos se deba exclusivamente a variedad de datos. La posible memorización
previa del corpus público sigue abierta. DreamBank: CC BY-NC-SA 4.0, dreambank.net. Código MIT.
