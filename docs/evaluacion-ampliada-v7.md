# Evaluación ampliada del checkpoint 16, sin entrenar

## Resultado

El checkpoint generalizó sobre todo la capacidad de completar el formato. La cercanía
semántica mostró un aumento pequeño y heterogéneo, menor al observado en los cuatro
casos del piloto. No se siguió entrenando después de esta evaluación.

| Medida | Modelo base | Checkpoint 16 |
| --- | ---: | ---: |
| Generaciones | 72 | 72 |
| Formato válido | 52/72 (72,2%) | 66/72 (91,7%) |
| Coseno medio, sólo respuestas válidas | 0,6219 · n=52 | 0,6330 · n=66 |
| Score medio incluyendo cero por formato inválido | 0,5857 | 0,7485 |
| Truncamientos | 11 | 2 |
| Error relativo medio de longitud, sólo válidas | 48,7% | 37,9% |

En los **48 pares válidos en ambos modelos**, el cambio medio de coseno fue **+0,0102**.
Trece de los 23 relatos que tuvieron al menos un par válido mejoraron en su media pareada;
diez empeoraron. El relato restante no produjo ningún par válido porque sus tres respuestas
base fallaron de formato. Esto impide interpretar la mejora de score de +0,1628 como una
mejora equivalente de cercanía: gran parte corresponde a respuestas que ahora sí se completan.

El remuestreo por persona da un cambio medio de score de +0,1628, con intervalo exploratorio
95% [0,0864; 0,2299]. Para coseno, promediando primero pares dentro de cada relato y luego
relatos dentro de cada persona, el cambio es +0,0221 y el intervalo [0,0017; 0,0471].
**Este último estima una media con igual peso por persona y relato, distinta de +0,0102**, que
da igual peso a cada par válido. No es el intervalo de esa media de 48 pares. Los pocos grupos,
los formatos inválidos y el encoder compartido con el entrenamiento limitan la interpretación;
no se afirma una mejora general ni una prueba independiente de calidad onírica.

La auditoría recalculó los 144 scores desde vectores guardados, sin nuevas inferencias.
Ambas evaluaciones terminaron con pesos intactos, ningún optimizador y cero actualizaciones;
los adaptadores cargados coinciden exactamente con el checkpoint original. Las 139 pruebas
pasaron. [Resultados numéricos](resultados-v7.json). El informe completo y su gráfico están
en `results/dream-v7-eval/dream/report/`; la auditoría está en `results/dream-v7-eval/audit.json`.

## Diseño fijado antes de generar

Se comparan Qwen3-1.7B-Base original y el checkpoint del paso 16 del ensayo v6.
No se ejecutan actualizaciones, no se crea optimizador y se comprueba que los parámetros
de ambos modelos y el archivo de adaptadores queden intactos.

La nueva cohorte contiene 24 relatos de validación: cuatro de cada uno de los seis grupos
de personas reservados. Se excluyen los cinco relatos de validación que ya habían pasado
por alguna evaluación del generador, incluidos los cuatro del piloto v6. La selección
es determinista y balanceada por grupo, sin consultar scores ni elegir por contenido.
Las personas de entrenamiento no entran en esta cohorte. Los 200 relatos de prueba permanecen aparte.

La selección quedó fijada con hash
`5191f1bcb2fb7a832b4d84e8b82b16f9a375af0db0b2235fff82c1c02f39fbe1`.
El lote final de prompts, con semillas revisadas antes de generar, tiene hash
`7447aeaa4c594ab628b923df6cbc2879ee32a4645c45a7eb204d8cb2148293ee`.

## Consignas y generación

Se conserva la plantilla natural, la apertura de dream en la entrada y la longitud
aproximada de answer calculada a partir del relato humano. La referencia completa no
forma parte del prompt. Las semillas tienen entre 19 y 25 palabras.

Qwen3-8B congelado extrajo anclas de los 24 relatos. El asistente revisó su redacción y
las relaciones distintivas contra la fuente antes de evaluar cualquiera de los dos modelos.
Se conservaron las extracciones crudas y la versión anterior a esa revisión. La revisión
no es una anotación humana independiente y no se usaron resultados del generador para ajustar semillas.

Tres generaciones por caso y por modelo: 72 de base y 72 de checkpoint. Temperatura 0,8,
top-p 0,95 y hasta 1.024 tokens generados, igual que en v6. Ahora la semilla se fija por
consigna **y por índice de muestra**; cada par arranca con la misma semilla sin depender
de cuánto duraron las muestras anteriores. Los cuatro casos anteriores no se mezclan en el resumen.

## Comprobaciones

El checkpoint se carga con el rango, escala y capas LoRA originales. Se verifica la identidad
de sus adaptadores y de todas las matrices base. LoRA cambia nombres internos, por ejemplo
`q_proj.weight` pasa a `q_proj.linear.weight`; se normalizan esos nombres sólo para comparar
huellas, sin modificar las matrices. La primera comprobación se detuvo antes de generar
por esa diferencia de nombres; se conservó el intento y se corrigió el comparador.

La evaluación rechaza llamadas a train o prepare_training. Al terminar cada modelo se
vuelve a comprobar su huella completa. El informe exige exactamente los mismos prompts,
tokens de entrada, referencias y semillas en los 72 pares.

## Métricas e incertidumbre

Se informan coseno con la referencia humana, formato, score incluyendo cero por formato
inválido, truncamientos y longitud. Se comparan por separado los pares válidos en ambos modelos.
Los intervalos exploratorios remuestrean seis grupos de personas completos, con 20.000
réplicas; no se consideran las 144 generaciones como observaciones humanas independientes.

El encoder es el mismo MiniLM congelado utilizado por la recompensa de entrenamiento.
Por lo tanto, el resultado evalúa generalización de esa cercanía semántica a nuevas
referencias; no es una medida independiente de calidad onírica. No se penalizan coincidencias
textuales ni se puede descartar memorización previa de un corpus público.

## Comandos y artefactos

```sh
.venv/bin/python -m think_dream_lab holdout-prepare --config configs/dream-eval-v7.yaml --results results/dream-v7-eval
.venv/bin/python -m think_dream_lab holdout-evaluate --variant base --config configs/dream-eval-v7.yaml --results results/dream-v7-eval
.venv/bin/python -m think_dream_lab holdout-evaluate --variant checkpoint --config configs/dream-eval-v7.yaml --results results/dream-v7-eval
.venv/bin/python -m think_dream_lab holdout-report --config configs/dream-eval-v7.yaml --results results/dream-v7-eval
```

Se mantienen los ajustes limitados al proceso de Metal documentados para v6. El usuario
levantó el límite anterior de 15 minutos. El perfil usa una ventana operativa reanudable
de dos horas, conservando el contador y los intentos anteriores; no es un límite total
autorizado del proyecto.

Los registros se guardan bajo `results/dream-v7-eval/dream/holdout/`, con copias inmutables
del código y configuración. Las generaciones completas, referencias, extracciones y pesos
quedan fuera de Git. Fuente DreamBank, CC BY-NC-SA 4.0, atribución a dreambank.net. Código MIT.
