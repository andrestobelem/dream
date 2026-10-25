# Validación técnica del piloto DreamBank

25 de octubre de 2026. Es una comprobación operativa de un piloto, sin evidencia de mejora
ni validación de una medida universal de oniricidad.

## Protocolo

- Generador: `Qwen/Qwen3-1.7B-Base`, revisión `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Extractor: `mlx-community/Qwen3-8B-4bit`, revisión `545dc4251c05440727734bcd94334791f6ab0192`.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revisión `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Sólo D. Semilla 11; tres pasos GRPO, cuatro candidatos por grupo, LoRA rango 8.
- Prompt sin ejemplo resuelto. En el piloto actual termina con `<dream>` abierto:
  esa apertura pertenece a la entrada; el contenido, los cierres y el relato son muestreados.
- El score depende sólo de `<answer>` y del banco de entrenamiento. Se entrena con todos
  los tokens muestreados, sin reescribir ni recodificar sus IDs. No se puntúa el contenido
  de `<dream>` ni la fidelidad del relato respecto de él.

## Resultados comprobados

| Comprobación | Resultado |
| --- | --- |
| Corpus | 1.000/100/200 relatos; 30/6/7 grupos separados |
| Referencia exacta en el banco | Recompensa aproximadamente 1 |
| Pasos GRPO completados | 3 |
| Candidatos con formato válido | 4 de 12 |
| Grupos con recompensa variable | Pasos 2 y 3 |
| Gradientes distintos de cero | Pasos 2 y 3: normas 2,722 y 2,858 |
| Adaptadores modificados | Sí |
| Modelo base intacto | Sí |
| Recarga idéntica de adaptadores | Sí |
| Continuidad desde paso 2 | Diez comprobaciones aprobadas |
| Evaluación reservada antes/después | 0 de 2 formatos válidos en ambas |

Los fallos de formato reciben cero por estructura. En esa evaluación no hay relatos válidos
para comparar semánticamente: no se debe interpretar el cero como una mala puntuación onírica.
Tampoco se infiere mejora a partir de los cuatro candidatos válidos de entrenamiento.

La repetición desde el checkpoint intermedio reprodujo identidad, paso, adaptadores,
optimizador, estados aleatorios MLX/Python/NumPy, tokens acumulados y salidas finales.
El modelo base también permaneció intacto. Se ejecutó una actualización adicional de
verificación; no se contó como un cuarto paso del piloto principal.

## Ensayos anteriores conservados

El primer intento se interrumpió por tiempo durante la preparación de LoRA, sin actualizar
pesos. Tras ampliar el presupuesto se completó ese ensayo sin prefijo: sus doce candidatos
fallaron el formato, las ventajas y gradientes fueron cero y los adaptadores no cambiaron.
Se conserva en `results/dream-v5/dream/smoke-no-prefix-20261025/`.

El piloto con prefijo se guarda separado en `results/dream-v5/dream/smoke/`. No se mezclan
sus evaluaciones para atribuir mejoras al entrenamiento: el cambio de prompt también afecta
la generación. En particular, el formato válido de una de tres pruebas sin prefijo no es
la línea de base del piloto con prefijo.

## Revisión asistida de semillas

El asistente revisó las cinco consignas usadas frente a sus referencias, sin modificar las
salidas ni presentarlas como anotaciones humanas independientes:

| ID abreviado | Observación de desarrollo |
| --- | --- |
| `5495380bdc1a` | Conserva familia, lugares y fantasma; omite miedo y varios detalles. |
| `eb8d8d1a504c` | Conserva participantes, lugar y sustancias; omite malestar y hay una frase de acción poco clara. |
| `d575635dc5a1` | Conserva sándwich y sensaciones básicas; pierde ingredientes y la distinción de sabores. |
| `863b51027dc7` | Conserva elementos del escenario; diluye la relación entre arañas, huevos, luz y orientación. |
| `2013f9a8352e` | Conserva familia y lugar; omite relaciones y buena parte de la trayectoria emocional. |

Estas omisiones respaldan estudiar semillas más ricas. La cantidad adecuada sigue abierta:
no se ejecutó una comparación de tres niveles ni se estableció una cota definitiva.

El informe también registra un diagnóstico respecto del relato de origen, separado de la
recompensa al banco. Para los cuatro candidatos válidos de entrenamiento:

| Paso / candidato | Coseno al relato de origen | Coseno máximo al banco |
| --- | ---: | ---: |
| 2 / 1 | 0,3615 | 0,5200 |
| 2 / 2 | 0,4910 | 0,5323 |
| 3 / 0 | 0,6876 | 0,6876 |
| 3 / 3 | 0,5360 | 0,5360 |

Se usaron vectores ya guardados, sin nueva inferencia ni cambio de recompensa. Un candidato
puede acercarse más a otra referencia del banco; no se lo penaliza por eso. Esta muestra no
basta para concluir qué riqueza de semillas conviene.

## Artefactos y procedencia

Todos los archivos siguientes son locales e ignorados por Git; los relatos y los pesos
no se publican con este documento:

- `results/dream-v5/dream/data/manifest.json`: checksum, agrupaciones y particiones.
- `results/dream-v5/dream/seeds/cache/`: extracción cruda, selección e identidades.
- `results/dream-v5/dream/embeddings/diagnostic.json`: banco y controles.
- `results/dream-v5/dream/smoke/run.json`: identidad `2dc8a90f2dfc58f92b6a484e5f5ff0b0ebe3cc67fc50c6c2dc49ef6d42c8e9a6`.
- `results/dream-v5/dream/smoke/checkpoints/step-000003/`: adaptadores, optimizador y RNG.
- `training-validation.json`, `reload-validation.json`, `continuity-validation.json` en ese piloto.
- `results/dream-v5/dream/report/`: informe y diagnóstico respecto del relato de origen.
- `results/dream-v5/budget.json`: todas las ejecuciones, incluidas fallidas y de continuidad.

El piloto con prefijo ejecutó la copia inmutable de código con hash
`f10b3960b4a77fdcd3dcdc31a7d5415b6a56cd27078a161f134a24967e3ea342`.
La continuidad usó la copia `b1bde0cde46bb0331d242d39ea86f0e7602bcd416b20269c6605a02c676a7923`:
la identidad de la implementación de entrenamiento siguió coincidiendo. Las copias contienen
código completo, lock de dependencias y configuración efectiva.

El perfil acumuló 332,51 segundos de ejecución supervisada. La ampliación autorizada fue
hasta cinco minutos desde los 111,52 ya consumidos: se usaron unos 220,99 segundos adicionales.
No se reinició ningún contador. Las pruebas unitarias no cargan pesos reales del generador.

## Límites

No se probó entrenamiento largo ni eficacia generalizada. La estabilidad del formato, la riqueza
de las semillas, la diversidad de relatos y el alcance semántico de la recompensa siguen siendo
preguntas de investigación. Las categorías del extractor no son anotaciones humanas; el archivo
importado no aporta codificación temática universal por relato. DreamBank indica que sólo una
parte tiene codificación Hall/Van de Castle accesible mediante su circuito de investigación.

La posible memorización de un corpus público no se puede excluir. Una coincidencia alta es
favorable para esta recompensa y no se castiga automáticamente. Ninguna de estas comprobaciones
demuestra que el modelo tenga experiencia subjetiva.
