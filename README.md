# Dream

Un experimento local para pedirle a un modelo que **sueñe dentro de `<dream>` y después
cuente lo soñado dentro de `<answer>`**, aprendiendo de relatos humanos de DreamBank.

Se usa **Qwen/Qwen3-1.7B-Base**, el modelo del primer piloto, con revisión fijada.
Sólo se trabaja con Dream: no hay comparación con `<think>` ni problemas de aritmética.

Otro LLM extraerá unas pocas semillas de cada relato: personajes, lugares, objetos, acciones
y emociones. El generador recibe esas semillas, sin el relato humano ni una trama resuelta.
El contenido de `<dream>` queda libre. El diseño puntúa `<answer>` por cercanía mediante
embeddings al corpus de entrenamiento y asigna esa recompensa a toda la secuencia.
La fórmula de recompensa todavía está en desarrollo; no afirmamos que la similitud semántica
sea por sí sola una medida validada de oniricidad.

Una salida muy similar a un relato humano es favorable: **no hay penalización automática por
coincidencias o copias**. Para estudiar generalización se reservan personas, semillas y relatos
que no participan en las recompensas de entrenamiento. La posible memorización del corpus
público es una limitación al interpretar resultados.

## Estado y uso

La importación real de DreamBank está implementada: 1.000 relatos de entrenamiento,
100 de validación y 200 de prueba, sobre 43 grupos separados. Se deduplican colecciones
completas y subconjuntos; las series colectivas sin identidad individual resuelta quedan fuera.
El registro público de agrupación está en [configs/dreambank-groups.json](configs/dreambank-groups.json).

Con el entorno `.venv` del proyecto y el archivo local `.cache/corpora/dreambank.tar.xz`:

```sh
.venv/bin/python -m think_dream_lab corpus
.venv/bin/pytest -q
```

El perfil predeterminado es [configs/dream-v5.yaml](configs/dream-v5.yaml).
Los datos y su manifiesto se guardan en `results/dream-v5/dream/data/`, fuera de Git.
La carga verifica el SHA-256 del archivo; la reutilización verifica configuración, registro
y contenido de las particiones. No se descargan ni ejecutan archivos HTML al importarlos.

La extracción local ya produjo semillas para un relato de cada partición; todavía hay que
revisar su calidad y ampliar la muestra. Se conserva la salida del extractor y se seleccionan
hasta ocho frases sin reescribirlas. Para repetir esta prueba, con su consumo supervisado:

```sh
.venv/bin/python -m think_dream_lab seeds --limit 1
```

Faltan la elección y comprobación de la recompensa por embeddings y su conexión con el
entrenamiento y la evaluación. El prompt mínimo está
implementado pero todavía no se probó con el modelo real. La CLI impide ejecutar el juez
histórico como si fuera el nuevo protocolo. El [plan actualizado](docs/plan.md) registra
lo terminado, las tareas pendientes y el presupuesto acumulado.

La infraestructura histórica de GRPO/LoRA y checkpoints se conserva para adaptar. Un piloto
anterior comprobó cinco actualizaciones y recarga, sin demostrar mejora en los sueños.
Las diez generaciones con formato completo corresponden al prompt anterior con ejemplo,
no al actual.

## Fuentes e historia

Fuente principal: [DreamBank](https://dreambank.net/). Se conservan los metadatos originales;
no se inventan categorías ni se presentan indicadores automáticos como etiquetas humanas.

El protocolo anterior de DreamViews y su juez queda archivado en la rama
`archive/dreamviews-judge-20261025`. Sus documentos de [rúbrica](docs/rubrica-v4.md)
y [validación técnica](docs/validacion-tecnica.md) son históricos.
El [site del proyecto](https://del-think-al-drean.andrestobelem.chatgpt.site/proyecto-dream)
todavía describe ese protocolo y debe actualizarse.

Código **MIT**. Datos DreamBank **CC BY-NC-SA 4.0**, atribución a dreambank.net;
la licencia de los datos no cambia por usar este repositorio. No se publican automáticamente
relatos, pesos ni resultados locales.
