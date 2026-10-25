# Inspect AI + Inspect Flow para Dream

Usamos [Inspect AI](https://inspect.aisi.org.uk/tasks.html) para representar tareas mediante
dataset, solver y scorer, e [Inspect Flow](https://meridianlabs-ai.github.io/inspect_flow/)
para organizar sus condiciones, fases y corridas.

Esta primera integración **reproduce resultados locales ya guardados**. No genera relatos
nuevos, no recalifica con otro juez y no actualiza pesos. MLX/GRPO conserva el entrenamiento;
Inspect permite revisar, comparar y reutilizar sus evaluaciones con herramientas estándar.

## Instalar y ejecutar

Desde la raíz del repositorio:

```bash
uv sync --project evals --locked
uv run --project evals flow run evals/flow.py --dry-run
uv run --project evals flow run evals/flow.py
```

Versiones fijadas: Inspect AI 0.3.284 e Inspect Flow 0.14.0. Este proyecto usa su propio
`evals/.venv` y `evals/uv.lock`; no modifica el lock ni el entorno de entrenamiento.

La configuración descubre las corridas completas de `results/dream/smoke` y
`results/dream/runs/`. Construye sólo las celdas que tienen respuestas: A/B/C/D/neutral,
antes/después y rollouts, conservando la semilla. Excluye diagnósticos y corridas incompletas.

## Qué contiene cada tarea

- **Dataset:** consigna breve como entrada, referencia humana como target reservado e
  identificadores estables de problema, paso y candidato.
- **Solver:** recupera exactamente la salida original de MLX, sin llamar a `generate`.
- **Scorer:** conserva los seis componentes originales: oniricidad, alineación, fidelidad,
  calidad, formato y recompensa total. No crea scores nuevos ni corrige sus valores.
- **Procedencia:** identidad de corrida, semilla, versión del juez, tokens originales y
  un hash de las respuestas, referencias y código del adaptador.

El proveedor `mockllm/dream-replay` es únicamente el contexto local requerido por Inspect;
las respuestas son las guardadas, no salidas generadas por ese proveedor. La utilización
de modelos durante esta reproducción es cero. Los tokens de la ejecución original figuran
en `recorded_tokens`, separados del uso actual.

Las tareas se ejecutan secuencialmente. Los logs se guardan en `results/dream/inspect/`
y el índice de reutilización en `results/dream/inspect-store/`, ambos locales y fuera de Git.
Si cambian las respuestas, las referencias o el adaptador, cambia el snapshot y se invalida
la reutilización. Una celda inexistente no se convierte en un resultado de cero.

## Revisar resultados

Opcionalmente, abrir el visor local estándar:

```bash
uv run --project evals inspect view --log-dir results/dream/inspect
```

El visor muestra consigna, referencia, respuesta, scores y metadata. No se inicia un servidor
ni se publica contenido al ejecutar el Flow; el comando de visualización es una acción aparte.

Los rollouts de una misma corrida no son réplicas independientes de entrenamiento. Comparar
semillas y problemas emparejados sigue siendo necesario. El score neutral usa otra fórmula;
se comparan sus dimensiones comunes, no directamente su recompensa total.

## Verificación

```bash
uv run --project evals pytest -q -c evals/pyproject.toml evals/tests
```

Las pruebas verifican que la reproducción nunca invoque generación, preserve exactamente
los scores, invalide snapshots modificados, rechace scores inválidos y excluya celdas vacías
o corridas incompletas. La evaluación real de los modelos y cualquier cambio de rúbrica
continúan en el pipeline de Dream; aquí no se presentan scores guardados como evaluaciones nuevas.
