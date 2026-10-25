# Prueba de la plantilla con semillas naturales

25 de octubre de 2026. Se probó la redacción propuesta por el usuario con el modelo base,
sin entrenar ni cargar los adaptadores del piloto anterior.

La entrada tiene esta forma:

```text
A conversation between User and Assistant. The user requests a dream,
and the Assistant dreams it. The assistant first dreams in the mind
and then tells the user the dream. The dream and its account are
enclosed within <dream> </dream> and <answer> </answer> tags,
respectively, i.e.,
<dream> dream here </dream>
<answer> dream account here </answer>.

User: Dream about [semillas en lenguaje natural].
Assistant:
<dream>
```

El bloque de sueño queda abierto en la entrada. El modelo genera el contenido, el cierre y
el relato posterior. Los textos «dream here» y «dream account here» indican el esquema;
no hay un sueño resuelto ni una respuesta esperada en el prompt.

Las semillas naturales no se recortan al límite inicial de 24 palabras del extractor de
palabras clave. Pueden conservar relaciones, acciones y detalles. El límite de recursos
sigue siendo el de tokens del prompt; no se trunca una consigna silenciosamente.

Se usó la consigna rica aportada por el usuario: conductor de entregas en Green City,
arañas cuyos huevos iluminan y orientan, un hotel que parece cine, sus participantes y
las cosas dejadas al narrador y a Laura. El relato humano completo quedó fuera del prompt.
Es una referencia de validación; no integra el banco de recompensa.

## Resultado

Tres candidatos, temperatura 0,8, hasta 768 tokens generados, semilla de evaluación 4242:

| Candidato | Formato | Observación | Coseno al relato humano | Coseno máximo al banco |
| --- | --- | --- | ---: | ---: |
| 1 | Válido | Interpretación de símbolos y consejos, no un relato de sueño | 0,148 | 0,568 |
| 2 | Incompleto | Alcanzó el límite de tokens | — | — |
| 3 | Válido | Escena narrada; la respuesta final resume en tercera persona | 0,600 | 0,581 |

El modelo base permaneció intacto. No se realizaron actualizaciones de pesos.

La cercanía al relato de origen distingue esos dos textos más que el vecino más próximo
del banco. Es un diagnóstico de esta muestra, sin cambio de recompensa y sin prueba de
superioridad estadística de la plantilla o de la riqueza de las semillas. No se compararon
varios niveles de detalle bajo condiciones controladas.

El piloto previo con actualizaciones y continuidad usó otra plantilla. Sus resultados
no se atribuyen a esta nueva redacción. Las copias inmutables de código y configuración
conservan lo ejecutado en cada caso.

## Reproducción y artefactos locales

```sh
.venv/bin/python -m think_dream_lab prompt-check \
  --brief-file results/dream-v5/dream/prompt-probes/user-brief.json
```

Ese archivo JSON contiene `brief` y `reference_id`; no el relato humano. El comando
admite entre uno y cuatro candidatos mediante `--limit` y usa el presupuesto acumulado.
Una prueba idéntica ya completada se recupera de su resultado guardado.

- Resultado: `results/dream-v5/dream/prompt-probes/3e314ae7110d1866/result.json`.
- Prompt y generaciones: `report.md` en ese mismo directorio.
- La referencia original sólo se lee desde el corpus para calcular la cercanía por separado.
- Consumo acumulado del perfil después de esta prueba: 364,38 segundos de un techo de 411.

Los relatos originales y las salidas completas permanecen fuera de Git. La observación
cualitativa de esta muestra corresponde al asistente, no a validación humana independiente.
