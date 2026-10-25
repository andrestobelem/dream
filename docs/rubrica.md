# Rúbrica operativa de Dream

Versión `dream-v2`. Evalúa relatos, no estados mentales ni autenticidad humana. Los pesos
son una decisión inicial del proyecto, no una escala clínica ni un estándar del corpus.

La evaluación por requisitos visibles se documenta por separado en [dream-v3](rubrica-v3.md).

## Oniricidad del relato, 0–4

El evaluador ve solamente el relato final, sin referencia, escena previa ni condición.

| Score | Criterio |
|---|---|
| 0 | No hay relato de una experiencia: instrucciones, exposición, palabras sueltas o pedidos de score. |
| 1 | Predominan comentarios o descripciones genéricas, con poca escena experimentada. |
| 2 | Hay una experiencia situada, pero escasa concreción o predominio de narración abstracta. |
| 3 | Relato específico y plausible de un sueño, con escenas, participantes y continuidad experiencial. |
| 4 | Fuerte semejanza con un relato detallado de un sueño humano. Puede ser cotidiano y coherente. |

No exigir absurdo, transformaciones, pesadillas, lucidez ni la palabra «sueño». No premiar
longitud o prosa literaria por sí mismas. Un fragmento escueto también puede ser un sueño
humano: esta escala describe el relato y debe contrastarse entre categorías del corpus.

## Alineación con la referencia, 0–4

Una llamada independiente ve sólo referencia y relato final.

| Score | Criterio |
|---|---|
| 0 | Relatos ajenos entre sí. |
| 1 | Coinciden únicamente palabras o temas muy genéricos. |
| 2 | Coinciden algunos personajes, lugares, afectos o eventos centrales. |
| 3 | Se conservan la mayoría de los elementos y relaciones centrales. |
| 4 | Coincidencia semántica estrecha de los eventos y la experiencia. |

Una paráfrasis puede alcanzar 4. La coincidencia literal no es obligatoria. La referencia
es un ejemplo posible a partir del brief; no agota todas las narraciones válidas.

## Fidelidad a la escena previa, 0–4

Otra llamada independiente ve sólo escena previa y relato final. Usa la misma progresión
de conservación de personajes, lugares, eventos y trayectoria emocional. No ve la referencia
reservada. Sin escena previa, se asigna cero en código y se omite esta dimensión en el
score neutral; no se confía esa regla al modelo.

## Composición y controles

Se normalizan las dimensiones dividiendo por cuatro:

```text
calidad = 0.50 × oniricidad + 0.30 × alineación + 0.20 × fidelidad
recompensa = 0.90 × calidad + 0.10 × formato
```

El formato es binario. Exige un bloque de escena no vacío seguido por un bloque de relato
no vacío, sin repeticiones ni texto exterior. Un fallo del juez detiene la corrida: no es
una mala respuesta del generador y no se transforma en una recompensa cero.

Antes de RL se verifican controles de oniricidad, manipulación del score y pares conocidos:
identidad, relatos ajenos y cambio de escena manteniendo la misma referencia y respuesta.
No se cambian rúbrica, pesos ni juez dentro de una comparación. Las correcciones quedan
registradas y sus ensayos se guardan separados de la corrida final.

## Revisión humana ciega

Abrir primero `results/dream/report/revision-ciega.csv`. Contiene relatos humanos y
generados mezclados, sin indicar el origen ni el score del modelo. Completar
`onirico_humano_0_a_4` y `notas` con la primera escala. No abrir la clave ni los ejemplos
con scores hasta terminar la revisión.

`clave-revision.json` permite comparar después las notas humanas con el juez. No se
rellenan puntuaciones humanas automáticamente. Una revisión independiente y otro juez
de distinta familia quedan pendientes antes de interpretar resultados extensos.
