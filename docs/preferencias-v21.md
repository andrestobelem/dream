# Preferencias v21: mejora de calidad, sin promoción por fallo de formato

Ensayo completado el 25 de octubre de 2026: **19 actualizaciones desde el 32**. En la
revisión de 96 comparaciones nuevas, el piloto ganó 30, el 32 ganó 11 y hubo 55 empates.
Superó el criterio de calidad, pero falló formato en una prueba de dependencia:
**17/18 en vez de 18/18**. Se conserva el 32 según el protocolo previo; el nuevo queda
guardado como candidato experimental. [Informe local completo](http://127.0.0.1:62877/report.html).

La preparación produjo 19 pares claros y cinco empates fuera del entrenamiento. Se
generaron y revisaron cuatro candidatos para cada uno de 24 pedidos nuevos, provenientes
de 24 personas de entrenamiento. Las secciones de diseño siguientes conservan lo fijado
antes de ejecutar; el resultado está al final.

Vista local con los 24 prompts exactos, los 96 dreams y answers completos y las notas de
selección: [abrir el lote](http://127.0.0.1:62877/index.html). Las cuatro alternativas se
mantienen visibles; no se muestran sólo las ganadoras. Son preferencias del asistente,
no validación humana independiente. La fuente 32 era conocida; los índices de muestreo
no se mostraron durante la lectura y las notas se congelaron antes de consultar su clave.

## Con qué se entrenaría

- **Origen:** Qwen/Qwen3-1.7B-Base, revisión `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`,
  con los adaptadores exactos del checkpoint 32. No se continúa desde el piloto DPO fallido.
- **Datos:** una preferencia entre dos continuaciones reales por cada uno de los 19 pedidos
  elegidos. Las otras dos alternativas se conservan para inspección. Los cinco pedidos
  empatados quedan fuera. Hay tres preferencias de confianza alta y dieciséis medias.
- **Selección:** se revisaron primero las cuatro alternativas por calidad de experiencia.
  No se eligió primero el par de longitudes más próximas. Once dreams preferidos son más
  largos que el rechazado y ocho más cortos; no se forzaron ni invirtieron etiquetas para
  obtener esa distribución.
- **Actualizaciones:** una sola pasada, 19 actualizaciones DPO; beta 0,1, learning rate
  5e-6, AdamW nuevo, weight decay cero y recorte de gradiente a norma 1. LoRA heredado:
  rango 8, escala 16, dropout cero, proyecciones q/v en las mismas capas del origen.
- **Señal:** probabilidad de la continuación completa, dream y answer; prompt y apertura
  prellenada sin pérdida. Preferencia revisada sobre dream. Referencia estacionaria: el 32.
  Sin penalización automática por copia ni longitud y sin una pérdida nueva por segmentos.

Los 96 candidatos conservaron formato y ninguno se truncó. Las dos trayectorias de los 19
pares elegidos entran completas en 1.536 tokens, incluyendo prompt: no se recodificaron,
recortaron ni reescribieron. El límite es operativo, no una regla de calidad.

Los pedidos no repiten los 24 de v20 ni los ocho originales. Se excluyeron también las
referencias humanas de calibración de v19. Se consultaron las identidades de las personas
reservadas sólo para comprobar separación; ningún relato reservado entró en generación,
selección o evaluación. Los 200 relatos de prueba siguen sin usarse.

## Plantilla y semillas

Se conserva la plantilla de conversación con `<dream>` y `<answer>`, una semilla natural
breve de 20 a 30 palabras en este lote y el tamaño orientativo del relato humano en palabras
para **answer**. El modelo no recibe el relato humano completo ni se le fija la longitud
de dream. El tamaño orientativo heredado varía entre 41 y 326 palabras.

Ejemplo real del pedido N04, con answer orientado a 41 palabras:

```text
User: Dream about a man chasing someone who is somehow you, running downstairs to your room, hiding under a sofa, and being caught.
Assistant:
<dream>
```

En ese caso se prefirió el candidato D, que narra esconderse, ser atrapado e intentar
escapar, frente a C, casi una reproducción de la consigna. Los textos exactos están en
la vista local. Hay también contrastes entre escena y explicación genérica, o entre
experiencia y reiteración emocional. No todos son contrastes puros: N03, por ejemplo,
conserva mejor los roles además de reducir reiteración; las notas lo declaran.

No se rechazaron fragmentos sólo por ser breves, quietos, cotidianos, discontinuos o
físicamente imposibles. N07, N08, N11, N16 y N22 quedaron como empates. En N19 aparece
una alternativa en chino: se revisó su escena y no se descartó por cambiar de idioma.

## Evaluación fijada

**24 pedidos nuevos de desarrollo, escritos antes de generar los candidatos de entrenamiento,
con cuatro muestras por modelo y pedido: 96 comparaciones antes/después.** Ninguno se usará
para actualizar pesos. Los doce pedidos observados en v20 quedan como diagnóstico histórico,
no se reutilizan como confirmación nueva. Sus prompts, semillas y reglas se congelaron
antes de generar las 192 continuaciones de evaluación, que ahora están completas.

Se comparará el 32 exacto con el único checkpoint final de la pasada. Semillas 54242,
54343, 54444 y 54545; temperatura 0,8, top-p 0,95, sin top-k ni min-p, límite total de
2.048 tokens y answer orientado a cien palabras. Misma configuración a ambos lados.

La calidad se revisará por pares con procedencia oculta, empates y dudas permitidos,
congelando notas antes de revelar claves. El orden de modelos deberá sortearse por caso,
sin alternancia deducible. La variedad tendrá una revisión aparte de los dos grupos de
cuatro muestras por pedido. Se informará el cegado realmente logrado, sin prometer
independencia del revisor que también prepara el experimento.

No hay un único relato humano esperado para estos pedidos nuevos: la métrica primaria es
calidad de dream. Cumplir motivos se revisa por separado. No se calculará una cercanía a
una referencia inexistente ni se mezclarán estas cifras con los cosenos históricos de v17.

## Criterio de mejora, fijado antes de cualquier actualización

Todos los requisitos siguientes son necesarios para considerar al nuevo checkpoint un
candidato provisional. Si falla uno, falta revisión o no hay evidencia suficiente de
calidad, se conserva el 32:

1. **Calidad:** ventaja neta media ≥0,15, con +1 si gana el nuevo, −1 si gana el 32 y cero
   para empate o duda. En 96 comparaciones exige al menos quince victorias netas. Además,
   el límite inferior del intervalo exploratorio del 95% debe ser estrictamente positivo.
   Se promedia primero dentro de cada pedido y luego entre los 24 pedidos; el bootstrap
   remuestrea 24 pedidos completos, 20.000 veces, semilla 20261031. No se eliminan fallos
   ni se tratan las cuatro muestras de un pedido como independientes.
2. **Formato:** al menos 95/96 salidas válidas y no menos que el 32.
3. **Motivos:** no aumentar el número de muestras con una omisión o contradicción importante
   de personajes, objetos o relaciones centrales del pedido. Se revisan las 96 por modelo,
   incluida cualquier mera repetición del pedido. La calidad se puntúa separadamente.
4. **Variedad:** sobre los 24 pedidos, al menos tantas preferencias por el nuevo grupo como
   por el del 32; no aumentar los grupos con menos de dos episodios utilizables. La variedad
   trata de episodios y relaciones diferentes, no de distancia léxica ni rareza. Se informan
   los empates y las dudas; este control pequeño no demuestra preservación poblacional.
5. **Dependencia local:** repetir las 18 condiciones agregado/cambiado/paráfrasis de v19
   con los mismos contextos y semillas. Al menos 16/18 hechos completos y formato 18/18;
   no conceder aciertos por hechos parciales.

Se conserva una sola pasada y un único punto final. No se prolongará por mirar fluctuaciones,
no se escogerá otro checkpoint sobre estos pedidos y no se cambiarán umbrales después de ver
las salidas. El criterio está en [JSON](preferencias-v21-protocolo.json) y su cálculo de calidad
y controles en `experiments/criteria_preferences_v21.py`.

## Qué queda sin resolver

Este lote se seleccionó con juicio del mismo asistente que evaluará después. Las preferencias
medias son discutibles; seleccionar el mejor contraste entre cuatro también condiciona el
muestreo. Los targets de answer siguen variando y se entrena answer con una preferencia sobre
dream. No se solucionó eso cambiando pérdidas sin evidencia.

La auditoría estructural del pool anterior encontró 207 pares de dreams distintos y válidos;
151 excedían la razón 1,5 en dream o answer. No se revisó la calidad de todos esos pares
descartados: el conteo muestra la restricción, no que excluyera mejores sueños ni que causara
la regresión. [Conteos de la auditoría](preferencias-v21-auditoria-pool.json).

El próximo entrenamiento tendría 19 actualizaciones frente a las 14 de v20. Cambian los datos,
la elección de contraste y el número de pasos: es un ensayo exploratorio del nuevo lote, no
una prueba causal que aísle cada factor. No se promete mejora ni experiencia subjetiva.

## Auditoría y estado operativo

Las notas y el lote quedaron congelados. Huella de los pares:
`21e603aee86d4f4c2385d03e92002cb0209c7c1011860e206fd819ee27189dab`.
Se verificó contra las muestras originales cada dream, prompt y trayectoria elegida;
los cuatro archivos del checkpoint 32 conservan sus huellas. No se creó optimizador
ni se actualizó el modelo durante esa preparación; después se ejecutó la única pasada autorizada.

[Resumen del lote](preferencias-v21-lote.json), [auditoría de preparación](preferencias-v21-auditoria.json),
[integridad de trayectorias](preferencias-v21-integridad.json).

Validación: **187 pruebas aprobadas y una omitida** por el paquete opcional Torch ausente;
incluye nueve pruebas nuevas de preparación y criterio, además de las pruebas previas de
DPO. Ruff y comprobación de la vista local aprobados. Los datos completos, las referencias
y los pesos quedan fuera de Git. DreamBank: CC BY-NC-SA 4.0, dreambank.net; código MIT.

## Resultado del ensayo cerrado

Se recorrieron una sola vez los 19 pares elegidos, con el orden y los hiperparámetros
fijados. La pérdida DPO entrenada bajó de 0,6931 a 0,2219. Los adaptadores cambiaron;
la base y los cuatro archivos del checkpoint 32 conservaron sus huellas; recarga exacta.
No se continuó el entrenamiento después de mirar la evaluación.

| Medida | Checkpoint 32 | Piloto v21 |
| --- | ---: | ---: |
| Preferencias de calidad a favor, 96 pares | 11 | 30 |
| Empates compartidos | 55 | 55 |
| Formato en desarrollo | 95/96 | 95/96 |
| Muestras con pérdida importante de motivos | 28/96 | 26/96 |
| Preferencias de variedad, 24 grupos | 4 | 8 |
| Empates de variedad compartidos | 12 | 12 |
| Grupos con menos de dos episodios utilizables | 0 | 0 |
| Hechos completos de dependencia | 16/18 | 17/18 |
| Formato de dependencia | 18/18 | **17/18** |
| Palabras en dream, media | 81,8 | 109,8 |
| Palabras en answer, media | 64,0 | 78,9 |

La ventaja neta de calidad fue **+0,1979**, intervalo exploratorio por pedido
**[+0,09375; +0,30208]**, frente al umbral ≥0,15 y límite inferior >0. Son preferencias
del mismo asistente, no validación humana independiente. Calidad y variedad se congelaron
antes de consultar claves privadas sorteadas; no se usó una alternancia deducible.
No hubo dudas en estas notas; los empates se conservaron.

No se observó ninguna copia literal completa del pedido en dream en ninguno de los dos
modelos. Eso no constituye una métrica integral de calidad. Los motivos perdidos bajaron
en dos muestras; es un control descriptivo, no una mejora de fidelidad establecida.

### El control que impide promoverlo

En P11, la prueba de dependencia conserva el hecho de que el narrador escondió la moneda,
pero **no cierra answer, inventa otros turnos User/Assistant y alcanza 1.024 tokens**.
El hecho correcto y el formato fallido se cuentan por separado. P07 omite la ubicación
de la llave; las otras 17 respuestas conservan el hecho completo. Por condición: agregado
6/6, cambiado 6/6 y paráfrasis 5/6. No se repitió ninguna generación para reemplazar el fallo.

El requisito era formato 18/18 en dependencia. No se cambia por 17/18 después de observarlo.
El piloto satisface los demás controles numéricos, pero la decisión conjunta es **keep32**.
El checkpoint experimental queda en
`results/dream-v21/dpo/checkpoints/step-000019`, con identidad distinta del origen; no es
el checkpoint 51 de una continuación GRPO histórica.

### Sensibilidad posterior y límites de lectura

En el desarrollo ambos modelos fallaron una vez, en pedidos diferentes. La salida del 32
omite el cierre de dream: hay texto en bruto, pero el extractor deja el bloque vacío.
Una victoria primaria del nuevo incluye ese fallo. El piloto omite el cierre de answer
en otro pedido, conservando un dream legible. No se repararon etiquetas ni se descartaron
esos casos de la métrica primaria.

Como diagnóstico posterior, en los **94 pares con formato válido en ambos lados** quedan
29 victorias del piloto, 11 del 32 y 54 empates, ventaja por par +0,1915. No reemplaza las
96 comparaciones, no se usa para promover y no incorpora una prueba confirmatoria nueva.
La dirección observada no depende sólo de aquella victoria por extracción vacía.

La variedad favoreció al piloto en ocho pedidos y al 32 en cuatro, con doce empates.
No demuestra preservación poblacional de diversidad; sólo aplica el control pequeño fijado.
El fallo de P11 apunta a estudiar estabilidad de cierre y de turnos en otro ensayo separado,
con protocolo fijado y evaluación nueva. No se realizó ese ensayo ni se añadieron épocas.

[Resultados agregados](preferencias-v21-resultados.json),
[auditoría del resultado](preferencias-v21-resultados-auditoria.json).
Validación final: **189 pruebas aprobadas y una omitida** por Torch opcional ausente;
Ruff y auditoría de la corrida real aprobados. Los 200 relatos de prueba siguen sin usarse.
