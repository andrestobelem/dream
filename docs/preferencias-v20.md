# Piloto de preferencias de calidad desde el checkpoint 32

25 de octubre de 2026. **Piloto completado: resultado desfavorable en la revisión del asistente.**
Se entrenó una pasada de 14 actualizaciones desde el 32. En 24 comparaciones de desarrollo
preferí el 32 doce veces, el piloto cuatro y empaté ocho. Se mantiene el 32 como referencia
práctica y el checkpoint DPO como resultado experimental, sin prolongar esta corrida.

El protocolo y lote revisado se registraron antes de entrenar en `944a6bd`. Las notas de
evaluación se congelaron antes de revelar la clave. Son juicios del asistente, no validación
humana independiente. [Resultados agregados](preferencias-v20-resultados.json),
[auditoría](preferencias-v20-auditoria.json).

## Objetivo

Aumentar la frecuencia de escenas y experiencias que se parezcan a relatos humanos de sueños
y utilicen los motivos pedidos. No se exige rareza, una trama literaria, mayor emoción o longitud.
Dream sigue libre; la referencia humana completa queda fuera del prompt. La cercanía a un
relato humano específico y el juez Phi-4 anterior no participan en esta nueva selección.

La revisión es del asistente. No es validación humana independiente; las preferencias son
juicios discutibles, y la misma fuente de revisión evaluará después. No se afirma que este
piloto vaya a mejorar generalización. Los 200 relatos de prueba siguen sin usarse.

## Lote previo al entrenamiento

Se seleccionaron por hash 24 pedidos de 24 personas de la partición de entrenamiento,
excluyendo los ocho relatos originales y las referencias humanas de calibración de v19.
Se conservaron semillas naturales breves, plantilla y longitud aproximada de answer del lote
revisado. Las personas de validación/prueba quedaron excluidas de los pares de entrenamiento.

Se generaron cuatro candidatos por pedido desde el 32. Si no había dos dreams distintos,
válidos y con longitud comparable tanto en dream como en answer, se generaron cuatro más,
hasta ocho. Se eligió el par que minimizaba la mayor razón de longitudes entre ambos bloques,
sin scores de calidad. Fue necesario generar **108 candidatos** para formar 24 pares.
Es una selección condicionada a formato y comparabilidad de longitud, no un muestreo representativo.

El asistente revisó sólo los dos bloques dream y el pedido. Conocía que ambos venían del 32;
los índices de candidato, semillas aleatorias y la clave de procedencia estaban ocultos.
Las 24 notas se congelaron antes de consultar esa clave. Se eligieron **14 preferencias claras
(5 altas y 9 medias) y 10 empates**. Los catorce pares también cumplieron razón de longitudes
≤1,5 y límite operativo de secuencia completa de 1.536 tokens. No se recortó ninguna trayectoria.

La comparación de longitud es control del diseño, no penalización ni recompensa. No se
castiga copia. Las notas favorecen acciones y percepciones situadas frente a explicaciones
genéricas, integrando motivos como acontecimientos. Los empates no se convirtieron en preferencias.

[Protocolo numérico](preferencias-v20-protocolo.json), [lote y hashes](preferencias-v20-lote.json).
Vista y notas completas locales: `results/dream-v20/index.html` y `preferences/annotations.json`.

## Entrenamiento fijado

- Inicio: adaptadores exactos del checkpoint 32, base Qwen3-1.7B-Base intacta.
- DPO sigmoid estándar: beta 0,1, suma de log-probabilidades de tokens de la continuación.
- Una pasada por las 14 preferencias claras: **14 actualizaciones**, sin selección intermedia por validación.
- AdamW nuevo, momentos reiniciados, learning rate 5e-6, weight decay cero, recorte de gradiente a norma 1.
- LoRA y modelo conservan rango, escala, capas, dropout cero y revisiones anteriores.
- La referencia es el 32 fijo: se guardan sus log-probabilidades antes de la primera actualización.
- El optimizador y los cuatro archivos del origen no se modifican.

Se entrena la continuación muestreada **completa**, dream y answer, con una preferencia
basada en dream. No hay pérdida por segmentos. Prompt y prefill no reciben pérdida;
se conservan los tokens originales muestreados, incluido el token final que contiene el cierre.
No se recodifica ni recorta la respuesta. Evaluaremos por separado posibles efectos en answer.

Para limitar memoria se calculan los gradientes de chosen/rejected secuencialmente, usando
la derivada exacta del mismo objetivo DPO. Se comprobó numéricamente contra `dpo_loss`
del backend instalado, con secuencias de distinta longitud y máscara del prompt.
También se comprobó que un paso de descenso mejora el margen preferido en un modelo pequeño.
La caché estacionaria de referencia evita mantener un segundo modelo durante el entrenamiento.

No se heredaron momentos de GRPO: cambió el objetivo. El piloto tiene una identidad nueva.
Cada actualización guarda adaptadores, optimizador y aleatoriedad transaccionalmente.

## Evaluación fijada antes de entrenar

Doce pedidos de desarrollo nuevos, escritos por el asistente antes del piloto y no derivados
de referencias reservadas. Son distintos de los 24 pedidos de entrenamiento. Se pide un
answer de aproximadamente cien palabras, manteniendo el contenido de dream libre.
Dos generaciones por pedido y modelo, mismas semillas 4242 y 4343, temperatura 0,8,
top-p 0,95 y límite total de 2.048 tokens: 24 pares antes/después, 48 generaciones.

Métrica primaria: preferencia del asistente por la calidad de dream, con procedencia del
modelo oculta hasta congelar notas, permitiendo empates y dudas. Se informa formato sobre
todas las generaciones y diversidad entre las dos muestras por pedido como diagnóstico
exploratorio; no se toma distancia de embeddings como una recompensa por producir rareza.
La revisión no podrá establecer ausencia general de pérdida de diversidad con esta muestra.

Se repiten después las 18 condiciones agregado/cambiado/paráfrasis de los tres pedidos y
semillas de v19, comparándolas con las salidas iniciales ya guardadas. Es comprobación local
de dependencia, no prueba de fidelidad completa ni experiencia subjetiva. Los contextos se
reconstruyen desde cero y las etiquetas del asistente se congelan antes de revelar condición.

No se ajustará la duración ni se elegirá otro checkpoint mirando estas fluctuaciones.
Los datos de prueba de DreamBank no se usan. Este conjunto nuevo de pedidos sigue siendo
desarrollo y no ofrece una evaluación humana independiente.
No hay un relato humano único esperado para estos doce pedidos nuevos, por lo que aquí no
se recalculó la cercanía MiniLM usada en v17. Las métricas no se mezclan con aquella cohorte.

## Resultados observados

Se completaron los 14 pasos fijados, una visita por par elegido. La pérdida DPO media sobre
esos pares bajó de **0,6931 a 0,2500**. Los adaptadores cambiaron; los pesos base y los cuatro
archivos del checkpoint 32 permanecieron idénticos. La recarga del checkpoint DPO coincidió.
No se seleccionó duración mirando la evaluación ni se entrenó otra pasada después.

| Medida de desarrollo | Checkpoint 32 | Piloto DPO |
| --- | ---: | ---: |
| Preferencias a favor, 24 pares | 12 | 4 |
| Empates compartidos | 8 | 8 |
| Formato válido | 24/24 | 24/24 |
| Dream que repite literalmente el pedido | 1/24 | 5/24 |
| Palabras en dream, media | 100,8 | 68,9 |
| Palabras en answer, media | 80,5 | 73,1 |

Entre los 16 pares decisivos el piloto obtuvo 4/16 preferencias. Incluyendo empates como cero,
el cambio neto fue **−0,3333**, donde +1 favorece al piloto y −1 al 32. El intervalo exploratorio
por pedido fue **[-0,5833; −0,0417]**, remuestreando los doce pedidos completos con 20.000 réplicas,
semilla 20261029. No se trataron las dos generaciones por pedido como personas independientes.
Ese intervalo no incluye incertidumbre entre revisores ni convierte doce pedidos escritos
por el asistente en evidencia poblacional de calidad humana.

La reducción de longitud no se usó como penalización. Un fragmento breve puede ser un buen
relato; el caso problemático es sustituir la vivencia por la consigna. El contador de repeticiones
es un diagnóstico posterior, no una recompensa por evitar coincidencias textuales. La selección
de pares de entrenamiento por longitud comparable tampoco demuestra ausencia de sesgos de longitud.

### Variedad: no se acreditó preservación

Sobre doce comparaciones de dos muestras por modelo, la revisión exploratoria favoreció
al 32 en dos, empató cinco y quedó indeterminada en cinco. En esos cinco faltaban dos escenas
utilizables en al menos un grupo; una consigna repetida no se contó como una escena diferente.

Esta revisión **no se presenta como cegada**: el mismo asistente implementó un orden alternado
de fuentes que se podía deducir desde el código y detectó esa limitación durante la lectura.
La revisión primaria de calidad ya estaba congelada antes de detectar el problema. No se
cambiaron sus notas. La variedad no permite afirmar que el piloto la haya conservado.

La similitud léxica media de grupos de cuatro palabras entre las dos muestras fue 0,00644
antes y 0,00761 después; no hubo pares completamente idénticos dentro de un modelo. Es un
proxy superficial, no una medición de diversidad semántica. No se usa para premiar rareza.

### Dependencia textual: conservada en esta prueba pequeña

| Hecho completo observado | Antes | Después |
| --- | ---: | ---: |
| Detalle agregado | 5/6 | 6/6 |
| Detalle cambiado | 5/6 | 5/6 |
| Detalle parafraseado | 6/6 | 6/6 |
| Total | 16/18 | 17/18 |
| Ambos lados de la edición correctos | 4/6 | 5/6 |

Las 18 respuestas posteriores conservaron el formato. Las nuevas etiquetas se congelaron
sin consultar la condición. En el caso parcial posterior, el texto dice que nosotros llegamos
a la costa y que el barco descarga sin hundirse, pero no afirma inequívocamente que el barco
llegó: no se inventó esa relación para conceder el acierto. La prueba mide hechos agregados
artificialmente y no demuestra fidelidad completa al dream ni compensa la baja de calidad.

## Diagnóstico posterior del objetivo aprendido

Se volvieron a calcular, sin actualizar pesos, las probabilidades de las continuaciones de
los 14 pares de entrenamiento con el 32 y con el piloto. La caché de referencia original
coincidió con el replay, con diferencia máxima de **0,0000223** en log-probabilidad acumulada.

La ganancia media de margen chosen−rejected respecto del 32 fue **+16,2951**: +11,4557 en
tokens de dream, +4,8089 en answer y +0,0306 en tokens de frontera. En los 14 pares ambos
bloques aumentaron el margen a favor del elegido. Los tokens originales que cruzan cierres
se conservaron enteros en un grupo de frontera para este diagnóstico; no se recortaron ni
se crearon máscaras nuevas de entrenamiento.

Por tanto, **dream sí recibió señal de aprendizaje**. El resultado no se explica simplemente
por ausencia de crédito a ese bloque, ni demuestra que separar la pérdida sea una solución.
El margen es una medida sobre datos entrenados, no calidad generada, y no identifica la causa
de la regresión. Sobreajuste, preferencias poco transferibles, selección de pares y diferencias
entre pedidos de entrenamiento y desarrollo quedan como hipótesis, no hallazgos causales.

## Decisión y límites

Se conserva el **checkpoint 32** como referencia práctica. No se aumentaron épocas ni se
promovió el último checkpoint porque bajara la pérdida entrenada. El piloto cumplió el
procedimiento, pero **no cumplió el objetivo de mejorar calidad sin perder variedad**.

El mismo asistente preparó preferencias y revisó resultados, los pares son pocos y la
evaluación es de desarrollo. Las etiquetas pueden discutirse en la vista local. Antes de
otro entrenamiento conviene revisar el diseño de los contrastes —incluido escena frente
a repetición de motivos— y su transferencia a pedidos nuevos. No se relabeló este lote
después de ver los resultados y no se inició otro ensayo.

## Artefactos locales

Revisión posterior de los cuatro aciertos, sus contracasos y el próximo diseño propuesto:
[aprendizajes de v20](aprendizajes-v20.md). No modifica las notas ni inicia otro entrenamiento.

```sh
.venv/bin/python -m experiments.preferences_v20 prepare
.venv/bin/python -m experiments.preferences_v20 train-pool
.venv/bin/python -m experiments.preferences_v20 before
.venv/bin/python -m experiments.review_preferences_v20 prepare
.venv/bin/python -m experiments.review_preferences_v20 freeze
.venv/bin/python -m experiments.review_preferences_v20 build
.venv/bin/python -m experiments.dpo_preferences_v20
.venv/bin/python -m experiments.preferences_v20 after
.venv/bin/python -m experiments.evaluate_preferences_v20 prepare
.venv/bin/python -m experiments.evaluate_preferences_v20 probes
.venv/bin/python -m experiments.dpo_logratio_v20
.venv/bin/python -m experiments.report_preferences_v20
.venv/bin/python -m experiments.view_preferences_v20
```

Freeze/build requieren las anotaciones reales del asistente; no las generan automáticamente.
La calidad, variedad y dependencia también requieren sus notas y hashes congelados.
Vista de resultados: `results/dream-v20/report.html`; la vista del lote sigue en `index.html`.
El lote, las referencias, las continuaciones completas y los pesos se mantienen fuera de Git.
Validación final: **178 pruebas aprobadas y una omitida** porque el paquete opcional Torch
no estaba instalado. Esa prueba corresponde al encoder de embeddings anterior, no usado
en este piloto DPO. Las pruebas nuevas de gradiente, máscara, congelación y agrupación
por pedido aprobaron; también se auditó la corrida real y el replay de referencia.
DreamBank: CC BY-NC-SA 4.0, dreambank.net. Código MIT.
