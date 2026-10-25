# Piloto de preferencias de calidad desde el checkpoint 32

25 de octubre de 2026. Protocolo y lote revisado antes de entrenar. Estado: piloto autorizado
por el usuario, lote congelado y evaluación inicial guardada; falta completar entrenamiento y comparación.

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

## Artefactos locales

```sh
.venv/bin/python -m experiments.preferences_v20 prepare
.venv/bin/python -m experiments.preferences_v20 train-pool
.venv/bin/python -m experiments.preferences_v20 before
.venv/bin/python -m experiments.review_preferences_v20 prepare
.venv/bin/python -m experiments.review_preferences_v20 freeze
.venv/bin/python -m experiments.review_preferences_v20 build
.venv/bin/python -m experiments.dpo_preferences_v20
.venv/bin/python -m experiments.preferences_v20 after
```

Freeze/build requieren las anotaciones reales del asistente; no las generan automáticamente.
El lote, las referencias, las continuaciones completas y los pesos se mantienen fuera de Git.
DreamBank: CC BY-NC-SA 4.0, dreambank.net. Código MIT.
