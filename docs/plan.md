# Plan de implementación de Dream

Actualizado el 25 de octubre de 2026.

El objetivo es generar una experiencia dentro de `<dream>` y contar esa misma experiencia en `<answer>`. Los relatos humanos del corpus sirven para construir las consignas y evaluar los resultados. El experimento de aritmética queda como antecedente técnico.

El circuito histórico ejecutó un piloto de cinco actualizaciones GRPO con Qwen3 y `dream-v2`. El generador actual es SmolLM2-1.7B base, sin posentrenamiento de razonamiento. Se auditaron tokenizer, carga y generación, con pesos intactos y sin juez ni entrenamiento. El nuevo protocolo `dream-v4` usa clasificación cerrada por probabilidades y abstención para cumplimiento y fidelidad. La CLI lo selecciona por defecto y mantiene su entrenamiento bloqueado mientras se completa integración y calibración.

Estado verificado: 88 pruebas del proyecto y cinco de Inspect aprobadas. La primera prueba real del clasificador cerrado coincidió con cuatro de ocho situaciones y la última con cinco de ocho. Repitió exactamente los logits de etiquetas en un forward independiente y dejó los pesos intactos. La corrección semántica y los umbrales siguen sin validar. Los ensayos anteriores por JSON, selección de evidencia y razonamiento del juez quedaron registrados; ninguno habilitó entrenamiento. El contraste exploratorio con Phi falló en formato de respuesta y no se adoptó como evaluador.

El [site ya presenta el proyecto](https://del-think-al-drean.andrestobelem.chatgpt.site/proyecto-dream), con el origen, corpus, método y límites de los resultados. La actualización del nuevo generador y evaluador se mantiene junto con esta revisión.

## Implementado

- [x] Repositorio público, licencia MIT y configuración de Python.
- [x] Importación del corpus DreamViews, filtros y separación por autor entre entrenamiento, validación y prueba. Selección actual: 1.000, 100 y 200 relatos, respectivamente.
- [x] Consignas construidas con descripciones breves extraídas de los relatos y categorías del corpus.
- [x] Cuatro condiciones que cruzan instrucción de pensar o soñar con etiquetas de proceso, más una condición neutral de evaluación.
- [x] Juez local congelado con evaluaciones separadas de carácter onírico, cercanía al relato humano y fidelidad entre experiencia y relato final.
- [x] GRPO con LoRA, límites de recursos, checkpoints e informes reproducibles.
- [x] Piloto de cinco actualizaciones en una condición y una semilla. Verificación de cambios en adaptadores, base y juez sin cambios, y recarga del checkpoint.
- [x] Integración de Inspect AI e Inspect Flow para reproducir y revisar resultados guardados, sin nueva inferencia.
- [x] Informes que distinguen scores globales y scores de respuestas extraíbles; paquete de revisión humana ciega preparado.

El piloto valida el funcionamiento técnico, pero no demuestra una mejora en los sueños generados. La evaluación comparativa usó dos muestras por celda. En la condición entrenada, el formato completo pasó de una de dos respuestas a ninguna de dos.

## 1. Mejorar la evaluación de cercanía

- [x] Versionar un protocolo nuevo y conservar los resultados de `dream-v2` como antecedentes separados.
- [x] Extraer una lista explícita de requisitos y dar al generador los mismos elementos que se usarán para puntuar su cumplimiento. Las citas de procedencia se verifican en el relato fuente.
- [x] Evaluar cada elemento como preservado, omitido o contradicho, registrando evidencia textual. Se implementó una llamada independiente por elemento.
- [x] Mantener la cercanía global al relato humano como diagnóstico y definir fórmula y pesos: cumplimiento = máximo(0, (preservados − 0.5 × contradichos) / total); aporta 30% de calidad.
- [x] Mantener separadas la fidelidad entre `<dream>` y `<answer>` y la semejanza con relatos de sueños humanos.
- [x] Presentar por separado cumplimiento, carácter onírico, fidelidad y tasa de formato válido. Una respuesta no extraíble se registra como fallo de respuesta.

Criterio para avanzar: protocolo documentado, fórmula explícita y puntuaciones auditables por elemento. No se debe penalizar un detalle del relato reservado que no apareció en la consigna.

### Actualización acordada para dream-v4

- [x] Seleccionar un generador base sin posentrenamiento de razonamiento y verificar que think/dream no sean tokens especiales. Modelo, revisión y auditoría están en [modelo-base.md](modelo-base.md).
- [x] Implementar clasificación cerrada por logits con modelo congelado, sin muestreo ni generación de explicaciones.
- [x] Verificar etiquetas de un token bajo el contexto exacto, rotar el mapeo A/B/C y registrar probabilidades, masa total y margen.
- [x] Implementar fragmentos identificados y reglas deterministas de agregación, conflicto y abstención. Una abstención devuelve `None`, no cero.
- [x] Comprobar repetibilidad de un forward independiente en este equipo, distinguiéndola de corrección semántica.
- [ ] Integrar el clasificador en cumplimiento y fidelidad del circuito completo; definir requisitos de fidelidad y manejo de grupos GRPO con abstenciones sin castigar al generador.
- [ ] Calibrar umbrales con pares revisados por humanos y validar exactitud y cobertura en un conjunto separado.

Los detalles y los fallos observados se documentan en [rubrica-v4.md](rubrica-v4.md). Los umbrales provisionales no autorizan entrenamiento.

## 2. Validar el juez

- [x] Preparar controles con paráfrasis, omisiones, contradicciones, cambios de participantes y relatos ajenos que compartan vocabulario. Incluye validación estricta de citas y bloqueo de RL ante fallos.
- [ ] Comprobar que agregar detalles compatibles no perjudique el cumplimiento y que las contradicciones relevantes sí lo reduzcan.
- [ ] Resolver los falsos positivos ante vocabulario sin experiencia relatada. Repetir los controles después del cambio y ampliar con ejemplos independientes de los usados para ajustar el evaluador.
- [x] Sustituir las citas generadas libremente por selección de fragmentos identificados del texto, verificables en código. Los controles semánticos de esas selecciones siguen pendientes de aprobación.
- [ ] Completar una revisión humana ciega y medir el acuerdo con el juez por dimensión.
- [ ] Contrastar una muestra con un juez de otra familia y revisar desacuerdos.
- [ ] Revisar las categorías del corpus y los indicadores de contenido: las etiquetas nativas y las etiquetas automáticas débiles deben permanecer diferenciadas.

Criterio para avanzar: controles con resultados esperados definidos de antemano y discrepancias relevantes resueltas o documentadas. La semejanza textual no prueba que una experiencia sea un sueño real.

## 3. Estabilizar el formato y la recompensa

- [ ] Resolver la omisión o mala delimitación de los bloques.
- [ ] Revisar la recompensa para que omitir `<dream>` no resulte conveniente y para evitar premiar copia o extensión sin contenido.
- [ ] Ejecutar un piloto acotado con el protocolo nuevo y verificar formato, puntuaciones y ejemplos.

La comprobación inicial de SmolLM2 produjo formato válido en cero de cinco condiciones con la plantilla conversacional y cinco de cinco con un ejemplo común, una sola consigna por condición. Fueron pruebas de generación sin entrenamiento ni juez. Falta ampliar la muestra antes de dar por estable el formato.

Criterio para avanzar: fijar previamente el umbral de formato aceptable y comprobarlo con una muestra mayor que las dos respuestas del piloto inicial.

## 4. Completar la validación operativa

- [ ] Demostrar continuidad de una corrida de sueños interrumpida y reanudada, comparada con una corrida equivalente sin interrupción. La prueba completa anterior corresponde al circuito de aritmética.
- [ ] Conectar Inspect con evaluación nueva de modelos y jueces; actualmente reproduce resultados existentes.
- [ ] Completar el análisis de condiciones y semillas, con diferencias pareadas e incertidumbre.

La ruta de verificación de continuidad para sueños ya está implementada: compara adaptadores, optimizador, PRNG de Python/NumPy/MLX, tokens y rollouts lógicos, excluyendo latencias y memoria. Falta ejecutarla contra un nuevo smoke completo con SmolLM2 y el protocolo congelado.

## 5. Ejecutar la comparación completa

- [ ] Fijar presupuesto, pasos, conjunto de evaluación, protocolo y criterios de éxito antes de lanzar las corridas.
- [ ] Ejecutar las cuatro condiciones con las tres semillas configuradas y evaluar también la condición neutral.
- [ ] Informar resultados por categoría, variación entre semillas, fallos de formato y revisión humana, además de las medias.
- [ ] Analizar diversidad y posible copia de los relatos del corpus.

Las doce corridas comparativas todavía no se lanzaron. DreamViews aporta suficientes relatos para la primera etapa; los textos surrealistas quedan como alternativa posterior, con procedencia y evaluación separadas.

La estimación histórica de `dream-v2` fue de unas 15 horas de evaluación para las doce corridas, además del entrenamiento. `dream-v3` usa más llamadas por relato: requiere un benchmark nuevo antes de fijar un plazo o presupuesto. La validación humana tampoco tiene un tiempo de finalización fijado.

## Entrega del trabajo

- [ ] Crear commits atómicos con los cambios implementados y revisados.
- [ ] Subirlos a GitHub y comprobar que la documentación describa el estado publicado.

La implementación actual está local, sin commits ni publicación de estos cambios. Este plan distingue trabajo implementado, validación pendiente y propuestas todavía sin programar.
