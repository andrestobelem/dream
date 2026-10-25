# Plan de implementación de Dream

Actualizado el 25 de octubre de 2026, después de la aclaración de alcance.

## Objetivo actual

Usar un modelo base sin posentrenamiento de razonamiento para **soñar dentro de `<dream>` y después contar ese sueño dentro de `<answer>`**. Aprender a partir de relatos humanos y evaluar qué tan oníricos son los relatos, si cumplen la consigna y si cuentan la experiencia escrita en el primer bloque.

La única condición activa es **D: Soñar + `<dream>`**. Las variantes A/B/C y neutral se conservan como antecedentes; no forman parte de las corridas actuales. Tampoco se requiere la comparación de doce corridas del diseño anterior. La aritmética es un antecedente técnico.

## Hecho

- [x] Repositorio público con licencia MIT y configuración de Python.
- [x] Corpus DreamViews: 1.000 relatos de entrenamiento, 100 de validación y 200 de prueba, separados por autor; categorías originales conservadas.
- [x] Consignas derivadas de los relatos, con requisitos visibles para el generador y el evaluador.
- [x] SmolLM2-1.7B base con revisión fijada; carga, tokenización y generación comprobadas. `<dream>` no es un token especial ni un modo de razonamiento del modelo.
- [x] Generación en dos bloques y extracción de escena y relato.
- [x] Configuración actual limitada a D; el piloto y el entrenamiento con esta configuración tampoco agregan la evaluación neutral.
- [x] Implementación local de GRPO con LoRA, checkpoints y límites de recursos. El piloto ejecutado corresponde al modelo Qwen3 anterior, no demuestra aprendizaje en SmolLM2.
- [x] Integración del clasificador cerrado en cumplimiento y fidelidad; una abstención se conserva como ausencia de score. Un grupo con recompensa desconocida se detiene antes de actualizar adaptadores.
- [x] Informes y revisión humana ciega preparados; los resultados históricos se pueden revisar con Inspect.
- [x] [Presentación del proyecto publicada](https://del-think-al-drean.andrestobelem.chatgpt.site/proyecto-dream). La versión 8 acotó el método a D; la versión 9 registra los controles nuevos del juez y el diagnóstico de formato de diez generaciones.

## Lo que falta para entrenar y evaluar Dream

1. **Validar la evaluación.** La variante con ejemplos del clasificador cerrado pasó siete de ocho controles conocidos y tres de cuatro nuevos. Siguen pendientes errores relevantes y calibración humana. Se prepararon 29 pares de requisitos y relatos para revisar cumplimiento y fidelidad; no tienen anotaciones humanas. El paquete anterior de seis relatos para evaluar carácter onírico tampoco fue puntuado y es una revisión distinta.
2. **Estabilizar generación D.** La nueva prueba de diez consignas sintéticas obtuvo nueve respuestas completas y una sin cierre de `<dream>`; no aprobó el criterio fijado de 10/10. Falta revisar calidad, copia y diversidad con relatos del corpus.
3. **Ejecutar un piloto pequeño de entrenamiento D con SmolLM2**, una vez que la evaluación sea utilizable. Comparar relatos antes y después, conservar componentes del score y comprobar que sólo cambien los adaptadores.
4. **Verificar recuperación** del piloto de sueños y documentar resultados y límites. La prueba completa de interrupción y reanudación anterior corresponde a aritmética.
5. **Publicar los resultados del piloto** en el repo y el site, distinguiendo pruebas de código y evidencia de aprendizaje.

El entrenamiento de `dream-v4` permanece bloqueado porque el evaluador aún no está validado. La infraestructura opcional de evaluación no debe demorar ni ampliar estas tareas. No se lanzarán comparaciones entre instrucciones o delimitadores como parte de este alcance.

## Antecedentes y límites

El piloto histórico de Qwen3 tuvo cinco actualizaciones, con dos muestras de evaluación por celda. En D, el formato completo pasó de una de dos respuestas a ninguna de dos: no hay evidencia de mejora. Los intentos anteriores de juez por JSON y el contraste exploratorio con Phi tampoco habilitaron el nuevo entrenamiento.

El contraste posterior con DeBERTa NLI acertó 26/32 pares y cinco de ocho situaciones completas, con errores en paráfrasis y listas de palabras. No se adoptó como reemplazo del juez. El [diagnóstico del evaluador](rubrica-v4.md) conserva revisión del modelo, procedimiento y límites.

El contraste de etiquetas cerradas con Phi tampoco pasó: cero de doce situaciones completas, 17 abstenciones y nueve decisiones incorrectas entre 40 pares. Se mantiene Qwen3 como configuración de desarrollo; ninguno de estos resultados habilita entrenamiento.

DreamViews tiene suficientes relatos para esta etapa. Los textos surrealistas quedan como alternativa posterior si hicieran falta, con procedencia y evaluación separadas. La semejanza textual no demuestra experiencia subjetiva del modelo.

Detalles: [modelo base](modelo-base.md), [rúbrica actual](rubrica-v4.md) y [validación técnica histórica](validacion-tecnica.md).

Verificación actual: 103 pruebas del proyecto y ocho de la integración opcional de Inspect aprobadas, incluyendo rotación de ejemplos y preservación de anotaciones humanas. La nueva inferencia a través de Inspect todavía no se probó con modelos reales y no es requisito del próximo piloto D.
