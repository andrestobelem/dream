# Plan de implementación de Dream

Actualizado el 25 de octubre de 2026 con el cambio acordado a DreamBank y embeddings.

## Objetivo y decisiones vigentes

El modelo sueña dentro de `<dream>` y después cuenta lo soñado dentro de `<answer>`.
La única condición es D. No hay comparación con `<think>` ni experimento de aritmética.
Conservamos **Qwen/Qwen3-1.7B-Base**, revisión
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, del primer piloto.

1. **Corpus principal: DreamBank.** Se deduplican relatos y se separan personas y grupos reales
   identificados en los metadatos públicos. Una serie no se considera automáticamente una persona.
   Alias conocidos y hogares identificados permanecen juntos. Las colecciones colectivas sin
   identificación individual se catalogan, pero quedan fuera del piloto.
2. **Semillas mínimas extraídas por otro LLM.** Personajes, lugares, objetos, acciones y emociones:
   el primer ensayo usa de tres a ocho frases, hasta cuatro palabras por frase y 24 palabras totales.
   La riqueza adecuada queda abierta: esas cotas no son un requisito definitivo. No se extrae una
   trama ni se entrega el relato humano al generador.
3. **Prompt mínimo, sin ejemplo resuelto.** `<dream>` queda libre. No se evalúa su contenido ni
   se exige fidelidad detallada entre bloques. El límite de tokens limita recursos, no prescribe
   la longitud o el estilo del sueño.
4. **Recompensa basada en embeddings de `<answer>` respecto del corpus.** Se aplica a toda la
   secuencia generada, incluidos los tokens de `<dream>`. No interviene el juez histórico de
   cumplimiento, fidelidad u oniricidad. La primera fórmula exploratoria está fijada: máximo coseno con el banco de entrenamiento,
   reescalado a [0,1], usando MiniLM congelado. Fue comprobada operativamente, sin validar oniricidad.
5. **La similitud alta es favorable.** No se penalizan automáticamente coincidencias textuales ni
   se introduce evitar copias como objetivo. Una coincidencia literal no permite distinguir generación
   y memorización previa: es una limitación interpretativa, no un castigo en la recompensa.
6. **Generalización reservada.** El banco de recompensa se construye sólo con entrenamiento.
   Las personas, semillas y referencias de validación y prueba no contribuyen a esa recompensa.
   Los contrastes temáticos sirven para diagnosticar qué mide la fórmula, no constituyen una fórmula
   ya aprobada ni un requisito de separar perfectamente sueños de relatos de vigilia.

## Implementado y comprobado

- [x] Repositorio público, código MIT y entorno Python/MLX.
- [x] Archivo DreamBank fijado por SHA-256; HTML leído sin ejecutarlo ni extraer rutas del tar.
- [x] Registro explícito de 96 series; se admiten 71 series individuales en inglés, con alias
  y hogares agrupados. Se excluyen del piloto 25 series colectivas o en otros idiomas.
- [x] Importación real: 45.058 entradas de series, 11.463 excluidas por colección/idioma,
  10.221 repeticiones normalizadas entre las admitidas. Quedan 23.374 relatos únicos antes
  del filtro de longitud; 6.685 quedan fuera del intervalo de 40 a 350 palabras.
- [x] Selección balanceada por grupo: 1.000/100/200 relatos en entrenamiento/validación/prueba,
  sobre 30/6/7 grupos separados. Se conserva la procedencia de cada duplicado.
- [x] Manifiesto congelado que verifica datos, configuración y registro de grupos al reutilizarlos.
- [x] Prompt mínimo y validación de semillas implementados, sin acceso al relato original.
- [x] Extracción local con Qwen3-8B congelado y temperatura cero: un relato de cada partición
  produjo semillas válidas estructuralmente. Se conservan salida cruda, modelo, hash y selección.
  Se seleccionan hasta ocho frases mediante rondas por categoría, sin escribir ni completar términos.
  Estas tres extracciones comprueban el circuito; no validan la calidad semántica de las categorías.
- [x] Perfil `configs/dream-v5.yaml` como predeterminado. La CLI impide ejecutar por accidente
  el juez histórico con el nuevo protocolo.
- [x] Encoder MiniLM versionado, procesamiento completo de fragmentos, caché y banco de 1.000
  referencias de entrenamiento. Diagnóstico real de doce casos por grupo conservado.
- [x] Recompensa del relato conectada a GRPO para todos los tokens. Las pruebas verifican
  independencia respecto del contenido de dream y separación de las reservas.
- [x] Prompt mínimo sin ejemplo probado: un formato válido de tres, una truncación, pesos base intactos.
- [x] Informe local por CLI que distingue estado, diagnóstico y entrenamiento incompleto.
- [x] Site publicado, versión 11, con DreamBank, embeddings, riqueza de semillas abierta y límites.
- [x] Infraestructura histórica de GRPO, LoRA, checkpoints y cierre textual disponible para adaptar.
  Su funcionamiento anterior no demuestra eficacia del nuevo protocolo.

## Trabajo pendiente, en orden

1. Revisar y ampliar semillas: su riqueza queda abierta. Una comparación de niveles breve,
   intermedio y rico para los mismos relatos es una propuesta para estudiar, no una decisión
   cerrada ni un ensayo ejecutado. Deben aportar suficiente contexto sin entregar una trama.
2. Completar el piloto de tres actualizaciones. Su primer intento completó la evaluación inicial,
   pero se interrumpió por tiempo durante la preparación de LoRA, sin actualizaciones ni checkpoint.
   Falta verificar cambios de adaptadores, pesos base congelados, recarga y evaluación posterior.
3. Verificar continuidad real con el protocolo nuevo. Las pruebas con dobles comprueban el
   recorrido de interrupción/reanudación; todavía no sustituyen la ejecución con pesos reales.
4. Ampliar el diagnóstico de cercanía: distinguir vecino del banco y relato que originó la semilla.
   La recompensa actual usa el banco; la métrica respecto del relato fuente puede registrarse
   aparte. No incorporar validación/prueba al banco ni penalizaciones por coincidencias.
5. Publicar los resultados efectivos del piloto una vez que existan; el site ya describe el
   método actual y su estado incompleto.

El diagnóstico encontró recompensa media 1,000 para referencias exactas, 0,777 para relatos
reservados y 0,793 para listas temáticas. Los grupos tienen distintos referentes; los números
no prueban una preferencia por listas frente a sueños. Sí advierten que la cercanía temática
puede puntuar alto. Detalles en [recompensa por embeddings](recompensa-embeddings.md).

## Recursos y límites

Se conservan todos los consumos previos: el presupuesto original es de 2.700 segundos de ejecución
supervisada. Tras detener el último piloto quedaban aproximadamente 136 segundos. La prueba de extracción
consumió 11,37 segundos adicionales, incluido el primer intento fallido. Junto con embeddings,
pruebas de generación y el intento de piloto, el nuevo perfil acumula 111,52 de sus 120 segundos.
El supervisor conserva diez segundos para detener el proceso, por lo que ese perfil ya no permite
una ejecución útil. Se solicitó autorización para hasta cinco minutos adicionales, todavía pendiente
al redactar este estado. La preparación del corpus no realiza inferencias. No se reinician los
contadores anteriores para simular un presupuesto nuevo. Si el piloto completo no cabe, se
presentará el trabajo preparado y la necesidad concreta de ampliar recursos.

DreamBank declara **CC BY-NC-SA 4.0**, con atribución a [dreambank.net](https://dreambank.net/).
La licencia MIT del código no reemplaza la del corpus. Relatos, pesos y resultados locales quedan
fuera de Git. No se necesitan textos surrealistas mientras el corpus alcance.

Las agrupaciones se basan en las identidades públicas de la fuente; no descartan alias desconocidos.
El carácter público de los relatos impide excluir memorización previa. La cercanía semántica no
prueba experiencia subjetiva ni constituye por sí sola una medida validada de oniricidad.

## Historia preservada

DreamViews, el juez de cumplimiento/fidelidad y sus calibraciones quedan como antecedentes,
no como requisitos del diseño actual. El trabajo previo está archivado en la rama
`archive/dreamviews-judge-20261025`, commit `2f445fa`. El último piloto se interrumpió sin
completar actualizaciones. Los resultados de SmolLM2 y de la comparación histórica se conservan
separados. El piloto histórico Qwen tuvo cinco actualizaciones, sin evidencia de mejora.

Detalles históricos: [modelo base](modelo-base.md), [rúbrica v4](rubrica-v4.md),
[validación técnica](validacion-tecnica.md). Las diez respuestas con formato válido del protocolo
anterior no validan el nuevo prompt sin ejemplo.

Verificación de esta etapa: importación real y primera extracción por partición, más 124 pruebas
automatizadas de corpus, recompensa, conexión con GRPO e informes, incluidas identidad, fuga,
duplicados, manifiestos, prompt y caché. La recompensa por embeddings se ejecutó; el entrenamiento aún no completó actualizaciones.
