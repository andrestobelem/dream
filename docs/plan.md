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

## Resultados y siguiente trabajo

- [x] El piloto con apertura de `<dream>` completó tres pasos. Cuatro de doce candidatos
  cumplieron el formato; los pasos 2 y 3 tuvieron recompensas variables y gradientes no nulos.
  Los adaptadores cambiaron y la base quedó intacta. La recarga reprodujo los pesos guardados.
- [x] Continuidad real desde el paso 2: diez comprobaciones aprobadas, incluidas salidas,
  adaptadores, optimizador y estados aleatorios. No se atribuye esto a la nueva plantilla natural.
- [x] Evaluación antes y después: dos muestras reservadas por fase, todas con formato inválido.
  No hay evidencia de mejora. El cero de formato no es un juicio semántico del sueño.
- [x] Revisión de desarrollo asistida de las cinco consignas usadas: contenido conservado,
  con omisiones de emociones y relaciones. No son anotaciones humanas independientes.
- [x] Diagnóstico separado respecto del relato de origen, calculado con vectores ya guardados.
  En dos candidatos el vecino más cercano del banco difiere del relato de origen.
- [x] Site versión 12 publicado con resultados, continuidad y limitaciones del piloto anterior.
- [x] Nueva plantilla pedida por el usuario y semillas en lenguaje natural probadas con el modelo
  base, sin entrenamiento: dos formatos válidos de tres, uno interpretativo y otro narrativo;
  un tercero truncado. Referencia humana fuera del prompt. [Detalle](plantilla-natural.md).

La nueva redacción cambia la consigna: sus pruebas se conservan separadas del piloto entrenado.
Antes de atribuirle resultados de aprendizaje habría que entrenarla y evaluarla con su propia
identidad. Ese entrenamiento no se ejecutó en esta prueba de plantilla.

Quedan abiertas la riqueza adecuada de las semillas y la estabilidad del formato. Comparar
niveles breve, intermedio y rico es una propuesta, no un ensayo aprobado o ejecutado. La
cercanía al relato de origen y al banco sigue registrada por separado; no se cambió la recompensa.

El diagnóstico del encoder encontró recompensa media 1,000 para referencias exactas, 0,777
para relatos reservados y 0,793 para listas temáticas. Los grupos tienen distintos referentes;
los números no prueban preferencia por listas frente a sueños. Sí advierten que la cercanía
temática puede puntuar alto. [Método](recompensa-embeddings.md) y
[validación del piloto](validacion-dreambank.md).

## Recursos y límites

Se conservan todos los consumos previos: el presupuesto original es de 2.700 segundos de ejecución
supervisada. Tras detener el último piloto quedaban aproximadamente 136 segundos. La prueba de extracción
consumió 11,37 segundos adicionales, incluido el primer intento fallido. Junto con embeddings,
pruebas de generación y el intento de piloto, el nuevo perfil acumula 111,52 de sus 120 segundos.
El supervisor conserva diez segundos para detener el proceso. El usuario autorizó hasta cinco minutos adicionales desde el consumo actual;
el perfil amplió su techo acumulado a 411 segundos, sin reiniciar el contador. El piloto y
la continuidad llevaron el consumo a 332,51 segundos; la nueva prueba de plantilla lo llevó
a 364,38 segundos. La preparación del corpus no realiza inferencias. No se reinician los
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

Verificación de esta etapa: importación real y primera extracción por partición, más 128 pruebas
automatizadas de corpus, recompensa, conexión con GRPO e informes, incluidas identidad, fuga,
duplicados, manifiestos, prompt y caché. La recompensa, el piloto anterior y su continuidad se ejecutaron; la plantilla natural nueva
se probó sin entrenamiento. Los resultados no demuestran mejora.
