# Proyecto: Clasificación de Tuercas, Tornillos y Arandelas (OpenCV clásico)

## 📋 Descripción del Proyecto
Sistema de visión artificial capaz de clasificar automáticamente piezas mecánicas (tuercas, tornillos y arandelas) en imágenes, independientemente de su orientación o ángulo de captura.

## 🎯 Evolución de la Estrategia de Clasificación

### ❌ Estrategia Inicial (Compleja - Descartada)
```
DETECCIÓN POR ÁNGULO + MODELOS ESPECIALIZADOS
├── Dataset dividido por ángulos
│   ├── frontal/
│   ├── lateral/
│   └── angulado/
├── Múltiples clasificadores
│   ├── ClasificadorFrontal
│   ├── ClasificadorLateral
│   └── ClasificadorAngulado
└── Sistema de ensamble complejo
```
Problemas identificados:
- Complejidad excesiva: demasiados modelos y reglas.
- Fragilidad: si falla la detección de ángulo, se propagan errores.
- Mantenimiento difícil: varios modelos a actualizar y versionar.
- Sobreenfocamiento: modelos muy específicos pierden generalización.
- Dataset artificial: en la práctica, las imágenes vienen mezcladas.

### ✅ Estrategia Final (Simplificada - Implementada)
```
CARACTERÍSTICAS ROBUSTAS + CLASIFICACIÓN ÚNICA
├── Dataset simple por clase
│   ├── tuercas/
│   ├── tornillos/
│   └── arandelas/
├── Un clasificador principal
│   └── ClasificadorPiezas
├── Características multi-ángulo
│   └── Invariantes a rotación
└── Reglas de backup simples
```
Ventajas:
- Simplicidad: un solo modelo fácil de mantener.
- Robustez: características que funcionan en cualquier ángulo.
- Generalización: mejor rendimiento en casos nuevos.
- Mantenibilidad: fácil de depurar y mejorar.
- Realismo: se adapta a datasets del mundo real.

## 📁 Estructura y Archivos Clave

Estructura de carpetas esperada:
```
dataset/
  arandelas/
  tornillos/
  tuercas/
modelos/
resultados/
codigo/
  config.py
  utils.py
  preprocesamiento.py
  segmentacion.py
  extraccion_caracteristicas.py
  clasificacion.py
  evaluacion.py
	main.py
```

Descripción de archivos:
1) `codigo/config.py` (⚙️): Configuración centralizada.
	- Tamaños de imagen y preprocesamiento
	- Umbrales para segmentación/filtrado y reglas
	- Hiperparámetros de SVM/KNN/Random Forest
	- Rutas del dataset y modelos (por defecto `./dataset`, `./modelos`)

2) `codigo/utils.py` (🛠️): Utilidades generales.
	- Carga de dataset, guardado/carga de modelos, visualización de contornos
	- Manejo de directorios y guardado de imágenes procesadas

3) `codigo/preprocesamiento.py` (🖼️): Preparación de imágenes.
	- Redimensionado, conversión a gris, filtro gaussiano, ecualización, normalización
	- Operaciones morfológicas y binarización (Otsu)

4) `codigo/segmentacion.py` (✂️): Detección de objetos.
	- Umbralización Otsu, contornos `RETR_EXTERNAL + CHAIN_APPROX_SIMPLE`
	- Filtrado por área y relación de aspecto, extracción de ROI, orientación básica

5) `codigo/extraccion_caracteristicas.py` (🔍): Descriptores de forma y textura.
	- Relación de aspecto, solidez, circularidad, compacidad, rectangularidad, excentricidad
	- Aristas (polígono aproximado), detección de agujero, momentos de Hu, textura/uniformidad

6) `codigo/clasificacion.py` (🧠): Clasificación ML + reglas.
	- SVM/KNN/RandomForest con `StandardScaler`
	- Clasificación por reglas (backup) y combinación ML+reglas

7) `codigo/evaluacion.py` (📊): Métricas y análisis.
	- Accuracy, precisión, recall, F1, matriz de confusión, curvas de aprendizaje
	- Evaluación por clase y reporte de errores visual


8) `codigo/main.py` (🎮): Orquestador CLI.
	- `--entrenar`, `--predecir`, `--evaluar`, `--evaluar-todo` y `--modelo [svm|knn|rf]`
	- `--tune` para activar GridSearchCV (búsqueda de hiperparámetros)
	- `--debug N` para guardar N ejemplos por clase (binarizados y contornos)
	- Nota: el `main.py` reside dentro de `codigo/`. Las rutas (`./dataset`, `./modelos`, `./resultados`) se resuelven automáticamente respecto a la raíz del proyecto, independientemente del directorio actual.

## 🔄 Flujo de Procesamiento
```
IMAGEN ORIGINAL
	 ↓
preprocesamiento.py      → Mejora calidad
	 ↓
segmentacion.py          → Detecta objetos
	 ↓
extraccion_caracteristicas.py → Extrae features
	 ↓
clasificacion.py         → Clasifica pieza
	 ↓
RESULTADO: "tuerca" | "tornillo" | "arandela"
```

## 📦 Requisitos e Instalación

Requisitos mínimos: Python 3.10+, Linux/macOS/Windows.

Instala dependencias (recomendado en un entorno virtual):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Si tienes problemas al mostrar/guardar imágenes con OpenCV en Linux, instala `libgl1`:
```bash
sudo apt-get update && sudo apt-get install -y libgl1
```

## ▶️ Cómo Ejecutar

Entrenamiento (SVM por defecto):
```bash
# desde la raíz del proyecto
python codigo/main.py --entrenar --modelo svm

# o situándote dentro de 'codigo/'
cd codigo && python main.py --entrenar --modelo svm
```

Predicción sobre una imagen:
```bash
# desde la raíz del proyecto
python codigo/main.py --predecir ./dataset/tuercas/ejemplo.jpg --modelo svm
```

Evaluación de un modelo existente:
```bash
# desde la raíz del proyecto
python codigo/main.py --evaluar --modelo svm
# alias explícito para evaluar TODO el dataset
python codigo/main.py --evaluar-todo --modelo svm
```

Notas:
- Estructura del dataset: coloca imágenes directamente dentro de `dataset/arandelas`, `dataset/tornillos`, `dataset/tuercas` (sin subcarpetas por ángulo). El código resuelve `./dataset` siempre respecto a la raíz del repo, aunque ejecutes desde `codigo/`.
- Los modelos se guardan en `./modelos/` y resultados/figuras en `./resultados/`.
- Si `scikit-learn` no está instalado, el sistema puede degradarse a reglas simples.

### Modos de entrenamiento y flags

- Básico (rápido):
	- `python codigo/main.py --entrenar --modelo svm`
	- Entrena un pipeline con StandardScaler + SVM/KNN/RF usando parámetros de `config.py`.

- Con búsqueda de hiperparámetros (recomendado para subir accuracy):
	- `python codigo/main.py --entrenar --modelo svm --tune`
	- Ejecuta GridSearchCV con las parrillas de `config.py` (`GRID_SVM`, `GRID_KNN`, `GRID_RF`), selecciona el mejor estimador y lo deja guardado en el pipeline.

- Con depuración de segmentación (ahorra iteraciones de ajuste):
	- `python codigo/main.py --entrenar --modelo svm --debug 5`
	- Guarda 5 ejemplos por clase de binarizados y contornos en `resultados/debug_entrenamiento_<timestamp>/` para inspeccionar si la segmentación descarta o confunde piezas.

### Evaluación e informes

Al ejecutar evaluación se crea una carpeta por ejecución:

```
resultados/estadisticas_evaluaciones/<YYYY-MM-DD_HH-MM-SS>/
	├── metricas_<modelo>.txt             # accuracy, precision_macro, recall_macro, f1_macro, f1_weighted
	├── classification_report.txt         # reporte detallado por clase
	├── confusion_matrix.png              # matriz de confusión
	├── per_class_metrics.png             # barras de precision/recall/f1 por clase
	├── predictions.csv                   # filas: ruta,real,pred (cada imagen procesada)
	└── coverage.txt                      # total, procesadas, saltadas y listado de saltadas (hasta 200)
```

Además, al final de la evaluación se imprime el Accuracy en consola y queda registrado en logs.

## 🔬 Características extraídas (features)

- Geométricas: relación de aspecto, solidez, circularidad, compacidad, rectangularidad, excentricidad.
- Estructurales: número de lados aproximado (approxPolyDP), índice de aristas (hexagonal≈1), relación de agujero.
- Invariantes (Hu): 7 momentos de Hu con log-transform y signo.
- Textura: suavidad (std normalizada) y uniformidad (energía del histograma) sobre ROI.
- HOG (opcional, activado por defecto):
	- Se calcula sobre el ROI con padding (`ROI_PADDING`) para capturar bordes del objeto.
	- Parámetros en `config.py`: `HOG_ORIENTACIONES`, `HOG_PIXELS_PER_CELL`, `HOG_CELDAS_X`, `HOG_CELDAS_Y`.

El ROI se extrae del contorno mayor tras filtros por área y relación de aspecto; puede expandirse con `ROI_PADDING` para capturar el objeto completo antes de HOG/texture.

## ⚙️ Segmentación y parámetros importantes (config.py)

- Umbralización: Otsu por defecto (`UMBRAL_OTSU=True`). Alternativas: `binarizar_imagen(..., metodo='adaptive' | 'binary')`.
- Morfología post-umbral (recomendado):
	- `MORFOLOGIA_POST_UMBRAL=True`, `MORFOLOGIA_OPERACION="cierre"`, `MORFOLOGIA_KERNEL=3|5`.
	- Mejora máscaras, une bordes y reduce ruido.
- Filtro por área: `AREA_MIN`, `AREA_MAX` para descartar ruido o objetos minúsculos.
- Filtro por aspecto: `ASPECTO_MIN`, `ASPECTO_MAX` para suprimir contornos muy extremos.
- ROI con padding: `ROI_PADDING` (p.ej., 0.05) para no recortar bordes útiles del objeto.

## 🧠 Modelos y tuning

- SVM (por defecto): `SVM_PARAMS` con `class_weight='balanced'` y `probability=True`.
- KNN: `KNN_PARAMS` (vecinos, métrica, weights).
- Random Forest: `RF_PARAMS` con `class_weight='balanced'`.
- Grid search (`--tune`) usa:
	- `GRID_SVM`: C, gamma, kernel.
	- `GRID_KNN`: n_neighbors, weights, p.
	- `GRID_RF`: n_estimators, max_depth, max_features.

## 🧭 Cómo subir el accuracy (guía rápida)

1) Segmentación primero:
	 - Activa morfología y ajusta `MORFOLOGIA_KERNEL` (3→5) si ves huecos o cortes.
	 - Ajusta `AREA_MIN` para no descartar piezas pequeñas ni aceptar ruido.
	 - Revisa `ASPECTO_MIN/MAX` si hay variabilidad alta.
	 - Usa `--debug 5` para generar binarizados/contornos por clase y ajustar rápido.

2) ROI + HOG:
	 - Asegura `USE_HOG=True` y que el ROI tenga `ROI_PADDING` suficiente para no cortar bordes.
	 - Ajusta rejilla HOG (`HOG_CELDAS_X/Y`) y bins (`HOG_ORIENTACIONES`).

3) Tuning de modelo:
	 - Lanza `--tune` con SVM; guarda el mejor estimador automáticamente.
	 - Si hay desbalance, `class_weight='balanced'` ya viene activado en SVM/RF.

4) Validación:
	 - Revisa `classification_report.txt` y `confusion_matrix.png` en cada ejecución.
	 - Si una clase falla mucho, considera más datos o ajustar umbrales/ROI.

## 🎯 Características Clave del Diseño Final
1) Robustez Angular (🌀): características geométricas invariantes a rotación; no depende de la orientación.
2) Clasificación Híbrida (⚡): ML para patrones complejos; reglas simples como respaldo.
3) Mantenibilidad (🔧): configuración centralizada, módulos desacoplados, fácil de depurar.
4) Escalabilidad (📈): sencillo añadir nuevas clases y evaluar el impacto.

## 🚀 Resultados Esperados (objetivos)
| Métrica               | Objetivo |
|-----------------------|----------|
| Accuracy general      | > 80%    |
| Precisión tuercas     | > 85%    |
| Recall tornillos      | > 75%    |
| F1-score arandelas    | > 80%    |

## 💡 Lecciones Aprendidas
- Simplicidad > Complejidad: menos componentes, menos puntos de fallo.
- Características robustas > muchos modelos específicos.
- Dataset real > dataset ideal: adaptarse a los datos disponibles.
- Sistema híbrido > enfoque puro: ML + reglas = mejor robustez.

## 🧪 Consejos y Troubleshooting
- "No hay contornos": revisa iluminación y binarización; prueba `pre.binarizar_imagen(..., metodo='adaptive')`.
- "predict_proba no disponible": habilitado para SVM con `probability=True` (por defecto en `config`).
- "No module named sklearn": ejecuta `pip install -r requirements.txt` en tu entorno.

## 📈 Experiencias, problemas y soluciones aplicadas

Durante el desarrollo se observaron dos escenarios relevantes que impactaban al rendimiento:

1) Baja cobertura de procesamiento (muchas imágenes saltadas)
	 - Síntomas: `coverage.txt` mostraba muy pocas `procesadas` y muchas `saltadas` (p. ej., 71/472 procesadas; 401 saltadas).
	 - Causas probables: binarización no robusta a iluminación/fondo, morfología insuficiente, filtros de área/aspecto demasiado estrictos.
	 - Soluciones implementadas:
		 - Segmentación robusta: se intentan múltiples métodos de binarización (Otsu, adaptativa, binaria fija), con y sin inversión, y distintas morfologías (kernel 3/5/7).
		 - Relajación progresiva: se relajan umbrales de área y aspecto por etapas si no se encuentran contornos válidos.
		 - Resultado: cobertura 100% (0 saltadas) en el dataset de ejemplo.

2) Caída de accuracy tras aumentar cobertura
	 - Síntomas: al pasar a 0 saltadas el accuracy cayó (p. ej., ~0.41) por mayor ruido en las muestras procesadas.
	 - Causas probables: contornos subóptimos elegidos (fondo, sombras), alta dimensionalidad de HOG y ruido.
	 - Soluciones implementadas:
		 - Selección de contorno por score de calidad: se elige el mejor contorno según combinación de solidez, rectangularidad y área, evitando tomar contornos de mala calidad aunque pasen filtros.
		 - PCA opcional en el pipeline: reducción de dimensionalidad antes del clasificador (útil con HOG) para mejorar la generalización y reducir ruido. Controlable vía `USE_PCA`, `PCA_COMPONENTS` y `PCA_WHITEN` en `config.py`.
		 - Re-entrenar con `--tune`: GridSearchCV para hallar hiperparámetros adecuados con validación cruzada, clave tras cambios de features/segmentación.

### Recomendaciones de entrenamiento/validación

- Mantén validación honesta mientras buscas hiperparámetros:
	- Usa `--tune` para GridSearchCV (CV interna).
	- No entrenes con todo el dataset hasta fijar hiperparámetros con CV.

- Modelo final (para despliegue):
	- Una vez elegidos hiperparámetros, reentrena con todo el dataset de entrenamiento (puedo añadir `--entrenar-final` si lo deseas).

### Parámetros útiles a ajustar en `config.py`

- Segmentación:
	- `MORFOLOGIA_POST_UMBRAL=True`, `MORFOLOGIA_OPERACION="cierre"`, `MORFOLOGIA_KERNEL=3|5|7`
	- `AREA_MIN`, `AREA_MAX` (objetos muy pequeños o grandes)
	- `ASPECTO_MIN`, `ASPECTO_MAX` (variabilidad de formas)
	- `ROI_PADDING` (evitar recortes demasiado ajustados)

- Features (HOG/Dimensionalidad):
	- `USE_HOG=True`, `HOG_ORIENTACIONES`, `HOG_PIXELS_PER_CELL`, `HOG_CELDAS_X/Y`
	- `USE_PCA=True`, `PCA_COMPONENTS=100`, `PCA_WHITEN=False`

- Clasificadores/Tuning:
	- `SVM_PARAMS`, `KNN_PARAMS`, `RF_PARAMS`
	- `GRID_SVM`, `GRID_KNN`, `GRID_RF` para `--tune`

### Auditoría de resultados por ejecución

Cada evaluación crea una carpeta con timestamp en `resultados/estadisticas_evaluaciones/<fecha_hora>/` con:

- `metricas_<modelo>.txt`, `classification_report.txt`, `confusion_matrix.png`, `per_class_metrics.png`
- `predictions.csv`: (ruta, real, pred) para revisar errores concretos.
- `coverage.txt`: total/procesadas/saltadas, y listado de saltadas (hasta 200) con motivo (p. ej., `sin_caracteristicas`).

Usa estos archivos para localizar patrones de error por clase o por condiciones de imagen (iluminación, foco, fondo).

