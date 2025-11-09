# Proyecto: Clasificación de Tuercas, Tornillos y Arandelas

Pipeline completo de visión por computador para detectar objetos en imágenes (segmentación) y clasificarlos mediante descriptores (geométricos, HOG, ORB, histogramas) y modelos clásicos (SVM/KNN).

## Estructura

```
Proyecto_Vision_Tuercas_Tornillos/
├── codigo/
│   ├── main.py                    # Orquestador CLI (entrenar, predecir, evaluar)
│   ├── preprocesamiento.py        # Filtros, gris, morfología, resize
│   ├── segmentacion.py            # Otsu, contornos, filtros por área/aspecto
│   ├── extraccion_caracteristicas.py  # Geometría, Hu, HOG, ORB, histogramas
│   ├── clasificacion.py           # Pipelines SVM/KNN + StandardScaler
│   ├── evaluacion.py              # Métricas, matriz confusión, curvas aprendizaje
│   ├── utils.py                   # Carga dataset, guardados, plotting de contornos
│   └── config.py                  # Parámetros globales (constantes)
├── dataset/
│   ├── tuercas/
│   ├── tornillos/
│   └── arandelas/
├── modelos/
├── resultados/
└── tests/
```

## Requisitos

- Python 3.9+
- Paquetes:
  - numpy
  - opencv-python
  - scikit-learn
  - matplotlib
  - pillow
  - joblib

Instalación rápida:

```bash
pip install numpy opencv-python scikit-learn matplotlib pillow joblib
```

Si no usas entorno virtual, te recomiendo crear uno antes.

## Configuración

Edita `codigo/config.py` para ajustar:
- Tamaño de imagen (`IMAGE_SIZE`), interpolación (`RESIZE_INTERPOLATION`)
- Filtro gaussiano (`GAUSSIAN_KERNEL_SIZE`, `GAUSSIAN_SIGMA_X/Y`)
- Segmentación (`THRESH_METHOD`, umbrales adaptativos, morfología)
- Filtros de contornos (`MIN_CONTOUR_AREA`, `MAX_CONTOUR_AREA`)
- HOG (`HOG_*`), SVM/KNN (`SVM_*`, `KNN_*`)
- Rutas (`DATASET_DIR`, `MODELOS_DIR`, `RESULTADOS_DIR`)

## Uso

Desde la carpeta raíz del proyecto, ejecuta el módulo `codigo.main`:

- Entrenamiento (SVM por defecto):

```bash
python -m codigo.main --entrenar --dataset ./dataset --modelo svm
```

- Predicción sobre una imagen:

```bash
python -m codigo.main --predecir --imagen ./dataset/tuercas/ejemplo.jpg --modelo svm
```

- Evaluación de un modelo guardado:

```bash
python -m codigo.main --evaluar --dataset ./dataset --modelo svm --modelo-ruta ./modelos/svm_hog.joblib
```

Artefactos generados:
- Modelos: `modelos/` (p. ej., `svm_hog.joblib`, `knn_hog.joblib`)
- Resultados: `resultados/metricas*.txt`, `resultados/matriz_confusion*.png`

## Detalles del pipeline

1. Preprocesamiento: resize → gris → Gaussiano → morfología (opcional)
2. Segmentación: Otsu → contornos externos (RETR_EXTERNAL) → filtro por área/aspecto
3. Extracción: geométricas (relación aspecto, solidez, circularidad, agujeros), Hu, histograma (HSV), HOG, ORB (mean+std)
4. Clasificación: StandardScaler + SVM/KNN
5. Evaluación: accuracy, precision, recall, f1, matriz de confusión, curvas de aprendizaje

## Solución de problemas

- "No se detectaron contornos": ajusta `MIN_CONTOUR_AREA`, `GAUSSIAN_KERNEL_SIZE` o iluminación/umbral.
- OpenCV headless (servidores): para mostrar/guardar figuras, usa matplotlib; no dependemos de `cv2.imshow`.
- ImportError scikit-learn: instala dependencias indicadas en Requisitos.

## Licencia

Uso académico/educativo. Ajusta según tus necesidades.
