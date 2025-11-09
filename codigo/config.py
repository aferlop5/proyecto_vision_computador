"""
Configuración del proyecto de visión para clasificación de tuercas, tornillos y arandelas.

Todas las variables son constantes en MAYÚSCULAS para facilitar su uso y trazabilidad.
"""

from pathlib import Path

# =============================
# RUTAS DEL PROYECTO
# =============================
# Raíz del proyecto: carpeta padre de "codigo/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rutas relativas a datos y modelos (resueltas desde PROJECT_ROOT)
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELOS_DIR = PROJECT_ROOT / "modelos"
RESULTADOS_DIR = PROJECT_ROOT / "resultados"

# Clases disponibles en el dataset
CLASSES = ("tuercas", "tornillos", "arandelas")
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASSES)}


# =============================
# PARÁMETROS DE IMAGEN
# =============================
# Tamaño para redimensionamiento (ancho, alto)
IMAGE_SIZE = (256, 256)

# Interpolación para redimensionar: "nearest", "linear", "area", "cubic", "lanczos"
RESIZE_INTERPOLATION = "area"


# =============================
# PREPROCESAMIENTO: FILTRO GAUSSIANO
# =============================
# Tamaño de kernel impar (kx, ky) y sigmas
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA_X = 0.0
GAUSSIAN_SIGMA_Y = 0.0


# =============================
# SEGMENTACIÓN: THRESHOLDS Y MORFOLOGÍA
# =============================
# Método de umbralización: "fixed", "adaptive-mean", "adaptive-gaussian", "otsu"
THRESH_METHOD = "otsu"

# Umbral fijo (si THRESH_METHOD == "fixed")
THRESH_BINARY = 127
THRESH_MAXVAL = 255

# Umbralización adaptativa (si THRESH_METHOD empieza por "adaptive-")
ADAPTIVE_METHOD = "gaussian"  # "mean" o "gaussian"
ADAPTIVE_BLOCK_SIZE = 11       # Debe ser impar y >= 3
ADAPTIVE_C = 2                 # Constante restada del promedio local

# Operaciones morfológicas para limpiar máscaras
MORPH_OP = "open"              # "open", "close", "erode", "dilate"
MORPH_KERNEL_SIZE = (3, 3)
MORPH_ITERATIONS = 1


# =============================
# CONTORNOS
# =============================
# Áreas mínima y máxima para filtrar contornos (en píxeles)
MIN_CONTOUR_AREA = 200.0
MAX_CONTOUR_AREA = 1_000_000.0


# =============================
# EXTRACCIÓN DE CARACTERÍSTICAS: HOG
# =============================
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"      # "L1", "L1-sqrt", "L2", "L2-Hys"
HOG_TRANSFORM_SQRT = True


# =============================
# CLASIFICACIÓN: HIPERPARÁMETROS
# =============================
# SVM
SVM_C = 1.0
SVM_KERNEL = "rbf"              # "linear", "poly", "rbf", "sigmoid"
SVM_GAMMA = "scale"            # "scale", "auto" o un valor float


# KNN
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"       # "uniform" o "distance"
KNN_METRIC = "minkowski"       # "euclidean", "manhattan", "minkowski", etc.
KNN_P = 2                       # p=2 => euclídea, p=1 => manhattan


# =============================
# EXPERIMENTACIÓN / REPRODUCIBILIDAD
# =============================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5


# =============================
# NOMBRADO DE MODELOS / ARTEFACTOS
# =============================
MODEL_FILENAME_SVM = "svm_hog.joblib"
MODEL_FILENAME_KNN = "knn_hog.joblib"


__all__ = [
	# Rutas
	"PROJECT_ROOT", "DATASET_DIR", "MODELOS_DIR", "RESULTADOS_DIR",
	# Dataset
	"CLASSES", "CLASS_TO_LABEL",
	# Imagen
	"IMAGE_SIZE", "RESIZE_INTERPOLATION",
	# Gaussiano
	"GAUSSIAN_KERNEL_SIZE", "GAUSSIAN_SIGMA_X", "GAUSSIAN_SIGMA_Y",
	# Segmentación
	"THRESH_METHOD", "THRESH_BINARY", "THRESH_MAXVAL",
	"ADAPTIVE_METHOD", "ADAPTIVE_BLOCK_SIZE", "ADAPTIVE_C",
	"MORPH_OP", "MORPH_KERNEL_SIZE", "MORPH_ITERATIONS",
	# Contornos
	"MIN_CONTOUR_AREA", "MAX_CONTOUR_AREA",
	# HOG
	"HOG_ORIENTATIONS", "HOG_PIXELS_PER_CELL", "HOG_CELLS_PER_BLOCK",
	"HOG_BLOCK_NORM", "HOG_TRANSFORM_SQRT",
	# Clasificación
	"SVM_C", "SVM_KERNEL", "SVM_GAMMA",
	"KNN_N_NEIGHBORS", "KNN_WEIGHTS", "KNN_METRIC", "KNN_P",
	# Experimentos
	"RANDOM_STATE", "TEST_SIZE", "N_FOLDS",
	# Artefactos
	"MODEL_FILENAME_SVM", "MODEL_FILENAME_KNN",
]

