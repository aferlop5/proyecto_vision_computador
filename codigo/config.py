"""
Configuración centralizada para la clasificación de tuercas, tornillos y arandelas.

Todas las variables aquí son constantes (MAYÚSCULAS) pensadas para ser importadas
por los distintos módulos del proyecto.
"""
from typing import Tuple, Dict, List

# ==========================
# RUTAS DE DATOS Y MODELOS
# ==========================
DATASET_PATH: str = "./dataset"
MODELOS_PATH: str = "./modelos"
# Carpeta de resultados (si no existe, se crea en runtime)
RESULTADOS_PATH: str = "./resultados"

# ==========================
# PREPROCESADO DE IMAGEN
# ==========================
# Tamaño de imagen para redimensionamiento (ancho, alto)
IMAGE_SIZE: Tuple[int, int] = (500, 500)

# Parámetros del filtro Gaussiano
GAUSSIAN_KERNEL: Tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: float = 1.5

# Morfología posterior al umbral para mejorar máscaras
MORFOLOGIA_POST_UMBRAL: bool = True
MORFOLOGIA_OPERACION: str = "cierre"  # opciones: 'apertura', 'cierre', 'erosion', 'dilatacion'
MORFOLOGIA_KERNEL: int = 3

# ==========================
# SEGMENTACIÓN
# ==========================
# Método de umbralado: Otsu. Si se usan otros métodos, cambiar esta bandera o añadir opciones.
UMBRAL_OTSU: bool = True

# ==========================
# CONTORNOS Y FILTRADO POR ÁREA
# ==========================
AREA_MIN: int = 1000
AREA_MAX: int = 50000

# Filtro de aspecto para descartar ruidos extremos (w/h vs h/w)
ASPECTO_MIN: float = 0.3
ASPECTO_MAX: float = 12.0

# Padding relativo alrededor del ROI del contorno
ROI_PADDING: float = 0.05

# ==========================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================
# Configuración para HOG
USE_HOG: bool = True
HOG_ORIENTACIONES: int = 9
HOG_PIXELS_PER_CELL: Tuple[int, int] = (8, 8)
HOG_CELDAS_X: int = 8  # número de celdas horizontales para dividir el ROI
HOG_CELDAS_Y: int = 8  # número de celdas verticales

# ==========================
# CLASIFICADORES
# ==========================
# Hiperparámetros para SVM y KNN
SVM_PARAMS: Dict[str, object] = {
    "C": 1.0,
    "kernel": "rbf",
    "class_weight": "balanced",
    "probability": True,
}

KNN_PARAMS: Dict[str, int] = {
    "n_neighbors": 5,
}

RF_PARAMS: Dict[str, object] = {
    "n_estimators": 300,
    "random_state": 42,
    "class_weight": "balanced",
}

# Parrillas de búsqueda para GridSearchCV (se usan si --tune)
GRID_SVM: Dict[str, List[object]] = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    "clf__kernel": ["rbf"],
}
GRID_KNN: Dict[str, List[object]] = {
    "clf__n_neighbors": [3, 5, 7, 11],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],
}
GRID_RF: Dict[str, List[object]] = {
    "clf__n_estimators": [200, 400, 800],
    "clf__max_depth": [None, 10, 20],
    "clf__max_features": ["sqrt", "log2"],
}

# Reducción de dimensionalidad opcional (útil con HOG)
USE_PCA: bool = True
PCA_COMPONENTS: int = 120
PCA_WHITEN: bool = False

# Embeddings profundos (pretrained CNN)
USE_DEEP_FEATURES: bool = True
DEEP_MODEL_NAME: str = "resnet18"
DEEP_FEATURES_DIM: int = 512

# ==========================================================
# UMBRALES PARA CLASIFICACIÓN POR CARACTERÍSTICAS (HEURÍSTICOS)
# ==========================================================
# Objetos alargados tienden a ser tornillos
UMBRAL_ASPECTO_TORNILLO: float = 1.7

# Mínimo de detección de aristas para tuercas (por ejemplo, fracción de píxeles borde/total)
UMBRAL_ARISTAS_TUERCA: float = 0.5

# Diferenciación suavizada entre tuercas y arandelas
UMBRAL_RATIO_AGUJERO_TUERCA_MAX: float = 0.7
UMBRAL_RATIO_AGUJERO_ARANDELA_MIN: float = 0.6
UMBRAL_SOLIDEZ_TUERCA_MAX: float = 0.9
UMBRAL_RECTANGULARIDAD_TUERCA_MIN: float = 0.5

# Relación área del agujero / área total para considerar arandela
UMBRAL_AGUJERO_ARANDELA: float = 0.3

# Para distinguir formas compactas (p. ej., perímetro^2 / (4π area))
UMBRAL_COMPACIDAD: float = 1.5

# ==========================
# CARACTERÍSTICAS PRINCIPALES
# ==========================
HOG_VECTOR_LENGTH: int = HOG_CELDAS_X * HOG_CELDAS_Y * HOG_ORIENTACIONES
HU_MOMENTS_LENGTH: int = 7

CARACTERISTICAS_PRINCIPALES: List[str] = [
    "relacion_aspecto",
    "circularidad",
    "compacidad",
    "solidez",
    "rectangularidad",
    "excentricidad",
    "numero_lados_aprox",
    "suavidad",
    "uniformidad",
    "hu_moments",
    "hog",
    "deep_features",
    "indice_aristas",
    "ratio_agujero",
    "tiene_aristas",
    "tiene_agujero",
]

