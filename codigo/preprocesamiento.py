"""
Módulo de preprocesamiento de imágenes usando parámetros de configuracion en codigo/config.py.
Incluye funciones: redimensionar, conversión a gris, filtro gaussiano, ecualización, normalización,
morfología y binarización (Otsu por defecto).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Any

# Importación robusta de config
try:
    from . import config as config  # type: ignore
except Exception:
    try:
        import config  # type: ignore
    except Exception:
        config = None  # type: ignore

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

LOGGER = logging.getLogger(__name__)


def redimensionar_imagen(imagen: Any, tamaño: Optional[Tuple[int, int]] = None) -> Any:
    """
    Redimensiona la imagen al tamaño especificado; por defecto usa config.IMAGE_SIZE.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible: %s", e)
        return imagen

    target = tamaño or (getattr(config, "IMAGE_SIZE", None) if config else None) or (500, 500)
    try:
        return cv2.resize(imagen, target, interpolation=cv2.INTER_AREA)
    except Exception as e:
        LOGGER.error("Error redimensionando a %s: %s", target, e)
        return imagen


def convertir_gris(imagen: Any) -> Any:
    """Convierte una imagen BGR/RGB a gris. Si ya es 1 canal, la devuelve tal cual."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible: %s", e)
        return imagen

    try:
        shape = getattr(imagen, "shape", None)
        if shape is None:
            return imagen
        if len(shape) == 2:
            return imagen
        # Asumimos BGR (convención OpenCV)
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        LOGGER.error("Error convirtiendo a gris: %s", e)
        return imagen


def aplicar_filtro_gaussiano(imagen: Any, kernel: Optional[Tuple[int, int]] = None, sigma: Optional[float] = None) -> Any:
    """
    Aplica un filtro Gaussiano usando parámetros de config si no se pasan explícitamente.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible: %s", e)
        return imagen

    k = kernel or (getattr(config, "GAUSSIAN_KERNEL", None) if config else None) or (5, 5)
    s = sigma if sigma is not None else ((getattr(config, "GAUSSIAN_SIGMA", None) if config else None) or 1.5)
    try:
        return cv2.GaussianBlur(imagen, k, s)
    except Exception as e:
        LOGGER.error("Error aplicando filtro gaussiano (k=%s, sigma=%s): %s", k, s, e)
        return imagen


def ecualizar_histograma(imagen_gris: Any) -> Any:
    """
    Ecualiza histograma de una imagen en escala de grises con cv2.equalizeHist.
    Si recibe color, primero convierte a gris.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible: %s", e)
        return imagen_gris

    try:
        # Asegurar gris
        shape = getattr(imagen_gris, "shape", None)
        if shape is None:
            return imagen_gris
        if len(shape) != 2:
            imagen_gris = convertir_gris(imagen_gris)
        return cv2.equalizeHist(imagen_gris)
    except Exception as e:
        LOGGER.error("Error en ecualización de histograma: %s", e)
        return imagen_gris


def normalizar_imagen(imagen: Any) -> Any:
    """
    Normaliza la imagen al rango [0,255] y tipo uint8 si no lo está.
    Mantiene la imagen si ya es uint8.
    """
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias para normalización no disponibles: %s", e)
        return imagen

    try:
        if imagen.dtype == np.uint8:
            return imagen
        # Normalización min-max a [0,255]
        norm = cv2.normalize(imagen, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    except Exception as e:
        LOGGER.error("Error normalizando imagen: %s", e)
        return imagen


def aplicar_morfologia(imagen: Any, operacion: str = "apertura", kernel_size: int = 3) -> Any:
    """
    Aplica operación morfológica: 'apertura', 'cierre', 'erosion', 'dilatacion'.
    kernel_size es el tamaño de un kernel cuadrado de unos.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV/NumPy no disponibles: %s", e)
        return imagen

    op = operacion.lower()
    k = max(1, int(kernel_size))
    kernel = np.ones((k, k), np.uint8)

    try:
        if op == "apertura":
            return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
        if op == "cierre":
            return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
        if op == "erosion":
            return cv2.erode(imagen, kernel, iterations=1)
        if op == "dilatacion":
            return cv2.dilate(imagen, kernel, iterations=1)
        LOGGER.warning("Operación morfológica desconocida '%s'. Devolviendo imagen original.", operacion)
        return imagen
    except Exception as e:
        LOGGER.error("Error aplicando morfología '%s': %s", operacion, e)
        return imagen


def binarizar_imagen(imagen: Any, metodo: str = "otsu") -> Any:
    """
    Binariza la imagen.
    - metodo='otsu' (por defecto): usa Otsu si config.UMBRAL_OTSU es True o si se fuerza por parámetro.
    - metodo='binary': umbral fijo 127.
    - metodo='adaptive': umbral adaptativo gaussiano.
    Devuelve una imagen binaria (uint8 0/255).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV/NumPy no disponibles: %s", e)
        return imagen

    try:
        # Asegurar escala de grises uint8
        gris = convertir_gris(imagen)
        gris = normalizar_imagen(gris)

        m = metodo.lower()
        if m == "otsu" and ((getattr(config, "UMBRAL_OTSU", True) if config else True)):
            _thr, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Morfología posterior para consolidar regiones
            if getattr(config, "MORFOLOGIA_POST_UMBRAL", False) if config else False:
                try:
                    op = getattr(config, "MORFOLOGIA_OPERACION", "cierre") if config else "cierre"
                    ks = int(getattr(config, "MORFOLOGIA_KERNEL", 3) if config else 3)
                    binaria = aplicar_morfologia(binaria, operacion=op, kernel_size=ks)
                except Exception:
                    pass
            return binaria
        elif m == "adaptive":
            binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            if getattr(config, "MORFOLOGIA_POST_UMBRAL", False) if config else False:
                try:
                    op = getattr(config, "MORFOLOGIA_OPERACION", "cierre") if config else "cierre"
                    ks = int(getattr(config, "MORFOLOGIA_KERNEL", 3) if config else 3)
                    binaria = aplicar_morfologia(binaria, operacion=op, kernel_size=ks)
                except Exception:
                    pass
            return binaria
        else:  # 'binary' u otros
            _thr, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
            if getattr(config, "MORFOLOGIA_POST_UMBRAL", False) if config else False:
                try:
                    op = getattr(config, "MORFOLOGIA_OPERACION", "cierre") if config else "cierre"
                    ks = int(getattr(config, "MORFOLOGIA_KERNEL", 3) if config else 3)
                    binaria = aplicar_morfologia(binaria, operacion=op, kernel_size=ks)
                except Exception:
                    pass
            return binaria
    except Exception as e:
        LOGGER.error("Error en binarización (metodo=%s): %s", metodo, e)
        return imagen


def preprocesar_para_segmentacion(imagen: Any) -> Any:
    """Preprocesado específico para segmentación robusta.

    Pipeline pensado para obtener bordes más estables frente a cambios de
    iluminación y ruido:
    - Conversión a gris
    - Ecualización adaptativa del histograma (CLAHE)
    - Suavizado que preserve bordes (mediana o bilateral según config)
    Devuelve imagen en gris uint8 lista para binarización/edges.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias para preprocesar segmentacion no disponibles: %s", e)
        return imagen

    try:
        gris = convertir_gris(imagen)
        gris = normalizar_imagen(gris)

        # CLAHE para compensar variaciones de iluminación
        clip = float(getattr(config, "CLAHE_CLIP", 2.0) if config else 2.0)
        tile = getattr(config, "CLAHE_TILE_GRID", (8, 8)) if config else (8, 8)
        try:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
            gris = clahe.apply(gris)
        except Exception:
            pass

        # Suavizado que preserve bordes
        blur_tipo = str(getattr(config, "SEGMENTACION_BLUR_TIPO", "mediana") if config else "mediana").lower()
        k = int(getattr(config, "SEGMENTACION_BLUR_K", 5) if config else 5)
        k = k if k % 2 == 1 else k + 1  # aseguramos impar
        try:
            if blur_tipo == "bilateral":
                gris = cv2.bilateralFilter(gris, d=k, sigmaColor=75, sigmaSpace=75)
            elif blur_tipo == "gauss":
                gris = cv2.GaussianBlur(gris, (k, k), 1.5)
            else:  # mediana por defecto
                gris = cv2.medianBlur(gris, k)
        except Exception:
            pass
        return gris
    except Exception as e:
        LOGGER.error("Error en preprocesar_para_segmentacion: %s", e)
        return imagen
