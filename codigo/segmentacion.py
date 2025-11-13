"""
Módulo de segmentación y análisis de contornos.
Incluye umbralización Otsu, detección de contornos y filtros por área/relación de aspecto,
así como utilidades para extracción de ROI y clasificación simple de orientación.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple, Optional, Any

# Imports robustos de módulos locales
try:
    from . import config as config  # type: ignore
except Exception:
    try:
        import config  # type: ignore
    except Exception:
        config = None  # type: ignore

# Intento de importar funciones de preprocesamiento
try:
    from .preprocesamiento import convertir_gris, normalizar_imagen  # type: ignore
except Exception:
    try:
        from preprocesamiento import convertir_gris, normalizar_imagen  # type: ignore
    except Exception:
        convertir_gris = None  # type: ignore
        normalizar_imagen = None  # type: ignore

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

LOGGER = logging.getLogger(__name__)


def _ensure_gray_uint8(imagen: Any) -> Any:
    """Garantiza imagen en escala de grises y uint8."""
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias no disponibles para estandarizar imagen: %s", e)
        return imagen

    img = imagen
    try:
        if convertir_gris is not None:
            img = convertir_gris(img)
        else:
            if len(getattr(img, "shape", [])) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if normalizar_imagen is not None:
            img = normalizar_imagen(img)
        else:
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        return img
    except Exception as e:
        LOGGER.error("Error asegurando gris/uint8: %s", e)
        return imagen


def umbralizar_otsu(imagen_gris: Any) -> Any:
    """Aplica umbralización Otsu y devuelve imagen binaria 0/255."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return imagen_gris

    img = _ensure_gray_uint8(imagen_gris)
    try:
        _thr, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaria
    except Exception as e:
        LOGGER.error("Error en umbralización Otsu: %s", e)
        return imagen_gris


def encontrar_contornos(imagen_binaria: Any) -> List[Any]:
    """
    Encuentra contornos usando cv2.findContours con RETR_EXTERNAL y CHAIN_APPROX_SIMPLE.
    Devuelve lista de contornos.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return []

    try:
        # Compatibilidad con distintas firmas de OpenCV
        res = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 3:
            _img, contornos, _hier = res
        else:
            contornos, _hier = res
        return contornos
    except Exception as e:
        LOGGER.error("Error encontrando contornos: %s", e)
        return []


def filtrar_contornos_por_area(
    contornos: Sequence[Any],
    area_min: Optional[float] = None,
    area_max: Optional[float] = None,
) -> List[Any]:
    """Filtra contornos cuyo área esté dentro de [area_min, area_max]."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return list(contornos)

    amin = area_min if area_min is not None else (getattr(config, "AREA_MIN", 1000) if config else 1000)
    amax = area_max if area_max is not None else (getattr(config, "AREA_MAX", 50000) if config else 50000)

    filtrados: List[Any] = []
    for c in contornos:
        try:
            a = float(cv2.contourArea(c))
            if amin <= a <= amax:
                filtrados.append(c)
        except Exception:
            continue
    return filtrados


def _aspect_ratio_from_contour(contorno: Any) -> Optional[float]:
    """Calcula la relación de aspecto max(w,h)/min(w,h) a partir del bounding box del contorno."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    try:
        x, y, w, h = cv2.boundingRect(contorno)
        if w == 0 or h == 0:
            return None
        a = max(w, h) / float(min(w, h))
        return a
    except Exception:
        return None


def filtrar_contornos_por_aspecto(
    contornos: Sequence[Any],
    umbral_min: float = 1.0,
    umbral_max: float = 3.0,
) -> List[Any]:
    """
    Filtra contornos cuya relación de aspecto esté en [umbral_min, umbral_max].
    """
    filtrados: List[Any] = []
    for c in contornos:
        ra = _aspect_ratio_from_contour(c)
        if ra is None:
            continue
        if umbral_min <= ra <= umbral_max:
            filtrados.append(c)
    return filtrados


def extraer_roi(imagen_original: Any, contorno: Any, padding_rel: Optional[float] = None) -> Any:
    """Extrae el recorte rectangular (ROI) que envuelve al contorno con padding opcional.

    padding_rel: fracción del tamaño (0..0.5 aprox) a expandir alrededor del bbox.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias no disponibles para extraer ROI: %s", e)
        return imagen_original

    try:
        x, y, w, h = cv2.boundingRect(contorno)
        h_img, w_img = imagen_original.shape[:2]
        pad = float(padding_rel if padding_rel is not None else (getattr(config, "ROI_PADDING", 0.0) if config else 0.0))
        dx = int(w * pad)
        dy = int(h * pad)
        x0 = max(0, x - dx)
        y0 = max(0, y - dy)
        x1 = min(w_img, x + w + dx)
        y1 = min(h_img, y + h + dy)
        if x0 >= x1 or y0 >= y1:
            return imagen_original
        return imagen_original[y0:y1, x0:x1].copy()
    except Exception as e:
        LOGGER.error("Error extrayendo ROI: %s", e)
        return imagen_original


def dibujar_contornos(imagen: Any, contornos: Sequence[Any]) -> Any:
    """Dibuja los contornos sobre una copia de la imagen y la devuelve."""
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return imagen

    try:
        # Si la imagen es gris, convertir a BGR para colorear contornos
        if len(getattr(imagen, "shape", [])) == 2:
            copia = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            copia = imagen.copy()
        cv2.drawContours(copia, contornos, -1, (0, 255, 0), 2)
        return copia
    except Exception as e:
        LOGGER.error("Error dibujando contornos: %s", e)
        return imagen


def determinar_orientacion_objeto(contorno: Any) -> str:
    """
    Clasificación simplificada de orientación:
    - "lateral" si la relación de aspecto >= UMBRAL_ASPECTO_TORNILLO (más alargado)
    - "frontal" en caso contrario.
    """
    umbral = getattr(config, "UMBRAL_ASPECTO_TORNILLO", 1.7) if config else 1.7
    ra = _aspect_ratio_from_contour(contorno)
    if ra is None:
        return "frontal"
    return "lateral" if ra >= umbral else "frontal"


def estimar_posicion_relativa(contorno: Any) -> str:
    """
    Determina si el objeto está de frente o de lado (heurística por relación de aspecto).
    Devuelve: "de_frente" o "de_lado".
    """
    orient = determinar_orientacion_objeto(contorno)
    return "de_lado" if orient == "lateral" else "de_frente"
