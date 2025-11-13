"""
Extracción de características geométricas, de forma y textura para objetos (tuercas/tornillos/arandelas).
Incluye utilidades para aristas, agujeros y momentos invariantes de Hu.
"""
from __future__ import annotations

import logging
from math import pi, sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ======================
# CARACTERÍSTICAS GEOMÉTRICAS
# ======================

def relacion_aspecto(contorno: Any) -> Optional[float]:
    """Relación de aspecto = max(w,h)/min(w,h) del bounding box."""
    try:
        import cv2  # type: ignore
        x, y, w, h = cv2.boundingRect(contorno)
        if w == 0 or h == 0:
            return None
        return max(w, h) / float(min(w, h))
    except Exception as e:
        LOGGER.error("Error calculando relación de aspecto: %s", e)
        return None


def solidez(contorno: Any) -> Optional[float]:
    """Solidez = área / área del casco convexo."""
    try:
        import cv2  # type: ignore
        area = float(cv2.contourArea(contorno))
        hull = cv2.convexHull(contorno)
        hull_area = float(cv2.contourArea(hull))
        if hull_area == 0:
            return None
        return area / hull_area
    except Exception as e:
        LOGGER.error("Error calculando solidez: %s", e)
        return None


def circularidad(contorno: Any) -> Optional[float]:
    """Circularidad = 4π·área / perímetro^2 (1 para círculo)."""
    try:
        import cv2  # type: ignore
        area = float(cv2.contourArea(contorno))
        per = float(cv2.arcLength(contorno, True))
        if per == 0:
            return None
        return (4.0 * pi * area) / (per * per)
    except Exception as e:
        LOGGER.error("Error calculando circularidad: %s", e)
        return None


def compacidad(contorno: Any) -> Optional[float]:
    """Compacidad = perímetro^2 / (4π·área); = 1 para círculo, >1 menos compacto."""
    try:
        import cv2  # type: ignore
        area = float(cv2.contourArea(contorno))
        per = float(cv2.arcLength(contorno, True))
        if area == 0:
            return None
        return (per * per) / (4.0 * pi * area)
    except Exception as e:
        LOGGER.error("Error calculando compacidad: %s", e)
        return None


def rectangularidad(contorno: Any) -> Optional[float]:
    """Rectangularidad = área / (ancho*alto) del rectángulo envolvente."""
    try:
        import cv2  # type: ignore
        area = float(cv2.contourArea(contorno))
        x, y, w, h = cv2.boundingRect(contorno)
        denom = float(w * h)
        if denom == 0:
            return None
        return area / denom
    except Exception as e:
        LOGGER.error("Error calculando rectangularidad: %s", e)
        return None


def excentricidad(contorno: Any) -> Optional[float]:
    """Excentricidad de la elipse ajustada: sqrt(1 - (b/a)^2), 0 circular, ->1 alargado."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        if len(contorno) < 5:
            # Fallback con momentos para elipses
            M = cv2.moments(contorno)
            mu20 = M.get("mu20", 0.0)
            mu02 = M.get("mu02", 0.0)
            mu11 = M.get("mu11", 0.0)
            cov_xx = mu20 / M.get("m00", 1.0)
            cov_yy = mu02 / M.get("m00", 1.0)
            cov_xy = mu11 / M.get("m00", 1.0)
            # Autovalores de la covarianza (aprox proporcionales a a^2 y b^2)
            trace = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy * cov_xy
            disc = max(trace * trace - 4 * det, 0.0)
            l1 = 0.5 * (trace + sqrt(disc))
            l2 = 0.5 * (trace - sqrt(disc))
            a2 = max(l1, l2)
            b2 = min(l1, l2)
            if a2 <= 0:
                return None
            return float(sqrt(max(1.0 - (b2 / a2), 0.0)))
        ellipse = cv2.fitEllipse(contorno)
        (cx, cy), (maj, min_), angle = ellipse
        a = max(maj, min_) / 2.0
        b = min(maj, min_) / 2.0
        if a == 0:
            return None
        return float(sqrt(max(1.0 - (b / a) ** 2, 0.0)))
    except Exception as e:
        LOGGER.error("Error calculando excentricidad: %s", e)
        return None


# ======================
# DETECCIÓN DE CARACTERÍSTICAS CLAVE
# ======================

def _approx_vertices(contorno: Any, epsilon_ratio: float = 0.02) -> int:
    """Devuelve número de vértices de la aproximación poligonal."""
    try:
        import cv2  # type: ignore
        peri = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon_ratio * peri, True)
        return int(len(approx))
    except Exception:
        return 0


def detectar_aristas(contorno: Any, imagen: Any) -> float:
    """
    Estima un índice de 'aristas' usando la aproximación poligonal.
    Devuelve un valor en [0,1] aproximando a 1 cuando el nº de lados >= 6 (hexagonal).
    """
    n = _approx_vertices(contorno)
    # Normalización simple: 6 lados -> 1.0; menos caras proporcionalmente; tope en 1.0
    if n <= 0:
        return 0.0
    score = min(n / 6.0, 1.0)
    return float(score)


def detectar_agujero(contorno: Any, imagen_binaria: Any) -> float:
    """
    Calcula la relación área_agujero/área_total dentro del contorno utilizando una máscara.
    Requiere imagen binaria (0/255) con el objeto en 255 y fondo en 0.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        # Máscara del contorno relleno
        x, y, w, h = cv2.boundingRect(contorno)
        if w == 0 or h == 0:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        contorno_shift = contorno.copy()
        contorno_shift[:, 0, 0] -= x
        contorno_shift[:, 0, 1] -= y
        cv2.drawContours(mask, [contorno_shift], -1, 255, thickness=-1)

        # Recorte correspondiente del binario
        crop = imagen_binaria[y : y + h, x : x + w]
        # Píxeles esperados como objeto (mask==255) que están a 0 en la imagen -> agujero
        total_region = int((mask == 255).sum())
        if total_region == 0:
            return 0.0
        agujero_pix = int(((mask == 255) & (crop == 0)).sum())
        return float(agujero_pix) / float(total_region)
    except Exception as e:
        LOGGER.error("Error detectando agujero: %s", e)
        return 0.0


def calcular_numero_lados(contorno: Any) -> int:
    """Número de lados aproximado mediante approxPolyDP."""
    return _approx_vertices(contorno)


def extraer_momentos_hu(contorno: Any) -> List[float]:
    """Devuelve los 7 momentos de Hu en escala logarítmica con signo preservado."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        M = cv2.moments(contorno)
        hu = cv2.HuMoments(M).flatten()
        # Escala log con signo
        out: List[float] = []
        for v in hu:
            s = -1.0 if v < 0 else 1.0
            val = 0.0 if v == 0 else s * float(abs(v))
            # Para estabilidad, usar log10 si v!=0
            if v != 0:
                import math
                val = s * math.log10(abs(v))
            out.append(val)
        return out
    except Exception as e:
        LOGGER.error("Error calculando momentos de Hu: %s", e)
        return [0.0] * 7


# ======================
# CARACTERÍSTICAS DE TEXTURA
# ======================

def _to_gray_uint8(imagen: Any) -> Any:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        if len(getattr(imagen, "shape", [])) == 2:
            gray = imagen
        else:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        return gray
    except Exception:
        return imagen


def calcular_suavidad(imagen: Any) -> Optional[float]:
    """Suavidad como desviación estándar normalizada de intensidades en gris (0..1 aprox)."""
    try:
        import numpy as np  # type: ignore
        g = _to_gray_uint8(imagen)
        std = float(np.std(g))
        return std / 255.0
    except Exception as e:
        LOGGER.error("Error calculando suavidad: %s", e)
        return None


def calcular_uniformidad(imagen: Any) -> Optional[float]:
    """Uniformidad (energía del histograma): sum(p_i^2) sobre 256 bins; 1 si un solo nivel."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        g = _to_gray_uint8(imagen)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        total = float(hist.sum()) or 1.0
        p = hist / total
        return float((p * p).sum())
    except Exception as e:
        LOGGER.error("Error calculando uniformidad: %s", e)
        return None


# ======================
# HOG SIMPLE (sin dependencias externas)
# ======================

def _hog_simple(
    imagen_gray_uint8: Any,
    bins: int = 9,
    celdas_x: int = 8,
    celdas_y: int = 8,
) -> List[float]:
    """
    Calcula un descriptor HOG simple:
    - Gradientes con Sobel
    - Orientaciones en [0,180)
    - Acumulación por celdas (rejilla celdas_x x celdas_y)
    Devuelve el vector concatenado (celdas_x*celdas_y*bins,) normalizado L2.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias para HOG no disponibles: %s", e)
        return []

    try:
        g = imagen_gray_uint8
        h, w = g.shape[:2]
        if h == 0 or w == 0:
            return []
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # Orientaciones unsigned [0,180)
        ang = (ang % 180.0)

        # Rejilla
        cell_w = max(1, w // celdas_x)
        cell_h = max(1, h // celdas_y)
        vec: List[float] = []
        bin_width = 180.0 / float(max(1, bins))

        for cy in range(celdas_y):
            for cx in range(celdas_x):
                x0 = cx * cell_w
                y0 = cy * cell_h
                x1 = w if cx == celdas_x - 1 else (x0 + cell_w)
                y1 = h if cy == celdas_y - 1 else (y0 + cell_h)
                m = mag[y0:y1, x0:x1].reshape(-1)
                a = ang[y0:y1, x0:x1].reshape(-1)
                # Histograma por celdas
                hist = [0.0] * bins
                for mi, ai in zip(m.tolist(), a.tolist()):
                    b = int(min(bins - 1, max(0, int(ai // bin_width))))
                    hist[b] += float(mi)
                # Normalización L2 por celda
                import math
                norm = math.sqrt(sum(hh * hh for hh in hist)) or 1.0
                hist = [hh / norm for hh in hist]
                vec.extend(hist)
        # Normalización global ligera
        import math
        n2 = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n2 for v in vec]
    except Exception as e:
        LOGGER.error("Error calculando HOG simple: %s", e)
        return []


# ======================
# FUNCIÓN PRINCIPAL
# ======================

def extraer_caracteristicas_completas(imagen: Any, contorno: Any, imagen_binaria: Optional[Any] = None) -> Dict[str, Any]:
    """
    Extrae un conjunto de características y devuelve un diccionario.
    Si se proporciona imagen_binaria, se calcula la relación de agujero.
    """
    feats: Dict[str, Any] = {}

    # Geométricas
    feats["relacion_aspecto"] = relacion_aspecto(contorno)
    feats["solidez"] = solidez(contorno)
    feats["circularidad"] = circularidad(contorno)
    feats["compacidad"] = compacidad(contorno)
    feats["rectangularidad"] = rectangularidad(contorno)
    feats["excentricidad"] = excentricidad(contorno)

    # Clave (aristas y agujero)
    indice_aristas = detectar_aristas(contorno, imagen)
    feats["indice_aristas"] = indice_aristas
    # booleano derivado si supera umbral de config
    umbral_aristas = getattr(config, "UMBRAL_ARISTAS_TUERCA", 0.6) if config else 0.6
    feats["tiene_aristas"] = bool(indice_aristas >= umbral_aristas)

    if imagen_binaria is not None:
        ratio_agujero = detectar_agujero(contorno, imagen_binaria)
    else:
        ratio_agujero = 0.0
    feats["ratio_agujero"] = ratio_agujero
    umbral_agujero = getattr(config, "UMBRAL_AGUJERO_ARANDELA", 0.3) if config else 0.3
    feats["tiene_agujero"] = bool(ratio_agujero >= umbral_agujero)

    # Número de lados aproximado y momentos de Hu
    n_lados = calcular_numero_lados(contorno)
    feats["numero_lados_aprox"] = n_lados
    feats["hu_moments"] = extraer_momentos_hu(contorno)

    # Textura en ROI si es posible
    try:
        import cv2  # type: ignore
        # ROI con padding configurable
        try:
            from .segmentacion import extraer_roi  # type: ignore
        except Exception:
            try:
                from segmentacion import extraer_roi  # type: ignore
            except Exception:
                extraer_roi = None  # type: ignore

        if extraer_roi is not None:
            pad = float(getattr(config, "ROI_PADDING", 0.05) if config else 0.05)
            roi = extraer_roi(imagen, contorno, padding_rel=pad)
        else:
            x, y, w, h = cv2.boundingRect(contorno)
            roi = imagen[y : y + h, x : x + w]
        feats["suavidad"] = calcular_suavidad(roi)
        feats["uniformidad"] = calcular_uniformidad(roi)

        # HOG opcional sobre ROI
        if getattr(config, "USE_HOG", True) if config else True:
            g = _to_gray_uint8(roi)
            # Redimensionar a rejilla exacta (celdas_x * px_cell, celdas_y * px_cell)
            cells_x = int(getattr(config, "HOG_CELDAS_X", 8) if config else 8)
            cells_y = int(getattr(config, "HOG_CELDAS_Y", 8) if config else 8)
            px_cell = getattr(config, "HOG_PIXELS_PER_CELL", (8, 8)) if config else (8, 8)
            target_w = max(8, int(cells_x * px_cell[0]))
            target_h = max(8, int(cells_y * px_cell[1]))
            g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_AREA)
            hog_bins = int(getattr(config, "HOG_ORIENTACIONES", 9) if config else 9)
            feats["hog"] = _hog_simple(g, bins=hog_bins, celdas_x=cells_x, celdas_y=cells_y)
        else:
            feats["hog"] = []
    except Exception as e:
        LOGGER.info("No se pudo extraer textura de ROI: %s", e)
        feats["suavidad"] = None
        feats["uniformidad"] = None
        feats["hog"] = []

    return feats
