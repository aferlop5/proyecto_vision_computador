"""Extracción de características geométricas y descriptores locales/globales.

Incluye:
- Características geométricas: relación de aspecto, solidez, circularidad, número de agujeros, momentos de Hu
- Descriptores: HOG, ORB (agregado), histograma de color (HSV)

Función principal:
	extraer_caracteristicas_completas(imagen, contorno)
Retorna un diccionario con componentes y un vector concatenado.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging

try:
	import cv2  # type: ignore
	import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
	cv2 = None  # type: ignore
	np = None  # type: ignore
	raise RuntimeError("Se requieren 'opencv-python' (cv2) y 'numpy' para extracción de características.") from e

try:
	from . import config as CFG
except Exception as e:  # pragma: no cover
	raise RuntimeError("No se pudo importar config.py") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# =============================
# Características geométricas
# =============================

def relacion_aspecto(contorno: Any) -> float:
	try:
		x, y, w, h = cv2.boundingRect(contorno)
		if h <= 0:
			return 0.0
		return float(w) / float(h)
	except Exception as e:
		logger.warning("relacion_aspecto falló: %s", e)
		return 0.0


def solidez(contorno: Any) -> float:
	try:
		area = float(cv2.contourArea(contorno))
		if area <= 0:
			return 0.0
		hull = cv2.convexHull(contorno)
		hull_area = float(cv2.contourArea(hull))
		if hull_area <= 0:
			return 0.0
		return area / hull_area
	except Exception as e:
		logger.warning("solidez falló: %s", e)
		return 0.0


def circularidad(contorno: Any) -> float:
	try:
		area = float(cv2.contourArea(contorno))
		per = float(cv2.arcLength(contorno, True))
		if per == 0:
			return 0.0
		circ = 4.0 * np.pi * area / (per * per)
		# Clamp a [0,1.5] por robustez numérica
		return float(np.clip(circ, 0.0, 1.5))
	except Exception as e:
		logger.warning("circularidad falló: %s", e)
		return 0.0


def numero_agujeros(contorno: Any, imagen_binaria: Any) -> int:
	"""Cuenta agujeros internos al contorno dibujándolo como máscara.

	No depende estrictamente de imagen_binaria; crea su propia máscara a partir del contorno.
	Estrategia: dibujar el contorno lleno en una máscara del rectángulo envolvente, invertir, flood-fill
	desde el borde para eliminar fondo externo y contar componentes conectados restantes (agujeros).
	"""
	try:
		x, y, w, h = cv2.boundingRect(contorno)
		if w <= 0 or h <= 0:
			return 0
		# Máscara del ROI
		roi_mask = np.zeros((h, w), dtype=np.uint8)
		# Desplazar contorno a coords locales del ROI
		contorno_local = contorno - np.array([[x, y]])
		cv2.drawContours(roi_mask, [contorno_local], -1, 255, thickness=cv2.FILLED)

		# Invertir: fondo=255 dentro agujeros y fuera del objeto
		bg = 255 - roi_mask

		# FloodFill desde bordes para quitar fondo externo
		ff = bg.copy()
		mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
		cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=64)
		cv2.floodFill(ff, mask, seedPoint=(w - 1, 0), newVal=64)
		cv2.floodFill(ff, mask, seedPoint=(0, h - 1), newVal=64)
		cv2.floodFill(ff, mask, seedPoint=(w - 1, h - 1), newVal=64)

		# Los píxeles 255 restantes son agujeros internos
		holes = np.zeros_like(ff, dtype=np.uint8)
		holes[ff == 255] = 1
		num, _labels = cv2.connectedComponents(holes, connectivity=8)
		# connectedComponents cuenta el fondo como 0, el resto son componentes
		return int(num)
	except Exception as e:
		logger.warning("numero_agujeros falló: %s", e)
		return 0


def momentos_hu(contorno: Any):
	try:
		m = cv2.moments(contorno)
		hu = cv2.HuMoments(m).flatten()
		# Transformación logarítmica con signo preservado
		with np.errstate(divide='ignore'):
			hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
		return hu_log.astype(np.float32)
	except Exception as e:
		logger.warning("momentos_hu falló: %s", e)
		return np.zeros(7, dtype=np.float32)


# =============================
# Descriptores
# =============================

def _ensure_gray(img: Any) -> Any:
	if img.ndim == 2:
		return img
	if img.ndim == 3:
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img


def extraer_hog(imagen: Any):
	"""Extrae HOG con parámetros aproximados desde CFG.*.

	OpenCV requiere que winSize sea múltiplo del cellSize. Se ajusta a la baja.
	"""
	try:
		# Preparar tamaños
		H, W = imagen.shape[:2]
		cx, cy = CFG.HOG_PIXELS_PER_CELL
		bx, by = CFG.HOG_CELLS_PER_BLOCK
		win_w = (W // cx) * cx
		win_h = (H // cy) * cy
		if win_w < cx * bx:
			win_w = cx * bx
		if win_h < cy * by:
			win_h = cy * by
		img = cv2.resize(imagen, (win_w, win_h), interpolation=cv2.INTER_AREA)
		img_gray = _ensure_gray(img)

		winSize = (win_w, win_h)
		cellSize = (cx, cy)
		blockSize = (bx * cx, by * cy)
		blockStride = (cx, cy)
		nbins = int(CFG.HOG_ORIENTATIONS)

		hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
		hog.gamma_correction = bool(getattr(CFG, "HOG_TRANSFORM_SQRT", True))
		desc = hog.compute(img_gray)
		if desc is None:
			return np.zeros(1, dtype=np.float32)
		return desc.flatten().astype(np.float32)
	except Exception as e:
		logger.warning("extraer_hog falló: %s", e)
		return np.zeros(1, dtype=np.float32)


def extraer_orb(imagen: Any, n_features: int = 500):
	"""Extrae descriptores ORB y los agrega (media y desviación), tamaño fijo 64.

	Si no hay descriptores, retorna vector de ceros.
	"""
	try:
		orb = cv2.ORB_create(nfeatures=n_features)
		img_gray = _ensure_gray(imagen)
		kps, des = orb.detectAndCompute(img_gray, None)
		if des is None or len(des) == 0:
			return np.zeros(64, dtype=np.float32)
		des = des.astype(np.float32)
		mean = des.mean(axis=0)
		std = des.std(axis=0)
		feat = np.concatenate([mean, std], axis=0)
		return feat.astype(np.float32)
	except Exception as e:
		logger.warning("extraer_orb falló: %s", e)
		return np.zeros(64, dtype=np.float32)


def extraer_histograma(imagen: Any, espacio: str = "hsv", bins: Tuple[int, int, int] = (8, 8, 8)):
	"""Histograma de color normalizado (HSV por defecto)."""
	try:
		if espacio.lower() == "hsv":
			img = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) if imagen.ndim == 3 else cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			ranges = [0, 180, 0, 256, 0, 256]
		else:  # rgb
			img = imagen if imagen.ndim == 3 else cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
			ranges = [0, 256, 0, 256, 0, 256]
		hist = cv2.calcHist([img], [0, 1, 2], None, list(bins), ranges)
		hist = cv2.normalize(hist, None).flatten().astype(np.float32)
		return hist
	except Exception as e:
		logger.warning("extraer_histograma falló: %s", e)
		return np.zeros(int(np.prod(bins)), dtype=np.float32)


# =============================
# Función principal
# =============================

def _recortar_roi(imagen: Any, contorno: Any) -> Any:
	x, y, w, h = cv2.boundingRect(contorno)
	return imagen[y:y + h, x:x + w]


def extraer_caracteristicas_completas(imagen: Any, contorno: Any) -> Dict[str, Any]:
	"""Extrae conjunto combinado de características para clasificación.

	Retorna un diccionario con las claves: 'geom', 'hog', 'orb', 'hist', 'hu', 'vector'.
	"""
	try:
		roi = _recortar_roi(imagen, contorno)
		# Descriptores de imagen (ROI)
		hog = extraer_hog(roi)
		orb = extraer_orb(roi)
		hist = extraer_histograma(roi, espacio="hsv", bins=(8, 8, 8))

		# Geometría basada en contorno
		ar = relacion_aspecto(contorno)
		sol = solidez(contorno)
		circ = circularidad(contorno)
		hu = momentos_hu(contorno)
		holes = numero_agujeros(contorno, None)
		geom = np.array([ar, sol, circ, float(holes)], dtype=np.float32)

		# Vector concatenado (orden: geom, hu, hist, hog, orb)
		vector = np.concatenate([geom, hu.astype(np.float32), hist, hog, orb], axis=0).astype(np.float32)

		return {
			"geom": geom,
			"hu": hu,
			"hist": hist,
			"hog": hog,
			"orb": orb,
			"vector": vector,
		}
	except Exception as e:
		logger.exception("Error en extraer_caracteristicas_completas: %s", e)
		# Fallback: vector vacío
		return {
			"geom": np.zeros(4, dtype=np.float32),
			"hu": np.zeros(7, dtype=np.float32),
			"hist": np.zeros(512, dtype=np.float32),
			"hog": np.zeros(1, dtype=np.float32),
			"orb": np.zeros(64, dtype=np.float32),
			"vector": np.zeros(4 + 7 + 512 + 1 + 64, dtype=np.float32),
		}


__all__ = [
	"relacion_aspecto",
	"solidez",
	"circularidad",
	"numero_agujeros",
	"momentos_hu",
	"extraer_hog",
	"extraer_orb",
	"extraer_histograma",
	"extraer_caracteristicas_completas",
]

