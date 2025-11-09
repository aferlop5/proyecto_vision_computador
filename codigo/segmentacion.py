"""Funciones de segmentación para detección de objetos.

Incluye:
- umbralizar_otsu(imagen_gris)
- encontrar_contornos(imagen_binaria)
- filtrar_contornos_por_area(contornos, area_min, area_max) con filtrado por relación de aspecto
- extraer_roi(imagen_original, contorno)
- dibujar_contornos(imagen, contornos)

cv2.findContours se usa con RETR_EXTERNAL y CHAIN_APPROX_SIMPLE.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple
import logging

try:
	import cv2  # type: ignore
	import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
	cv2 = None  # type: ignore
	np = None  # type: ignore
	raise RuntimeError("Se requieren 'opencv-python' (cv2) y 'numpy' para segmentación.") from e

try:
	from . import config as CFG
except Exception as e:  # pragma: no cover
	CFG = None  # type: ignore
	raise RuntimeError("No se pudo importar config.py") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def umbralizar_otsu(imagen_gris: Any) -> Any:
	"""Aplica umbralización Otsu y retorna imagen binaria (0/255)."""
	try:
		if imagen_gris.ndim != 2:
			raise ValueError("'umbralizar_otsu' requiere imagen en escala de grises (2D)")
		_thr, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return binaria
	except Exception as e:
		logger.exception("Error en umbralizar_otsu: %s", e)
		raise


def encontrar_contornos(imagen_binaria: Any) -> List[Any]:
	"""Encuentra contornos externos en una imagen binaria y los retorna como lista."""
	try:
		if imagen_binaria.ndim != 2:
			raise ValueError("'encontrar_contornos' requiere imagen binaria 2D")
		res = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contornos = res[0] if len(res) == 2 else res[1]
		contornos = list(contornos) if contornos is not None else []
		logger.info("Contornos encontrados: %d", len(contornos))
		return contornos
	except Exception as e:
		logger.exception("Error al encontrar contornos: %s", e)
		raise


def _aspect_ratio_ok(rect: Tuple[int, int, int, int],
					 aspect_min: float,
					 aspect_max: float) -> bool:
	x, y, w, h = rect
	if w <= 0 or h <= 0:
		return False
	ratio = (w / float(h)) if h > 0 else 0.0
	if ratio < aspect_min or ratio > aspect_max:
		return False
	return True


def filtrar_contornos_por_area(
	contornos: Sequence[Any],
	area_min: Optional[float] = None,
	area_max: Optional[float] = None,
	aspect_min: Optional[float] = None,
	aspect_max: Optional[float] = None,
) -> List[Any]:
	"""Filtra contornos por área y relación de aspecto del bounding box.

	- area_min/area_max por defecto toman valores de CFG.MIN_CONTOUR_AREA / CFG.MAX_CONTOUR_AREA si existen.
	- aspect_min/aspect_max definen el rango válido de w/h; si no se proporcionan, usan valores razonables
	  (o los de configuración si están definidos) para eliminar falsos positivos muy alargados o muy planos.
	"""
	try:
		a_min = float(area_min if area_min is not None else getattr(CFG, "MIN_CONTOUR_AREA", 1000.0))
		a_max = float(area_max if area_max is not None else getattr(CFG, "MAX_CONTOUR_AREA", 50000.0))
		ar_min = float(aspect_min if aspect_min is not None else getattr(CFG, "MIN_ASPECT_RATIO", 0.2))
		ar_max = float(aspect_max if aspect_max is not None else getattr(CFG, "MAX_ASPECT_RATIO", 5.0))

		filtrados: List[Any] = []
		for c in contornos:
			try:
				area = cv2.contourArea(c)
				if area < a_min or area > a_max:
					continue
				x, y, w, h = cv2.boundingRect(c)
				if not _aspect_ratio_ok((x, y, w, h), ar_min, ar_max):
					continue
				filtrados.append(c)
			except Exception:
				continue

		logger.info(
			"Contornos filtrados: %d (área:[%.1f, %.1f], aspect:[%.2f, %.2f])",
			len(filtrados), a_min, a_max, ar_min, ar_max,
		)
		return filtrados
	except Exception as e:
		logger.exception("Error filtrando contornos: %s", e)
		raise


def extraer_roi(imagen_original: Any, contorno: Any) -> Tuple[Any, Tuple[int, int, int, int]]:
	"""Extrae ROI rectangular (x,y,w,h) que encapsula el contorno desde la imagen original.

	Retorna (roi, (x, y, w, h)).
	"""
	try:
		x, y, w, h = cv2.boundingRect(contorno)
		roi = imagen_original[y:y + h, x:x + w]
		return roi, (x, y, w, h)
	except Exception as e:
		logger.exception("Error extrayendo ROI: %s", e)
		raise


def dibujar_contornos(
	imagen: Any,
	contornos: Iterable[Any],
	color: Tuple[int, int, int] = (0, 255, 0),
	thickness: int = 2,
	draw_bbox: bool = True,
) -> Any:
	"""Dibuja contornos (y opcionalmente bounding boxes) sobre una copia de la imagen.

	Retorna la imagen con anotaciones.
	"""
	try:
		out = imagen.copy()
		cv2.drawContours(out, list(contornos), -1, color, thickness)
		if draw_bbox:
			for c in contornos:
				try:
					x, y, w, h = cv2.boundingRect(c)
					cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
				except Exception:
					continue
		return out
	except Exception as e:
		logger.exception("Error dibujando contornos: %s", e)
		raise


__all__ = [
	"umbralizar_otsu",
	"encontrar_contornos",
	"filtrar_contornos_por_area",
	"extraer_roi",
	"dibujar_contornos",
]

