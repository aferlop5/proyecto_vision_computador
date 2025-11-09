"""Funciones de preprocesamiento de imágenes.

Cada función usa por defecto los parámetros definidos en config.py (CFG),
pero permite sobrescribirlos con argumentos opcionales.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any
import logging

try:
	import cv2  # type: ignore
	import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
	cv2 = None  # type: ignore
	np = None  # type: ignore
	raise RuntimeError(
		"Se requieren 'opencv-python' (cv2) y 'numpy' para preprocesamiento."
	) from e

try:
	from . import config as CFG
except Exception as e:  # pragma: no cover
	raise RuntimeError("No se pudo importar config.py") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# =============================
# Helpers internos
# =============================

def _get_interpolation(name: Optional[str]) -> int:
	"""Mapea nombre de interpolación a constante de cv2."""
	name = (name or CFG.RESIZE_INTERPOLATION or "area").lower()
	mapping = {
		"nearest": cv2.INTER_NEAREST,
		"linear": cv2.INTER_LINEAR,
		"bilinear": cv2.INTER_LINEAR,
		"area": cv2.INTER_AREA,
		"cubic": cv2.INTER_CUBIC,
		"bicubic": cv2.INTER_CUBIC,
		"lanczos": cv2.INTER_LANCZOS4,
		"lanczos4": cv2.INTER_LANCZOS4,
	}
	return mapping.get(name, cv2.INTER_AREA)


def _ensure_tuple_size(size: Optional[Tuple[int, int] | int], fallback: Tuple[int, int]) -> Tuple[int, int]:
	if size is None:
		return fallback
	if isinstance(size, int):
		return (size, size)
	return tuple(int(x) for x in size)


def _morph_params(operacion: Optional[str]) -> Tuple[int, int]:
	"""Devuelve (morph_op_code, iteraciones) a partir de nombre de operación.

	Acepta nombres en español e inglés (apertura/open, cierre/close, erosion/erode, dilatacion/dilate).
	"""
	name = (operacion or getattr(CFG, "MORPH_OP", "open")).lower()
	# Normalizar acentos y equivalentes
	name = (
		name.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
	)
	aliases = {
		"apertura": "open",
		"open": "open",
		"cierre": "close",
		"close": "close",
		"erosion": "erode",
		"erode": "erode",
		"dilatacion": "dilate",
		"dilate": "dilate",
	}
	op_std = aliases.get(name, "open")
	morph_map = {
		"open": cv2.MORPH_OPEN,
		"close": cv2.MORPH_CLOSE,
	}
	iteraciones = int(getattr(CFG, "MORPH_ITERATIONS", 1))
	if op_std in morph_map:
		return morph_map[op_std], iteraciones
	# Para erode/dilate usamos funciones dedicadas; devolveremos MORPH_ERODE como señal
	if op_std == "erode":
		return cv2.MORPH_ERODE, iteraciones
	if op_std == "dilate":
		return cv2.MORPH_DILATE, iteraciones
	return cv2.MORPH_OPEN, iteraciones


def _make_kernel(kernel: Optional[Union[int, Tuple[int, int]]]) -> Any:
	ktuple = _ensure_tuple_size(kernel, getattr(CFG, "MORPH_KERNEL_SIZE", (3, 3)))
	kx, ky = ktuple
	# Asegurar impares
	if kx % 2 == 0:
		kx += 1
	if ky % 2 == 0:
		ky += 1
	return cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))


# =============================
# Funciones públicas
# =============================

def redimensionar_imagen(
	imagen: Any,
	tamaño: Optional[Tuple[int, int]] = None,
) -> Any:
	"""Redimensiona una imagen al tamaño especificado (ancho, alto).

	Usa por defecto CFG.IMAGE_SIZE y CFG.RESIZE_INTERPOLATION.
	"""
	try:
		size = _ensure_tuple_size(tamaño, CFG.IMAGE_SIZE)
		inter = _get_interpolation(getattr(CFG, "RESIZE_INTERPOLATION", "area"))
		# cv2.resize espera (width, height)
		resized = cv2.resize(imagen, size, interpolation=inter)
		return resized
	except Exception as e:
		logger.exception("Error al redimensionar: %s", e)
		raise


def convertir_gris(imagen: Any) -> Any:
	"""Convierte a escala de grises si es necesario."""
	try:
		if imagen.ndim == 2:
			return imagen
		if imagen.ndim == 3 and imagen.shape[2] == 3:
			return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
		if imagen.ndim == 3 and imagen.shape[2] == 4:
			# Si tiene alfa, ignorarlo al convertir
			return cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
		# Fallback: intentar forzar a gris
		return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
	except Exception as e:
		logger.exception("Error al convertir a gris: %s", e)
		raise


def aplicar_filtro_gaussiano(
	imagen: Any,
	kernel: Optional[Tuple[int, int]] = None,
) -> Any:
	"""Aplica desenfoque Gaussiano con kernel y sigmas de CFG por defecto."""
	try:
		ksize = _ensure_tuple_size(kernel, CFG.GAUSSIAN_KERNEL_SIZE)
		kx, ky = ksize
		# Asegurar impares
		if kx % 2 == 0:
			kx += 1
		if ky % 2 == 0:
			ky += 1
		sigma_x = float(getattr(CFG, "GAUSSIAN_SIGMA_X", 0.0))
		sigma_y = float(getattr(CFG, "GAUSSIAN_SIGMA_Y", 0.0))
		return cv2.GaussianBlur(imagen, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)
	except Exception as e:
		logger.exception("Error al aplicar filtro Gaussiano: %s", e)
		raise


def ecualizar_histograma(imagen_gris: Any) -> Any:
	"""Ecualiza histograma de una imagen en escala de grises."""
	try:
		if imagen_gris.ndim != 2:
			raise ValueError("'ecualizar_histograma' requiere imagen en escala de grises (2D)")
		return cv2.equalizeHist(imagen_gris)
	except Exception as e:
		logger.exception("Error al ecualizar histograma: %s", e)
		raise


def normalizar_imagen(imagen: Any) -> Any:
	"""Normaliza la imagen a rango [0, 1] en float32."""
	try:
		if imagen.dtype.kind in ("u", "i"):
			return (imagen.astype(np.float32) / 255.0).clip(0.0, 1.0)
		# Si ya es float, asegurar rango [0,1]
		img = imagen.astype(np.float32)
		# Reescalar si supera 1.0
		if img.max() > 1.0:
			img = img / 255.0
		img = np.clip(img, 0.0, 1.0)
		return img
	except Exception as e:
		logger.exception("Error al normalizar imagen: %s", e)
		raise


def aplicar_morfologia(
	imagen: Any,
	operacion: str = "apertura",
	kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Any:
	"""Aplica operación morfológica (apertura, cierre, erosion, dilatacion).

	Usa por defecto parámetros de CFG.MORPH_* y soporta override por argumentos.
	"""
	try:
		morph_code, iters = _morph_params(operacion)
		kernel = _make_kernel(kernel_size)
		if morph_code == cv2.MORPH_ERODE:
			return cv2.erode(imagen, kernel, iterations=iters)
		if morph_code == cv2.MORPH_DILATE:
			return cv2.dilate(imagen, kernel, iterations=iters)
		return cv2.morphologyEx(imagen, morph_code, kernel, iterations=iters)
	except Exception as e:
		logger.exception("Error al aplicar morfología '%s': %s", operacion, e)
		raise


__all__ = [
	"redimensionar_imagen",
	"convertir_gris",
	"aplicar_filtro_gaussiano",
	"ecualizar_histograma",
	"normalizar_imagen",
	"aplicar_morfologia",
]

