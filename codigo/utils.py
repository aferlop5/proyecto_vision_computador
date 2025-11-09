"""Utilidades para el proyecto de visión por computador.

Incluye funciones para:
- Carga de dataset desde subcarpetas por clase
- División train/test
- Guardado de imágenes procesadas por etapa
- Gestión de directorios de salida
- Persistencia de modelos (guardar/cargar)
- Visualización de contornos para depuración

Todas las funciones implementan manejo de errores (try/except) y logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import logging

# Logger básico del módulo
logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)

try:
	import cv2  # type: ignore
except Exception as e:  # pragma: no cover - dependencia opcional
	cv2 = None  # type: ignore
	logger.warning("OpenCV (cv2) no está disponible: %s", e)

try:
	import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
	np = None  # type: ignore
	logger.warning("NumPy no está disponible: %s", e)

try:
	import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
	plt = None  # type: ignore
	logger.warning("matplotlib no está disponible: %s", e)

try:
	from PIL import Image  # type: ignore
except Exception as e:  # pragma: no cover
	Image = None  # type: ignore
	logger.warning("Pillow (PIL) no está disponible: %s", e)

try:
	from . import config as CFG
except Exception as e:  # pragma: no cover
	CFG = None  # type: ignore
	logger.error("No se pudo importar config.py: %s", e)


def _ensure_dir(path: Path) -> None:
	"""Crea el directorio si no existe."""
	try:
		path.mkdir(parents=True, exist_ok=True)
	except Exception as e:
		logger.error("No se pudo crear el directorio %s: %s", path, e)
		raise


def crear_directorios() -> None:
	"""Asegura que existan las carpetas de modelos y resultados."""
	if CFG is None:
		raise RuntimeError("La configuración (CFG) no está disponible.")
	try:
		_ensure_dir(CFG.MODELOS_DIR)
		_ensure_dir(CFG.RESULTADOS_DIR)
		logger.info("Directorios verificados: %s, %s", CFG.MODELOS_DIR, CFG.RESULTADOS_DIR)
	except Exception:
		logger.exception("Error creando directorios de salida")
		raise


def _listar_imagenes(carpeta: Path) -> List[Path]:
	"""Lista rutas de imágenes con extensiones comunes dentro de una carpeta."""
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
	archivos: List[Path] = []
	try:
		for p in carpeta.glob("**/*"):
			if p.is_file() and p.suffix.lower() in exts:
				archivos.append(p)
	except Exception as e:
		logger.error("Error listando imágenes en %s: %s", carpeta, e)
	return archivos


def cargar_dataset(ruta_base: Optional[Path | str] = None) -> Tuple[List, List[int], List[Path]]:
	"""Carga imágenes y etiquetas desde subcarpetas por clase.

	Estructura esperada:
		ruta_base/
		  tuercas/
		  tornillos/
		  arandelas/

	Retorna:
		- imagenes: lista de imágenes (np.ndarray o PIL.Image según disponibilidad)
		- etiquetas: lista de enteros (según CFG.CLASS_TO_LABEL)
		- rutas: lista de rutas de archivos correspondientes (Path)
	"""
	if CFG is None:
		raise RuntimeError("La configuración (CFG) no está disponible.")

	base = Path(ruta_base) if ruta_base is not None else CFG.DATASET_DIR
	imagenes: List = []
	etiquetas: List[int] = []
	rutas: List[Path] = []

	if not base.exists():
		logger.error("La ruta base del dataset no existe: %s", base)
		return imagenes, etiquetas, rutas

	for clase in CFG.CLASSES:
		carpeta_clase = base / clase
		if not carpeta_clase.exists():
			logger.warning("Carpeta de clase no encontrada: %s", carpeta_clase)
			continue
		for ruta in _listar_imagenes(carpeta_clase):
			try:
				img = None
				if cv2 is not None:
					img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
					if img is None:
						raise ValueError("cv2.imread devolvió None")
				elif Image is not None:
					img = Image.open(ruta).convert("RGB")
				else:
					raise RuntimeError("No hay backend de imagen disponible (cv2 o PIL)")

				imagenes.append(img)
				etiquetas.append(CFG.CLASS_TO_LABEL[clase])
				rutas.append(ruta)
			except Exception as e:
				logger.warning("No se pudo leer %s: %s", ruta, e)

	logger.info(
		"Dataset cargado: %d imágenes, clases=%s desde %s",
		len(imagenes), CFG.CLASSES, base,
	)
	return imagenes, etiquetas, rutas


def dividir_dataset(
	caracteristicas: Sequence,
	etiquetas: Sequence[int],
	test_size: float = 0.2,
	stratify: bool = True,
):
	"""Divide características y etiquetas en train/test.

	Si scikit-learn no está disponible, emite error con instrucción.
	"""
	try:
		from sklearn.model_selection import train_test_split  # type: ignore
	except Exception as e:  # pragma: no cover
		logger.error(
			"scikit-learn no está instalado o falló al importar: %s. Instala con 'pip install scikit-learn'",
			e,
		)
		raise

	y = list(etiquetas)
	X = list(caracteristicas)
	strat = y if stratify and len(set(y)) > 1 else None
	try:
		X_tr, X_te, y_tr, y_te = train_test_split(
			X, y,
			test_size=test_size,
			random_state=getattr(CFG, "RANDOM_STATE", 42),
			stratify=strat,
		)
		logger.info(
			"Split: train=%d, test=%d (test_size=%.2f)", len(X_tr), len(X_te), test_size
		)
		return X_tr, X_te, y_tr, y_te
	except Exception as e:
		logger.exception("Error al dividir el dataset: %s", e)
		raise


def guardar_imagen_procesada(
	imagen,
	nombre: str,
	etapa: str,
) -> Optional[Path]:
	"""Guarda una imagen procesada en resultados/etapa/nombre.

	- imagen: np.ndarray (BGR o GRAY) si se usa OpenCV, o PIL.Image.
	- nombre: nombre de archivo (se añadirá .png si no tiene extensión).
	- etapa: subcarpeta para agrupar resultados por paso del pipeline.
	"""
	if CFG is None:
		raise RuntimeError("La configuración (CFG) no está disponible.")

	try:
		ext = Path(nombre).suffix
		fname = nombre if ext else f"{nombre}.png"
		carpeta = CFG.RESULTADOS_DIR / etapa
		_ensure_dir(carpeta)
		destino = carpeta / fname

		if cv2 is not None and np is not None and isinstance(imagen, np.ndarray):
			arr = imagen
			# Normalización para float
			if arr.dtype.kind == "f":
				arr = np.clip(arr, 0.0, 1.0)
				arr = (arr * 255).astype("uint8")
			ok = cv2.imwrite(str(destino), arr)
			if not ok:
				raise IOError("cv2.imwrite devolvió False")
		elif Image is not None and isinstance(imagen, Image.Image):
			imagen.save(destino)
		else:
			raise TypeError("Tipo de imagen no soportado; se espera np.ndarray o PIL.Image")

		logger.info("Imagen guardada: %s", destino)
		return destino
	except Exception as e:
		logger.exception("No se pudo guardar imagen procesada '%s' en etapa '%s': %s", nombre, etapa, e)
		return None


def guardar_modelo(modelo, ruta: Path | str) -> Optional[Path]:
	"""Guarda un modelo con joblib si está disponible, sino pickle.

	Retorna la ruta final si tuvo éxito, None en caso de error.
	"""
	path = Path(ruta)
	try:
		_ensure_dir(path.parent)

		try:
			import joblib  # type: ignore

			joblib.dump(modelo, path)
			logger.info("Modelo guardado (joblib): %s", path)
			return path
		except Exception as e_joblib:  # fallback a pickle
			import pickle

			with open(path, "wb") as f:
				pickle.dump(modelo, f)
			logger.info("Modelo guardado (pickle, fallback por error joblib: %s): %s", e_joblib, path)
			return path
	except Exception as e:
		logger.exception("No se pudo guardar el modelo en %s: %s", path, e)
		return None


def cargar_modelo(ruta: Path | str):
	"""Carga un modelo previamente guardado con joblib o pickle."""
	path = Path(ruta)
	if not path.exists():
		logger.error("La ruta del modelo no existe: %s", path)
		return None
	try:
		try:
			import joblib  # type: ignore

			modelo = joblib.load(path)
			logger.info("Modelo cargado (joblib): %s", path)
			return modelo
		except Exception as e_joblib:  # fallback a pickle
			import pickle

			with open(path, "rb") as f:
				modelo = pickle.load(f)
			logger.info("Modelo cargado (pickle, fallback por error joblib: %s): %s", e_joblib, path)
			return modelo
	except Exception as e:
		logger.exception("No se pudo cargar el modelo desde %s: %s", path, e)
		return None


def plot_contornos(imagen, contornos: Iterable, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
	"""Genera una figura con contornos superpuestos para depuración.

	- imagen: np.ndarray (BGR o GRAY) o PIL.Image
	- contornos: iterable de contornos (como los de cv2.findContours)
	- color: color BGR para cv2 o RGB para matplotlib

	Retorna (fig, ax) de matplotlib, o None si matplotlib no está disponible.
	"""
	if plt is None:
		logger.warning("matplotlib no disponible; no se puede generar la visualización de contornos")
		return None

	try:
		if cv2 is not None and np is not None and isinstance(imagen, np.ndarray):
			img = imagen.copy()
			# Si es gris, convertir a BGR para dibujar en color
			if len(img.shape) == 2:
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			# Dibujar contornos
			try:
				cv2.drawContours(img, list(contornos), -1, color, thickness)
			except Exception:
				# Compatibilidad si contornos no es compatible con drawContours
				for c in contornos:
					try:
						cv2.drawContours(img, [c], -1, color, thickness)
					except Exception:
						continue
			# Convertir a RGB para plt
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow(img_rgb)
			ax.axis('off')
			ax.set_title('Contornos (OpenCV)')
			return fig, ax
		elif Image is not None and isinstance(imagen, Image.Image):
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.imshow(imagen)
			ax.axis('off')
			# Dibujo básico de contornos si son arrays Nx1x2 o Nx2
			try:
				import numpy as _np  # local

				for c in contornos:
					arr = _np.array(c).squeeze()
					if arr.ndim == 2 and arr.shape[1] == 2:
						ax.plot(arr[:, 0], arr[:, 1], color=_color_to_hex(color), linewidth=1.5)
			except Exception:
				pass
			ax.set_title('Contornos (PIL)')
			return fig, ax
		else:
			logger.error("Tipo de imagen no soportado para plot_contornos")
			return None
	except Exception as e:
		logger.exception("Error generando visualización de contornos: %s", e)
		return None


def _color_to_hex(bgr: Tuple[int, int, int]) -> str:
	"""Convierte BGR/RGB (0-255) a hex para matplotlib."""
	b, g, r = bgr
	return f"#{r:02x}{g:02x}{b:02x}"


__all__ = [
	"cargar_dataset",
	"dividir_dataset",
	"guardar_imagen_procesada",
	"crear_directorios",
	"guardar_modelo",
	"cargar_modelo",
	"plot_contornos",
]

