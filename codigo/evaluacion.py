"""Módulo de evaluación de modelos de clasificación.

Incluye:
- calcular_metricas(y_real, y_pred)
- generar_matriz_confusion(y_real, y_pred, clases=['tuerca','tornillo','arandela'])
- generar_reporte_clasificacion(y_real, y_pred)
- visualizar_resultados(imagenes_test, predicciones, etiquetas_reales)
- guardar_metricas(metricas, ruta='resultados/metricas.txt')
- plot_curvas_aprendizaje(modelo, X, y)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
from pathlib import Path
import logging

try:
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover
	np = None  # type: ignore

try:
	import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
	plt = None  # type: ignore

try:
	from sklearn.metrics import (
		accuracy_score,
		precision_recall_fscore_support,
		confusion_matrix,
		ConfusionMatrixDisplay,
		classification_report,
	)  # type: ignore
	from sklearn.model_selection import learning_curve, KFold  # type: ignore
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"Se requiere scikit-learn para las funciones de evaluación. Instala con 'pip install scikit-learn'."
	) from e

try:
	import cv2  # type: ignore
except Exception:  # pragma: no cover
	cv2 = None  # type: ignore

try:
	from . import config as CFG
	from . import utils as U
except Exception as e:  # pragma: no cover
	CFG = None  # type: ignore
	U = None  # type: ignore
	raise RuntimeError("No se pudieron importar config o utils.") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def calcular_metricas(y_real: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
	"""Calcula accuracy, precision, recall y f1 (macro)."""
	try:
		acc = float(accuracy_score(y_real, y_pred))
		prec, rec, f1, _ = precision_recall_fscore_support(y_real, y_pred, average="macro", zero_division=0)
		metricas = {
			"accuracy": float(acc),
			"precision": float(prec),
			"recall": float(rec),
			"f1": float(f1),
		}
		logger.info("Métricas: %s", metricas)
		return metricas
	except Exception as e:
		logger.exception("Error calculando métricas: %s", e)
		raise


def generar_matriz_confusion(
	y_real: Sequence[int],
	y_pred: Sequence[int],
	clases: List[str] = ["tuerca", "tornillo", "arandela"],
):
	"""Genera y retorna (fig, ax) con la matriz de confusión."""
	if plt is None:
		logger.warning("matplotlib no disponible; no se puede generar la matriz de confusión")
		return None
	try:
		# Si existen clases en config, usarlas con nombres legibles
		target_names = list(getattr(CFG, "CLASSES", clases))
		# Si son plurales en config, se respetan; si se quiere singular, se puede pasar via parámetro
		labels = list(range(len(target_names)))
		cm = confusion_matrix(y_real, y_pred, labels=labels)
		fig, ax = plt.subplots(figsize=(5, 5))
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
		disp.plot(ax=ax, cmap="Blues", colorbar=False)
		ax.set_title("Matriz de confusión")
		ax.set_xlabel("Predicción")
		ax.set_ylabel("Real")
		fig.tight_layout()
		return fig, ax
	except Exception as e:
		logger.exception("Error generando matriz de confusión: %s", e)
		return None


def generar_reporte_clasificacion(y_real: Sequence[int], y_pred: Sequence[int]) -> str:
	"""Devuelve un string con el reporte de clasificación (precision/recall/f1 por clase)."""
	try:
		target_names = list(getattr(CFG, "CLASSES", [])) or None
		rep = classification_report(y_real, y_pred, target_names=target_names, zero_division=0)
		logger.info("Reporte de clasificación generado")
		return rep
	except Exception as e:
		logger.exception("Error generando reporte de clasificación: %s", e)
		return ""


def _to_rgb(img: Any):
	if cv2 is not None and isinstance(img, (np.ndarray if np is not None else tuple)):
		arr = img
		if arr.ndim == 2:
			return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
		if arr.shape[2] == 3:
			return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
		if arr.shape[2] == 4:
			return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
		return arr
	try:
		# PIL Image o similar
		from PIL import Image as _Image  # type: ignore

		if isinstance(img, _Image.Image):
			return img
	except Exception:
		pass
	return img


def visualizar_resultados(
	imagenes_test: Sequence[Any],
	predicciones: Sequence[int],
	etiquetas_reales: Sequence[int],
	max_cols: int = 4,
):
	"""Muestra un grid con imágenes y etiquetas predicha/real.

	Retorna (fig, axes) o None si no hay matplotlib.
	"""
	if plt is None:
		logger.warning("matplotlib no disponible; no se pueden visualizar resultados")
		return None
	try:
		n = min(len(imagenes_test), len(predicciones), len(etiquetas_reales))
		cols = max(1, int(max_cols))
		rows = (n + cols - 1) // cols
		fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
		axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

		names = list(getattr(CFG, "CLASSES", [])) or None

		for i in range(rows * cols):
			ax = axes[i] if i < len(axes) else None
			if ax is None:
				continue
			ax.axis("off")
			if i >= n:
				continue
			img = _to_rgb(imagenes_test[i])
			ax.imshow(img)
			y_p = predicciones[i]
			y_t = etiquetas_reales[i]
			pred_name = names[y_p] if names and y_p < len(names) else str(y_p)
			true_name = names[y_t] if names and y_t < len(names) else str(y_t)
			ax.set_title(f"Pred: {pred_name}\nReal: {true_name}")

		fig.tight_layout()
		return fig, axes
	except Exception as e:
		logger.exception("Error visualizando resultados: %s", e)
		return None


def guardar_metricas(metricas: Dict[str, float], ruta: Any = None) -> Path:
	"""Guarda métricas en un archivo de texto. Ruta por defecto en resultados/metricas.txt."""
	try:
		if ruta is None:
			ruta = Path(getattr(CFG, "RESULTADOS_DIR", Path("resultados"))) / "metricas.txt"
		else:
			ruta = Path(ruta)
		ruta.parent.mkdir(parents=True, exist_ok=True)
		# Formatear métricas ordenadas por clave
		lines = ["Métricas de evaluación:\n"]
		for k in sorted(metricas.keys()):
			lines.append(f"{k}: {metricas[k]:.4f}\n")
		with open(ruta, "w", encoding="utf-8") as f:
			f.writelines(lines)
		logger.info("Métricas guardadas en %s", ruta)
		return ruta
	except Exception as e:
		logger.exception("No se pudieron guardar las métricas: %s", e)
		raise


def plot_curvas_aprendizaje(modelo: Any, X: Sequence[Any], y: Sequence[int], n_splits: int = None):
	"""Genera figura con curvas de aprendizaje (train/test) usando learning_curve.

	Retorna (fig, ax) o None si matplotlib no está disponible.
	"""
	if plt is None:
		logger.warning("matplotlib no disponible; no se pueden generar curvas de aprendizaje")
		return None
	try:
		X_arr = np.asarray(X) if np is not None else X
		y_arr = np.asarray(y) if np is not None else y

		n_splits = int(n_splits) if n_splits is not None else int(getattr(CFG, "N_FOLDS", 5))
		cv = KFold(n_splits=n_splits, shuffle=True, random_state=getattr(CFG, "RANDOM_STATE", None))

		train_sizes = np.linspace(0.1, 1.0, 5) if np is not None else [0.1, 0.325, 0.55, 0.775, 1.0]
		sizes, train_scores, test_scores = learning_curve(
			modelo, X_arr, y_arr, cv=cv, train_sizes=train_sizes, n_jobs=-1, shuffle=True, random_state=getattr(CFG, "RANDOM_STATE", None)
		)

		train_mean = train_scores.mean(axis=1)
		train_std = train_scores.std(axis=1)
		test_mean = test_scores.mean(axis=1)
		test_std = test_scores.std(axis=1)

		fig, ax = plt.subplots(figsize=(6, 4))
		ax.plot(sizes, train_mean, "-o", label="Train")
		ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
		ax.plot(sizes, test_mean, "-o", label="Validation")
		ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
		ax.set_xlabel("Tamaño de entrenamiento")
		ax.set_ylabel("Score")
		ax.set_title("Curvas de aprendizaje")
		ax.legend()
		ax.grid(True, linestyle=":", alpha=0.5)
		fig.tight_layout()
		return fig, ax
	except Exception as e:
		logger.exception("Error generando curvas de aprendizaje: %s", e)
		return None


__all__ = [
	"calcular_metricas",
	"generar_matriz_confusion",
	"generar_reporte_clasificacion",
	"visualizar_resultados",
	"guardar_metricas",
	"plot_curvas_aprendizaje",
]

