"""Clasificador para tuercas/tornillos/arandelas con SVM y KNN.

Incluye normalización de características con StandardScaler mediante Pipeline.
Permite entrenar, predecir y persistir modelos (guardar/cargar) usando utils.
"""

from __future__ import annotations

from typing import Any, Sequence, Union
from pathlib import Path
import logging

try:
	from sklearn.svm import SVC  # type: ignore
	from sklearn.neighbors import KNeighborsClassifier  # type: ignore
	from sklearn.preprocessing import StandardScaler  # type: ignore
	from sklearn.pipeline import Pipeline  # type: ignore
	import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"Se requiere scikit-learn y numpy. Instala con: pip install scikit-learn numpy"
	) from e

try:
	from . import config as CFG
	from . import utils as U
except Exception as e:  # pragma: no cover
	raise RuntimeError("No se pudieron importar configuraciones o utilidades.") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class ClasificadorTornillos:
	"""Contiene dos clasificadores: SVM y KNN, cada uno con su Pipeline (StandardScaler + modelo)."""

	def __init__(self) -> None:
		# Parámetros desde config
		svm_c = getattr(CFG, "SVM_C", 1.0)
		svm_kernel = getattr(CFG, "SVM_KERNEL", "rbf")
		svm_gamma = getattr(CFG, "SVM_GAMMA", "scale")

		knn_k = getattr(CFG, "KNN_N_NEIGHBORS", 5)
		knn_weights = getattr(CFG, "KNN_WEIGHTS", "distance")
		knn_metric = getattr(CFG, "KNN_METRIC", "minkowski")
		knn_p = getattr(CFG, "KNN_P", 2)

		# Pipelines con StandardScaler
		self.svm: Pipeline = Pipeline(
			steps=[
				("scaler", StandardScaler()),
				(
					"clf",
					SVC(
						C=float(svm_c),
						kernel=str(svm_kernel),
						gamma=svm_gamma,
						probability=False,
						random_state=getattr(CFG, "RANDOM_STATE", None),
					),
				),
			]
		)

		self.knn: Pipeline = Pipeline(
			steps=[
				("scaler", StandardScaler()),
				(
					"clf",
					KNeighborsClassifier(
						n_neighbors=int(knn_k),
						weights=str(knn_weights),
						metric=str(knn_metric),
						p=int(knn_p),
					),
				),
			]
		)

		logger.info("Clasificador inicializado: SVM=%s, KNN=%s", self.svm, self.knn)

	# =============================
	# Entrenamiento
	# =============================

	def entrenar_svm(self, X_train: Sequence, y_train: Sequence[int]) -> "ClasificadorTornillos":
		X = np.asarray(X_train)
		y = np.asarray(y_train)
		self.svm.fit(X, y)
		logger.info("SVM entrenado con %d muestras", len(X))
		return self

	def entrenar_knn(self, X_train: Sequence, y_train: Sequence[int]) -> "ClasificadorTornillos":
		X = np.asarray(X_train)
		y = np.asarray(y_train)
		self.knn.fit(X, y)
		logger.info("KNN entrenado con %d muestras", len(X))
		return self

	# =============================
	# Predicción
	# =============================

	def predecir(self, caracteristicas: Union[Sequence, Any], modelo: str = "svm") -> Any:
		"""Predice etiquetas para un vector o matriz de características usando el modelo indicado."""
		X = np.asarray(caracteristicas)
		if X.ndim == 1:
			X = X.reshape(1, -1)
		if modelo.lower() == "svm":
			return self.svm.predict(X)
		elif modelo.lower() == "knn":
			return self.knn.predict(X)
		else:
			raise ValueError("modelo debe ser 'svm' o 'knn'")

	# =============================
	# Persistencia
	# =============================

	def guardar_modelo(self, ruta: Union[str, Path], modelo: str = "svm") -> Path:
		"""Guarda el pipeline completo (scaler + modelo) en la ruta indicada usando utils."""
		U.crear_directorios()
		path = Path(ruta)
		obj = self.svm if modelo.lower() == "svm" else self.knn if modelo.lower() == "knn" else None
		if obj is None:
			raise ValueError("modelo debe ser 'svm' o 'knn'")
		res = U.guardar_modelo(obj, path)
		if res is None:
			raise IOError(f"No se pudo guardar el modelo en {path}")
		return res

	def cargar_modelo(self, ruta: Union[str, Path], modelo: str = "svm") -> "ClasificadorTornillos":
		"""Carga el pipeline completo (scaler + modelo) desde la ruta indicada usando utils."""
		path = Path(ruta)
		obj = U.cargar_modelo(path)
		if obj is None:
			raise IOError(f"No se pudo cargar el modelo desde {path}")
		if modelo.lower() == "svm":
			self.svm = obj
		elif modelo.lower() == "knn":
			self.knn = obj
		else:
			raise ValueError("modelo debe ser 'svm' o 'knn'")
		logger.info("Modelo '%s' cargado desde %s", modelo, path)
		return self


__all__ = [
	"ClasificadorTornillos",
]

