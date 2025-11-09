"""Orquestador principal del proyecto de visión (entrenar, predecir, evaluar).

Ejemplos:
	python main.py --entrenar --dataset ./dataset --modelo svm
	python main.py --predecir --imagen ./dataset/tuercas/ejemplo.jpg --modelo svm
	python main.py --evaluar --dataset ./dataset --modelo svm --modelo-ruta ./modelos/svm_hog.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
	import numpy as np  # type: ignore
	import cv2  # type: ignore
except Exception as e:  # pragma: no cover
	raise RuntimeError("Se requieren numpy y opencv-python.") from e

try:
	from . import config as CFG
	from . import utils as U
	from . import preprocesamiento as P
	from . import segmentacion as S
	from . import extraccion_caracteristicas as F
	from .clasificacion import ClasificadorTornillos
	from . import evaluacion as E
except Exception as e:  # pragma: no cover
	raise RuntimeError("No se pudieron importar módulos internos del proyecto.") from e


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# =============================
# Utilidades de pipeline
# =============================

def _procesar_imagen_para_contornos(img: Any) -> Tuple[Any, List[Any]]:
	"""Preprocesa y segmenta una imagen para obtener contornos filtrados."""
	# Preprocesamiento básico
	img_res = P.redimensionar_imagen(img, CFG.IMAGE_SIZE)
	img_gray = P.convertir_gris(img_res)
	img_blur = P.aplicar_filtro_gaussiano(img_gray, CFG.GAUSSIAN_KERNEL_SIZE)

	# Segmentación por Otsu
	bin_ = S.umbralizar_otsu(img_blur)
	contornos = S.encontrar_contornos(bin_)
	cont_filtrados = S.filtrar_contornos_por_area(
		contornos,
		area_min=getattr(CFG, "MIN_CONTOUR_AREA", 200.0),
		area_max=getattr(CFG, "MAX_CONTOUR_AREA", 1_000_000.0),
	)
	return img_res, cont_filtrados


def _extraer_features_imagen(img: Any, label: int) -> Tuple[List[np.ndarray], List[int]]:
	"""Extrae features para todos los contornos válidos en una imagen y retorna (X, y)."""
	try:
		img_proc, contornos = _procesar_imagen_para_contornos(img)
		X: List[np.ndarray] = []
		y: List[int] = []
		if len(contornos) == 0:
			logger.warning("Sin contornos válidos; imagen omitida")
			return X, y
		for c in contornos:
			feats = F.extraer_caracteristicas_completas(img_proc, c)
			X.append(feats["vector"])  # vector concatenado
			y.append(label)
		return X, y
	except Exception as e:
		logger.warning("Fallo extrayendo características de una imagen: %s", e)
		return [], []


def _construir_dataset_caracteristicas(imagenes: Sequence[Any], etiquetas: Sequence[int]) -> Tuple[List[np.ndarray], List[int]]:
	"""Convierte un conjunto de imágenes en un dataset de vectores de características con sus etiquetas."""
	X_total: List[np.ndarray] = []
	y_total: List[int] = []
	for img, lab in zip(imagenes, etiquetas):
		X, y = _extraer_features_imagen(img, lab)
		if X:
			X_total.extend(X)
			y_total.extend(y)
	logger.info("Dataset de características: %d muestras", len(X_total))
	return X_total, y_total


def _guardar_fig_matriz_confusion(y_true, y_pred, out_path: Path) -> None:
	try:
		fig_ax = E.generar_matriz_confusion(y_true, y_pred)
		if fig_ax is None:
			return
		fig, _ax = fig_ax
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=150)
		logger.info("Matriz de confusión guardada en %s", out_path)
	except Exception as e:
		logger.warning("No se pudo guardar matriz de confusión: %s", e)


# =============================
# Modos de ejecución
# =============================

def modo_entrenar(args: argparse.Namespace) -> int:
	t0 = time.perf_counter()
	try:
		U.crear_directorios()
		dataset_dir = Path(args.dataset) if args.dataset else CFG.DATASET_DIR
		logger.info("Cargando dataset desde %s", dataset_dir)
		imgs, y, paths = U.cargar_dataset(dataset_dir)
		t1 = time.perf_counter()
		logger.info("Carga dataset: %.2fs", t1 - t0)

		# Construir dataset de características
		X, y_feat = _construir_dataset_caracteristicas(imgs, y)
		t2 = time.perf_counter()
		logger.info("Extracción de características: %.2fs", t2 - t1)

		if len(X) == 0:
			logger.error("No se generaron características. Revisa segmentación/contornos.")
			return 2

		# Dividir
		X_tr, X_te, y_tr, y_te = U.dividir_dataset(X, y_feat, test_size=getattr(CFG, "TEST_SIZE", 0.2))
		t3 = time.perf_counter()
		logger.info("Split train/test: %.2fs", t3 - t2)

		# Entrenar
		clf = ClasificadorTornillos()
		modelo = (args.modelo or "svm").lower()
		if modelo == "svm":
			clf.entrenar_svm(X_tr, y_tr)
		elif modelo == "knn":
			clf.entrenar_knn(X_tr, y_tr)
		else:
			logger.error("Modelo no soportado: %s", args.modelo)
			return 2
		t4 = time.perf_counter()
		logger.info("Entrenamiento (%s): %.2fs", modelo, t4 - t3)

		# Evaluación
		y_pred = clf.predecir(X_te, modelo=modelo)
		metricas = E.calcular_metricas(y_te, y_pred)
		rep = E.generar_reporte_clasificacion(y_te, y_pred)
		E.guardar_metricas(metricas, CFG.RESULTADOS_DIR / "metricas.txt")
		_guardar_fig_matriz_confusion(y_te, y_pred, CFG.RESULTADOS_DIR / "matriz_confusion.png")
		t5 = time.perf_counter()
		logger.info("Evaluación: %.2fs", t5 - t4)

		# Guardar modelo
		ruta_modelo = Path(args.modelo_ruta) if args.modelo_ruta else (
			CFG.MODELOS_DIR / (CFG.MODEL_FILENAME_SVM if modelo == "svm" else CFG.MODEL_FILENAME_KNN)
		)
		clf.guardar_modelo(ruta_modelo, modelo=modelo)
		logger.info("Modelo guardado en: %s", ruta_modelo)

		logger.info("Tiempo total: %.2fs", time.perf_counter() - t0)
		# Imprimir resumen breve en stdout
		print("Resumen métricas:", metricas)
		print(rep)
		return 0
	except Exception as e:
		logger.exception("Error en modo entrenamiento: %s", e)
		return 1


def _predecir_imagen(clf: ClasificadorTornillos, ruta_imagen: Path, modelo: str) -> List[int]:
	img = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
	img_proc, contornos = _procesar_imagen_para_contornos(img)
	if len(contornos) == 0:
		logger.warning("No se detectaron objetos en %s", ruta_imagen)
		return []
	predicciones: List[int] = []
	for c in contornos:
		feats = F.extraer_caracteristicas_completas(img_proc, c)["vector"]
		pred = clf.predecir(feats, modelo=modelo)
		predicciones.append(int(pred[0]))
	return predicciones


def modo_predecir(args: argparse.Namespace) -> int:
	try:
		modelo = (args.modelo or "svm").lower()
		ruta_modelo = Path(args.modelo_ruta) if args.modelo_ruta else (
			CFG.MODELOS_DIR / (CFG.MODEL_FILENAME_SVM if modelo == "svm" else CFG.MODEL_FILENAME_KNN)
		)
		clf = ClasificadorTornillos()
		clf.cargar_modelo(ruta_modelo, modelo=modelo)

		ruta_imagen = Path(args.imagen)
		preds = _predecir_imagen(clf, ruta_imagen, modelo)
		names = list(getattr(CFG, "CLASSES", []))
		etiquetas = [names[p] if p < len(names) else str(p) for p in preds]
		print({"imagen": str(ruta_imagen), "predicciones": preds, "clases": etiquetas})
		return 0
	except Exception as e:
		logger.exception("Error en modo predicción: %s", e)
		return 1


def modo_evaluar(args: argparse.Namespace) -> int:
	try:
		# Cargar dataset y modelo y solo evaluar
		dataset_dir = Path(args.dataset) if args.dataset else CFG.DATASET_DIR
		imgs, y, _paths = U.cargar_dataset(dataset_dir)
		X, y_feat = _construir_dataset_caracteristicas(imgs, y)
		if len(X) == 0:
			logger.error("No se generaron características para evaluación.")
			return 2
		modelo = (args.modelo or "svm").lower()
		ruta_modelo = Path(args.modelo_ruta) if args.modelo_ruta else (
			CFG.MODELOS_DIR / (CFG.MODEL_FILENAME_SVM if modelo == "svm" else CFG.MODEL_FILENAME_KNN)
		)
		clf = ClasificadorTornillos().cargar_modelo(ruta_modelo, modelo=modelo)
		y_pred = clf.predecir(X, modelo=modelo)
		metricas = E.calcular_metricas(y_feat, y_pred)
		rep = E.generar_reporte_clasificacion(y_feat, y_pred)
		E.guardar_metricas(metricas, CFG.RESULTADOS_DIR / "metricas_eval.txt")
		_guardar_fig_matriz_confusion(y_feat, y_pred, CFG.RESULTADOS_DIR / "matriz_confusion_eval.png")
		print("Métricas evaluación:", metricas)
		print(rep)
		return 0
	except Exception as e:
		logger.exception("Error en modo evaluación: %s", e)
		return 1


# =============================
# CLI
# =============================

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Orquestador principal: entrenamiento, predicción y evaluación.")
	g = p.add_mutually_exclusive_group(required=True)
	g.add_argument("--entrenar", action="store_true", help="Ejecuta el pipeline de entrenamiento")
	g.add_argument("--predecir", action="store_true", help="Predice sobre una imagen")
	g.add_argument("--evaluar", action="store_true", help="Evalúa un modelo sobre el dataset")

	p.add_argument("--dataset", type=str, default=str(CFG.DATASET_DIR), help="Ruta al dataset")
	p.add_argument("--modelo", type=str, choices=["svm", "knn"], default="svm", help="Modelo a usar")
	p.add_argument("--modelo-ruta", type=str, default=None, help="Ruta del archivo de modelo para guardar/cargar")
	p.add_argument("--imagen", type=str, default=None, help="Ruta de la imagen para predicción")

	return p.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv if argv is not None else sys.argv[1:])
	if args.predecir and not args.imagen:
		logger.error("--predecir requiere --imagen /ruta/a/imagen")
		return 2
	if args.entrenar:
		return modo_entrenar(args)
	if args.predecir:
		return modo_predecir(args)
	if args.evaluar:
		return modo_evaluar(args)
	logger.error("No se seleccionó un modo válido.")
	return 2


if __name__ == "__main__":
	raise SystemExit(main())

