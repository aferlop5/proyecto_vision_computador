"""
Orquestador principal para entrenamiento, predicción y evaluación.
Ejemplos:
  python main.py --entrenar --modelo svm
  python main.py --predecir ./dataset/tuercas/img_01.jpg --modelo rf
  python main.py --evaluar --modelo knn
"""
from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Raíz del proyecto: padre de este archivo (estamos en 'codigo/')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))

# Imports del proyecto (robustos)
try:
    import config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    import utils  # type: ignore
    import preprocesamiento as pre  # type: ignore
    import segmentacion as seg  # type: ignore
    import extraccion_caracteristicas as ext  # type: ignore
    import clasificacion as cls  # type: ignore
    import evaluacion as ev  # type: ignore
except Exception as e:
    print(f"Error importando módulos: {e}")
    raise

# Logging básico
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
LOGGER = logging.getLogger("main")


# ======================
# Utilidades de pipeline
# ======================

def _procesar_imagen_basico(img: Any) -> Dict[str, Any]:
    """Aplica preprocesado básico y devuelve dict con etapas útiles."""
    out: Dict[str, Any] = {"original": img}
    try:
        img_r = pre.redimensionar_imagen(img)
        img_g = pre.convertir_gris(img_r)
        img_f = pre.aplicar_filtro_gaussiano(img_g)
        img_b = pre.binarizar_imagen(img_f, metodo="otsu")
        out.update({
            "redimensionada": img_r,
            "gris": img_g,
            "filtrada": img_f,
            "binaria": img_b,
        })
    except Exception as e:
        LOGGER.error("Error en preprocesado básico: %s", e)
    return out


def _extraer_mejor_contorno(img_binaria: Any) -> Optional[Any]:
    try:
        conts = seg.encontrar_contornos(img_binaria)
        conts = seg.filtrar_contornos_por_area(conts)
        if not conts:
            return None
        # Elegir el de mayor área
        import cv2  # type: ignore
        conts = sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)
        # Filtro de aspecto suave para evitar ruido extremo
        conts = seg.filtrar_contornos_por_aspecto(
            conts,
            umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
            umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0),
        )
        return conts[0] if conts else None
    except Exception as e:
        LOGGER.error("Error extrayendo mejor contorno: %s", e)
        return None


def _segmentar_robusto(img_color: Any) -> Optional[Tuple[Any, Any, Any]]:
    """
    Intenta varias estrategias de binarización y morfología y relaja filtros de área/aspecto
    hasta encontrar un contorno razonable. Devuelve (contorno, imagen_binaria_usada, img_redimensionada).
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return None

    try:
        # Etapas base
        img_r = pre.redimensionar_imagen(img_color)
        img_g = pre.convertir_gris(img_r)
        img_f = pre.aplicar_filtro_gaussiano(img_g)

        # Parámetros y candidatos de binarización
        methods = ["otsu", "adaptive", "binary"]
        ks_base = int(getattr(config, "MORFOLOGIA_KERNEL", 3) if config else 3)
        ks_list = sorted({max(1, ks_base), 3, 5, 7})
        op = getattr(config, "MORFOLOGIA_OPERACION", "cierre") if config else "cierre"

        candidatos = []
        for m in methods:
            try:
                b0 = pre.binarizar_imagen(img_f, metodo=m)
            except Exception:
                continue
            if b0 is None:
                continue
            for inv in (False, True):
                try:
                    b = cv2.bitwise_not(b0) if inv else b0
                except Exception:
                    b = b0
                for ks in ks_list:
                    try:
                        if getattr(config, "MORFOLOGIA_POST_UMBRAL", True) if config else True:
                            b2 = pre.aplicar_morfologia(b, operacion=op, kernel_size=int(ks))
                        else:
                            b2 = b
                        candidatos.append((b2, m, inv, ks))
                    except Exception:
                        candidatos.append((b, m, inv, ks))

        # Rango base y relajado
        amin0 = float(getattr(config, "AREA_MIN", 1000) if config else 1000.0)
        amax0 = float(getattr(config, "AREA_MAX", 50000) if config else 50000.0)
        aspecto_min0 = float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3)
        aspecto_max0 = float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0)
        relajas = [
            (1.0, 1.0, aspecto_min0, aspecto_max0),
            (0.5, 1.0, max(0.2, aspecto_min0 * 0.8), min(15.0, aspecto_max0 * 1.1)),
            (0.25, 2.0, 0.1, 20.0),
            (0.1, 3.0, 0.05, 25.0),
        ]

        import cv2  # ensure available
        for binaria, m, inv, ks in candidatos:
            try:
                conts = seg.encontrar_contornos(binaria)
                if not conts:
                    continue
                # ordenar por área desc
                conts = sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)
                found = None
                for fac_min, fac_max, asp_min, asp_max in relajas:
                    c_fil = seg.filtrar_contornos_por_area(conts, area_min=amin0 * fac_min, area_max=amax0 * fac_max)
                    if not c_fil:
                        continue
                    c_fil = seg.filtrar_contornos_por_aspecto(c_fil, umbral_min=asp_min, umbral_max=asp_max)
                    if c_fil:
                        # Elegir mejor contorno por score de calidad (solidez*rectangularidad*sqrt(area))
                        def _score(c):
                            try:
                                area = float(cv2.contourArea(c))
                                x, y, w, h = cv2.boundingRect(c)
                                rect_area = float(max(1, w * h))
                                rectangularidad = area / rect_area
                                hull = cv2.convexHull(c)
                                hull_area = float(cv2.contourArea(hull)) or 1.0
                                solidez = area / hull_area
                                # penalizar contornos que llenan demasiado o demasiado poco el bbox
                                fill = rectangularidad
                                import math
                                return (solidez * rectangularidad) * math.sqrt(max(area, 1.0)) * (0.5 + 0.5 * fill)
                            except Exception:
                                return 0.0
                        best = max(c_fil, key=_score)
                        found = best
                        break
                if found is not None:
                    return found, binaria, img_r
            except Exception:
                continue
    except Exception as e:
        LOGGER.info("Segmentación robusta fallida: %s", e)
        return None
    return None


def _extraer_features_por_imagen(img_color: Any) -> Optional[Dict[str, Any]]:
    """Extrae características intentando segmentación robusta si la básica falla."""
    # Intento robusto primero (más cobertura)
    robust = _segmentar_robusto(img_color)
    if robust is None:
        # Fallback al pipeline básico
        etapas = _procesar_imagen_basico(img_color)
        binaria = etapas.get("binaria")
        if binaria is None:
            return None
        cont = _extraer_mejor_contorno(binaria)
        if cont is None:
            return None
        img_ref = etapas.get("redimensionada", img_color)
        return ext.extraer_caracteristicas_completas(img_ref, cont, imagen_binaria=binaria)
    else:
        cont, binaria, img_r = robust
        return ext.extraer_caracteristicas_completas(img_r, cont, imagen_binaria=binaria)


def _crear_vector_desde_feats(feats: Dict[str, Any]) -> List[float]:
    # Orden según config.CARACTERISTICAS_PRINCIPALES
    orden = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
    vec: List[float] = []
    for k in orden:
        v = feats.get(k, 0)
        # Soporte para listas (p.ej., HOG), booleanos y escalares
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if isinstance(v, bool):
            vec.append(1.0 if v else 0.0)
        elif isinstance(v, (list, tuple)):
            for vi in v:
                try:
                    vec.append(float(vi))
                except Exception:
                    vec.append(0.0)
        elif np is not None and hasattr(np, 'ndarray') and isinstance(v, np.ndarray):  # type: ignore
            v = v.flatten().tolist()
            for vi in v:
                try:
                    vec.append(float(vi))
                except Exception:
                    vec.append(0.0)
        else:
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
    return vec


# ======================
# Pipelines
# ======================

def pipeline_entrenamiento(modelo: str = "svm", tune: bool = False, debug: int = 0) -> None:
    t0 = time.perf_counter()
    try:
        if hasattr(utils, "crear_directorios"):
            utils.crear_directorios()
    except Exception:
        pass

    # 1. Cargar dataset
    LOGGER.info("[1/6] Indexando dataset...")
    ds_path = _resolve_path(getattr(config, "DATASET_PATH", "./dataset") if config else "./dataset")
    rutas, labels = utils.listar_imagenes(ds_path)
    LOGGER.info("  -> %d rutas indexadas", len(rutas))

    # 2-4. Preprocesar, segmentar y extraer características
    LOGGER.info("[2-4/6] Extrayendo características...")
    X: List[List[float]] = []
    y: List[str] = []
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return

    # Preparar debug
    debug_count = {"arandelas": 0, "tornillos": 0, "tuercas": 0}
    debug_max = int(debug) if isinstance(debug, int) else 0
    debug_dir = None
    if debug_max > 0:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        debug_dir = os.path.join(_resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados"), f"debug_entrenamiento_{ts}")
        os.makedirs(debug_dir, exist_ok=True)

    for i, (ruta, lab) in enumerate(zip(rutas, labels)):
        try:
            img = cv2.imread(ruta, cv2.IMREAD_COLOR)
            if img is None:
                LOGGER.warning("No se pudo leer imagen: %s", ruta)
                continue
            # Debug opcional: guardar binaria y contornos
            feats = _extraer_features_por_imagen(img)
            if feats is None:
                continue
            X.append(_crear_vector_desde_feats(feats))
            y.append(lab)

            if debug_max > 0 and debug_dir is not None and debug_count.get(lab, 0) < debug_max:
                try:
                    import numpy as np  # type: ignore
                    # Reproducir etapas para obtener binaria y contornos
                    etapas = _procesar_imagen_basico(img)
                    binaria = etapas.get("binaria")
                    if binaria is not None:
                        import cv2  # type: ignore
                        # Guardar binaria
                        bpath = os.path.join(debug_dir, f"{lab}_{debug_count[lab]}_binaria.png")
                        cv2.imwrite(bpath, binaria)
                        # Contornos
                        conts = seg.encontrar_contornos(binaria)
                        conts = seg.filtrar_contornos_por_area(conts)
                        conts = seg.filtrar_contornos_por_aspecto(conts,
                            umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
                            umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0))
                        overlay = seg.dibujar_contornos(etapas.get("redimensionada", img), conts)
                        cpath = os.path.join(debug_dir, f"{lab}_{debug_count[lab]}_contornos.png")
                        cv2.imwrite(cpath, overlay)
                        debug_count[lab] += 1
                except Exception as e:
                    LOGGER.info("No se pudo guardar debug para %s: %s", lab, e)
        except Exception as e:
            LOGGER.warning("Fallo extrayendo características de imagen %d: %s", i, e)
    LOGGER.info("  -> %d muestras con características", len(X))

    if not X:
        LOGGER.error("No se pudieron extraer características. Abortando entrenamiento.")
        return

    # 5. Dividir y entrenar
    LOGGER.info("[5/6] Dividiendo dataset y entrenando (%s)...", modelo)
    X_train, X_test, y_train, y_test = utils.dividir_dataset(X, y, test_size=0.2)

    clasif = cls.ClasificadorPiezas()
    train_map = {
        "svm": clasif.entrenar_svm,
        "knn": clasif.entrenar_knn,
        "rf": clasif.entrenar_random_forest,
    }
    try:
        train_fn = train_map.get(modelo)
        if train_fn is None:
            LOGGER.warning("Modelo '%s' no reconocido. Usando 'svm'.", modelo)
            train_fn = clasif.entrenar_svm
            modelo = "svm"
        if tune:
            clasif.entrenar_con_gridsearch(X_train, y_train, tipo=modelo)
        else:
            train_fn(X_train, y_train)
    except Exception as e:
        LOGGER.error("Error entrenando modelo '%s': %s", modelo, e)
        LOGGER.info("Se puede usar clasificación por reglas como respaldo.")

    # 6. Evaluación y guardado
    LOGGER.info("[6/6] Evaluando y guardando modelo...")
    try:
        y_pred = [clasif.predecir(x, modelo=modelo) for x in X_test]
    except Exception as e:
        LOGGER.error("Error prediciendo con el modelo '%s': %s", modelo, e)
        y_pred = [clasif.clasificar_por_reglas(_to_feats_dict(x)) for x in X_test]

    # Guardar métricas
    try:
        metricas = ev.calcular_metricas(y_test, y_pred)
        ruta_metricas = os.path.join(_resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados"), "metricas.txt")
        ev.guardar_metricas(metricas, ruta=ruta_metricas)
        ev.generar_matriz_confusion(y_test, y_pred, clases=["arandelas", "tornillos", "tuercas"])  # figura opcional
    except Exception as e:
        LOGGER.info("Evaluación no disponible: %s", e)

    # Guardar modelo entrenado
    try:
        modelos_dir = _resolve_path(getattr(config, "MODELOS_PATH", "./modelos") if config else "./modelos")
        nombre = f"{modelo}_pipeline.joblib"
        ruta = os.path.join(modelos_dir, nombre)
        clasif.guardar_modelo(ruta, modelo=modelo)
    except Exception as e:
        LOGGER.info("No se pudo guardar el modelo: %s", e)

    t1 = time.perf_counter()
    LOGGER.info("Entrenamiento completo en %.2fs", t1 - t0)


def _to_feats_dict(vec: Sequence[float]) -> Dict[str, Any]:
    orden = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
    return {k: v for k, v in zip(orden, vec)}


def pipeline_prediccion(ruta_imagen: str, modelo: str = "svm") -> Optional[str]:
    t0 = time.perf_counter()
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias (cv2/numpy) no disponibles: %s", e)
        return None

    LOGGER.info("Cargando imagen: %s", ruta_imagen)
    img = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
    if img is None:
        LOGGER.error("No se pudo leer la imagen: %s", ruta_imagen)
        return None

    etapas = _procesar_imagen_basico(img)
    binaria = etapas.get("binaria")
    if binaria is None:
        LOGGER.error("No se pudo binarizar la imagen.")
        return None

    # Contornos y filtrados
    conts = seg.encontrar_contornos(binaria)
    conts = seg.filtrar_contornos_por_area(conts)
    conts = seg.filtrar_contornos_por_aspecto(
        conts,
        umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
        umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0),
    )

    # Preparar clasificador
    clasif = cls.ClasificadorPiezas()
    modelos_dir = _resolve_path(getattr(config, "MODELOS_PATH", "./modelos") if config else "./modelos")
    ruta_modelo = os.path.join(modelos_dir, f"{modelo}_pipeline.joblib")
    cargado = clasif.cargar_modelo(ruta_modelo, modelo=modelo)
    if cargado is None:
        LOGGER.warning("Modelo '%s' no cargado. Se usarán reglas.", modelo)

    # Dibujar y etiquetar
    salida = etapas.get("redimensionada", img).copy()
    try:
        for c in conts:
            x, y, w, h = cv2.boundingRect(c)
            feats = ext.extraer_caracteristicas_completas(etapas.get("redimensionada", img), c, imagen_binaria=binaria)
            pred = clasif.predecir(feats, modelo=modelo) if cargado is not None else clasif.clasificar_por_reglas(feats)
            cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(salida, pred, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        LOGGER.error("Error durante la clasificación/visualización: %s", e)

    # Guardar resultado
    out_dir = _resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ruta_imagen))[0]
    out_path = os.path.join(out_dir, f"pred_{base}.png")
    try:
        cv2.imwrite(out_path, salida)
        LOGGER.info("Resultado guardado en: %s", out_path)
    except Exception as e:
        LOGGER.error("No se pudo guardar el resultado: %s", e)
        out_path = None

    t1 = time.perf_counter()
    LOGGER.info("Predicción completa en %.2fs", t1 - t0)
    return out_path


def pipeline_evaluacion(modelo: str = "svm", debug: int = 0) -> None:
    t0 = time.perf_counter()
    # Reutiliza extracción de características (como en entrenamiento)
    LOGGER.info("Preparando datos para evaluación...")
    ds_path = _resolve_path(getattr(config, "DATASET_PATH", "./dataset") if config else "./dataset")
    rutas, labels = utils.listar_imagenes(ds_path)

    X: List[List[float]] = []
    y: List[str] = []
    procesadas: List[Tuple[str, str]] = []  # (ruta, etiqueta_real)
    saltadas: List[Tuple[str, str]] = []    # (ruta, razon)
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible: %s", e)
        return

    # Debug opcional de evaluación (muestras)
    debug_count = {"arandelas": 0, "tornillos": 0, "tuercas": 0}
    debug_max = int(debug) if isinstance(debug, int) else 0
    debug_dir = None
    if debug_max > 0:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        debug_dir = os.path.join(_resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados"), f"debug_evaluacion_{ts}")
        os.makedirs(debug_dir, exist_ok=True)

    for i, (ruta, lab) in enumerate(zip(rutas, labels)):
        try:
            img = cv2.imread(ruta, cv2.IMREAD_COLOR)
            if img is None:
                LOGGER.warning("No se pudo leer imagen: %s", ruta)
                continue
            feats = _extraer_features_por_imagen(img)
            if feats is None:
                saltadas.append((ruta, "sin_caracteristicas"))
                continue
            X.append(_crear_vector_desde_feats(feats))
            y.append(lab)
            procesadas.append((ruta, lab))

            if debug_max > 0 and debug_dir is not None and debug_count.get(lab, 0) < debug_max:
                try:
                    etapas = _procesar_imagen_basico(img)
                    binaria = etapas.get("binaria")
                    if binaria is not None:
                        import cv2  # type: ignore
                        # Guardar binaria
                        bpath = os.path.join(debug_dir, f"{lab}_{debug_count[lab]}_binaria.png")
                        cv2.imwrite(bpath, binaria)
                        conts = seg.encontrar_contornos(binaria)
                        conts = seg.filtrar_contornos_por_area(conts)
                        conts = seg.filtrar_contornos_por_aspecto(conts,
                            umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
                            umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0))
                        overlay = seg.dibujar_contornos(etapas.get("redimensionada", img), conts)
                        cpath = os.path.join(debug_dir, f"{lab}_{debug_count[lab]}_contornos.png")
                        cv2.imwrite(cpath, overlay)
                        debug_count[lab] += 1
                except Exception as e:
                    LOGGER.info("No se pudo guardar debug para %s: %s", lab, e)
        except Exception as e:
            LOGGER.warning("Fallo extrayendo características de imagen %d: %s", i, e)

    if not X:
        LOGGER.error("No hay datos para evaluar.")
        return

    # Cargar modelo y evaluar
    clasif = cls.ClasificadorPiezas()
    modelos_dir = _resolve_path(getattr(config, "MODELOS_PATH", "./modelos") if config else "./modelos")
    ruta_modelo = os.path.join(modelos_dir, f"{modelo}_pipeline.joblib")
    cargado = clasif.cargar_modelo(ruta_modelo, modelo=modelo)

    if cargado is None:
        LOGGER.warning("Modelo no disponible. Se evaluará con reglas (aproximación).")
        y_pred = [clasif.clasificar_por_reglas(_to_feats_dict(x)) for x in X]
    else:
        y_pred = [clasif.predecir(x, modelo=modelo) for x in X]

    try:
        # Generar paquete completo de estadísticas con carpeta/timestamp
        base_res = _resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados")
        rutas_out = ev.guardar_estadisticas_evaluacion(
            y_real=y,
            y_pred=y_pred,
            modelo=modelo,
            clases=["arandelas", "tornillos", "tuercas"],
            base_resultados=base_res,
        )

        # Guardar predicciones individuales y cobertura de procesamiento
        out_dir = rutas_out.get("carpeta_ejecucion") if isinstance(rutas_out, dict) else None
        if out_dir:
            try:
                pred_csv = os.path.join(out_dir, "predictions.csv")
                with open(pred_csv, "w", encoding="utf-8") as f:
                    f.write("ruta,real,pred\n")
                    for (ruta_i, real_i), pred_i in zip(procesadas, y_pred):
                        f.write(f"{ruta_i},{real_i},{pred_i}\n")
            except Exception as e:
                LOGGER.info("No se pudo guardar predictions.csv: %s", e)

            try:
                coverage_txt = os.path.join(out_dir, "coverage.txt")
                total = len(rutas)
                n_ok = len(procesadas)
                n_skip = len(saltadas)
                with open(coverage_txt, "w", encoding="utf-8") as f:
                    f.write(f"total_imagenes={total}\n")
                    f.write(f"procesadas={n_ok}\n")
                    f.write(f"saltadas={n_skip}\n")
                    if n_skip > 0:
                        f.write("saltadas_listado=\n")
                        for r, rsn in saltadas[:200]:
                            f.write(f"- {r} :: {rsn}\n")
            except Exception as e:
                LOGGER.info("No se pudo guardar coverage.txt: %s", e)

        # Mostrar accuracy al final por consola
        metricas = ev.calcular_metricas(y, y_pred)
        acc = metricas.get("accuracy") if metricas else None
        if acc is not None:
            LOGGER.info("Accuracy final: %.4f", acc)
            # También imprimir explícitamente para mayor visibilidad
            try:
                print(f"Accuracy: {acc:.4f}")
            except Exception:
                pass
        if out_dir:
            LOGGER.info("Resultados de evaluación guardados en: %s", out_dir)
    except Exception as e:
        LOGGER.info("Evaluación no disponible: %s", e)

    t1 = time.perf_counter()
    LOGGER.info("Evaluación completa en %.2fs", t1 - t0)


# ======================
# CLI
# ======================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestador de clasificación de piezas")
    parser.add_argument("--entrenar", action="store_true", help="Entrena nuevos modelos")
    parser.add_argument("--predecir", type=str, default=None, help="Ruta a una imagen para clasificar")
    parser.add_argument("--evaluar", action="store_true", help="Evalúa modelos existentes")
    parser.add_argument("--evaluar-todo", dest="evaluar_todo", action="store_true", help="Evalúa el modelo sobre TODO el dataset (alias de --evaluar)")
    parser.add_argument("--modelo", type=str, choices=["svm", "knn", "rf"], default="svm", help="Modelo a usar")
    parser.add_argument("--tune", action="store_true", help="Activa GridSearchCV para encontrar mejores hiperparámetros")
    parser.add_argument("--debug", type=int, default=0, help="Guarda N ejemplos por clase de binarizados y contornos")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.entrenar:
            pipeline_entrenamiento(modelo=args.modelo, tune=args.tune, debug=args.debug)
        if args.predecir:
            pipeline_prediccion(args.predecir, modelo=args.modelo)
        if args.evaluar or getattr(args, "evaluar_todo", False):
            pipeline_evaluacion(modelo=args.modelo, debug=args.debug)
        if not args.entrenar and not args.predecir and not args.evaluar and not getattr(args, "evaluar_todo", False):
            parser.print_help()
    except Exception as e:
        LOGGER.exception("Fallo en la ejecución principal: %s", e)


if __name__ == "__main__":
    main()
