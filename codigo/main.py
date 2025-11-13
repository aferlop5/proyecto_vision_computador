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
        # Etapas base: redimensionar y preprocesar específicamente para segmentación
        img_r = pre.redimensionar_imagen(img_color)
        img_seg = pre.preprocesar_para_segmentacion(img_r)

        # Parámetros y candidatos de binarización
        methods = ["otsu", "adaptive", "binary"]
        ks_base = int(getattr(config, "MORFOLOGIA_KERNEL", 3) if config else 3)
        ks_list = sorted({max(1, ks_base), 3, 5, 7})
        op = getattr(config, "MORFOLOGIA_OPERACION", "cierre") if config else "cierre"

        candidatos = []
        for m in methods:
            try:
                b0 = pre.binarizar_imagen(img_seg, metodo=m)
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

        # Máscara combinada: unimos distintas binarizaciones y contornos Canny
        def _build_combined_mask(b: Any) -> Any:
            try:
                base = b.copy()
                # Canny sobre la imagen preprocesada
                edges = cv2.Canny(img_seg, 50, 150)
                # Dilatar para engordar un poco los bordes
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges_d = cv2.dilate(edges, kernel, iterations=1)
                # Unión de máscara binaria y bordes dilatados
                combined = cv2.bitwise_or(base, edges_d)
                return combined
            except Exception:
                return b

        for binaria, m, inv, ks in candidatos:
            try:
                mask = _build_combined_mask(binaria)
                conts = seg.encontrar_contornos(mask)
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
                                # penalizar contornos pegados al borde de la imagen
                                h_img, w_img = mask.shape[:2]
                                margin = 3
                                if x <= margin or y <= margin or (x + w) >= w_img - margin or (y + h) >= h_img - margin:
                                    borde_penal = 0.5
                                else:
                                    borde_penal = 1.0
                                return (solidez * rectangularidad) * math.sqrt(max(area, 1.0)) * (0.5 + 0.5 * fill) * borde_penal
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
        y_pred = [clasif.predecir_con_validacion(_to_feats_dict(x), modelo=modelo) for x in X_test]
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
    feats: Dict[str, Any] = {}
    idx = 0
    total = len(vec)
    lengths: Dict[str, int] = {}
    if "hog" in orden:
        try:
            lengths["hog"] = (
                int(getattr(config, "HOG_CELDAS_X", 8) if config else 8)
                * int(getattr(config, "HOG_CELDAS_Y", 8) if config else 8)
                * int(getattr(config, "HOG_ORIENTACIONES", 9) if config else 9)
            )
        except Exception:
            lengths["hog"] = 0
    if "hu_moments" in orden:
        try:
            lengths["hu_moments"] = int(getattr(config, "HU_MOMENTS_LENGTH", 7) if config else 7)
        except Exception:
            lengths["hu_moments"] = 7
    if "deep_features" in orden:
        try:
            deep_len = int(getattr(config, "DEEP_FEATURES_DIM", 512) if config else 512)
        except Exception:
            deep_len = 0
        if deep_len > 0 and getattr(config, "USE_DEEP_FEATURES", False) if config else False:
            lengths["deep_features"] = deep_len

    for key in orden:
        expected = lengths.get(key, None)
        if expected is not None and expected > 0:
            remaining = total - idx
            if remaining >= expected:
                slice_end = idx + expected
                feats[key] = list(vec[idx:slice_end])
                idx = slice_end
                continue
            else:
                feats[key] = []
                continue

        val = float(vec[idx]) if idx < total else 0.0
        idx += 1
        if key.startswith("tiene_"):
            feats[key] = bool(val >= 0.5)
        else:
            feats[key] = val
    return feats


def pipeline_prediccion(
    ruta_imagen: str,
    modelo: str = "svm",
    clasificador: Optional[cls.ClasificadorPiezas] = None,
    destino_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
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

    # Usar segmentación robusta también en predicción
    etapas = _procesar_imagen_basico(img)
    robusto = _segmentar_robusto(img)
    if robusto is not None:
        cont_principal, binaria, img_r = robusto
        etapas["redimensionada"] = img_r
        conts = [cont_principal]
    else:
        binaria = etapas.get("binaria")
        if binaria is None:
            LOGGER.error("No se pudo binarizar la imagen.")
            return None
        conts = seg.encontrar_contornos(binaria)
        conts = seg.filtrar_contornos_por_area(conts)
        conts = seg.filtrar_contornos_por_aspecto(
            conts,
            umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
            umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0),
        )

    # Preparar clasificador
    clasif = clasificador if clasificador is not None else cls.ClasificadorPiezas()
    modelos_dir = _resolve_path(getattr(config, "MODELOS_PATH", "./modelos") if config else "./modelos")
    ruta_modelo = os.path.join(modelos_dir, f"{modelo}_pipeline.joblib")
    cargado = None
    try:
        if getattr(clasif, "_pipelines", {}).get(modelo) is None:
            cargado = clasif.cargar_modelo(ruta_modelo, modelo=modelo)
        else:
            cargado = getattr(clasif, "_pipelines", {}).get(modelo)
    except Exception:
        cargado = clasif.cargar_modelo(ruta_modelo, modelo=modelo)
    if cargado is None and getattr(clasif, "_pipelines", {}).get(modelo) is None:
        LOGGER.warning("Modelo '%s' no cargado. Se usarán reglas.", modelo)

    # Dibujar y etiquetar con contornos precisos
    salida = etapas.get("redimensionada", img).copy()
    colores = {
        "arandelas": (255, 0, 0),
        "tornillos": (0, 255, 0),
        "tuercas": (0, 165, 255),
    }
    predicciones: List[str] = []
    try:
        if not conts:
            cv2.putText(salida, "Sin contorno detectable", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for c in conts:
            feats = ext.extraer_caracteristicas_completas(salida, c, imagen_binaria=binaria)
            pred = clasif.predecir_con_validacion(feats, modelo=modelo) if cargado is not None else clasif.clasificar_por_reglas(feats)
            predicciones.append(pred)
            color = colores.get(pred, (0, 255, 255))
            cv2.drawContours(salida, [c], -1, color, 2)
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
            cv2.putText(salida, pred, (max(0, cx - 40), max(0, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Marcado adicional del agujero en caso de arandela
            if feats.get("tiene_agujero"):
                try:
                    (ccx, ccy), radius = cv2.minEnclosingCircle(c)
                    cv2.circle(salida, (int(ccx), int(ccy)), int(radius * 0.35), (255, 255, 255), 2)
                except Exception:
                    pass
    except Exception as e:
        LOGGER.error("Error durante la clasificación/visualización (contornos): %s", e)

    # Guardar resultado
    if destino_dir is not None:
        out_dir = destino_dir
    else:
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
    return {"output_path": out_path, "predicciones": predicciones, "contornos": len(conts or [])}


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
        y_pred = [clasif.predecir_con_validacion(_to_feats_dict(x), modelo=modelo) for x in X]

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

        # ==========================
        # Generar anotaciones multi-objeto por imagen
        # ==========================
        try:
            import cv2  # type: ignore
        except Exception as e:
            LOGGER.error("OpenCV no disponible para anotaciones multi-objeto: %s", e)
            cv2 = None  # type: ignore

        if cv2 is not None:
            try:
                resultado_eval_dir = os.path.join(_resolve_path(getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados"), "resultados_evaluacion")
                os.makedirs(resultado_eval_dir, exist_ok=True)
            except Exception as e:
                LOGGER.error("No se pudo crear carpeta resultados_evaluacion: %s", e)
                resultado_eval_dir = None

            if resultado_eval_dir is not None:
                LOGGER.info("Generando imágenes anotadas (multi-objeto) en %s", resultado_eval_dir)
                for (ruta_img, etiqueta_real), pred_global in zip(procesadas, y_pred):
                    try:
                        img_color = cv2.imread(ruta_img, cv2.IMREAD_COLOR)
                        if img_color is None:
                            continue
                        # Intento de segmentación robusta primero para obtener un contorno más fiel
                        robusto = _segmentar_robusto(img_color)
                        if robusto is not None:
                            cont_principal, binaria_loc, img_r_loc = robusto
                            base_vis = img_r_loc.copy()
                            conts_loc = [cont_principal]
                        else:
                            etapas_loc = _procesar_imagen_basico(img_color)
                            binaria_loc = etapas_loc.get("binaria")
                            base_vis = etapas_loc.get("redimensionada", img_color).copy()
                            if binaria_loc is None:
                                continue
                            # Usar máscara combinada también en el fallback
                            try:
                                img_seg_fallback = pre.preprocesar_para_segmentacion(base_vis)
                                b_otsu = pre.binarizar_imagen(img_seg_fallback, metodo="otsu")
                                edges_fb = cv2.Canny(img_seg_fallback, 50, 150)
                                kernel_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                edges_fb = cv2.dilate(edges_fb, kernel_fb, iterations=1)
                                mask_fb = cv2.bitwise_or(b_otsu, edges_fb)
                                conts_loc = seg.encontrar_contornos(mask_fb)
                            except Exception:
                                conts_loc = seg.encontrar_contornos(binaria_loc)
                            conts_loc = seg.filtrar_contornos_por_area(conts_loc)
                            conts_loc = seg.filtrar_contornos_por_aspecto(
                                conts_loc,
                                umbral_min=float(getattr(config, "ASPECTO_MIN", 0.3) if config else 0.3),
                                umbral_max=float(getattr(config, "ASPECTO_MAX", 12.0) if config else 12.0),
                            )
                        if not conts_loc:
                            # Marcar imagen sin objetos detectados
                            cv2.putText(base_vis, f"Sin objetos / Real: {etiqueta_real}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Colores por clase para contornos
                            colores = {
                                "arandelas": (255, 0, 0),     # azul (BGR)
                                "tornillos": (0, 255, 0),     # verde
                                "tuercas": (0, 165, 255),     # naranja
                            }
                            for c in conts_loc:
                                try:
                                    feats_loc = ext.extraer_caracteristicas_completas(base_vis, c, imagen_binaria=binaria_loc)
                                    pred_loc = clasif.predecir_con_validacion(feats_loc, modelo=modelo) if cargado is not None else clasif.clasificar_por_reglas(feats_loc)
                                    color = colores.get(pred_loc, (0, 255, 255))  # fallback amarillo
                                    # Dibujar contorno exacto
                                    cv2.drawContours(base_vis, [c], -1, color, 2)
                                    # Centroide para colocar etiqueta
                                    M = cv2.moments(c)
                                    if M.get("m00", 0) != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                    else:
                                        x, y, w, h = cv2.boundingRect(c)
                                        cx, cy = x + w // 2, y + h // 2
                                    etiqueta_txt = f"{pred_loc} / R:{etiqueta_real}"
                                    cv2.putText(base_vis, etiqueta_txt, (max(0, cx - 40), max(0, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                                    # Si se detecta agujero (arandela), remarcarlo con círculo interno
                                    if feats_loc.get("tiene_agujero"):
                                        try:
                                            # Buscar contorno interno (agujero) dentro del bbox usando la binaria
                                            x, y, w, h = cv2.boundingRect(c)
                                            crop_bin = binaria_loc[y:y+h, x:x+w]
                                            inv = cv2.bitwise_not(crop_bin)
                                            inner_conts = seg.encontrar_contornos(inv)
                                            inner_conts = seg.filtrar_contornos_por_area(inner_conts, area_min=50, area_max=w*h*0.5)
                                            if inner_conts:
                                                best_inner = max(inner_conts, key=lambda ic: cv2.contourArea(ic))
                                                inner_shift = best_inner.copy()
                                                inner_shift[:,0,0] += x
                                                inner_shift[:,0,1] += y
                                                cv2.drawContours(base_vis, [inner_shift], -1, (255, 255, 255), 2)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    LOGGER.info("Fallo anotando un contorno en %s: %s", ruta_img, e)
                        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
                        out_annot = os.path.join(resultado_eval_dir, f"annot_{nombre_base}.png")
                        try:
                            cv2.imwrite(out_annot, base_vis)
                        except Exception as e:
                            LOGGER.info("No se pudo guardar anotación de %s: %s", ruta_img, e)
                    except Exception as e:
                        LOGGER.info("Fallo procesando anotación multi-objeto para %s: %s", ruta_img, e)
                LOGGER.info("Imágenes anotadas multi-objeto generadas.")
    except Exception as e:
        LOGGER.info("Evaluación no disponible: %s", e)

    t1 = time.perf_counter()
    LOGGER.info("Evaluación completa en %.2fs", t1 - t0)


def pipeline_evaluacion_test(modelo: str = "svm") -> None:
    """Clasifica todas las imágenes del directorio dataset/test y guarda anotaciones y estadísticas."""
    test_base = _resolve_path(os.path.join(getattr(config, "DATASET_PATH", "./dataset") if config else "./dataset", "test"))
    if not os.path.isdir(test_base):
        LOGGER.error("Directorio de test no encontrado: %s", test_base)
        return

    archivos = [os.path.join(test_base, f) for f in sorted(os.listdir(test_base)) if os.path.isfile(os.path.join(test_base, f))]
    archivos = [f for f in archivos if utils._is_image_file(f)]  # type: ignore[attr-defined]
    if not archivos:
        LOGGER.warning("No se encontraron imágenes en %s", test_base)
        return

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_resultados = _resolve_path("resultados_test")
    out_dir = os.path.join(base_resultados, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    clasif = cls.ClasificadorPiezas()
    modelos_dir = _resolve_path(getattr(config, "MODELOS_PATH", "./modelos") if config else "./modelos")
    ruta_modelo = os.path.join(modelos_dir, f"{modelo}_pipeline.joblib")
    if clasif.cargar_modelo(ruta_modelo, modelo=modelo) is None:
        LOGGER.warning("Modelo '%s' no cargado; se aplicarán solo reglas.", modelo)

    from collections import Counter
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    conteo = Counter()
    registros: List[Tuple[str, str]] = []
    saltadas: List[str] = []

    for ruta in archivos:
        try:
            resultado = pipeline_prediccion(ruta, modelo=modelo, clasificador=clasif, destino_dir=out_dir)
            if resultado is None:
                saltadas.append(ruta)
                continue
            preds = resultado.get("predicciones", [])
            if preds:
                pred = preds[0]
            else:
                pred = "sin_prediccion"
            conteo[pred] += 1
            registros.append((ruta, pred))
        except Exception as e:
            LOGGER.info("Fallo procesando %s: %s", ruta, e)
            saltadas.append(ruta)

    stats_path = os.path.join(out_dir, "estadisticas.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        total = len(archivos)
        f.write(f"total_imagenes={total}\n")
        f.write(f"procesadas={total - len(saltadas)}\n")
        f.write(f"saltadas={len(saltadas)}\n")
        f.write("predicciones_por_clase=\n")
        for clase, cnt in conteo.most_common():
            f.write(f"- {clase}: {cnt}\n")
        if saltadas:
            f.write("saltadas_listado=\n")
            for ruta in saltadas:
                f.write(f"- {ruta}\n")

    csv_path = os.path.join(out_dir, "predicciones.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("ruta,pred\n")
        for ruta, pred in registros:
            f.write(f"{ruta},{pred}\n")

    # Graficas y visualizaciones
    if plt is not None and conteo:
        try:
            graf_dir = os.path.join(out_dir, "graficas")
            os.makedirs(graf_dir, exist_ok=True)

            clases = sorted(conteo.keys())
            valores = [conteo[c] for c in clases]
            fig, ax = plt.subplots(figsize=(6, 4))
            colores = ["#1f77b4", "#2ca02c", "#ff7f0e"]
            ax.bar(clases, valores, color=[colores[i % len(colores)] for i in range(len(clases))])
            ax.set_title("Predicciones por clase")
            ax.set_ylabel("Cantidad")
            ax.set_xlabel("Clase predicha")
            for idx, val in enumerate(valores):
                ax.text(idx, val + 0.05, str(val), ha="center", va="bottom")
            plt.tight_layout()
            graf_path = os.path.join(graf_dir, "predicciones_por_clase.png")
            fig.savefig(graf_path)
            plt.close(fig)

            if registros:
                nombres = [os.path.basename(r) for r, _ in registros]
                clases_pred = [p for _, p in registros]
                colores_map = {cl: colores[i % len(colores)] for i, cl in enumerate(clases)}
                fig, ax = plt.subplots(figsize=(max(6, len(nombres) * 0.35), 4))
                ax.bar(nombres, [1] * len(nombres), color=[colores_map.get(c, "#1f77b4") for c in clases_pred])
                ax.set_xticklabels(nombres, rotation=75, ha="right")
                ax.set_ylabel("Predicción")
                ax.set_title("Predicciones por imagen")
                legend_handles = [plt.Line2D([0], [0], color=colores_map[c], lw=4, label=c) for c in clases]
                ax.legend(handles=legend_handles, title="Clase")
                plt.tight_layout()
                distrib_path = os.path.join(graf_dir, "predicciones_por_imagen.png")
                fig.savefig(distrib_path)
                plt.close(fig)
        except Exception as e:
            LOGGER.info("No se pudo generar graficas para test: %s", e)

    LOGGER.info("Evaluación de test completada. Resultados en: %s", out_dir)


# ======================
# CLI
# ======================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestador de clasificación de piezas")
    parser.add_argument("--entrenar", action="store_true", help="Entrena nuevos modelos")
    parser.add_argument("--predecir", type=str, default=None, help="Ruta a una imagen para clasificar")
    parser.add_argument("--evaluar", action="store_true", help="Evalúa modelos existentes")
    parser.add_argument("--evaluar-todo", dest="evaluar_todo", action="store_true", help="Evalúa el modelo sobre TODO el dataset (alias de --evaluar)")
    parser.add_argument("--evaluar-test", dest="evaluar_test", action="store_true", help="Clasifica las imágenes de dataset/test y guarda anotaciones en resultados_test")
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
        if getattr(args, "evaluar_test", False):
            pipeline_evaluacion_test(modelo=args.modelo)
        if (
            not args.entrenar
            and not args.predecir
            and not args.evaluar
            and not getattr(args, "evaluar_todo", False)
            and not getattr(args, "evaluar_test", False)
        ):
            parser.print_help()
    except Exception as e:
        LOGGER.exception("Fallo en la ejecución principal: %s", e)


if __name__ == "__main__":
    main()
