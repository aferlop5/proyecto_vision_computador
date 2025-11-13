"""
Funciones auxiliares para el proyecto de visión por computador.
Incluye carga de dataset, particionado, guardado de artefactos, serialización de modelos
y utilidades de visualización y etiquetas.
"""
from __future__ import annotations

import os
import sys
import logging
from typing import List, Tuple, Sequence, Optional, Any

# Raíz del proyecto (padre de este directorio 'codigo')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    """Resuelve rutas relativas respecto a la raíz del proyecto."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))

# Intentamos importar la configuración desde distintas ubicaciones posibles
try:  # Si 'codigo' es paquete
    from . import config as config  # type: ignore
except Exception:  # Si no es paquete, intentamos resolver por nombre
    try:
        import config  # type: ignore
    except Exception:  # Como último recurso, seguimos sin config
        config = None  # type: ignore

# Configuración básica de logging (solo si no existe una configuración previa)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

LOGGER = logging.getLogger(__name__)

# Extensiones de imagen soportadas
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Rutas por defecto si no hay config
DATASET_PATH_DEFAULT = "./dataset"
MODELOS_PATH_DEFAULT = "./modelos"
RESULTADOS_PATH_DEFAULT = "./resultados"


def _ensure_dir(path: str) -> None:
    """Crea el directorio si no existe (anidado)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        LOGGER.error("No se pudo crear el directorio '%s': %s", path, e)
        raise


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def crear_directorios() -> None:
    """
    Asegura que existan los directorios de modelos y resultados.
    Usa rutas de config si existen; si no, usa valores por defecto.
    """
    modelos_path = _resolve_path(getattr(config, "MODELOS_PATH", MODELOS_PATH_DEFAULT) if config else MODELOS_PATH_DEFAULT)
    resultados_path = _resolve_path(getattr(config, "RESULTADOS_PATH", RESULTADOS_PATH_DEFAULT) if config else RESULTADOS_PATH_DEFAULT)

    LOGGER.info("Creando directorios si no existen: modelos='%s', resultados='%s'", modelos_path, resultados_path)
    _ensure_dir(modelos_path)
    _ensure_dir(resultados_path)


def guardar_imagen_procesada(imagen: Any, nombre: str, etapa: str) -> Optional[str]:
    """
    Guarda una imagen de alguna etapa de pre/procesado para documentación.

    - imagen: matriz de imagen (esperado: numpy.ndarray, BGR o RGB)
    - nombre: nombre de archivo (con o sin extensión)
    - etapa: subcarpeta bajo 'resultados'

    Devuelve la ruta escrita si tiene éxito, o None si falla.
    """
    try:
        import cv2  # local import para reducir dependencias si no se usa
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible: %s", e)
        return None

    resultados_base = _resolve_path(getattr(config, "RESULTADOS_PATH", RESULTADOS_PATH_DEFAULT) if config else RESULTADOS_PATH_DEFAULT)
    destino_dir = os.path.join(resultados_base, etapa)
    _ensure_dir(destino_dir)

    root, ext = os.path.splitext(nombre)
    if not ext:
        ext = ".png"
    destino_path = os.path.join(destino_dir, root + ext)

    try:
        ok = cv2.imwrite(destino_path, imagen)
        if not ok:
            LOGGER.warning("cv2.imwrite devolvió False para '%s'", destino_path)
            return None
        LOGGER.info("Imagen guardada: %s", destino_path)
        return destino_path
    except Exception as e:
        LOGGER.error("Fallo al guardar imagen '%s': %s", destino_path, e)
        return None


def cargar_dataset(ruta_base: Optional[str] = None) -> Tuple[List[Any], List[str]]:
    """
    Carga imágenes y etiquetas desde las carpetas directas: arandelas/, tornillos/, tuercas/
    sin descender a subcarpetas (p. ej., de ángulos). Devuelve (imagenes, etiquetas).

    - ruta_base: ruta al directorio del dataset (por defecto usa config.DATASET_PATH o './dataset')
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV (cv2) no disponible para leer imágenes: %s", e)
        return [], []

    base = _resolve_path(ruta_base or (getattr(config, "DATASET_PATH", DATASET_PATH_DEFAULT) if config else DATASET_PATH_DEFAULT))
    clases = ["arandelas", "tornillos", "tuercas"]

    imagenes: List[Any] = []
    etiquetas: List[str] = []

    for clase in clases:
        clase_dir = os.path.join(base, clase)
        if not os.path.isdir(clase_dir):
            LOGGER.warning("Directorio de clase no encontrado: %s", clase_dir)
            continue
        try:
            for fname in os.listdir(clase_dir):
                fpath = os.path.join(clase_dir, fname)
                if not os.path.isfile(fpath):
                    # Ignorar subcarpetas u otros
                    continue
                if not _is_image_file(fpath):
                    continue
                try:
                    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                    if img is None:
                        LOGGER.warning("No se pudo leer imagen: %s", fpath)
                        continue
                    imagenes.append(img)
                    etiquetas.append(clase)
                except Exception as e:
                    LOGGER.error("Error leyendo '%s': %s", fpath, e)
        except Exception as e:
            LOGGER.error("Error listando '%s': %s", clase_dir, e)

    LOGGER.info("Cargadas %d imágenes de %s", len(imagenes), base)
    return imagenes, etiquetas


def listar_imagenes(ruta_base: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Devuelve listas paralelas (rutas_imagenes, etiquetas) sin cargar en memoria las imágenes.
    Útil para pipelines con grandes datasets (menor uso de RAM).
    """
    base = _resolve_path(ruta_base or (getattr(config, "DATASET_PATH", DATASET_PATH_DEFAULT) if config else DATASET_PATH_DEFAULT))
    clases = ["arandelas", "tornillos", "tuercas"]

    rutas: List[str] = []
    etiquetas: List[str] = []

    for clase in clases:
        clase_dir = os.path.join(base, clase)
        if not os.path.isdir(clase_dir):
            LOGGER.warning("Directorio de clase no encontrado: %s", clase_dir)
            continue
        try:
            for fname in os.listdir(clase_dir):
                fpath = os.path.join(clase_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if not _is_image_file(fpath):
                    continue
                rutas.append(fpath)
                etiquetas.append(clase)
        except Exception as e:
            LOGGER.error("Error listando '%s': %s", clase_dir, e)

    LOGGER.info("Listadas %d rutas de %s", len(rutas), base)
    return rutas, etiquetas


def dividir_dataset(
    caracteristicas: Sequence[Any],
    etiquetas: Sequence[Any],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Divide en entrenamiento y prueba. Intenta usar sklearn para estratificación; si no está
    disponible, realiza una división simple conservando orden tras barajar con semilla fija.

    Devuelve: X_train, X_test, y_train, y_test
    """
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        return train_test_split(
            list(caracteristicas), list(etiquetas),
            test_size=test_size, random_state=random_state, stratify=list(etiquetas)
        )
    except Exception as e:
        LOGGER.warning("sklearn no disponible o error en train_test_split (%s). Usando fallback.", e)
        # Fallback manual
        import random
        n = len(caracteristicas)
        idx = list(range(n))
        random.Random(random_state).shuffle(idx)
        cut = int(n * (1 - test_size))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        X = list(caracteristicas)
        y = list(etiquetas)
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test


def guardar_modelo(modelo: Any, ruta: str) -> Optional[str]:
    """Guarda un modelo (SVM/KNN/etc.). Intenta usar joblib, si falla usa pickle."""
    # Crear directorio destino
    ruta_resuelta = _resolve_path(ruta)
    destino_dir = os.path.dirname(ruta_resuelta) or _resolve_path(getattr(config, "MODELOS_PATH", MODELOS_PATH_DEFAULT) if config else MODELOS_PATH_DEFAULT)
    if destino_dir:
        _ensure_dir(destino_dir)

    # Intento con joblib
    try:
        import joblib  # type: ignore
        joblib.dump(modelo, ruta_resuelta)
        LOGGER.info("Modelo guardado con joblib: %s", ruta_resuelta)
        return ruta_resuelta
    except Exception as e:
        LOGGER.warning("joblib no disponible o fallo al guardar (%s). Usando pickle.", e)

    # Fallback con pickle
    try:
        import pickle
        with open(ruta_resuelta, "wb") as f:
            pickle.dump(modelo, f)
        LOGGER.info("Modelo guardado con pickle: %s", ruta_resuelta)
        return ruta_resuelta
    except Exception as e:
        LOGGER.error("No se pudo guardar el modelo en '%s': %s", ruta, e)
        return None


def cargar_modelo(ruta: str) -> Optional[Any]:
    """Carga un modelo serializado (intenta joblib y luego pickle)."""
    ruta_resuelta = _resolve_path(ruta)
    if not os.path.exists(ruta_resuelta):
        LOGGER.error("Ruta de modelo no encontrada: %s", ruta_resuelta)
        return None

    try:
        import joblib  # type: ignore
        modelo = joblib.load(ruta_resuelta)
        LOGGER.info("Modelo cargado con joblib: %s", ruta_resuelta)
        return modelo
    except Exception:
        pass

    try:
        import pickle
        with open(ruta_resuelta, "rb") as f:
            modelo = pickle.load(f)
        LOGGER.info("Modelo cargado con pickle: %s", ruta_resuelta)
        return modelo
    except Exception as e:
        LOGGER.error("No se pudo cargar el modelo '%s': %s", ruta, e)
        return None


def plot_contornos(imagen: Any, contornos: Sequence[Any]) -> None:
    """
    Dibuja contornos sobre la imagen para depuración. Si matplotlib está disponible, muestra una ventana.
    En cualquier caso, también intenta guardar una versión con contornos en 'resultados/debug_contornos/'.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        LOGGER.error("OpenCV no disponible para dibujar contornos: %s", e)
        return

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("NumPy no disponible: %s", e)
        return

    # Aseguramos una copia en color
    if len(getattr(imagen, "shape", [])) == 2:
        canvas = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        canvas = imagen.copy()

    try:
        cv2.drawContours(canvas, contornos, -1, (0, 255, 0), 2)
    except Exception as e:
        LOGGER.error("Error dibujando contornos: %s", e)
        return

    # Mostrar si es posible
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure("Contornos")
        # Convertir BGR->RGB para matplotlib
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        plt.imshow(canvas_rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)
    except Exception as e:
        LOGGER.info("Matplotlib no disponible para mostrar: %s", e)

    # Guardar para debug
    guardar_imagen_procesada(canvas, "contornos_debug.png", etapa="debug_contornos")


def generar_etiquetas_desde_ruta(ruta_imagen: str) -> Optional[str]:
    """
    Extrae la etiqueta (clase) a partir de la ruta del archivo.
    Busca alguno de: 'arandelas', 'tornillos', 'tuercas' en los componentes del path.
    Si no encuentra, intenta usar el directorio padre inmediato.
    """
    objetivo = {"arandelas", "tornillos", "tuercas"}
    partes = [p.lower() for p in ruta_imagen.replace("\\", "/").split("/") if p]

    for p in partes:
        if p in objetivo:
            return p

    # Si no hay coincidencia directa, usa el padre inmediato del archivo
    try:
        padre = os.path.basename(os.path.dirname(ruta_imagen))
        padre_l = padre.lower()
        if padre_l in objetivo:
            return padre_l
    except Exception:
        pass

    LOGGER.warning("No se pudo inferir etiqueta desde la ruta: %s", ruta_imagen)
    return None
