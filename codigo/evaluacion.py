"""
Utilidades de evaluación del clasificador:
- Métricas (accuracy, precision, recall, f1)
- Matriz de confusión y reporte
- Visualización de resultados
- Curvas de aprendizaje
- Evaluaciones específicas por clase y análisis de errores
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

# Imports de config y utils (robustos)
try:
    from . import config as config  # type: ignore
except Exception:
    try:
        import config  # type: ignore
    except Exception:
        config = None  # type: ignore

try:
    from .utils import crear_directorios  # type: ignore
except Exception:
    try:
        from utils import crear_directorios  # type: ignore
    except Exception:
        crear_directorios = None  # type: ignore

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_CLASSES = ["arandelas", "tornillos", "tuercas"]


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        LOGGER.error("No se pudo crear directorio '%s': %s", path, e)


# ======================
# MÉTRICAS BÁSICAS
# ======================

def calcular_metricas(y_real: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    """Devuelve accuracy, precision_macro, recall_macro, f1_macro y f1_weighted."""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
    except Exception as e:
        LOGGER.error("scikit-learn no disponible: %s", e)
        return {}

    acc = float(accuracy_score(y_real, y_pred))
    prec = float(precision_score(y_real, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_real, y_pred, average="macro", zero_division=0))
    f1m = float(f1_score(y_real, y_pred, average="macro", zero_division=0))
    f1w = float(f1_score(y_real, y_pred, average="weighted", zero_division=0))
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1m,
        "f1_weighted": f1w,
    }


def generar_matriz_confusion(
    y_real: Sequence[Any],
    y_pred: Sequence[Any],
    clases: Optional[Sequence[str]] = None,
):
    """
    Calcula y devuelve (matriz, fig) si matplotlib está disponible.
    """
    try:
        from sklearn.metrics import confusion_matrix  # type: ignore
    except Exception as e:
        LOGGER.error("scikit-learn no disponible: %s", e)
        return None, None

    labels = list(clases) if clases is not None else list(DEFAULT_CLASSES)
    cm = confusion_matrix(y_real, y_pred, labels=labels)

    fig = None
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=labels, yticklabels=labels,
               ylabel="Real", xlabel="Predicho",
               title="Matriz de confusión")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = "d"
        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
    except Exception as e:
        LOGGER.info("Matplotlib no disponible para graficar matriz de confusión: %s", e)

    return cm, fig


def generar_reporte_clasificacion(y_real: Sequence[Any], y_pred: Sequence[Any]) -> Optional[str]:
    """Devuelve el texto del classification_report si está disponible."""
    try:
        from sklearn.metrics import classification_report  # type: ignore
        return classification_report(y_real, y_pred, zero_division=0)
    except Exception as e:
        LOGGER.error("scikit-learn no disponible: %s", e)
        return None


# ======================
# PAQUETE DE ESTADÍSTICAS (carpeta con timestamp)
# ======================

def guardar_estadisticas_evaluacion(
    y_real: Sequence[Any],
    y_pred: Sequence[Any],
    modelo: str = "svm",
    clases: Optional[Sequence[str]] = None,
    base_resultados: Optional[str] = None,
) -> Dict[str, str]:
    """
    Crea una carpeta resultados/estadisticas_evaluaciones/<timestamp> y guarda:
      - metricas_{modelo}.txt
      - classification_report.txt
      - confusion_matrix.png
      - per_class_metrics.png (si matplotlib disponible)

    Devuelve un dict con rutas generadas. Si alguna no se pudo crear, no aparece.
    """
    rutas: Dict[str, str] = {}

    # Resolver base de resultados
    try:
        base_root = base_resultados or getattr(config, "RESULTADOS_PATH", "./resultados") if config else "./resultados"
    except Exception:
        base_root = base_resultados or "./resultados"

    base_root = os.path.abspath(base_root)
    estad_dir = os.path.join(base_root, "estadisticas_evaluaciones")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(estad_dir, ts)
    _ensure_dir(run_dir)

    # 1) Métricas agregadas
    metricas = calcular_metricas(y_real, y_pred)
    if metricas:
        ruta_metricas = os.path.join(run_dir, f"metricas_{modelo}.txt")
        if guardar_metricas(metricas, ruta=ruta_metricas):
            rutas["metricas"] = ruta_metricas

    # 2) Reporte de clasificación (texto)
    rep = generar_reporte_clasificacion(y_real, y_pred)
    if rep is not None:
        ruta_rep = os.path.join(run_dir, "classification_report.txt")
        try:
            with open(ruta_rep, "w", encoding="utf-8") as f:
                f.write(rep)
            rutas["classification_report"] = ruta_rep
        except Exception as e:
            LOGGER.error("No se pudo guardar classification_report: %s", e)

    # 3) Matriz de confusión (imagen)
    cm, fig = generar_matriz_confusion(y_real, y_pred, clases=clases)
    if fig is not None:
        ruta_cm = os.path.join(run_dir, "confusion_matrix.png")
        try:
            fig.savefig(ruta_cm, bbox_inches="tight")
            rutas["confusion_matrix"] = ruta_cm
        except Exception as e:
            LOGGER.error("No se pudo guardar imagen de matriz de confusión: %s", e)
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.close(fig)
        except Exception:
            pass

    # 4) Métricas por clase (barras)
    try:
        per_cls = evaluar_por_clase(y_real, y_pred, clases=clases)
    except Exception:
        per_cls = {}

    if per_cls:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            labels = list(per_cls.keys())
            precision = [per_cls[c]["precision"] for c in labels]
            recall = [per_cls[c]["recall"] for c in labels]
            f1 = [per_cls[c]["f1"] for c in labels]

            x = range(len(labels))
            width = 0.25
            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.bar([i - width for i in x], precision, width=width, label="precision")
            ax.bar(x, recall, width=width, label="recall")
            ax.bar([i + width for i in x], f1, width=width, label="f1")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title("Métricas por clase")
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc="best")
            fig2.tight_layout()
            ruta_percls = os.path.join(run_dir, "per_class_metrics.png")
            fig2.savefig(ruta_percls, bbox_inches="tight")
            rutas["per_class_metrics"] = ruta_percls
            plt.close(fig2)
        except Exception as e:
            LOGGER.info("No se pudieron graficar métricas por clase: %s", e)

    LOGGER.info("Estadísticas de evaluación guardadas en %s", run_dir)
    rutas["carpeta_ejecucion"] = run_dir
    return rutas


# ======================
# VISUALIZACIÓN DE RESULTADOS
# ======================

def visualizar_resultados(
    imagenes_test: Sequence[Any],
    predicciones: Sequence[str],
    etiquetas_reales: Sequence[str],
    cols: int = 5,
    max_muestras: int = 25,
    guardar_en: Optional[str] = None,
):
    """
    Muestra una cuadrícula de imágenes con etiqueta real y predicción. Devuelve fig si puede.
    Si 'guardar_en' se proporciona, guarda la figura en esa ruta.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        LOGGER.error("Matplotlib/NumPy no disponibles: %s", e)
        return None

    n = min(len(imagenes_test), len(predicciones), len(etiquetas_reales), max_muestras)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            img = imagenes_test[i]
            # Convertir BGR->RGB si parece BGR (heurística: usar como está)
            try:
                import numpy as np  # type: ignore
                if len(getattr(img, "shape", [])) == 3 and img.shape[2] == 3:
                    # Intentar mostrar tal cual; si colores raros, usuario puede convertir antes
                    ax.imshow(img[..., ::-1])  # BGR->RGB
                else:
                    ax.imshow(img, cmap="gray")
            except Exception:
                ax.imshow(img, cmap="gray")
            ax.set_title(f"Real: {etiquetas_reales[i]}\nPred: {predicciones[i]}")
            ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout()

    if guardar_en:
        out_dir = os.path.dirname(guardar_en) or "."
        _ensure_dir(out_dir)
        try:
            fig.savefig(guardar_en, bbox_inches="tight")
        except Exception as e:
            LOGGER.error("No se pudo guardar figura en '%s': %s", guardar_en, e)
    return fig


# ======================
# PERSISTENCIA DE MÉTRICAS
# ======================

def guardar_metricas(metricas: Dict[str, Any], ruta: str = "resultados/metricas.txt") -> Optional[str]:
    """Guarda un dict de métricas en texto clave=valor por línea."""
    out_dir = os.path.dirname(ruta) or "."
    if crear_directorios is not None and os.path.basename(out_dir) in {"resultados", "modelos"}:
        try:
            crear_directorios()
        except Exception:
            _ensure_dir(out_dir)
    else:
        _ensure_dir(out_dir)

    try:
        with open(ruta, "w", encoding="utf-8") as f:
            for k, v in metricas.items():
                f.write(f"{k}={v}\n")
        LOGGER.info("Métricas guardadas en %s", ruta)
        return ruta
    except Exception as e:
        LOGGER.error("No se pudieron guardar métricas: %s", e)
        return None


# ======================
# CURVAS DE APRENDIZAJE
# ======================

def plot_curvas_aprendizaje(modelo: Any, X: Sequence[Sequence[float]], y: Sequence[Any], cv: int = 5, n_jobs: int = -1):
    """
    Genera curva de aprendizaje (train vs validation) usando sklearn.model_selection.learning_curve.
    Devuelve (train_sizes, train_scores_mean, val_scores_mean, fig) si es posible.
    """
    try:
        import numpy as np  # type: ignore
        from sklearn.model_selection import learning_curve  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        LOGGER.error("Dependencias para curvas de aprendizaje no disponibles: %s", e)
        return None, None, None, None

    try:
        train_sizes, train_scores, val_scores = learning_curve(modelo, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5))
        train_scores_mean = train_scores.mean(axis=1)
        val_scores_mean = val_scores.mean(axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_scores_mean, "o-", label="Entrenamiento")
        ax.plot(train_sizes, val_scores_mean, "o-", label="Validación")
        ax.set_xlabel("Tamaño de entrenamiento")
        ax.set_ylabel("Exactitud")
        ax.set_title("Curva de aprendizaje")
        ax.legend(loc="best")
        fig.tight_layout()
        return train_sizes, train_scores_mean, val_scores_mean, fig
    except Exception as e:
        LOGGER.error("Error generando curva de aprendizaje: %s", e)
        return None, None, None, None


# ======================
# EVALUACIÓN ESPECÍFICA
# ======================

def evaluar_por_clase(y_real: Sequence[Any], y_pred: Sequence[Any], clases: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Devuelve métricas por clase: precision, recall, f1 y soporte por clase."""
    try:
        from sklearn.metrics import precision_recall_fscore_support  # type: ignore
    except Exception as e:
        LOGGER.error("scikit-learn no disponible: %s", e)
        return {}

    labels = list(clases) if clases is not None else sorted(set(list(y_real) + list(y_pred)))
    p, r, f1, s = precision_recall_fscore_support(y_real, y_pred, labels=labels, zero_division=0)
    out: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(labels):
        out[str(c)] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "soporte": float(s[i])}
    return out


def analizar_errores_comunes(
    errores: Sequence[Any],
    caracteristicas: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Analiza errores agrupando por (y_real, y_pred) y contando ocurrencias.
    Si se proporcionan características (dicts o vectores), calcula medias por grupo.

    errores: lista de tuplas (idx, y_real, y_pred) o dicts con llaves 'idx','y_real','y_pred'.
    """
    from collections import defaultdict
    import numpy as np  # type: ignore

    grupos = defaultdict(list)
    for e in errores:
        try:
            if isinstance(e, dict):
                idx = int(e.get("idx"))
                yr = e.get("y_real")
                yp = e.get("y_pred")
            else:
                idx, yr, yp = e  # type: ignore
            grupos[(yr, yp)].append(idx)
        except Exception:
            continue

    resumen: Dict[str, Any] = {}
    for (yr, yp), idxs in grupos.items():
        entry: Dict[str, Any] = {"count": len(idxs), "indices": idxs}
        if caracteristicas is not None:
            try:
                # Si las características son dicts, promediar cada clave numérica
                if isinstance(caracteristicas[0], dict):
                    acum: Dict[str, List[float]] = defaultdict(list)
                    for i in idxs:
                        d = caracteristicas[i]
                        for k, v in d.items():
                            if isinstance(v, (int, float)):
                                acum[k].append(float(v))
                    entry["mean_features"] = {k: float(np.mean(vs)) if len(vs) else 0.0 for k, vs in acum.items()}
                else:
                    # Asumir vectores
                    mats = [caracteristicas[i] for i in idxs]
                    entry["mean_vector"] = np.asarray(mats, dtype=float).mean(axis=0).tolist()
            except Exception:
                pass
        resumen[f"{yr}->{yp}"] = entry
    return resumen


def generar_reporte_errores(
    errores: Sequence[Any],
    imagenes: Sequence[Any],
    max_muestras: int = 16,
    ruta_salida: str = "resultados/errores.png",
):
    """
    Genera una cuadrícula con ejemplos de errores (y_real, y_pred) y guarda la imagen.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        LOGGER.error("Matplotlib no disponible: %s", e)
        return None

    # Preparar muestras
    muestras: List[Tuple[int, Any, Any]] = []
    for e in errores:
        try:
            if isinstance(e, dict):
                idx, yr, yp = int(e.get("idx")), e.get("y_real"), e.get("y_pred")
            else:
                idx, yr, yp = e  # type: ignore
            muestras.append((idx, yr, yp))
        except Exception:
            continue
    muestras = muestras[:max_muestras]

    cols = 4
    rows = max(1, (len(muestras) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < len(muestras):
            idx, yr, yp = muestras[i]
            img = imagenes[idx]
            try:
                ax.imshow(img[..., ::-1]) if getattr(img, "ndim", 2) == 3 else ax.imshow(img, cmap="gray")
            except Exception:
                ax.imshow(img, cmap="gray")
            ax.set_title(f"Real: {yr}\nPred: {yp}")
            ax.axis("off")
        else:
            ax.axis("off")

    fig.tight_layout()

    out_dir = os.path.dirname(ruta_salida) or "."
    _ensure_dir(out_dir)
    try:
        fig.savefig(ruta_salida, bbox_inches="tight")
        LOGGER.info("Reporte de errores guardado en %s", ruta_salida)
    except Exception as e:
        LOGGER.error("No se pudo guardar reporte de errores: %s", e)
        return None
    return ruta_salida
