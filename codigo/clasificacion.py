"""
Clasificación de piezas mediante modelos de ML con un respaldo heurístico ligero.
Incluye pipelines para SVM, KNN y Random Forest, con opcional PCA y embeddings profundos.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Importaciones robustas de dependencias internas
try:  # ejecución como paquete (python -m codigo.main)
    from . import config as config  # type: ignore
except Exception:  # ejecución directa desde carpeta codigo/
    try:
        import config  # type: ignore
    except Exception:
        config = None  # type: ignore

try:
    from .utils import guardar_modelo as _guardar_modelo, cargar_modelo as _cargar_modelo  # type: ignore
except Exception:
    try:
        from utils import guardar_modelo as _guardar_modelo, cargar_modelo as _cargar_modelo  # type: ignore
    except Exception:
        _guardar_modelo = None  # type: ignore
        _cargar_modelo = None  # type: ignore

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

CLASES: List[str] = ["arandelas", "tornillos", "tuercas"]


def _merge_params(defaults: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(defaults)
    if not cfg:
        return params
    try:
        params.update(dict(cfg))
    except Exception:
        LOGGER.debug("Parámetros inválidos en config; usando defaults.")
    return params


class ClasificadorPiezas:
    """Gestiona pipelines ML y heurísticas de respaldo para clasificar piezas."""

    def __init__(self) -> None:
        default_svm = {"C": 1.0, "kernel": "rbf", "class_weight": "balanced", "probability": True}
        default_knn = {"n_neighbors": 5}
        default_rf = {"n_estimators": 300, "random_state": 42, "class_weight": "balanced"}

        cfg_svm = getattr(config, "SVM_PARAMS", None) if config else None
        cfg_knn = getattr(config, "KNN_PARAMS", None) if config else None
        cfg_rf = getattr(config, "RF_PARAMS", None) if config else None

        svm_params = _merge_params(default_svm, cfg_svm)
        svm_params.setdefault("probability", True)
        knn_params = _merge_params(default_knn, cfg_knn)
        rf_params = _merge_params(default_rf, cfg_rf)
        if "class_weight" not in rf_params:
            rf_params["class_weight"] = "balanced"

        self._pipelines: Dict[str, Any] = {"svm": None, "knn": None, "rf": None}
        self._params: Dict[str, Dict[str, Any]] = {"svm": svm_params, "knn": knn_params, "rf": rf_params}

    # ============================
    # Utilidades internas
    # ============================
    def _vectorizar(self, caracteristicas: Union[Sequence[float], Dict[str, Any]]) -> List[List[float]]:
        """Normaliza la entrada a un vector [[…]] respetando el orden configurado."""
        import numpy as np  # type: ignore

        if isinstance(caracteristicas, dict):
            orden: List[str] = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
            vec: List[float] = []
            for key in orden:
                valor = caracteristicas.get(key, 0)
                if isinstance(valor, bool):
                    vec.append(1.0 if valor else 0.0)
                elif isinstance(valor, (list, tuple)):
                    for item in valor:
                        try:
                            vec.append(float(item))
                        except Exception:
                            vec.append(0.0)
                elif isinstance(valor, np.ndarray):
                    for item in valor.flatten().tolist():
                        try:
                            vec.append(float(item))
                        except Exception:
                            vec.append(0.0)
                else:
                    try:
                        vec.append(float(valor))
                    except Exception:
                        vec.append(0.0)
            return [vec]

        arr = np.asarray(list(caracteristicas), dtype=float).reshape(1, -1)
        return arr.tolist()

    def _hog_length(self) -> int:
        """Calcula la longitud esperada del vector HOG según config."""
        try:
            cx = int(getattr(config, "HOG_CELDAS_X", 8) if config else 8)
            cy = int(getattr(config, "HOG_CELDAS_Y", 8) if config else 8)
            bins = int(getattr(config, "HOG_ORIENTACIONES", 9) if config else 9)
            return max(0, cx) * max(0, cy) * max(0, bins)
        except Exception:
            return 0

    def _feature_lengths(self) -> Dict[str, int]:
        lengths: Dict[str, int] = {}
        hog_len = self._hog_length()
        if hog_len > 0:
            lengths["hog"] = hog_len
        try:
            hu_len = int(getattr(config, "HU_MOMENTS_LENGTH", 7) if config else 7)
        except Exception:
            hu_len = 7
        if hu_len > 0:
            lengths["hu_moments"] = hu_len
        try:
            if getattr(config, "USE_DEEP_FEATURES", False) if config else False:
                deep_len = int(getattr(config, "DEEP_FEATURES_DIM", 512) if config else 512)
                if deep_len > 0:
                    lengths["deep_features"] = deep_len
        except Exception:
            pass
        return lengths

    def _vector_a_dict(self, vector: Sequence[float]) -> Dict[str, Any]:
        """Reconstruye un dict de características a partir del vector plano."""
        orden: List[str] = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
        feats: Dict[str, Any] = {}
        idx = 0
        total = len(vector)
        lengths = self._feature_lengths()
        for key in orden:
            if key in lengths:
                length = lengths[key]
                slice_end = min(total, idx + length)
                feats[key] = list(vector[idx:slice_end])
                idx = slice_end
                continue
            val = float(vector[idx]) if idx < total else 0.0
            idx += 1
            if key.startswith("tiene_"):
                feats[key] = bool(val >= 0.5)
            else:
                feats[key] = val
        return feats

    def _caracteristicas_a_dict(self, caracteristicas: Union[Sequence[float], Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(caracteristicas, dict):
            return dict(caracteristicas)
        return self._vector_a_dict(caracteristicas)

    def _get_pipeline(self, tipo: str):
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except Exception:
            PCA = None  # type: ignore

        if self._pipelines.get(tipo) is not None:
            return self._pipelines[tipo]

        if tipo == "svm":
            from sklearn.svm import SVC  # type: ignore

            params = dict(self._params["svm"])
            params.setdefault("class_weight", "balanced")
            params.setdefault("probability", True)
            clf = SVC(**params)
        elif tipo == "knn":
            from sklearn.neighbors import KNeighborsClassifier  # type: ignore

            clf = KNeighborsClassifier(**self._params["knn"])
        elif tipo == "rf":
            from sklearn.ensemble import RandomForestClassifier  # type: ignore

            params = dict(self._params["rf"])
            params.setdefault("class_weight", "balanced")
            clf = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {tipo}")

        steps = [("scaler", StandardScaler())]
        try:
            use_pca = bool(getattr(config, "USE_PCA", False) if config else False)
            if use_pca and PCA is not None:
                n_comp = int(getattr(config, "PCA_COMPONENTS", 100) if config else 100)
                whiten = bool(getattr(config, "PCA_WHITEN", False) if config else False)
                steps.append(("pca", PCA(n_components=n_comp, whiten=whiten)))
        except Exception:
            LOGGER.debug("PCA no disponible; continuando sin reducción")
        steps.append(("clf", clf))

        pipe = Pipeline(steps)
        self._pipelines[tipo] = pipe
        return pipe

    def _predict_proba(self, tipo: str, X: List[List[float]]) -> Optional[List[float]]:
        try:
            pipe = self._pipelines.get(tipo)
            if pipe is None:
                return None
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X)
            else:
                named_steps = getattr(pipe, "named_steps", {})
                clf = named_steps.get("clf") if isinstance(named_steps, dict) else None
                if clf is None or not hasattr(clf, "predict_proba"):
                    return None
                datos = X
                if isinstance(named_steps, dict):
                    if "scaler" in named_steps:
                        datos = named_steps["scaler"].transform(datos)
                    if "pca" in named_steps:
                        datos = named_steps["pca"].transform(datos)
                proba = clf.predict_proba(datos)
            if proba is None:
                return None
            classes = list(getattr(pipe, "classes_", []))
            scores = [0.0] * len(CLASES)
            row = proba[0].tolist()
            for i, prob in enumerate(row):
                label = classes[i] if i < len(classes) else None
                if label in CLASES:
                    scores[CLASES.index(label)] = float(prob)
            total = sum(scores) or 1.0
            return [s / total for s in scores]
        except Exception as exc:
            LOGGER.info("predict_proba no disponible (%s): %s", tipo, exc)
            return None

    # ============================
    # Entrenamiento
    # ============================
    def entrenar_svm(self, X_train: Sequence[Sequence[float]], y_train: Sequence[str]) -> None:
        pipe = self._get_pipeline("svm")
        pipe.fit(X_train, y_train)
        LOGGER.info("SVM entrenado")

    def entrenar_knn(self, X_train: Sequence[Sequence[float]], y_train: Sequence[str]) -> None:
        pipe = self._get_pipeline("knn")
        pipe.fit(X_train, y_train)
        LOGGER.info("KNN entrenado")

    def entrenar_random_forest(self, X_train: Sequence[Sequence[float]], y_train: Sequence[str]) -> None:
        pipe = self._get_pipeline("rf")
        pipe.fit(X_train, y_train)
        LOGGER.info("Random Forest entrenado")

    def entrenar_con_gridsearch(
        self,
        X_train: Sequence[Sequence[float]],
        y_train: Sequence[str],
        tipo: str = "svm",
    ) -> Dict[str, Any]:
        """Ejecuta GridSearchCV si está disponible y guarda el mejor estimador."""
        try:
            from sklearn.model_selection import GridSearchCV  # type: ignore
        except Exception as exc:
            LOGGER.error("GridSearchCV no disponible: %s", exc)
            train_map = {"svm": self.entrenar_svm, "knn": self.entrenar_knn, "rf": self.entrenar_random_forest}
            train_map.get(tipo, self.entrenar_svm)(X_train, y_train)
            return {}

        pipe = self._get_pipeline(tipo)
        grid: Dict[str, Any] = {}
        try:
            if tipo == "svm":
                grid = getattr(config, "GRID_SVM", {}) if config else {}
            elif tipo == "knn":
                grid = getattr(config, "GRID_KNN", {}) if config else {}
            elif tipo == "rf":
                grid = getattr(config, "GRID_RF", {}) if config else {}
        except Exception:
            grid = {}

        if not grid:
            LOGGER.warning("Parrilla vacía para '%s'; entrenando sin búsqueda.", tipo)
            pipe.fit(X_train, y_train)
            return {}

        gs = GridSearchCV(pipe, grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=0)
        gs.fit(X_train, y_train)
        self._pipelines[tipo] = gs.best_estimator_
        LOGGER.info("GridSearch '%s': best_score=%.4f, best_params=%s", tipo, float(gs.best_score_), gs.best_params_)
        return {"best_score": float(gs.best_score_), "best_params": dict(gs.best_params_)}

    # ============================
    # Predicción
    # ============================
    def predecir(self, caracteristicas: Union[Sequence[float], Dict[str, Any]], modelo: str = "svm") -> str:
        X = self._vectorizar(caracteristicas)
        pipe = self._pipelines.get(modelo)
        if pipe is None:
            LOGGER.warning("Modelo '%s' no entrenado. Usando reglas de respaldo.", modelo)
            return self.clasificar_por_reglas(caracteristicas)
        try:
            pred = pipe.predict(X)
            return str(pred[0])
        except Exception as exc:
            LOGGER.error("Error en predicción con '%s': %s", modelo, exc)
            return self.clasificar_por_reglas(caracteristicas)

    def predecir_con_validacion(
        self,
        caracteristicas: Union[Sequence[float], Dict[str, Any]],
        modelo: str = "svm",
        w_ml: float = 0.65,
        w_reglas: float = 0.35,
    ) -> str:
        """Combina modelo y reglas, rechazando predicciones incoherentes."""
        feats = self._caracteristicas_a_dict(caracteristicas)
        pred_ml = self.predecir(caracteristicas, modelo=modelo)

        vector = self._vectorizar(caracteristicas)
        proba = self._predict_proba(modelo, vector)
        top_score = 0.0
        if proba:
            idx_max = max(range(len(proba)), key=lambda i: proba[i])
            top_score = float(proba[idx_max])

        if top_score >= 0.45 and self.validar_prediccion(feats, pred_ml):
            return pred_ml

        pred_reglas = self.clasificar_por_reglas(feats)
        if top_score < 0.25:
            return pred_reglas

        if not self.validar_prediccion(feats, pred_ml) and self.validar_prediccion(feats, pred_reglas):
            LOGGER.info("Ajuste heurístico: %s → %s", pred_ml, pred_reglas)
            return pred_reglas

        combinado = self.predecir_combinado(feats, modelo=modelo, w_ml=w_ml, w_reglas=w_reglas)
        if combinado != pred_ml and self.validar_prediccion(feats, combinado) and top_score < 0.45:
            LOGGER.info("Ajuste combinado: %s → %s", pred_ml, combinado)
            return combinado

        return pred_ml

    # ============================
    # Persistencia
    # ============================
    def guardar_modelo(self, ruta: str, modelo: str = "svm") -> Optional[str]:
        pipe = self._pipelines.get(modelo)
        if pipe is None:
            LOGGER.error("No hay modelo '%s' para guardar.", modelo)
            return None
        if _guardar_modelo is None:
            LOGGER.error("Función guardar_modelo no disponible.")
            return None
        return _guardar_modelo(pipe, ruta)

    def cargar_modelo(self, ruta: str, modelo: str = "svm") -> Optional[Any]:
        if _cargar_modelo is None:
            LOGGER.error("Función cargar_modelo no disponible.")
            return None
        modelo_cargado = _cargar_modelo(ruta)
        if modelo_cargado is not None:
            self._pipelines[modelo] = modelo_cargado
        return modelo_cargado

    # ============================
    # Reglas de respaldo
    # ============================
    def clasificar_por_reglas(self, caracteristicas: Union[Sequence[float], Dict[str, Any]]) -> str:
        """Reglas heurísticas suaves usadas solo como respaldo."""
        feats = self._caracteristicas_a_dict(caracteristicas)

        aspecto = float(feats.get("relacion_aspecto", 0) or 0)
        tiene_aristas = bool(feats.get("tiene_aristas", False))
        indice_aristas = float(feats.get("indice_aristas", 0) or 0)
        tiene_agujero = bool(feats.get("tiene_agujero", False))
        ratio_agujero = float(feats.get("ratio_agujero", 0) or 0)
        circularidad = float(feats.get("circularidad", 0) or 0)
        solidez = float(feats.get("solidez", 0) or 0)
        rectangularidad = float(feats.get("rectangularidad", 0) or 0)

        umbral_aspecto = float(getattr(config, "UMBRAL_ASPECTO_TORNILLO", 1.7) if config else 1.7)
        umbral_circ_alta = float(getattr(config, "UMBRAL_CIRCULARIDAD_ALTA", 0.75) if config else 0.75)
        umbral_aristas = float(getattr(config, "UMBRAL_ARISTAS_TUERCA", 0.5) if config else 0.5)
        ratio_tuerca_max = float(getattr(config, "UMBRAL_RATIO_AGUJERO_TUERCA_MAX", 0.7) if config else 0.7)
        ratio_arandela_min = float(getattr(config, "UMBRAL_RATIO_AGUJERO_ARANDELA_MIN", 0.6) if config else 0.6)
        solidez_tuerca_max = float(getattr(config, "UMBRAL_SOLIDEZ_TUERCA_MAX", 0.9) if config else 0.9)
        rect_tuerca_min = float(getattr(config, "UMBRAL_RECTANGULARIDAD_TUERCA_MIN", 0.5) if config else 0.5)

        if aspecto > umbral_aspecto * 1.1:
            return "tornillos"

        if tiene_aristas and indice_aristas >= umbral_aristas:
            if tiene_agujero:
                if ratio_agujero <= ratio_tuerca_max or solidez <= solidez_tuerca_max or rectangularidad <= rect_tuerca_min:
                    return "tuercas"
            else:
                return "tuercas"

        if tiene_agujero:
            if ratio_agujero >= ratio_arandela_min and circularidad >= umbral_circ_alta and solidez > solidez_tuerca_max:
                return "arandelas"
            if ratio_agujero <= ratio_tuerca_max and (tiene_aristas or indice_aristas >= umbral_aristas):
                return "tuercas"
            if ratio_agujero >= ratio_arandela_min and circularidad >= umbral_circ_alta:
                return "arandelas"

        for modelo in ("svm", "rf", "knn"):
            if self._pipelines.get(modelo) is not None:
                try:
                    return self.predecir(caracteristicas, modelo=modelo)
                except Exception:
                    continue
        return "tuercas"

    # ============================
    # Combinado ML + Reglas
    # ============================
    def predecir_combinado(
        self,
        caracteristicas: Union[Sequence[float], Dict[str, Any]],
        modelo: str = "svm",
        w_ml: float = 0.6,
        w_reglas: float = 0.4,
    ) -> str:
        """Combina predicción ML con reglas como voto ponderado."""
        X = self._vectorizar(caracteristicas)
        pred_reglas = self.clasificar_por_reglas(caracteristicas)
        proba_ml = self._predict_proba(modelo, X)

        score_regla = [0.0] * len(CLASES)
        if pred_reglas in CLASES:
            score_regla[CLASES.index(pred_reglas)] = 1.0

        if proba_ml is None:
            pred_ml = self.predecir(caracteristicas, modelo=modelo)
            score_ml = [0.0] * len(CLASES)
            if pred_ml in CLASES:
                score_ml[CLASES.index(pred_ml)] = 1.0
        else:
            score_ml = proba_ml

        total = (w_ml or 0) + (w_reglas or 0)
        if total <= 0:
            w_ml = w_reglas = 0.5
        else:
            w_ml /= total
            w_reglas /= total

        scores = [w_ml * m + w_reglas * r for m, r in zip(score_ml, score_regla)]
        idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return CLASES[idx]

    def validar_prediccion(self, caracteristicas: Union[Sequence[float], Dict[str, Any]], prediccion: str) -> bool:
        """Detecta incoherencias obvias para evitar sobrecorrecciones."""
        feats = self._caracteristicas_a_dict(caracteristicas)

        aspecto = float(feats.get("relacion_aspecto", 0) or 0)
        ratio_agujero = float(feats.get("ratio_agujero", 0) or 0)
        indice_aristas = float(feats.get("indice_aristas", 0) or 0)

        umbral_aspecto = float(getattr(config, "UMBRAL_ASPECTO_TORNILLO", 1.7) if config else 1.7)
        umbral_tuerca = float(getattr(config, "UMBRAL_ARISTAS_TUERCA", 0.5) if config else 0.5)

        # Solo bloqueamos cuando hay evidencia fuerte de otra clase.
        if prediccion == "tornillos" and ratio_agujero >= 0.65 and aspecto < (umbral_aspecto - 0.5):
            LOGGER.info("Ajuste: 'tornillos' con agujero grande y aspecto bajo")
            return False
        if prediccion == "tuercas" and indice_aristas < (umbral_tuerca * 0.6) and ratio_agujero >= 0.7:
            LOGGER.info("Ajuste: 'tuercas' sin aristas claras y agujero grande")
            return False
        return True


__all__ = ["ClasificadorPiezas", "CLASES"]
