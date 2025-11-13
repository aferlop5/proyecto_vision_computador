"""
Clasificación de piezas mediante SVM, KNN y Random Forest con normalización (StandardScaler),
además de una ruta de respaldo por reglas heurísticas basadas en config.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Importaciones robustas de dependencias internas
try:
    from . import config as config  # type: ignore
except Exception:
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

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# Clases esperadas en el dataset
CLASES: List[str] = ["arandelas", "tornillos", "tuercas"]


class ClasificadorPiezas:
    """
    Entrena y evalúa SVM, KNN y RandomForest con un StandardScaler.
    Ofrece clasificación por reglas y una combinación ML+reglas.
    """

    def __init__(self) -> None:
        try:
            from sklearn.svm import SVC  # type: ignore
            from sklearn.neighbors import KNeighborsClassifier  # type: ignore
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except Exception as e:
            LOGGER.error("scikit-learn no disponible: %s", e)
            # Demora la importación real hasta entrenar
            SVC = KNeighborsClassifier = RandomForestClassifier = Pipeline = StandardScaler = None  # type: ignore

        # Parámetros desde config
        svm_params: Dict[str, Any] = dict(getattr(config, "SVM_PARAMS", {}) if config else {})
        # Asegurar probabilidad para combinaciones
        svm_params.setdefault("probability", True)
        knn_params: Dict[str, Any] = dict(getattr(config, "KNN_PARAMS", {}) if config else {})
        rf_params: Dict[str, Any] = dict(getattr(config, "RF_PARAMS", {"n_estimators": 100, "random_state": 42}) if config else {"n_estimators": 100, "random_state": 42})

        # Construcción diferida de pipelines al entrenar
        self._pipelines: Dict[str, Any] = {
            "svm": None,
            "knn": None,
            "rf": None,
        }
        self._params: Dict[str, Dict[str, Any]] = {
            "svm": svm_params,
            "knn": knn_params,
            "rf": rf_params,
        }

    # ============================
    # Utilidades internas
    # ============================
    def _vectorizar(self, caracteristicas: Union[Sequence[float], Dict[str, Any]]) -> List[List[float]]:
        """
        Convierte un dict de características a un vector ordenado según config.CARACTERISTICAS_PRINCIPALES.
        Si recibe una secuencia numérica, la envuelve como [[...]].
        """
        import numpy as np  # type: ignore

        if isinstance(caracteristicas, dict):
            orden: List[str] = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
            vec: List[float] = []
            for k in orden:
                v = caracteristicas.get(k, 0)
                if isinstance(v, bool):
                    v = 1.0 if v else 0.0
                try:
                    vec.append(float(v))
                except Exception:
                    vec.append(0.0)
            return [vec]
        # Si ya es vector/array
        arr = np.asarray(list(caracteristicas), dtype=float).reshape(1, -1)
        return arr.tolist()

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
            params = dict(self._params["svm"])  # type: ignore
            # Asegurar class_weight balanced si se pidió en config
            if "class_weight" not in params:
                try:
                    cw = getattr(config, "SVM_PARAMS", {}).get("class_weight") if config else None  # type: ignore
                except Exception:
                    cw = None
                if cw is None:
                    params["class_weight"] = "balanced"
            clf = SVC(**params)  # type: ignore
        elif tipo == "knn":
            from sklearn.neighbors import KNeighborsClassifier  # type: ignore
            clf = KNeighborsClassifier(**self._params["knn"])  # type: ignore
        elif tipo == "rf":
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            params = dict(self._params["rf"])  # type: ignore
            if "class_weight" not in params:
                params["class_weight"] = "balanced"
            clf = RandomForestClassifier(**params)  # type: ignore
        else:
            raise ValueError(f"Tipo de modelo desconocido: {tipo}")
        steps = [("scaler", StandardScaler())]
        # PCA opcional para reducir dimensionalidad de HOG y ruido
        try:
            use_pca = bool(getattr(config, "USE_PCA", False) if config else False)
            if use_pca and PCA is not None:
                n_comp = int(getattr(config, "PCA_COMPONENTS", 100) if config else 100)
                whiten = bool(getattr(config, "PCA_WHITEN", False) if config else False)
                steps.append(("pca", PCA(n_components=n_comp, whiten=whiten)))
        except Exception:
            pass
        steps.append(("clf", clf))
        pipe = Pipeline(steps)
        self._pipelines[tipo] = pipe
        return pipe

    def _predict_proba(self, tipo: str, X: List[List[float]]) -> Optional[List[float]]:
        try:
            import numpy as np  # type: ignore
            pipe = self._pipelines.get(tipo)
            if pipe is None:
                return None
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X)
            else:
                # Intentar acceder al estimador final del pipeline
                clf = getattr(pipe, "named_steps", {}).get("clf", None)
                if clf is not None and hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(pipe.named_steps["scaler"].transform(X))
                else:
                    return None
            if proba is None:
                return None
            # Alinear a orden CLASES si es posible
            if hasattr(pipe, "classes_"):
                classes = list(pipe.classes_)
            else:
                classes = []
            vec = [0.0] * len(CLASES)
            for i, p in enumerate(proba[0].tolist()):
                c = classes[i] if i < len(classes) else None
                if c in CLASES:
                    vec[CLASES.index(c)] = float(p)
            s = sum(vec) or 1.0
            return [v / s for v in vec]
        except Exception as e:
            LOGGER.info("predict_proba no disponible (%s): %s", tipo, e)
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

    def entrenar_con_gridsearch(self, X_train: Sequence[Sequence[float]], y_train: Sequence[str], tipo: str = "svm") -> Dict[str, Any]:
        """
        Ejecuta GridSearchCV sobre el pipeline correspondiente con las parrillas definidas en config.
        Devuelve dict con 'best_params_' y 'best_score_' y deja el mejor estimador en self._pipelines[tipo].
        """
        try:
            from sklearn.model_selection import GridSearchCV  # type: ignore
        except Exception as e:
            LOGGER.error("GridSearchCV no disponible: %s", e)
            # fallback: entreno normal
            train_map = {"svm": self.entrenar_svm, "knn": self.entrenar_knn, "rf": self.entrenar_random_forest}
            train_map.get(tipo, self.entrenar_svm)(X_train, y_train)
            return {}

        pipe = self._get_pipeline(tipo)
        # Seleccionar grid
        grid = {}
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
            LOGGER.warning("Parrilla de búsqueda vacía para '%s'. Entrenando sin grid.", tipo)
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
            y = pipe.predict(X)
            return str(y[0])
        except Exception as e:
            LOGGER.error("Error en predicción con '%s': %s", modelo, e)
            return self.clasificar_por_reglas(caracteristicas)

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
        """
        Lógica heurística:
        - Si aspecto > UMBRAL_ASPECTO_TORNILLO → "tornillos"
        - Si tiene_aristas y no tiene_agujero → "tuercas"
        - Si tiene_agujero y circularidad alta → "arandelas"
        - Si nada aplica → predicción ML si disponible, o "tuercas" por defecto.
        """
        # Obtener características como dict si es posible
        feats: Dict[str, Any] = {}
        if isinstance(caracteristicas, dict):
            feats = caracteristicas
        else:
            # mapear si hay orden conocido
            orden = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
            try:
                feats = {k: v for k, v in zip(orden, caracteristicas)}
            except Exception:
                feats = {}

        aspecto = float(feats.get("relacion_aspecto", 0) or 0)
        tiene_aristas = bool(feats.get("tiene_aristas", False))
        tiene_agujero = bool(feats.get("tiene_agujero", False))
        circularidad = float(feats.get("circularidad", 0) or 0)

        umbral_aspecto = float(getattr(config, "UMBRAL_ASPECTO_TORNILLO", 1.7) if config else 1.7)
        umbral_circ_alta = float(getattr(config, "UMBRAL_CIRCULARIDAD_ALTA", 0.75) if config else 0.75)

        if aspecto > umbral_aspecto:
            return "tornillos"
        if tiene_aristas and not tiene_agujero:
            return "tuercas"
        if tiene_agujero and circularidad >= umbral_circ_alta:
            return "arandelas"

        # fallback: usar modelo entrenado si existe
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
    def predecir_combinado(self, caracteristicas: Union[Sequence[float], Dict[str, Any]], modelo: str = "svm", w_ml: float = 0.6, w_reglas: float = 0.4) -> str:
        """
        Combina predicción de ML con la de reglas (pesos w_ml y w_reglas suman 1 aprox.).
        Si no hay proba de ML, se usa predicción dura del modelo.
        """
        X = self._vectorizar(caracteristicas)
        regla = self.clasificar_por_reglas(caracteristicas)
        proba_ml = self._predict_proba(modelo, X)

        import numpy as np  # type: ignore
        score_regla = [0.0] * len(CLASES)
        if regla in CLASES:
            score_regla[CLASES.index(regla)] = 1.0

        if proba_ml is None:
            # Sin probas, combinar como voto ponderado duro
            pred_ml = self.predecir(caracteristicas, modelo=modelo)
            score_ml = [0.0] * len(CLASES)
            if pred_ml in CLASES:
                score_ml[CLASES.index(pred_ml)] = 1.0
        else:
            score_ml = proba_ml

        # Normalizar pesos
        s = (w_ml or 0) + (w_reglas or 0)
        if s <= 0:
            w_ml, w_reglas = 0.5, 0.5
        else:
            w_ml, w_reglas = w_ml / s, w_reglas / s

        scores = [w_ml * m + w_reglas * r for m, r in zip(score_ml, score_regla)]
        idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return CLASES[idx]

    def validar_prediccion(self, caracteristicas: Union[Sequence[float], Dict[str, Any]], prediccion: str) -> bool:
        """
        Verifica coherencia entre predicción y características básicas.
        Devuelve True si es coherente, False en caso contrario.
        """
        if isinstance(caracteristicas, dict):
            feats = caracteristicas
        else:
            orden = list(getattr(config, "CARACTERISTICAS_PRINCIPALES", []) if config else [])
            feats = {k: v for k, v in zip(orden, caracteristicas)}

        aspecto = float(feats.get("relacion_aspecto", 0) or 0)
        tiene_aristas = bool(feats.get("tiene_aristas", False))
        tiene_agujero = bool(feats.get("tiene_agujero", False))
        circularidad = float(feats.get("circularidad", 0) or 0)

        umbral_aspecto = float(getattr(config, "UMBRAL_ASPECTO_TORNILLO", 1.7) if config else 1.7)
        umbral_circ_alta = float(getattr(config, "UMBRAL_CIRCULARIDAD_ALTA", 0.75) if config else 0.75)

        coherente = True
        if prediccion == "tornillos" and not (aspecto > umbral_aspecto):
            LOGGER.info("Incoherencia: 'tornillos' con aspecto=%.2f <= %.2f", aspecto, umbral_aspecto)
            coherente = False
        if prediccion == "tuercas" and not (tiene_aristas and not tiene_agujero):
            LOGGER.info("Incoherencia: 'tuercas' con tiene_aristas=%s, tiene_agujero=%s", tiene_aristas, tiene_agujero)
            coherente = False
        if prediccion == "arandelas" and not (tiene_agujero and circularidad >= umbral_circ_alta):
            LOGGER.info("Incoherencia: 'arandelas' con tiene_agujero=%s, circularidad=%.2f < %.2f", tiene_agujero, circularidad, umbral_circ_alta)
            coherente = False
        return coherente
