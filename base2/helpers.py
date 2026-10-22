"""
BASE 2 - CLASIFICACION (Hotel Booking) + K-MEANS
Modulo compartido: constantes, transformadores personalizados, metricas y
preprocesadores. Se importa desde cada script numerado (03_..., 04_..., etc.)
para no duplicar codigo, y su contenido se inyecta tal cual en la celda de
"configuracion" del notebook final.

Disciplina metodologica de todo el proyecto (heredada y verificada como
correcta en el ejercicio de regresion):
- Hold-Out 70/30 ANTES de imputar/escalar/agrupar categorias.
- Todo lo que se "aprende" de los datos (medianas, categorias frecuentes,
  percentiles de recorte) vive DENTRO de un Pipeline/ColumnTransformer para
  que Cross-Validation lo recalcule de forma independiente en cada fold.
- TEST se toca una sola vez, al final.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    make_scorer,
)

RANDOM_STATE = 42
N_SPLITS = 5
PRIMARY_METRIC = "f1"

RESULTS_DIR = Path("resultados_clasificacion")
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Transformadores personalizados (fit SOLO con lo que reciban -> a
# prueba de fugas cuando viven dentro de un Pipeline)
# ------------------------------------------------------------------
class TopCategoryBucketer(BaseEstimator, TransformerMixin):
    """Agrupa una columna categorica de alta cardinalidad: conserva las
    `top_n` categorias mas frecuentes VISTAS EN FIT y colapsa el resto en
    'Other'. Los nulos se mapean a 'Desconocido'. Pensado para 'country'
    (177 categorias en el dataset completo, 157 de ellas con <0.5% de
    las filas cada una)."""

    def __init__(self, top_n=15):
        self.top_n = top_n

    def fit(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.ravel(X))
        counts = col.value_counts(dropna=True)
        self.top_categories_ = set(counts.head(self.top_n).index)
        return self

    def transform(self, X):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.ravel(X))
        out = col.where(col.isin(self.top_categories_), other="Other")
        out = out.where(~col.isna(), other="Desconocido")
        out = out.rename("country_agrupado")
        return out.to_frame()

    def get_feature_names_out(self, input_features=None):
        return np.array(["country_agrupado"])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta (winsoriza) valores por encima del percentil `upper_q`
    APRENDIDO EN FIT. Se usa solo sobre 'adr', que tiene un valor extremo
    (5400, ~53x la mediana) que de otro modo domina la varianza que ve
    StandardScaler para KNN."""

    def __init__(self, upper_q=0.995):
        self.upper_q = upper_q

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.caps_ = np.quantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        return np.minimum(arr, self.caps_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def make_ohe():
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessors(numeric_cols, categorical_cols, scale_numeric):
    """Construye el ColumnTransformer. `scale_numeric=False` para
    Arbol/Random Forest (no lo necesitan); `True` para KNN (distancias)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    other_cat = [c for c in categorical_cols if c != "country"]

    num_steps = [("winsor_adr", Winsorizer(upper_q=0.995))] if "adr" in numeric_cols else []
    # Nota: Winsorizer solo tiene sentido aplicado a 'adr'; para el resto de
    # columnas numericas actua como no-operacion salvo que tengan outliers
    # extremos, por lo que se aisla en una rama propia mas abajo.

    branches = []

    numeric_sin_adr = [c for c in numeric_cols if c != "adr"]
    if numeric_sin_adr:
        num_pipeline_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_pipeline_steps.append(("scaler", StandardScaler()))
        branches.append(("num", Pipeline(num_pipeline_steps), numeric_sin_adr))

    if "adr" in numeric_cols:
        adr_steps = [("winsor", Winsorizer(upper_q=0.995)),
                     ("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            adr_steps.append(("scaler", StandardScaler()))
        branches.append(("adr", Pipeline(adr_steps), ["adr"]))

    if "country" in categorical_cols:
        branches.append(("country", Pipeline([
            ("bucket", TopCategoryBucketer(top_n=15)),
            ("ohe", make_ohe()),
        ]), ["country"]))

    if other_cat:
        branches.append(("cat", make_ohe(), other_cat))

    return ColumnTransformer(branches)


# ------------------------------------------------------------------
# Metricas de clasificacion
# ------------------------------------------------------------------
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


specificity_scorer = make_scorer(specificity_score)

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "specificity": specificity_scorer,
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def classification_metrics(y_true, y_pred, y_prob=None):
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        result["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        result["PR_AUC"] = average_precision_score(y_true, y_prob)
    return result


def build_smotenc_pipeline(numeric_cols, categorical_cols, model, scale_numeric, sampling_strategy=1.0):
    """Pipeline con SMOTENC (imbalanced-learn) como estrategia adicional de
    balanceo, agregada a pedido explicito del estudiante tras revisar el
    ejercicio de un companero. SMOTENC exige que las variables categoricas
    ya esten codificadas numericamente (no acepta strings ni NaN), asi que
    el orden es:

      1) pre_smote: igual disciplina que en `build_preprocessors` (bucket
         top-N + winsorizacion de 'adr'), pero terminando en
         OrdinalEncoder para las categoricas en vez de One-Hot;
      2) SMOTENC: genera observaciones sinteticas de la clase minoritaria
         SOLO dentro de cada fold de entrenamiento (va dentro de un
         imblearn.Pipeline, que solo re-muestrea en .fit(), nunca en
         .transform()/.predict(), por lo que TEST y el fold de validacion
         de cada iteracion de CV nunca se tocan);
      3) post_smote: One-Hot (+ StandardScaler si scale_numeric=True) sobre
         las columnas ya sinteticas, para que el modelo final reciba el
         mismo tipo de entrada que en `build_preprocessors`.

    Devuelve un imblearn.pipeline.Pipeline completo: preprocesador +
    SMOTENC + modelo, listo para usar en GridSearchCV/cross_validate.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC

    other_cat = [c for c in categorical_cols if c != "country"]
    numeric_sin_adr = [c for c in numeric_cols if c != "adr"]
    has_adr = "adr" in numeric_cols
    has_country = "country" in categorical_cols

    def make_ordinal():
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    pre_branches = []
    if numeric_sin_adr:
        pre_branches.append(("num", SimpleImputer(strategy="median"), numeric_sin_adr))
    if has_adr:
        pre_branches.append(("adr", SkPipeline([
            ("winsor", Winsorizer(upper_q=0.995)),
            ("imputer", SimpleImputer(strategy="median")),
        ]), ["adr"]))
    if has_country:
        pre_branches.append(("country", SkPipeline([
            ("bucket", TopCategoryBucketer(top_n=15)),
            ("ordinal", make_ordinal()),
        ]), ["country"]))
    if other_cat:
        pre_branches.append(("cat", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", make_ordinal()),
        ]), other_cat))

    pre_smote = ColumnTransformer(pre_branches, verbose_feature_names_out=False)

    n_num = len(numeric_sin_adr) + (1 if has_adr else 0)
    n_cat = (1 if has_country else 0) + len(other_cat)
    numeric_indices = list(range(n_num))
    categorical_indices = list(range(n_num, n_num + n_cat))

    post_num_step = StandardScaler() if scale_numeric else "passthrough"
    post_smote = ColumnTransformer([
        ("num", post_num_step, numeric_indices),
        ("cat", make_ohe(), categorical_indices),
    ], verbose_feature_names_out=False)

    smote = SMOTENC(categorical_features=categorical_indices,
                     sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)

    return ImbPipeline([
        ("pre_smote", pre_smote),
        ("smote", smote),
        ("post_smote", post_smote),
        ("model", model),
    ])


def cv_summary(cv_result, model_name):
    row = {"Modelo": model_name}
    for metric in SCORING:
        values = cv_result[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()
    row["fit_time_mean_s"] = cv_result["fit_time"].mean()
    row["score_time_mean_s"] = cv_result["score_time"].mean()
    return row
