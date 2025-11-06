import os, sys, types, numpy as np
from typing import List, Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Query
import json
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from joblib import load
import pickle
import re
import spacy
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
import sys, types
from formales import *
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Configuración fija (usa la URI que indicaste) ---
TRACKING_URI = "http://ec2-34-236-154-42.compute-1.amazonaws.com:3000"
RUN_ID = "a4fc0d31fb374ef68bc309378caeb5a6"

try:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"No se pudo crear MlflowClient: {e}")

# Obtener run
try:
    run = client.get_run(RUN_ID)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"No se pudo obtener el run '{RUN_ID}': {e}")
    
try:
    IMAGES_DIR = "./static/images"
    os.makedirs(IMAGES_DIR, exist_ok=True)

    artifact_path_model = "mlflow-artifacts:/568376531369023149/a4fc0d31fb374ef68bc309378caeb5a6/artifacts/model/lgbm_tfidf_bigram.joblib"
    artifact_path_column = "mlflow-artifacts:/568376531369023149/a4fc0d31fb374ef68bc309378caeb5a6/artifacts/vectorizer/tfidf_bigram.joblib"
    #artifact_path_selector = "mlflow-artifacts:/568376531369023149/9626f4dcb5e348f49a7d710effedb1d4/artifacts/selector.joblib"
    
    #Imagenes
    artifact_path_confusion_matrix = "mlflow-artifacts:/568376531369023149/a4fc0d31fb374ef68bc309378caeb5a6/artifacts/evaluation/confusion_matrix.png"
    artifact_path_confusion_matrix_RL = "mlflow-artifacts:/568376531369023149/19a0e0342f0a4bdaaab2ebf89b62ca92/artifacts/plots/confusion_matrix_counts.png"
    artifact_path_learning_curve_RL = "mlflow-artifacts:/568376531369023149/19a0e0342f0a4bdaaab2ebf89b62ca92/artifacts/plots/precision_recall_curve.png"
    artifact_path_confusion_matrix_K = "mlflow-artifacts:/568376531369023149/ea1dbfe391ee4b7f9f41857daf6ce009/artifacts/plots/confusion_matrix_counts_test.png"
    artifact_path_confusion_matrixPc_K = "mlflow-artifacts:/568376531369023149/ea1dbfe391ee4b7f9f41857daf6ce009/artifacts/plots/confusion_matrix_normalized_test.png"
    
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_model, dst_path="./Modelo")
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_confusion_matrix, dst_path=IMAGES_DIR)
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_confusion_matrix_RL, dst_path=IMAGES_DIR)
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_confusion_matrix_K, dst_path=IMAGES_DIR)
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_confusion_matrixPc_K, dst_path=IMAGES_DIR)
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_learning_curve_RL, dst_path=IMAGES_DIR)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_column, dst_path="./Modelo")
    #local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_selector, dst_path="./Modelo")
    #local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path_selector, dst_path="./Modelo")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"No se pudo descargar el archivo")


app = FastAPI(title="El mejor modelo del mundo")

# Estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========= Rutas HTML =========
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ========= Rutas de Imágenes =========
@app.get("/confusion-matrix")
async def get_confusion_matrix():
    return FileResponse("./static/images/confusion_matrix.png")


def _format_lgbm_params(params: dict) -> List[Dict[str, str]]:
    """
    Formatea los parámetros de LightGBM (sin prefijo o con 'lgb__') a una tabla legible.
    También muestra partes del pipeline (TF-IDF, shapes, etc.).
    """

    # --------- Mapas legibles ---------
    LGBM_MAP = {
        # Núcleo LightGBM
        "boosting_type": "Boosting",
        "objective": "Objetivo",
        "n_estimators": "Nº de Árboles (n_estimators)",
        "learning_rate": "Learning Rate",
        "num_leaves": "Nº de Hojas (num_leaves)",
        "max_depth": "Profundidad Máxima (max_depth)",
        "min_data_in_leaf": "Mín. datos por hoja (min_data_in_leaf)",
        "min_child_samples": "Mín. datos por hoja (min_child_samples)",
        "min_child_weight": "Peso mínimo por hoja (min_child_weight)",
        "min_split_gain": "Ganancia mínima por split (min_split_gain)",
        "feature_fraction": "Fracción de features (feature_fraction)",
        "colsample_bytree": "Colsample by tree",
        "subsample": "Subsample (row fraction)",
        "subsample_freq": "Frecuencia de subsample",
        "subsample_for_bin": "Muestras p/ binning (subsample_for_bin)",
        "reg_alpha": "L1 (Regularización alpha)",
        "reg_lambda": "L2 (Regularización lambda)",
        "importance_type": "Tipo de importancia",
        "class_weight": "Peso de clases",
        "verbose": "Verbose",
        "force_col_wise": "Force Col-wise",
        # Paralelismo / reproducibilidad
        "n_jobs": "No. Trabajadores",
        "threads": "Hilos",
        "random_state": "Semilla (random_state)",
        "random_seed": "Semilla (random_seed)",
        # Vectorización / TF-IDF (pipeline)
        "tfidf_lowercase": "TF-IDF lowercase",
        "tfidf_token_pattern": "TF-IDF patron de token",
        "tfidf_ngram_range": "TF-IDF N-Grama",
        "vectorizer": "Vectorizador",
        # Shapes útiles del pipeline
        "X_tr_shape": "Tamaño Train (X_tr_shape)",
        "X_val_shape": "Tamaño Validación (X_val_shape)",
        "X_test_tfidf_shape": "Tamaño Test TF-IDF (X_test_tfidf_shape)",
        "X_train_tfidf_shape": "Tamaño Train TF-IDF (X_train_tfidf_shape)",
    }

    def _to_native(v: Any) -> Optional[Any]:
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() == "none":
            return None
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # ints
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                pass
        # floats
        try:
            return float(s)
        except Exception:
            return s

    def _fmt_value(key: str, v: Any) -> str:
        if v is None:
            return "<i>(No definido)</i>"
        if isinstance(v, bool):
            return "Sí" if v else "No"
        # Porcentajes en fracciones [0,1]
        if key in ("feature_fraction", "subsample", "colsample_bytree"):
            try:
                vf = float(v)
                if 0.0 <= vf <= 1.0:
                    return f"{vf:g} ({vf*100:.1f}%)"
            except Exception:
                pass
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _get_param(keys: List[str]) -> Optional[Any]:
        """
        Busca la clave 'tal_cual' o con prefijo 'lgb__tal_cual'.
        Devuelve el primero que exista.
        """
        for k in keys:
            if k in params:
                return _to_native(params[k])
            lgbk = f"lgb__{k}"
            if lgbk in params:
                return _to_native(params[lgbk])
        return None

    formatted: List[Dict[str, str]] = []

    # Recorremos el mapa y agregamos si existe en params (con o sin prefijo lgb__)
    for raw_key, nice in LGBM_MAP.items():
        v = _get_param([raw_key])
        if v is not None:
            formatted.append({"name": nice, "value": _fmt_value(raw_key, v)})

    # Deducciones útiles:
    # - Si no viene n_estimators, pero hay best_iteration, muéstralo como proxy.
    have_estimators = any(d["name"].startswith("Nº de Árboles") for d in formatted)
    best_it = _get_param(["best_iteration"])
    if not have_estimators and best_it is not None:
        formatted.append({"name": "Mejor iteración (proxy de árboles)", "value": _fmt_value("best_iteration", best_it)})

    if not formatted:
        return [{"name": "Parámetros", "value": "No se encontraron parámetros relevantes para LightGBM."}]
    return formatted


@app.get("/model_info")
def model_info():
    """
    Devuelve JSON con:
      - params: todos los parámetros del run
      - metrics: métricas clave de LightGBM (si existen)
      - tags, artifacts
      - run metadata (start/end/status)
      - descripción de preprocesamiento
      - formatted_params: Parámetros formateados para display (LightGBM)
    """
    # ---------- Parámetros (todos) ----------
    params = dict(run.data.params or {})

    # ---------- Métricas ----------
    # Ajustado a tu run: f1, precision, best_iteration, accuracy, auc, recall_macro
    all_metrics = dict(run.data.metrics or {})
    wanted = [
        "recall_macro",
        "f1",
        "precision",
        "accuracy",
        "auc",
        "best_iteration",
    ]
    metrics = {k: all_metrics.get(k, None) for k in wanted}

    # ---------- Tags ----------
    tags = dict(run.data.tags or {})

    # ---------- Artefactos (lista superficial) ----------
    artifacts = []
    try:
        for a in client.list_artifacts(RUN_ID, path=""):
            artifacts.append({
                "path": a.path,
                "is_dir": a.is_dir,
                "file_size": getattr(a, "file_size", None)
            })
    except Exception:
        pass  # No romper si falla listar artefactos

    # ---------- Run meta ----------
    info = {
        "run_id": run.info.run_id,
        "experiment_id": getattr(run.info, "experiment_id", None),
        "status": getattr(run.info, "status", None),
        "start_time": getattr(run.info, "start_time", None),
        "end_time": getattr(run.info, "end_time", None),
    }

    # ---------- Descripción (puedes personalizarla) ----------
    preprocessing = (
        "Vectorización TF-IDF con bigramas (ngram_range=(2,2)), sin conversión a minúsculas y "
        "token_pattern='\\S+' para respetar la tokenización previa. "
        "Las formas reportadas en MLflow indican matrices dispersas muy grandes: "
        "X_train_tfidf_shape, X_val_shape y X_test_tfidf_shape. "
        "El modelo LightGBM (GBDT) usa num_leaves, feature_fraction, subsample, reg_alpha/lambda y "
        "n_estimators=1000 con learning_rate=0.1; la mejor iteración reportada es 'best_iteration'."
    )

    payload = {
        "tracking_uri": TRACKING_URI,
        "run_id": RUN_ID,
        "run_info": info,
        "params": params,
        "formatted_params": _format_lgbm_params(params),  # <-- NUEVO: LightGBM
        "metrics": metrics,
        "tags": tags,
        "artifacts": artifacts,
        "preprocessing_description": preprocessing,
    }

    return payload

# Asegura que las clases picklables estén visibles bajo __main__ (por si tu TF-IDF/Modelo las requiere)
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")

# Publica transformadores de formales.py en __main__
setattr(sys.modules["__main__"], "ElongationNormalizerV5", ElongationNormalizerV5)
setattr(sys.modules["__main__"], "BaselineCleanerV5", BaselineCleanerV5)
setattr(sys.modules["__main__"], "EmojiHandlerV5", EmojiHandlerV5)
setattr(sys.modules["__main__"], "SpaCyTokenizerV5", SpaCyTokenizerV5)

# Rutas locales (ya descargadas por tu bloque anterior)
LOCAL_MODEL = "./Modelo/lgbm_tfidf_bigram.joblib"
LOCAL_TFIDF = "./Modelo/tfidf_bigram.joblib"

# Umbral por defecto (env var opcional)
DEFAULT_THRESHOLD = float(os.getenv("BEST_THRESHOLD", "0.6"))

def _try_joblib_or_pickle(path: str):
    try:
        return load(path)  # joblib.load
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# Carga artefactos
try:
    MODEL = _try_joblib_or_pickle(LOCAL_MODEL)
    MODEL_SUPPORTS_PROBA = hasattr(MODEL, "predict_proba")
    MODEL_CLASSES = getattr(MODEL, "classes_", None)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {e}")

try:
    TFIDF = _try_joblib_or_pickle(LOCAL_TFIDF)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"No se pudo cargar el vectorizador: {e}")

# -------- Preprocesamiento para runtime (replica flags de entrenamiento) --------
# Estos flags deben reflejar tu entrenamiento (según el código que compartiste)
LEMATIZACION_ON = False
DROP_STOPWORDS = False
DROP_PUNTUACION = False
NORMALIZACION_ALARGAMIENTOS_ON = True
KEEP_EMOJIS = False  # mantenías False
# NOTA: en entrenamiento usabas TRANSLATE_EMOJI=False, así que aquí solo removemos/reemplazamos según EmojiHandlerV5

# Construimos una mini-pipeline con tus transformadores para inferencia
# (sin TF-IDF, que ya está cargado aparte)
_elon = ElongationNormalizerV5()
_emo = EmojiHandlerV5(keep_emojis=KEEP_EMOJIS)
_base = BaselineCleanerV5()
_tok  = SpaCyTokenizerV5(
    lemmatize=LEMATIZACION_ON,
    drop_stopwords=DROP_STOPWORDS,
    drop_punct=DROP_PUNTUACION,
    model="en_core_web_md"  
)

def preprocess_runtime(texts: List[str]) -> List[str]:
    # Aplica exactamente el mismo orden usado en tu entrenamiento "ligero":
    # 1) alargamientos  2) emojis   3) baseline   4) tokenización spaCy
    x = _elon.transform(texts)
    x = _emo.transform(x)
    x = _base.transform(x)
    x = _tok.transform(x)
    return x

# -------- Esquemas de IO --------
class PredictRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list, description="Textos a predecir (1..N)")

class PredictResponse(BaseModel):
    inputs: List[str]
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[Any]] = None
    threshold: float
    run_id: str

# -------- Health / Labels --------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "vectorizer_loaded": TFIDF is not None,
        "supports_proba": MODEL_SUPPORTS_PROBA,
        "default_threshold": DEFAULT_THRESHOLD,
        "run_id": RUN_ID
    }

@app.get("/labels")
def labels():
    if MODEL_CLASSES is None:
        raise HTTPException(status_code=500, detail="El modelo no expone classes_.")
    # normaliza a str para que sea estable en front
    return {"classes": [str(c) for c in MODEL_CLASSES]}

# -------- /predict (batch) --------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, threshold: Optional[float] = Query(None, ge=0.0, le=1.0)):
    if MODEL is None or TFIDF is None:
        raise HTTPException(status_code=500, detail="Modelo/TFIDF no cargado.")

    texts = req.inputs
    if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="'inputs' debe ser lista de strings.")
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="Proporciona al menos un texto.")
    if len(texts) > 1024:
        raise HTTPException(status_code=400, detail="Máximo 1024 textos por petición.")

    thr = float(threshold if threshold is not None else DEFAULT_THRESHOLD)

    # 1) Preproc runtime (igual al entrenamiento)
    texts_proc = preprocess_runtime(texts)

    # 2) Vectoriza con TF-IDF cargado
    try:
        X = TFIDF.transform(texts_proc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en TFIDF.transform: {e}")

    # 3) Probabilidades + predicción con umbral
    try:
        probs_mat = None
        if MODEL_SUPPORTS_PROBA:
            probs_mat = MODEL.predict_proba(X)
            if probs_mat.ndim == 1:
                probs_mat = probs_mat.reshape(-1, 1)
        else:
            if hasattr(MODEL, "decision_function"):
                df_scores = np.atleast_2d(MODEL.decision_function(X))
                if df_scores.shape[1] == 1:
                    p1 = 1.0 / (1.0 + np.exp(-df_scores.ravel()))
                    probs_mat = np.stack([1.0 - p1, p1], axis=1)
                else:
                    z = df_scores - df_scores.max(axis=1, keepdims=True)
                    exp_z = np.exp(z)
                    probs_mat = exp_z / exp_z.sum(axis=1, keepdims=True)
            else:
                preds = MODEL.predict(X)
                return PredictResponse(
                    inputs=texts,
                    predictions=(preds.tolist() if hasattr(preds, "tolist") else list(preds)),
                    probabilities=None,
                    classes=[str(c) for c in MODEL_CLASSES] if MODEL_CLASSES is not None else None,
                    threshold=thr,
                    run_id=RUN_ID
                )

        # índice de la clase positiva = "1" si está en classes_
        if MODEL_CLASSES is not None and (1 in MODEL_CLASSES or "1" in MODEL_CLASSES):
            # soporta que classes_ sean ints o strings
            if 1 in MODEL_CLASSES:
                pos_idx = int(np.where(MODEL_CLASSES == 1)[0][0])
            else:
                pos_idx = int(np.where(MODEL_CLASSES == "1")[0][0])
        else:
            pos_idx = 1 if probs_mat.shape[1] > 1 else 0

        p1 = probs_mat[:, pos_idx]
        preds_list = (p1 >= thr).astype(int).tolist()

        probabilities = probs_mat.astype(float).tolist()
        classes_out = [str(c) for c in MODEL_CLASSES] if MODEL_CLASSES is not None else (["0", "1"] if probs_mat.shape[1] == 2 else None)

        return PredictResponse(
            inputs=texts,
            predictions=preds_list,
            probabilities=probabilities,
            classes=classes_out,
            threshold=thr,
            run_id=RUN_ID
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predict: {e}")

# -------- /predict_text (para un input de formulario) --------
@app.get("/predict_text")
def predict_text(q: str = Query(..., min_length=1), threshold: Optional[float] = Query(None, ge=0.0, le=1.0)):
    resp = predict(PredictRequest(inputs=[q]), threshold=threshold)
    # devuelve compacto para UI
    proba_pos = None
    if resp.probabilities and resp.classes and "1" in resp.classes:
        idx1 = resp.classes.index("1")
        proba_pos = resp.probabilities[0][idx1]
    return {
        "text": q,
        "prediction": int(resp.predictions[0]),
        "proba_pos": proba_pos,
        "threshold": resp.threshold,
        "classes": resp.classes,
        "run_id": resp.run_id
    }

EXPERIMENT_ID_BASELINE = "344603012050082888"

LABEL_MAPPING = {
    "Lema": "Lematización",
    "Stop": "Stopwords",
    "Punt": "Puntuación",
    "Alarg": "Alargamientos",
    "TranslateEmoji": "Traducir Emojis",
    "KeepEmoji": "Mantener Emojis"
}

def _format_run_label(long_label: str) -> str:
    if '_' not in long_label:
        return long_label
    try:
        parts = long_label.split('_')
        active = []
        for i in range(0, len(parts), 2):
            k = parts[i]
            v = parts[i+1]
            if v.lower() == 'true' and k in LABEL_MAPPING:
                active.append(LABEL_MAPPING[k])
        return "Modelo Base (Sin Preprocesamiento)" if not active else " + ".join(active)
    except Exception:
        return long_label

def _normalize_ngram(val: str) -> str:
    """
    Normaliza '(1, 1)' -> '1,1', '(2,2)' -> '2,2'. Si no matchea, devuelve ''.
    """
    if val is None:
        return ""
    s = str(val).strip()
    s = s.replace(" ", "")  # '(1, 1)' -> '(1,1)'
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]  # '1,1'
    return s if s in ("1,1", "2,2") else ""

def _process_experiment_by_ngram(experiment_id: str) -> dict:
    """
    Devuelve dos buckets:
      - unigram: labels/data para params.tfidf_ngram_range == (1,1)
      - bigram:  labels/data para params.tfidf_ngram_range == (2,2)
    """
    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

    # Columnas esperadas
    col_name = "tags.mlflow.runName"
    col_recall = "metrics.recall_macro"
    col_ngram = "params.tfidf_ngram_range"

    if runs_df.empty or col_recall not in runs_df.columns or col_ngram not in runs_df.columns:
        empty = {"labels": [], "data": []}
        return {"unigram": empty, "bigram": empty}

    df = runs_df[[col_name, col_recall, col_ngram]].copy()
    df.rename(columns={col_name: "label", col_recall: "value", col_ngram: "ngram"}, inplace=True)

    # Normaliza ngram y filtra los que no sean 1,1 o 2,2
    df["bucket"] = df["ngram"].apply(_normalize_ngram)
    df = df[df["bucket"].isin(["1,1", "2,2"])]
    if df.empty:
        empty = {"labels": [], "data": []}
        return {"unigram": empty, "bigram": empty}

    # Formatea etiquetas y limpia
    df["label"] = df["label"].apply(_format_run_label)
    df = df.dropna(subset=["value"])

    # Unigrama
    df_uni = df[df["bucket"] == "1,1"].copy()
    df_uni.sort_values(by="value", ascending=False, inplace=True)
    unigram = {
        "labels": df_uni["label"].tolist(),
        "data": df_uni["value"].tolist()
    }

    # Bigrama
    df_bi = df[df["bucket"] == "2,2"].copy()
    df_bi.sort_values(by="value", ascending=False, inplace=True)
    bigram = {
        "labels": df_bi["label"].tolist(),
        "data": df_bi["value"].tolist()
    }

    return {"unigram": unigram, "bigram": bigram}

@app.get("/ablation_summary")
def get_ablation_summary():
    """
    Devuelve los datos separados por codificación:
      - baseline_unigram: tfidf_ngram_range == (1,1)
      - baseline_bigram:  tfidf_ngram_range == (2,2)
    Mantiene 'codificacion' vacío para compatibilidad con el frontend previo.
    """
    try:
        grouped = _process_experiment_by_ngram(EXPERIMENT_ID_BASELINE)
        payload = {
            "baseline_unigram": grouped.get("unigram", {"labels": [], "data": []}),
            "baseline_bigram": grouped.get("bigram", {"labels": [], "data": []}),
            "codificacion": {"labels": [], "data": []}
        }
        return JSONResponse(content=payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos: {e}")
#-------------------------------Ablacion

COMPARISON_JSON = "./summary.json"

@app.get("/comparison")
def comparison():
    if not os.path.exists(COMPARISON_JSON):
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el archivo: {COMPARISON_JSON}"
        )

    try:
        with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo leer JSON: {e}")

    text_top    = data.get("text_top") or ""
    text_bottom = data.get("text_bottom") or ""
    table       = data.get("table") or {}
    headers     = table.get("headers") or []
    rows        = table.get("rows") or []

    return JSONResponse(content={
        "text_top": text_top,
        "table": {"headers": headers, "rows": rows},
        "text_bottom": text_bottom
    })