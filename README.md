# AutoResearch-X

> 一個給統計學家 & 資料科學家用的自動化 ML / DL / TS 框架  
> 給我資料路徑，我幫你搞定剩下的事。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![sklearn](https://img.shields.io/badge/sklearn-1.3%2B-orange)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.6%2B-blueviolet)](https://optuna.org/)

---

## 目錄

- [核心理念](#核心理念)
- [三大支柱](#三大支柱)
- [系統架構](#系統架構)
- [安裝](#安裝)
- [快速開始](#快速開始)
  - [監督學習 / 半監督學習](#監督學習--半監督學習)
  - [非監督分群](#非監督分群)
  - [時間序列預測](#時間序列預測)
  - [CLI 介面](#cli-介面)
- [CAKE Score](#cake-score)
- [超參數搜索](#超參數搜索)
- [實驗記錄](#實驗記錄)
- [輸出格式](#輸出格式)
- [套件依賴](#套件依賴)
- [目錄結構](#目錄結構)

---

## 核心理念

大部分 AutoML 框架要嘛太黑箱（你不知道它做了什麼），要嘛太複雜（光安裝就讓人崩潰）。  
AutoResearch-X 的設計原則：

- **自動但透明**：自動選模型、自動調參、自動 Stacking，但每一步都可以看到 leaderboard
- **不鎖套件**：ML 用 sklearn / XGBoost / LightGBM / CatBoost，DL 用 HuggingFace，TS 用 AutoGluon + StatsForecast。全部免費，不需要 API key
- **CAKE Score 驅動分群**：非監督學習用 CAKE（Confidence in Assignments via K-partition Ensembles）來選最佳分群參數，不再憑 inertia 猜
- **三個 Pillar 獨立但共享基礎**：ML / DL / TS 各自一套 pipeline，但共享 DataProfiler、ExperimentDB、CAKE

---

## 三大支柱

| Pillar | 任務 | 核心套件 |
|--------|------|----------|
| **ML** | 分類、回歸、半監督 | sklearn, XGBoost, LightGBM, CatBoost, Optuna |
| **DL** | 文字分類、回歸、NER | HuggingFace Transformers, PyTorch |
| **TS** | 時序預測、異常偵測 | AutoGluon-TimeSeries, StatsForecast, Darts |

---

## 系統架構

```
輸入資料
    │
    ▼
DataProfiler ──→ 決定 Pillar（ML / DL / TS）
    │              決定 Task（classification / regression / clustering / forecast）
    │              決定 Regime（large / medium / small × high / medium / low dim）
    │
    ├─── ML Pillar ──────────────────────────────────────────────────────
    │      AutoPreprocessor（KNNImputer + RobustScaler + OHE + 特徵選擇）
    │      ModelPool（依 regime 動態組合模型候選池）
    │      CVStrategy（StratifiedKFold / KFold，自動 scoring）
    │      ClusterAwareOptimizer（TPE / GP-BO，Optuna）
    │      EnsembleBuilder（Stacking 或 Blending）
    │      SemiSupervisedEngine（LabelSpreading / SelfTraining）
    │
    ├─── DL Pillar ──────────────────────────────────────────────────────
    │      DLBackend（HuggingFace Trainer API）
    │      AutoTokenizer + AutoModel（依任務自動選）
    │      EarlyStoppingCallback + FP16（GPU 自動啟用）
    │
    ├─── TS Pillar ──────────────────────────────────────────────────────
    │      TSBackend（三層路由）
    │        大資料 / 多序列 → AutoGluon-TimeSeries（medium_quality preset）
    │        小資料 / 單序列 → StatsForecast（ETS + ARIMA + Theta）
    │        無套件          → Seasonal Naive fallback
    │      Darts（異常偵測，KMeansScorer + QuantileDetector）
    │
    └─── Cluster Pillar ─────────────────────────────────────────────────
           ClusteringEngine（StandardScaler → CAKEParamSelector → fit）
           CAKE Score（√(Stability × Geometric_Fit)）
           支援：HDBSCAN / DBSCAN / KMeans / GMM / MeanShift
    │
    ▼
ExperimentDB（SQLite 自動記錄每次實驗）
    │
    ▼
MLResult / ClusterResult / ForecastResult
（predict / save / load / summary）
```

---

## 安裝

### 基礎（ML only）

```bash
git clone https://github.com/your-username/autoresearch-x.git
cd autoresearch-x
pip install -e .
```

### 加 DL 支援（HuggingFace + PyTorch）

```bash
pip install -e ".[dl]"
```

### 加 TS 支援（AutoGluon + Darts）

```bash
pip install -e ".[ts]"
```

### 全裝

```bash
pip install -e ".[all]"
```

> **需要 Python 3.10+**  
> GPU 可選，有 CUDA 的話 DL 訓練自動啟用 FP16

---

## 快速開始

### 監督學習 / 半監督學習

```python
from autoresearch_x import analyze

# 最簡單的用法：給 CSV 路徑 + 目標欄位
result = analyze("titanic.csv", target="Survived")

# 看摘要
print(result.summary())

# 查 leaderboard
print(result.leaderboard)

# 對新資料預測（自動套用相同前處理）
import pandas as pd
new_df = pd.read_csv("test.csv")
predictions = result.predict(new_df)

# 儲存模型（含前處理器）
result.save("./output/titanic")

# 之後重新載入，不需要重新訓練
result2 = MLResult.load("./output/titanic")
```

**半監督學習**（部分資料沒有標籤，用 `-1` 表示）：

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
# 80% 資料沒有標籤
df.loc[df.sample(frac=0.8).index, "label"] = np.nan
df["label"] = df["label"].fillna(-1)

result = analyze(df, target="label")
# DataProfiler 偵測到 label_ratio < 0.5 → 自動啟用 SemiSupervisedEngine
```

**進階選項**：

```python
result = analyze(
    "data.csv",
    target="price",
    strategy="gp_bo",    # 或 "tpe" / "random" / "auto"
    n_trials=100,        # Optuna trials 數（越多越準但越慢）
    ensemble=True,       # 建立 Stacking Ensemble
    output_dir="./out",  # 自動儲存
)
```

---

### 非監督分群

```python
from autoresearch_x import cluster

# 自動選方法（依資料量 / 維度決定）
result = cluster("customers.csv")

# 或指定方法
result = cluster("customers.csv", method="hdbscan")

# 查看結果
print(result.summary())
# → 顯示 cluster 數量、CAKE Score、各 cluster 大小

# 取得 label
labels = result.labels          # numpy array
cake_scores = result.cake_scores  # 每個樣本的信心分數 [0, 1]
suspicious = result.suspicious_idx  # CAKE < threshold 的可疑點

# 儲存（含 cluster_assignments.csv + cluster_metadata.json）
result.save("./cluster_output")
```

**支援的分群方法**：

| 方法 | 適用情境 | 參數選擇方式 |
|------|---------|-------------|
| `hdbscan` | 大資料、高維、不規則形狀 | CAKE max over (min_cluster_size, min_samples) |
| `dbscan` | 中小資料、噪聲點多 | k-distance graph + KneeLocator → CAKE refine |
| `kmeans` | 球形分布、已知大概群數 | Elbow + KneeLocator → CAKE 選 k |
| `gmm` | 橢球形、需要軟分群 | BIC + CAKE joint selection |
| `meanshift` | 不知道群數、小資料 | estimate_bandwidth → CAKE linear search |

---

### 時間序列預測

```python
from autoresearch_x import forecast

# 單序列預測
result = forecast(
    "sales.csv",
    target="revenue",
    horizon=30,          # 預測未來 30 步
    freq="D",            # 日頻率（H/D/W/M/Q/Y）
)

print(result.summary())
# → 顯示使用的後端、最佳模型、預測誤差

# 取得預測值（pandas Series，index 為未來日期）
forecast_values = result.forecast
print(forecast_values.head())

# 多序列預測
result = forecast(
    "multi_sales.csv",
    target="revenue",
    horizon=14,
    freq="D",
    item_id_col="store_id",   # 每個 store 獨立預測
)

# 加入異常偵測（需要 darts）
result = forecast(
    "sensor.csv",
    target="temperature",
    horizon=24,
    freq="H",
    detect_anomaly=True,
)
anomaly_positions = result.metadata["anomaly_indices"]
```

**後端路由邏輯**：

```
樣本數 > 500 或多序列 → AutoGluon-TimeSeries（medium_quality）
其他                  → StatsForecast（AutoETS + AutoARIMA + AutoTheta）
兩者都沒裝            → Seasonal Naive fallback
```

---

### CLI 介面

```bash
# 監督學習
python cli.py analyze data.csv --target label --strategy tpe --n-trials 50 --output ./out

# 分群
python cli.py cluster data.csv --method hdbscan --output ./out

# 時序預測
python cli.py forecast sales.csv --target revenue --horizon 30 --freq D

# 查看歷史實驗
python cli.py history --pillar ml --limit 10

# 詳細輸出（含 leaderboard）
python cli.py analyze data.csv --target label -v
```

安裝後可直接用 `arx` 指令：

```bash
arx analyze data.csv --target label
arx cluster data.csv --method kmeans
arx forecast data.csv --target value --horizon 7 --freq D
arx history
```

---

## CAKE Score

CAKE（Confidence in Assignments via K-partition Ensembles）是本框架用來評估分群品質的核心指標。

$$\text{CAKE} = \sqrt{\text{Stability} \times \text{Geometric\_Fit}}$$

| 子指標 | 計算方式 | 意義 |
|--------|----------|------|
| **Stability** | Bootstrap × 30 次，Hungarian 對齊後計算 label agreement | 分群結果的穩定性（0~1）|
| **Geometric_Fit** | Silhouette Score 正規化到 [0, 1] | 幾何結構的吻合度（0~1）|
| **CAKE** | 兩者的幾何平均 | 綜合信心分數（0~1，越高越好）|

**為什麼要用 CAKE 而不是 Silhouette？**

Silhouette 只看幾何結構，忽略了分群結果在不同子樣本下的穩定性。CAKE 同時考量兩者——一個好的分群不只要幾何漂亮，更要在資料有擾動時保持一致。

**每個樣本都有信心分數**：

```python
result = cluster("data.csv")

# 全局 CAKE
print(f"CAKE Score: {result.cake_score:.4f}")

# 每個樣本的信心
print(result.cake_scores[:10])

# 找出低信心樣本（邊界點、可能被錯誤分配）
suspicious = result.suspicious_idx
print(f"低信心樣本數：{len(suspicious)}")
```

---

## 超參數搜索

使用 `ClusterAwareOptimizer`（基於 Optuna）進行三層搜索：

1. **HDBSCAN**：將已跑過的 config 在超參數空間中分群
2. **Global GP-BO**：決定去探索哪個 cluster（exploration）
3. **Local TPE**：在該 cluster 內精細搜索（exploitation）

```python
from autoresearch_x.search.optimizer import ClusterAwareOptimizer, PARAM_SPACES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 自定義 objective
def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return scores.mean()

optimizer = ClusterAwareOptimizer(n_trials=100, seed=42)
best_params, best_score = optimizer.optimize(
    RandomForestClassifier,
    PARAM_SPACES["RandomForestClassifier"],
    objective,
    strategy="tpe",       # "tpe" / "gp_bo" / "random"
)
print(best_params, best_score)
```

**策略選擇建議**：

| 情境 | 推薦策略 |
|------|----------|
| 小資料 + 低維度 | `gp_bo`（GP-BO 在少 trial 下收斂快）|
| 大資料 + 多模型 | `tpe`（TPE 適合高維超參數空間）|
| 快速驗證 | `random`（隨機基準線）|
| 不知道選哪個 | `auto`（DataProfiler 自動決定）|

---

## 實驗記錄

每次 `analyze / cluster / forecast` 都會自動寫入 SQLite：

```python
from autoresearch_x.memory.experiment_db import ExperimentDB

db = ExperimentDB()   # 預設在 ~/.autoresearch_x/experiments.db

# 查詢歷史
rows = db.query(pillar="ml", task="classification", limit=20)
for r in rows:
    print(r["model_name"], r["metrics"])

# RAG：針對同一資料集找最佳歷史模型
import hashlib
dataset_hash = ExperimentDB.hash_dataframe(df)
best = db.best_for_dataset(dataset_hash, task="classification")
if best:
    print(f"歷史最佳：{best['model_name']}，score={best['metrics']}")
```

---

## 輸出格式

### MLResult

```python
result.model          # fitted 模型（sklearn / HF）
result.preprocessor   # fitted 前處理器
result.metrics        # dict：{"test_roc_auc": 0.92, ...}
result.predictions    # pd.Series
result.confidence     # pd.Series（0~1）
result.leaderboard    # pd.DataFrame（各模型 CV 分數）

result.predict(X_new) # 自動套用前處理 → 預測
result.save("./out")  # 儲存 model.joblib + preprocessor.joblib + metadata.json
result.load("./out")  # 重新載入，不用重新訓練
result.summary()      # 印出摘要
```

### ClusterResult

```python
result.labels          # np.ndarray（-1 = noise）
result.cake_score      # float：全局 CAKE
result.cake_scores     # np.ndarray：每個樣本的信心分數
result.suspicious_idx  # list：低信心樣本 index
result.metrics         # dict：{"n_clusters": 5, "cake": 0.73, ...}

result.save("./out")   # cluster_assignments.csv + cluster_metadata.json
result.summary()
```

### ForecastResult

```python
result.forecast        # pd.Series（未來時間點的預測值）
result.model           # fitted 模型
result.metrics         # dict：{"best_model": "AutoETS", "mae": 12.3}
result.leaderboard     # pd.DataFrame（各模型誤差比較）
result.metadata        # dict：{"backend": "statsforecast", "horizon": 30}

result.save("./out")   # forecast.csv + metadata.json
result.summary()
```

---

## 套件依賴

### 必裝（ML 核心）

| 套件 | 版本 | 用途 |
|------|------|------|
| scikit-learn | ≥ 1.3 | 模型、前處理、CV |
| numpy / scipy / pandas | 新版即可 | 基礎計算 |
| optuna | ≥ 3.6 | TPE / GP-BO 超參數搜索 |
| xgboost | ≥ 2.0 | XGBoost |
| lightgbm | ≥ 4.0 | LightGBM |
| catboost | ≥ 1.2 | CatBoost |
| hdbscan | ≥ 0.8 | HDBSCAN 分群 |
| kneed | ≥ 0.8 | Elbow / k-distance KneeLocator |
| statsforecast | ≥ 1.7 | ETS / ARIMA / Theta 統計預測 |

### 選裝

| 套件 | 用途 |
|------|------|
| `transformers` + `datasets` + `torch` | DL pillar（HuggingFace 微調）|
| `autogluon.timeseries` | TS pillar 大資料後端 |
| `darts` | 異常偵測 |

---

## 目錄結構

```
autoresearch-x/
├── autoresearch_x/
│   ├── __init__.py              # 匯出 analyze / cluster / forecast
│   ├── automl.py                # 三個主入口函式
│   ├── core/
│   │   ├── profiler.py          # DataProfiler（自動偵測 pillar / task / regime）
│   │   └── result.py            # MLResult / ClusterResult / ForecastResult
│   ├── clustering/
│   │   ├── cake.py              # CAKE Score 計算（Bootstrap + Hungarian）
│   │   ├── param_selector.py    # CAKE 驅動的參數自動選擇
│   │   └── methods.py           # ClusteringEngine 統一介面
│   ├── preprocessing/
│   │   └── pipeline.py          # AutoPreprocessor（sklearn Pipeline）
│   ├── ml/
│   │   ├── model_pool.py        # 依 DataProfile 動態組模型候選池
│   │   ├── cv.py                # CVStrategy（自動選 scoring / n_splits）
│   │   ├── ensemble.py          # Stacking / Blending
│   │   └── semi_supervised.py   # LabelSpreading / SelfTraining
│   ├── search/
│   │   └── optimizer.py         # ClusterAwareOptimizer（Optuna）
│   ├── ts/
│   │   └── backend.py           # AutoGluon → StatsForecast → Naive 三層路由
│   ├── dl/
│   │   └── backend.py           # HuggingFace Trainer 微調引擎
│   └── memory/
│       └── experiment_db.py     # SQLite 實驗記錄 + RAG 查詢
├── cli.py                        # arx CLI
├── requirements.txt
└── setup.py
```

---

## License

MIT
