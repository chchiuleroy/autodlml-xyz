from setuptools import setup, find_packages

setup(
    name="autoresearch-x",
    version="0.1.0",
    description="AutoML / AutoDL / AutoTS framework with CAKE-score clustering",
    author="Roy",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "optuna>=3.6",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "catboost>=1.2",
        "hdbscan>=0.8",
        "kneed>=0.8",
        "statsforecast>=1.7",
        "tqdm>=4.66",
    ],
    extras_require={
        "dl": [
            "transformers>=4.40",
            "datasets>=2.18",
            "torch>=2.2",
        ],
        "ts": [
            "autogluon.timeseries>=1.1",
            "darts>=0.29",
        ],
        "all": [
            "transformers>=4.40",
            "datasets>=2.18",
            "torch>=2.2",
            "autogluon.timeseries>=1.1",
            "darts>=0.29",
        ],
    },
    entry_points={
        "console_scripts": [
            "arx=cli:main",
        ],
    },
)
