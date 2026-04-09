#!/usr/bin/env python
"""
AutoResearch-X CLI
用法：
  python cli.py analyze  data.csv --target label
  python cli.py cluster  data.csv --method hdbscan
  python cli.py forecast data.csv --target sales --horizon 30
  python cli.py history  --pillar ml --limit 10
"""
import argparse
import sys
import json
from pathlib import Path


def _print_result(result, verbose: bool = False):
    """漂亮地列印結果摘要"""
    print("\n" + "═" * 55)
    print(result.summary())
    if verbose and hasattr(result, "leaderboard") and result.leaderboard is not None:
        print("\n[Leaderboard]")
        print(result.leaderboard.to_string(index=False))
    print("═" * 55)


def cmd_analyze(args):
    from autoresearch_x import analyze
    result = analyze(
        args.data,
        target=args.target,
        strategy=args.strategy,
        n_trials=args.n_trials,
        ensemble=not args.no_ensemble,
        output_dir=args.output,
    )
    _print_result(result, verbose=args.verbose)


def cmd_cluster(args):
    from autoresearch_x import cluster
    drop_cols = args.drop.split(",") if args.drop else None
    result = cluster(
        args.data,
        method=args.method,
        drop_cols=drop_cols,
        output_dir=args.output,
    )
    _print_result(result, verbose=args.verbose)


def cmd_forecast(args):
    from autoresearch_x import forecast
    result = forecast(
        args.data,
        target=args.target,
        horizon=args.horizon,
        freq=args.freq,
        item_id_col=args.item_id,
        detect_anomaly=args.anomaly,
        output_dir=args.output,
    )
    _print_result(result, verbose=args.verbose)


def cmd_history(args):
    from autoresearch_x.memory.experiment_db import ExperimentDB
    db = ExperimentDB()
    rows = db.query(
        pillar=args.pillar,
        task=args.task,
        limit=args.limit,
    )
    if not rows:
        print("（無歷史記錄）")
        return
    print(f"\n找到 {len(rows)} 筆實驗記錄：\n")
    for r in rows:
        import datetime
        ts = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        metrics_str = json.dumps(r["metrics"], ensure_ascii=False)[:60]
        print(f"  [{ts}] pillar={r['pillar']}  model={r['model_name']}  {metrics_str}")


def main():
    parser = argparse.ArgumentParser(
        prog="arx",
        description="AutoResearch-X：自動 ML / DL / TS 框架",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="顯示詳細結果")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyze ───────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="監督 / 半監督學習")
    p_analyze.add_argument("data",          help="CSV / xlsx / parquet 路徑")
    p_analyze.add_argument("--target", "-t", required=True, help="目標欄位名稱")
    p_analyze.add_argument("--strategy",    default="auto",
                           choices=["auto", "tpe", "gp_bo", "random"],
                           help="超參數搜索策略")
    p_analyze.add_argument("--n-trials",    type=int, default=50, help="Optuna trials 數")
    p_analyze.add_argument("--no-ensemble", action="store_true", help="關閉 Stacking Ensemble")
    p_analyze.add_argument("--output", "-o", default=None, help="輸出目錄")
    p_analyze.set_defaults(func=cmd_analyze)

    # ── cluster ───────────────────────────────────────────────
    p_cluster = sub.add_parser("cluster", help="非監督分群")
    p_cluster.add_argument("data",          help="CSV / xlsx / parquet 路徑")
    p_cluster.add_argument("--method", "-m", default="auto",
                           choices=["auto", "hdbscan", "dbscan", "kmeans", "gmm", "meanshift"],
                           help="分群方法")
    p_cluster.add_argument("--drop",        default=None, help="排除欄位（逗號分隔）")
    p_cluster.add_argument("--output", "-o", default=None, help="輸出目錄")
    p_cluster.set_defaults(func=cmd_cluster)

    # ── forecast ──────────────────────────────────────────────
    p_forecast = sub.add_parser("forecast", help="時間序列預測")
    p_forecast.add_argument("data",             help="CSV / xlsx / parquet 路徑")
    p_forecast.add_argument("--target", "-t",   required=True, help="目標欄位名稱")
    p_forecast.add_argument("--horizon", "-H",  type=int, required=True, help="預測步數")
    p_forecast.add_argument("--freq",           default=None, help="頻率（H/D/W/M/...）")
    p_forecast.add_argument("--item-id",        default=None, dest="item_id",
                            help="多序列 ID 欄位")
    p_forecast.add_argument("--anomaly",        action="store_true", help="開啟異常偵測")
    p_forecast.add_argument("--output", "-o",   default=None, help="輸出目錄")
    p_forecast.set_defaults(func=cmd_forecast)

    # ── history ───────────────────────────────────────────────
    p_history = sub.add_parser("history", help="查看實驗記錄")
    p_history.add_argument("--pillar",  default=None, choices=["ml", "dl", "ts", "cluster"])
    p_history.add_argument("--task",    default=None, help="任務類型（classification / regression...）")
    p_history.add_argument("--limit",   type=int, default=10, help="最多顯示幾筆")
    p_history.set_defaults(func=cmd_history)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
