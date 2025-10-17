import argparse
import json
import os
from src.model import train_and_save

def main():
    parser = argparse.ArgumentParser(
        description="Train model and save artifact + metrics."
    )
    parser.add_argument(
        "--kind", default="linear", choices=["linear"], help="Model kind"
    )
    parser.add_argument(
        "--out", default="artifacts/model.joblib", help="Output joblib path"
    )
    args = parser.parse_args()

    # 训练并保存模型
    result = train_and_save(kind=args.kind, out_path=args.out)

    # 保存 metrics.json
    metrics = {"rmse": result["rmse"], "kind": result["meta"]["kind"]}
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Saved model to {result['out_path']}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
