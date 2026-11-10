import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from training.data import TARGET, load_dataset
from training.tracking import as_mlflow_dataset, setup_mlflow

DELAYS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]


def _save(figure: plt.Figure, folder: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(folder / name, dpi=120)
    plt.close(figure)


def plot_target(frame: pd.DataFrame, folder: Path) -> None:
    figure, ax = plt.subplots(figsize=(4, 3))
    frame[TARGET].value_counts().sort_index().plot.bar(ax=ax, color="#4c72b0")
    ax.set(title="Баланс классов", xlabel="дефолт", ylabel="клиентов")
    _save(figure, folder, "target_balance.png")


def plot_default_by_delay(frame: pd.DataFrame, folder: Path) -> None:
    figure, ax = plt.subplots(figsize=(6, 3))
    frame.groupby("pay_0")[TARGET].mean().plot.bar(ax=ax, color="#c44e52")
    ax.set(
        title="Доля дефолтов по просрочке в последнем месяце",
        xlabel="pay_0",
        ylabel="доля дефолтов",
    )
    _save(figure, folder, "default_rate_by_pay_0.png")


def plot_limit(frame: pd.DataFrame, folder: Path) -> None:
    figure, ax = plt.subplots(figsize=(6, 3))
    for label, group in frame.groupby(TARGET):
        ax.hist(group["limit_bal"], bins=40, alpha=0.5, density=True, label=str(label))
    ax.set(title="Кредитный лимит по классам", xlabel="limit_bal")
    ax.legend(title="дефолт")
    _save(figure, folder, "limit_by_target.png")


def plot_correlations(frame: pd.DataFrame, folder: Path) -> None:
    corr = frame.corr(numeric_only=True)
    figure, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    figure.colorbar(image)
    ax.set_title("Корреляции признаков")
    _save(figure, folder, "correlations.png")


def summarize(frame: pd.DataFrame) -> str:
    corr = frame.corr(numeric_only=True)[TARGET].drop(TARGET)
    top = corr.abs().sort_values(ascending=False).head(5).index
    late = frame[frame["pay_0"] > 0][TARGET].mean()
    on_time = frame[frame["pay_0"] <= 0][TARGET].mean()
    bills = [f"bill_amt{i}" for i in range(1, 7)]
    bill_corr = frame[bills].corr().to_numpy()
    lines = [
        "# EDA: выводы",
        f"- Строк: {len(frame)}, признаков: {frame.shape[1] - 1}, "
        f"пропусков: {int(frame.isna().sum().sum())}, "
        f"дублей: {int(frame.duplicated().sum())}.",
        f"- Доля дефолтов {frame[TARGET].mean():.3f}: классы несбалансированы, "
        "поэтому основная метрика ROC-AUC, а не accuracy.",
        "- Сильнее всего с целевой переменной связаны: "
        + ", ".join(f"{name} ({corr[name]:+.3f})" for name in top)
        + ".",
        f"- С просрочкой в последнем месяце дефолт {late:.3f}, без неё {on_time:.3f}.",
        f"- bill_amt1..6 коррелируют между собой (минимум {bill_corr.min():.2f}): "
        "для линейной модели нужны производные отношения вместо сырых сумм.",
        "- Недокументированные коды: education "
        f"{sorted(set(frame['education']) - {1, 2, 3, 4})}, "
        f"marriage {sorted(set(frame['marriage']) - {1, 2, 3})} "
        "(считаем отдельными значениями, строки не удаляем).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_mlflow()
    frame = load_dataset()
    with mlflow.start_run(run_name="eda"), tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        mlflow.set_tag("stage", "eda")
        mlflow.log_input(as_mlflow_dataset(frame), context="eda")
        for plot in (plot_target, plot_default_by_delay, plot_limit, plot_correlations):
            plot(frame, folder)
        summary = summarize(frame)
        (folder / "eda_summary.md").write_text(summary, encoding="utf-8")
        frame.describe().T.to_csv(folder / "describe.csv")
        mlflow.log_artifacts(tmp, artifact_path="eda")
    print(summary)


if __name__ == "__main__":
    main()
