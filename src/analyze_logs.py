import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from constants import PREDICTION_LOG_DIR


def load_logs(log_file: Path) -> list[dict[str, Any]]:
    logs = []
    if not log_file.exists():
        print(f"Plik logów nie istnieje: {log_file}")
        return logs

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    log_entry = json.loads(line)
                    logs.append(log_entry)
                except json.JSONDecodeError as e:
                    print(f"Błąd parsowania linii: {e}")
                    continue

    return logs


def extract_predictions_by_model(logs: list[dict[str, Any]]) -> dict[str, list[float]]:
    predictions_by_model = defaultdict(list)

    for log_entry in logs:
        model_name = log_entry.get("model_name")
        prediction = log_entry.get("prediction")
        if model_name and prediction is not None:
            predictions_by_model[model_name].append(float(prediction))

    return dict(predictions_by_model)


def extract_predictions_with_actual(logs: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    data_by_model = defaultdict(lambda: {"predictions": [], "actuals": []})

    for log_entry in logs:
        model_name = log_entry.get("model_name")
        prediction = log_entry.get("prediction")
        input_data = log_entry.get("input_data", {})
        actual_rating = input_data.get("review_scores_rating")
        if model_name and prediction is not None and actual_rating is not None:
            try:
                data_by_model[model_name]["predictions"].append(float(prediction))
                data_by_model[model_name]["actuals"].append(float(actual_rating))
            except (ValueError, TypeError):
                continue

    result = {}
    for model_name, data in data_by_model.items():
        predictions = data["predictions"]
        actuals = data["actuals"]
        if predictions and actuals and len(predictions) == len(actuals):
            result[model_name] = (predictions, actuals)
    return result


def calculate_statistics(predictions: list[float]) -> dict[str, float]:
    if not predictions:
        return {}

    arr = np.array(predictions)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def plot_prediction_distribution(
    predictions_by_model: dict[str, list[float]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(len(predictions_by_model), 1, figsize=(12, 4 * len(predictions_by_model)))

    if len(predictions_by_model) == 1:
        axes = [axes]

    for idx, (model_name, predictions) in enumerate(predictions_by_model.items()):
        ax = axes[idx]
        ax.hist(predictions, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Predykcja")
        ax.set_ylabel("Częstość")
        ax.set_title(f"Rozkład predykcji - Model: {model_name} (n={len(predictions)})")
        ax.grid(True, alpha=0.3)

        stats_dict = calculate_statistics(predictions)
        stats_text = (
            f"Średnia: {stats_dict['mean']:.2f}\n"
            f"Mediana: {stats_dict['median']:.2f}\n"
            f"Std: {stats_dict['std']:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    output_path = output_dir / "prediction_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykres: {output_path}")
    plt.close()


def plot_model_comparison(
    predictions_by_model: dict[str, list[float]], output_dir: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    data_for_box = [predictions for predictions in predictions_by_model.values()]
    labels = list(predictions_by_model.keys())

    ax1.boxplot(data_for_box, labels=labels)
    ax1.set_ylabel("Predykcja")
    ax1.set_title("Porównanie rozkładu predykcji - Box Plot")
    ax1.grid(True, alpha=0.3)

    parts = ax2.violinplot(
        data_for_box,
        positions=range(len(labels)),
        showmeans=True,
        showmedians=True,
    )
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Predykcja")
    ax2.set_title("Porównanie rozkładu predykcji - Violin Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykres: {output_path}")
    plt.close()


def plot_predictions_vs_actual(
    data_by_model: dict[str, tuple[list[float], list[float]]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(
        len(data_by_model), 1, figsize=(10, 5 * len(data_by_model))
    )

    if len(data_by_model) == 1:
        axes = [axes]

    for idx, (model_name, (predictions, actuals)) in enumerate(data_by_model.items()):
        ax = axes[idx]
        if len(predictions) != len(actuals):
            ax.text(
                0.5,
                0.5,
                f"Błąd: niezgodna liczba predykcji ({len(predictions)}) "
                f"i rzeczywistych ocen ({len(actuals)})",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(f"Predykcje vs Rzeczywiste - Model: {model_name} (BŁĄD)")
            continue

        if not predictions or not actuals:
            ax.text(
                0.5,
                0.5,
                "Brak danych",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(f"Predykcje vs Rzeczywiste - Model: {model_name} (brak danych)")
            continue

        min_val = min(min(predictions), min(actuals))
        max_val = max(max(predictions), max(actuals))

        ax.scatter(actuals, predictions, alpha=0.5, s=20)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Idealna predykcja")

        try:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            correlation = np.corrcoef(actuals, predictions)[0, 1]

            ax.set_xlabel("Rzeczywista ocena")
            ax.set_ylabel("Predykcja")
            ax.set_title(
                f"Predykcje vs Rzeczywiste - Model: {model_name}\n"
                f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, Korelacja: {correlation:.3f}"
            )
        except Exception as e:
            ax.set_title(f"Predykcje vs Rzeczywiste - Model: {model_name} (błąd: {e})")

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "predictions_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykres: {output_path}")
    plt.close()


def print_statistics_table(predictions_by_model: dict[str, list[float]]) -> None:
    """Wypisuje tabelę ze statystykami dla każdego modelu."""
    print("\n" + "=" * 80)
    print("STATYSTYKI PREDYKCJI PO MODELACH")
    print("=" * 80)

    for model_name, predictions in predictions_by_model.items():
        stats_dict = calculate_statistics(predictions)
        print(f"\nModel: {model_name} (n={len(predictions)})")
        print(f"  Średnia:      {stats_dict['mean']:8.4f}")
        print(f"  Mediana:      {stats_dict['median']:8.4f}")
        print(f"  Odch. std.:   {stats_dict['std']:8.4f}")
        print(f"  Min:          {stats_dict['min']:8.4f}")
        print(f"  Max:          {stats_dict['max']:8.4f}")
        print(f"  Q25:          {stats_dict['q25']:8.4f}")
        print(f"  Q75:          {stats_dict['q75']:8.4f}")


def print_error_metrics(
    data_by_model: dict[str, tuple[list[float], list[float]]]
) -> None:
    """Wypisuje metryki błędów dla modeli z danymi rzeczywistymi."""
    if not data_by_model:
        print("\nBrak danych z rzeczywistymi ocenami do porównania.")
        return

    print("\n" + "=" * 80)
    print("METRYKI BŁĘDÓW (przy dostępnych rzeczywistych ocenach)")
    print("=" * 80)

    for model_name, (predictions, actuals) in data_by_model.items():
        if len(predictions) != len(actuals):
            print(
                f"\nModel: {model_name} - BŁĄD: niezgodna liczba predykcji ({len(predictions)}) "
                f"i rzeczywistych ocen ({len(actuals)})"
            )
            continue

        if not predictions or not actuals:
            print(f"\nModel: {model_name} - Brak danych do obliczenia metryk")
            continue

        try:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            correlation = np.corrcoef(actuals, predictions)[0, 1]

            print(f"\nModel: {model_name} (n={len(predictions)})")
            print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
            print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
            print(f"  Korelacja Pearsona:             {correlation:.4f}")
        except Exception as e:
            print(f"\nModel: {model_name} - Błąd obliczania metryk: {e}")
            continue


def perform_statistical_test(
    predictions_by_model: dict[str, list[float]]
) -> None:
    if len(predictions_by_model) < 2:
        print("\nPotrzebne są co najmniej 2 modele do porównania statystycznego.")
        return

    models = list(predictions_by_model.keys())
    print("\n" + "=" * 80)
    print("TESTY STATYSTYCZNE (A/B Testing)")
    print("=" * 80)

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]
            model_b = models[j]

            pred_a = np.array(predictions_by_model[model_a])
            pred_b = np.array(predictions_by_model[model_b])

            t_stat, p_value = stats.ttest_ind(pred_a, pred_b)

            u_stat, u_p_value = stats.mannwhitneyu(pred_a, pred_b)

            print(f"\nPorównanie: {model_a} vs {model_b}")
            print(f"  T-test:")
            print(f"    Statystyka t: {t_stat:.4f}")
            print(f"    P-value:      {p_value:.4f}")
            print(f"    Istotność:    {'TAK' if p_value < 0.05 else 'NIE'} (p < 0.05)")

            print(f"  Mann-Whitney U test:")
            print(f"    Statystyka U: {u_stat:.4f}")
            print(f"    P-value:      {u_p_value:.4f}")
            print(f"    Istotność:    {'TAK' if u_p_value < 0.05 else 'NIE'} (p < 0.05)")

            mean_a = np.mean(pred_a)
            mean_b = np.mean(pred_b)
            print(f"  Różnica średnich: {mean_a:.4f} vs {mean_b:.4f} (różnica: {abs(mean_a - mean_b):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza logów predykcji do testów A/B"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Ścieżka do pliku logów (domyślnie: src/service/logs/predictions.log)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Katalog na wykresy (domyślnie: analysis_output)",
    )

    args = parser.parse_args()

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = PREDICTION_LOG_DIR / "predictions.log"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wczytywanie logów z: {log_file}")
    logs = load_logs(log_file)

    if not logs:
        print("Brak danych w logu!")
        return

    print(f"Wczytano {len(logs)} wpisów z loga")

    predictions_by_model = extract_predictions_by_model(logs)
    data_with_actual = extract_predictions_with_actual(logs)

    if not predictions_by_model:
        print("Brak predykcji w logach!")
        return

    print(f"\nZnaleziono {len(predictions_by_model)} modeli:")
    for model_name, predictions in predictions_by_model.items():
        print(f"  - {model_name}: {len(predictions)} predykcji")

    print_statistics_table(predictions_by_model)
    print_error_metrics(data_with_actual)

    if len(predictions_by_model) >= 2:
        perform_statistical_test(predictions_by_model)

    print("\n" + "=" * 80)
    print("Generowanie wykresów...")
    print("=" * 80)

    plot_prediction_distribution(predictions_by_model, output_dir)

    if len(predictions_by_model) >= 2:
        plot_model_comparison(predictions_by_model, output_dir)

    if data_with_actual:
        plot_predictions_vs_actual(data_with_actual, output_dir)

    print(f"\nAnaliza zakończona! Wykresy zapisane w: {output_dir}")


if __name__ == "__main__":
    main()
