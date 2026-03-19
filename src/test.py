import csv
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from entities import Person, Meal, InsulinDose, Exercise, Stress
from models.minimal_model import MinimalModelState
from models.insulin_model import compute_bolus_units_from_carbs
from simulation.simulator import run_forecast

DT_MIN = 15
HORIZON_MIN = 180
MEAL_T_PEAK_MIN = 60
MEAL_DURATION_MIN = 180


def make_meal(time_min, carbs):
    return Meal(
        time_min=time_min,
        carbs_g=carbs,
        t_peak_min=MEAL_T_PEAK_MIN,
        duration_min=MEAL_DURATION_MIN
    )


def summarize_run(run_id, description, factors, result):
    glucose = result.glucose_mgdl
    return {
        "run_id": run_id,
        "description": description,
        "factors": factors,
        "min_glucose": round(min(glucose), 2),
        "max_glucose": round(max(glucose), 2),
        "end_glucose": round(glucose[-1], 2),
    }


def safe_filename(text):
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def plot_and_save_run(result, start_dt, scenario, output_path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    times_dt = result.times_dt
    glucose = result.glucose_mgdl

    ax.plot(
        times_dt,
        glucose,
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Glucose Forecast"
    )

    ax.fill_between(times_dt, glucose, [min(glucose)] * len(glucose), alpha=0.12)

    ax.axhspan(70, 180, alpha=0.08, label="Target Range")
    ax.axhline(70, linestyle="--", linewidth=1.2, alpha=0.85, label="Low Threshold")
    ax.axhline(180, linestyle="--", linewidth=1.2, alpha=0.85, label="High Threshold")

    g_min = min(glucose)
    g_max = max(glucose)
    y_pad = max(10, (g_max - g_min) * 0.20)
    y_bottom = max(20, g_min - y_pad)
    y_top = min(400, g_max + y_pad)
    ax.set_ylim(y_bottom, y_top)

    label_levels = [
        y_top - (y_top - y_bottom) * 0.10,
        y_top - (y_top - y_bottom) * 0.22,
        y_top - (y_top - y_bottom) * 0.34,
        y_top - (y_top - y_bottom) * 0.46
    ]

    used_labels = set()

    meals = scenario["meals"]
    doses = scenario["doses"]
    stress = scenario["stress"]
    exercise = scenario["exercise"]

    if meals:
        for meal in meals:
            meal_time = start_dt + timedelta(minutes=meal.time_min)
            legend_label = "Meal Event" if "Meal Event" not in used_labels else None
            ax.axvline(meal_time, linestyle="--", linewidth=1.5, alpha=0.95, label=legend_label)
            ax.text(
                meal_time,
                label_levels[0],
                "Meal",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.45)
            )
            used_labels.add("Meal Event")

    if doses:
        for dose in doses:
            dose_time = start_dt + timedelta(minutes=dose.time_min)
            legend_label = "Insulin Event" if "Insulin Event" not in used_labels else None
            ax.axvline(dose_time, linestyle=":", linewidth=1.7, alpha=0.95, label=legend_label)
            ax.text(
                dose_time,
                label_levels[1],
                "Insulin",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.45)
            )
            used_labels.add("Insulin Event")

    if stress is not None:
        stress_start = start_dt + timedelta(minutes=stress.start_min)
        legend_label = "Stress Event" if "Stress Event" not in used_labels else None
        ax.axvline(stress_start, linestyle="-.", linewidth=1.7, alpha=0.95, label=legend_label)
        ax.text(
            stress_start,
            label_levels[2],
            "Stress",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.45)
        )
        used_labels.add("Stress Event")

    if exercise is not None:
        exercise_start = start_dt + timedelta(minutes=exercise.start_min)
        legend_label = "Exercise Event" if "Exercise Event" not in used_labels else None
        ax.axvline(exercise_start, linestyle="-", linewidth=1.5, alpha=0.95, label=legend_label)
        ax.text(
            exercise_start,
            label_levels[3],
            "Exercise",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.45)
        )
        used_labels.add("Exercise Event")

    ax.scatter(times_dt[-1], glucose[-1], s=70, zorder=5)
    ax.text(
        times_dt[-1],
        glucose[-1] + 5,
        f"{glucose[-1]:.1f} mg/dL",
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="white", alpha=0.5)
    )

    summary_lines = [
        f"Run:   {scenario['run_id']}",
        f"Start: {glucose[0]:.1f} mg/dL",
        f"Min:   {min(glucose):.1f} mg/dL",
        f"Max:   {max(glucose):.1f} mg/dL",
        f"End:   {glucose[-1]:.1f} mg/dL"
    ]
    ax.text(
        0.015,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="white", alpha=0.45)
    )

    ax.set_title(
        f"Run {scenario['run_id']}: {scenario['description']}",
        fontsize=16,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=11)

    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.30)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    legend = ax.legend(loc="upper right", framealpha=0.35)
    legend.get_frame().set_edgecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    start_dt = datetime.now().replace(hour=22, minute=30, second=0, microsecond=0)

    person = Person(weight_kg=80.0, carb_ratio_g_per_unit=9.0)

    scenarios = []

    meal = make_meal(30, 50)
    units = compute_bolus_units_from_carbs(person, 50)
    scenarios.append({
        "run_id": 1,
        "description": "Baseline",
        "factors": "50g meal + auto insulin",
        "meals": [meal],
        "doses": [InsulinDose(time_min=30, units=units)],
        "exercise": None,
        "stress": None,
    })

    scenarios.append({
        "run_id": 2,
        "description": "Meal only",
        "factors": "50g meal, no insulin",
        "meals": [make_meal(30, 50)],
        "doses": [],
        "exercise": None,
        "stress": None,
    })

    scenarios.append({
        "run_id": 3,
        "description": "Insulin only",
        "factors": "5 units insulin, no meal",
        "meals": [],
        "doses": [InsulinDose(time_min=30, units=5.0)],
        "exercise": None,
        "stress": None,
    })

    meal = make_meal(30, 30)
    units = compute_bolus_units_from_carbs(person, 30)
    scenarios.append({
        "run_id": 4,
        "description": "Low carb",
        "factors": "30g meal + auto insulin",
        "meals": [meal],
        "doses": [InsulinDose(time_min=30, units=units)],
        "exercise": None,
        "stress": None,
    })

    meal = make_meal(30, 80)
    units = compute_bolus_units_from_carbs(person, 80)
    scenarios.append({
        "run_id": 5,
        "description": "High carb",
        "factors": "80g meal + auto insulin",
        "meals": [meal],
        "doses": [InsulinDose(time_min=30, units=units)],
        "exercise": None,
        "stress": None,
    })

    scenarios.append({
        "run_id": 6,
        "description": "Mild stress",
        "factors": "stress level 0.3 only",
        "meals": [],
        "doses": [],
        "exercise": None,
        "stress": Stress(start_min=55, duration_min=30, level=0.3),
    })

    scenarios.append({
        "run_id": 7,
        "description": "High stress",
        "factors": "stress level 0.8 only",
        "meals": [],
        "doses": [],
        "exercise": None,
        "stress": Stress(start_min=55, duration_min=30, level=0.8),
    })

    scenarios.append({
        "run_id": 8,
        "description": "Exercise only",
        "factors": "30 min exercise, intensity 0.6",
        "meals": [],
        "doses": [],
        "exercise": Exercise(start_min=45, duration_min=30, intensity=0.6),
        "stress": None,
    })

    meal = make_meal(30, 50)
    units = compute_bolus_units_from_carbs(person, 50)
    scenarios.append({
        "run_id": 9,
        "description": "Meal + insulin + mild stress",
        "factors": "50g meal + auto insulin + stress 0.3",
        "meals": [meal],
        "doses": [InsulinDose(time_min=30, units=units)],
        "exercise": None,
        "stress": Stress(start_min=55, duration_min=30, level=0.3),
    })

    meal = make_meal(30, 50)
    units = compute_bolus_units_from_carbs(person, 50)
    scenarios.append({
        "run_id": 10,
        "description": "Meal + insulin + exercise",
        "factors": "50g meal + auto insulin + exercise 0.6",
        "meals": [meal],
        "doses": [InsulinDose(time_min=30, units=units)],
        "exercise": Exercise(start_min=60, duration_min=30, intensity=0.6),
        "stress": None,
    })

    os.makedirs("run_graphs", exist_ok=True)

    rows = []

    for s in scenarios:
        start_state = MinimalModelState(G_mgdl=120.0, X_per_min=0.0)

        result = run_forecast(
            person=person,
            start_datetime=start_dt,
            start_state=start_state,
            horizon_min=HORIZON_MIN,
            dt_min=DT_MIN,
            meals=s["meals"],
            doses=s["doses"],
            exercise=s["exercise"],
            stress=s["stress"],
        )

        rows.append(
            summarize_run(
                s["run_id"],
                s["description"],
                s["factors"],
                result
            )
        )

        filename = f"run_{s['run_id']:02d}_{safe_filename(s['description'])}.png"
        output_path = os.path.join("run_graphs", filename)
        plot_and_save_run(result, start_dt, s, output_path)

    with open("run_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "description",
                "factors",
                "min_glucose",
                "max_glucose",
                "end_glucose",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved run_summary.csv")
    print("Saved graphs in run_graphs/")


if __name__ == "__main__":
    main()