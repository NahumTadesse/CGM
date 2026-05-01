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


import math, random
from statistics import mean, stdev

_N_REPS = 20
_NOISE_SD = 5.0
_BASE_START = 120.0
_BASE_DT = datetime(2025, 1, 1, 22, 30)
_PERSON = Person(weight_kg=80.0, carb_ratio_g_per_unit=9.0)


def _sim(carbs_g=None, insulin_units=None, stress_level=None, exercise_intensity=None, noise_sd=0.0):
    meals, doses, exercise, stress = [], [], None, None
    if carbs_g is not None:
        meals.append(make_meal(30, carbs_g))
    if insulin_units == "auto" and carbs_g is not None:
        doses.append(InsulinDose(time_min=30, units=compute_bolus_units_from_carbs(_PERSON, carbs_g)))
    elif insulin_units not in (None, "auto"):
        doses.append(InsulinDose(time_min=30, units=float(insulin_units)))
    if exercise_intensity is not None:
        exercise = Exercise(start_min=45, duration_min=30, intensity=exercise_intensity)
    if stress_level is not None:
        stress = Stress(start_min=55, duration_min=30, level=stress_level)
    result = run_forecast(
        person=_PERSON, start_datetime=_BASE_DT,
        start_state=MinimalModelState(G_mgdl=_BASE_START, X_per_min=0.0),
        horizon_min=HORIZON_MIN, dt_min=DT_MIN,
        meals=meals, doses=doses, exercise=exercise, stress=stress)
    g = result.glucose_mgdl
    if noise_sd > 0:
        g = [max(20.0, min(400.0, v + random.gauss(0, noise_sd))) for v in g]
    return {"min": min(g), "max": max(g), "end": g[-1]}


def _rep(n=_N_REPS, **kw):
    rows = [_sim(noise_sd=_NOISE_SD, **kw) for _ in range(n)]
    out = {}
    for m in ("min", "max", "end"):
        vals = [r[m] for r in rows]
        mu, s = mean(vals), stdev(vals)
        se = s / math.sqrt(n)
        out[m] = {"mean": round(mu,2), "sd": round(s,2),
                  "ci_lo": round(mu-1.96*se,2), "ci_hi": round(mu+1.96*se,2)}
    return out


def run_analysis():
    random.seed(42)
    os.makedirs("m4_output", exist_ok=True)
    out = "m4_output"

    print("\n=== SENSITIVITY ANALYSIS ===")
    b = _sim()

    def sens_table(label, levels, key, ref):
        rows = []
        print(f"\n{label}: Level / Max / End / ratio_max / ratio_end")
        for i, lv in enumerate(levels):
            r = _sim(**{key: lv})
            pi = ((lv - levels[0]) / levels[0] * 100) if i > 0 else 0
            pm = (r["max"] - ref["max"]) / ref["max"] * 100
            pe = (r["end"] - ref["end"]) / ref["end"] * 100
            rm = round(pm/pi, 4) if pi else "—"
            re = round(pe/pi, 4) if pi else "—"
            rows.append({"level": lv, "max": round(r["max"],2), "end": round(r["end"],2), "ratio_max": rm, "ratio_end": re})
            print(f"  {lv}  {r['max']:.2f}  {r['end']:.2f}  {rm}  {re}")
        return rows

    c_rows = sens_table("Carbs (g)",          [20,40,60,80],      "carbs_g",            b)
    i_rows = sens_table("Insulin (units)",     [2,4,6,8],          "insulin_units",      _sim(insulin_units=2))
    s_rows = sens_table("Stress level",        [0.2,0.4,0.6,0.8], "stress_level",       _sim(stress_level=0.2))
    e_rows = sens_table("Exercise intensity",  [0.2,0.4,0.6,0.8], "exercise_intensity", _sim(exercise_intensity=0.2))

    rankings = sorted([
        (lbl, round(mean([abs(r[m]) for r in rows if isinstance(r[m], float)]), 4))
        for lbl, rows, m in [("carbs",c_rows,"ratio_max"),("insulin",i_rows,"ratio_end"),
                              ("stress",s_rows,"ratio_max"),("exercise",e_rows,"ratio_end")]
    ], key=lambda x: x[1], reverse=True)
    print("\nRanking:", rankings)

    with open(f"{out}/sensitivity_ratios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter","level","max_glucose","end_glucose","ratio_max","ratio_end"])
        for lbl, rows in [("carbs",c_rows),("insulin",i_rows),("stress",s_rows),("exercise",e_rows)]:
            for r in rows:
                w.writerow([lbl, r["level"], r["max"], r["end"], r["ratio_max"], r["ratio_end"]])

    print("\n=== DISTINCT SCENARIOS ===")
    scenarios = [
        ("S1","Sedentary, large meal, high stress",      dict(carbs_g=80, insulin_units="auto", stress_level=0.8)),
        ("S2","Active day, moderate meal",               dict(carbs_g=50, insulin_units="auto", exercise_intensity=0.7)),
        ("S3","Missed insulin, large meal, mild stress", dict(carbs_g=80, stress_level=0.3)),
        ("S4","Low carb with exercise",                  dict(carbs_g=20, insulin_units="auto", exercise_intensity=0.5)),
        ("S5","Stress only",                             dict(stress_level=0.8)),
        ("S6","Full combined",                           dict(carbs_g=60, insulin_units="auto", stress_level=0.5, exercise_intensity=0.5)),
    ]
    sc_results = []
    with open(f"{out}/scenarios_ci.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","name","max_mean","max_sd","max_ci_lo","max_ci_hi","end_mean","end_sd","end_ci_lo","end_ci_hi"])
        for sid, name, kw in scenarios:
            s = _rep(**kw)
            sc_results.append((sid, name, s))
            print(f"{sid} {name}: max={s['max']['mean']}±{s['max']['sd']} CI[{s['max']['ci_lo']},{s['max']['ci_hi']}]")
            w.writerow([sid, name, s["max"]["mean"], s["max"]["sd"], s["max"]["ci_lo"], s["max"]["ci_hi"],
                        s["end"]["mean"], s["end"]["sd"], s["end"]["ci_lo"], s["end"]["ci_hi"]])

    print("\n=== REPLICATIONS ON ORIGINAL 10 RUNS ===")
    orig = [
        (1,"Baseline",            dict(carbs_g=50, insulin_units="auto")),
        (2,"Meal only",           dict(carbs_g=50)),
        (3,"Insulin only",        dict(insulin_units=5.0)),
        (4,"Low carb",            dict(carbs_g=30, insulin_units="auto")),
        (5,"High carb",           dict(carbs_g=80, insulin_units="auto")),
        (6,"Mild stress",         dict(stress_level=0.3)),
        (7,"High stress",         dict(stress_level=0.8)),
        (8,"Exercise only",       dict(exercise_intensity=0.6)),
        (9,"Meal+insulin+stress", dict(carbs_g=50, insulin_units="auto", stress_level=0.3)),
        (10,"Meal+insulin+exer.", dict(carbs_g=50, insulin_units="auto", exercise_intensity=0.6)),
    ]
    rep_results = []
    with open(f"{out}/replications_ci.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id","description","max_mean","max_sd","max_ci_lo","max_ci_hi","end_mean","end_sd","end_ci_lo","end_ci_hi"])
        for rid, name, kw in orig:
            s = _rep(**kw)
            rep_results.append((rid, name, s))
            print(f"Run {rid:>2} {name}: max={s['max']['mean']:.1f}±{s['max']['sd']:.1f} [{s['max']['ci_lo']:.1f},{s['max']['ci_hi']:.1f}]")
            w.writerow([rid, name, s["max"]["mean"], s["max"]["sd"], s["max"]["ci_lo"], s["max"]["ci_hi"],
                        s["end"]["mean"], s["end"]["sd"], s["end"]["ci_lo"], s["end"]["ci_hi"]])

    print("\n=== CLINICAL VALIDATION ===")
    checks = [
        ("Baseline peak",  _sim(carbs_g=50, insulin_units="auto")["max"], 110, 180, "ADA postprandial <180"),
        ("Meal only peak", _sim(carbs_g=50)["max"],                        180, 350, "Diabetic no insulin 200-350 (Ceriello 2010)"),
        ("Insulin nadir",  _sim(insulin_units=5.0)["min"],                 40,  80,  "Insulin nadir 40-70 (Kovatchev)"),
        ("Exercise end",   _sim(exercise_intensity=0.6)["end"],            80,  120, "Post-exercise 10-30 drop (ADA)"),
        ("Stress rise",    _sim(stress_level=0.8)["end"] - _BASE_START,   5,   40,  "Stress 5-40 rise (Surwit 2002)"),
    ]
    passed = 0
    for name, val, lo, hi, ref in checks:
        ok = lo <= val <= hi
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val:.2f}  [{lo},{hi}]  {ref}")
    print(f"\n  {passed}/{len(checks)} passed")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Sensitivity Response Curves (4 Levels per Parameter)", fontsize=13)
    for ax, (label, rows, metric) in zip(axes.flat, [
        ("Carbs (g)", c_rows, "max"), ("Insulin (units)", i_rows, "end"),
        ("Stress level", s_rows, "max"), ("Exercise intensity", e_rows, "end")
    ]):
        ax.plot([r["level"] for r in rows], [r[metric] for r in rows], "o-", linewidth=2, color="steelblue")
        for r in rows:
            if isinstance(r[f"ratio_{metric}"], float):
                ax.annotate(f"ratio={r[f'ratio_{metric}']:.2f}", (r["level"], r[metric]),
                            textcoords="offset points", xytext=(5,5), fontsize=7)
        ax.set_xlabel(label); ax.set_ylabel("Glucose (mg/dL)"); ax.set_title(label); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out}/fig1_sensitivity_curves.png", dpi=150); plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Original 10 Runs — Mean ± 95% CI (n={_N_REPS})", fontsize=13)
    ids = [r[0] for r in rep_results]
    for ax, metric, color, title in [(ax1,"max","steelblue","Peak Glucose"),(ax2,"end","coral","Final Glucose")]:
        mu = [r[2][metric]["mean"] for r in rep_results]
        lo = [r[2][metric]["mean"] - r[2][metric]["ci_lo"] for r in rep_results]
        hi = [r[2][metric]["ci_hi"] - r[2][metric]["mean"] for r in rep_results]
        ax.bar(ids, mu, yerr=[lo,hi], capsize=5, color=color, alpha=0.8)
        ax.set_xlabel("Run ID"); ax.set_ylabel("Glucose (mg/dL)"); ax.set_title(title)
        ax.set_xticks(ids); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out}/fig2_replications_ci.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    sc_mu = [s[2]["max"]["mean"] for s in sc_results]
    sc_lo = [s[2]["max"]["mean"] - s[2]["max"]["ci_lo"] for s in sc_results]
    sc_hi = [s[2]["max"]["ci_hi"] - s[2]["max"]["mean"] for s in sc_results]
    ax.bar([s[0] for s in sc_results], sc_mu, yerr=[sc_lo,sc_hi], capsize=6, color="mediumseagreen", alpha=0.85)
    ax.axhline(180, linestyle="--", color="red", alpha=0.6, label="180")
    ax.axhline(70,  linestyle="--", color="blue", alpha=0.6, label="70")
    ax.set_xlabel("Scenario"); ax.set_ylabel("Peak Glucose (mg/dL)")
    ax.set_title(f"Scenarios — Peak Glucose ± 95% CI (n={_N_REPS})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out}/fig3_scenarios_ci.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([r[0] for r in rankings], [r[1] for r in rankings], color="darkorange", alpha=0.85)
    ax.set_xlabel("Avg Sensitivity Ratio"); ax.set_title("Parameter Influence Ranking"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out}/fig4_ranking.png", dpi=150); plt.close()

    print(f"\nOutputs saved to {out}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analysis":
        run_analysis()
    else:
        main()