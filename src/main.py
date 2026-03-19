from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from entities import Person, Meal, InsulinDose, Exercise, Stress
from models.minimal_model import MinimalModelState
from models.insulin_model import compute_bolus_units_from_carbs
from simulation.simulator import run_forecast

DT_MIN = 15
HORIZON_MIN = 180  # 3 hours
MEAL_T_PEAK_MIN = 45
MEAL_DURATION_MIN = 180


def ask_int_range(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        s = input(f"{prompt}: ").strip()
        if s == "":
            v = default
        else:
            try:
                v = int(s)
            except ValueError:
                print("Enter a whole number.")
                continue
        if lo <= v <= hi:
            return v
        print(f"Value must be between {lo} and {hi}.")


def ask_float_range(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        s = input(f"{prompt}: ").strip()
        if s == "":
            v = default
        else:
            try:
                v = float(s)
            except ValueError:
                print("Enter a number.")
                continue
        if lo <= v <= hi:
            return v
        print(f"Value must be between {lo} and {hi}.")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    s = input(f"{prompt} (y/n): ").strip().lower()
    if s == "":
        return default_yes
    return s == "y"


def ask_ampm(default: str) -> str:
    s = input("  AM or PM: ").strip().upper()
    if s == "":
        s = default
    while s not in ("AM", "PM"):
        s = input("  Please enter AM or PM: ").strip().upper()
    return s


def ask_time_of_day(label: str, default_h: int, default_m: int, default_ampm: str):
    print(f"\n{label} time:")
    h = ask_int_range("  Hour (1-12)", default_h, 1, 12)
    m = ask_int_range("  Minute (0-59)", default_m, 0, 59)
    ampm = ask_ampm(default_ampm)

    if ampm == "AM":
        hour24 = 0 if h == 12 else h
    else:
        hour24 = 12 if h == 12 else h + 12
    return hour24, m


def minutes_since_start(start_dt: datetime, event_hour24: int, event_min: int) -> int:
    event_dt = start_dt.replace(hour=event_hour24, minute=event_min, second=0, microsecond=0)
    if event_dt < start_dt:
        event_dt += timedelta(days=1)
    return int((event_dt - start_dt).total_seconds() // 60)


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def time_range_str(start_dt: datetime, start_offset_min: int, duration_min: int) -> str:
    a = start_dt + timedelta(minutes=start_offset_min)
    b = a + timedelta(minutes=duration_min)
    return f"{fmt_time(a)}–{fmt_time(b)}"


def in_window(offset_min: int) -> bool:
    return 0 <= offset_min < HORIZON_MIN


def plot_cgm(times_dt, glucose, start_dt, meals=None, doses=None, exercise=None, stress=None):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Main glucose line
    ax.plot(
        times_dt,
        glucose,
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Glucose Forecast"
    )

    # Fill under curve slightly for a modern look
    ax.fill_between(times_dt, glucose, [min(glucose)] * len(glucose), alpha=0.12)

    # Target and threshold zones
    ax.axhspan(70, 180, alpha=0.08, label="Target Range")
    ax.axhline(70, linestyle="--", linewidth=1.2, alpha=0.85, label="Low Threshold")
    ax.axhline(180, linestyle="--", linewidth=1.2, alpha=0.85, label="High Threshold")

    # Dynamic y-limits with padding
    g_min = min(glucose)
    g_max = max(glucose)
    y_pad = max(10, (g_max - g_min) * 0.20)
    y_bottom = max(20, g_min - y_pad)
    y_top = min(400, g_max + y_pad)
    ax.set_ylim(y_bottom, y_top)

    # Event marker label positions
    label_levels = [
        y_top - (y_top - y_bottom) * 0.10,
        y_top - (y_top - y_bottom) * 0.22,
        y_top - (y_top - y_bottom) * 0.34,
        y_top - (y_top - y_bottom) * 0.46
    ]

    used_labels = set()

    # Meal markers
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

    # Insulin markers
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

    # Stress start marker
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

    # Exercise start marker
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

    # Last point highlight
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

    # Summary info box
    summary_lines = [
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

    ax.set_title("CGM Glucose Forecast", fontsize=16, fontweight="bold", pad=14)
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
    plt.show()


def main():
    sh24, sm = ask_time_of_day("Simulation START", 8, 24, "PM")
    today = datetime.now().date()
    start_dt = datetime(today.year, today.month, today.day, sh24, sm)

    start_glucose = ask_float_range("\nStart glucose (mg/dL)", 120.0, 20.0, 400.0)
    carb_ratio = ask_float_range("Carb-to-insulin ratio (grams per 1 unit)", 12.0, 1.0, 50.0)

    person = Person(weight_kg=80.0, carb_ratio_g_per_unit=carb_ratio)

    if start_glucose < 70:
        print("Warning: Starting glucose is below 70 mg/dL (hypoglycemia range).")
        if not ask_yes_no("Continue anyway?", True):
            print("Exiting.")
            return

    meals = []
    meal_summary = "Meal: (none)"
    if ask_yes_no("\nAdd a meal?", True):
        mh24, mm = ask_time_of_day("Meal", 8, 24, "PM")
        carbs = ask_float_range("Meal carbs (g)", 60.0, 0.0, 300.0)

        meal_offset = minutes_since_start(start_dt, mh24, mm)
        if not in_window(meal_offset):
            meal_dt = start_dt + timedelta(minutes=meal_offset)
            print(f"Meal at {fmt_time(meal_dt)} is outside the 3-hour window; ignoring it.")
        else:
            meals.append(
                Meal(
                    time_min=meal_offset,
                    carbs_g=carbs,
                    t_peak_min=MEAL_T_PEAK_MIN,
                    duration_min=MEAL_DURATION_MIN
                )
            )
            meal_dt = start_dt + timedelta(minutes=meal_offset)
            meal_summary = f"Meal: {carbs:.0f} g at {fmt_time(meal_dt)}"

    doses = []
    insulin_summary = "Insulin: (none)"
    if ask_yes_no("\nAdd insulin?", True):
        auto = ask_yes_no("Auto-calc insulin from meal carbs?", True)
        if auto and len(meals) > 0:
            units = compute_bolus_units_from_carbs(person, meals[0].carbs_g)
            doses.append(InsulinDose(time_min=meals[0].time_min, units=units))
            insulin_dt = start_dt + timedelta(minutes=meals[0].time_min)
            insulin_summary = f"Insulin: {units:.1f} units at {fmt_time(insulin_dt)} (auto)"
        else:
            ih24, im = ask_time_of_day("Insulin dose", 8, 24, "PM")
            units = ask_float_range("Insulin units", 5.0, 0.0, 50.0)
            dose_offset = minutes_since_start(start_dt, ih24, im)

            if not in_window(dose_offset):
                dose_dt = start_dt + timedelta(minutes=dose_offset)
                print(f"Insulin at {fmt_time(dose_dt)} is outside the 3-hour window; ignoring it.")
            else:
                doses.append(InsulinDose(time_min=dose_offset, units=units))
                dose_dt = start_dt + timedelta(minutes=dose_offset)
                insulin_summary = f"Insulin: {units:.1f} units at {fmt_time(dose_dt)}"

    exercise = None
    exercise_summary = "Exercise: (none)"
    if ask_yes_no("\nAdd exercise?", False):
        eh24, em = ask_time_of_day("Exercise START", 9, 24, "PM")
        ex_dur = ask_int_range("Exercise duration (min)", 30, 5, 240)
        ex_int = ask_float_range("Exercise intensity (0.0–1.0)", 0.7, 0.0, 1.0)

        ex_offset = minutes_since_start(start_dt, eh24, em)
        if not in_window(ex_offset):
            ex_dt = start_dt + timedelta(minutes=ex_offset)
            print(f"Exercise at {fmt_time(ex_dt)} is outside the 3-hour window; ignoring it.")
        else:
            exercise = Exercise(start_min=ex_offset, duration_min=ex_dur, intensity=ex_int)
            exercise_summary = f"Exercise: {time_range_str(start_dt, ex_offset, ex_dur)}, intensity {ex_int:.1f}"

    stress = None
    stress_summary = "Stress: (none)"
    if ask_yes_no("\nAdd stress?", False):
        sth24, stm = ask_time_of_day("Stress START", 10, 24, "PM")
        st_dur = ask_int_range("Stress duration (min)", 30, 5, 240)
        st_level = ask_float_range("Stress level (0.0-1.0)", 0.5, 0.0, 1.0)

        st_offset = minutes_since_start(start_dt, sth24, stm)
        if not in_window(st_offset):
            st_dt = start_dt + timedelta(minutes=st_offset)
            print(f"Stress at {fmt_time(st_dt)} is outside the 3-hour window; ignoring it.")
        else:
            stress = Stress(start_min=st_offset, duration_min=st_dur, level=st_level)
            stress_summary = f"Stress: {time_range_str(start_dt, st_offset, st_dur)}, level {st_level:.1f}"

    if start_glucose < 70 and len(meals) == 0 and (len(doses) > 0 or exercise is not None):
        print("WARNING: Low start glucose + insulin/exercise without carbs may cause dangerous hypoglycemia.")

    print("\nScenario:")
    print(f"  Start glucose: {start_glucose:.0f} mg/dL")
    print(f"  {meal_summary}")
    print(f"  {insulin_summary}")
    print(f"  {exercise_summary}")
    print(f"  {stress_summary}")

    state0 = MinimalModelState(G_mgdl=start_glucose, X_per_min=0.0)
    result = run_forecast(
        person=person,
        start_datetime=start_dt,
        start_state=state0,
        horizon_min=HORIZON_MIN,
        dt_min=DT_MIN,
        meals=meals,
        doses=doses,
        exercise=exercise,
        stress=stress
    )

    print("\n=== Forecast Results ===")
    for t_dt, g in zip(result.times_dt, result.glucose_mgdl):
        print(f"{fmt_time(t_dt)}  G={g:7.2f} mg/dL")

    plot_cgm(
        result.times_dt,
        result.glucose_mgdl,
        start_dt,
        meals=meals,
        doses=doses,
        exercise=exercise,
        stress=stress
    )


if __name__ == "__main__":
    main()