# CGM Glucose Simulation
For a full detailed report of the project including model design, analysis, and results, see the **[Final Report](docs/M5_Report.pdf)**.

A glucose simulation for diabetes management that models how meals, insulin, stress, and exercise affect blood glucose over a 3-hour window using a glucose-insulin minimal model.

## Installation

1. Clone the repository
```
git clone https://github.com/NahumTadesse/YOUR-REPO-NAME
cd YOUR-REPO-NAME
```

2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```
pip install matplotlib
```

## Usage

**Interactive mode** — enter your own inputs and see a 3-hour glucose forecast:
```
python main.py
```

**Test mode** — runs all 10 predefined scenarios and saves graphs and CSV:
```
python test.py
```

**Analysis mode** — runs sensitivity analysis, replications, and clinical validation:
```
python test.py --analysis
```

## Parameters

| Parameter | Description | Default |
|---|---|---|
| Start glucose | Starting blood glucose (mg/dL) | 120.0 |
| Carb ratio | Grams of carbs per 1 unit of insulin | 9.0 |
| Meal carbs | Carbohydrates in meal (g) | optional |
| Insulin units | Units of insulin dose | optional |
| Exercise intensity | Intensity level 0.0 to 1.0 | optional |
| Stress level | Stress level 0.0 to 1.0 | optional |

## Project Structure

```
CGM/
├── main.py               # Interactive simulation
├── test.py               # Test runs and analysis
├── entities/
│   ├── person.py         # Patient parameters
│   ├── meal.py           # Meal input
│   ├── insulin.py        # Insulin dose
│   ├── exercise.py       # Exercise input
│   └── stress.py         # Stress input
├── models/
│   ├── minimal_model.py  # Glucose-insulin ODE
│   ├── carb_absorption.py
│   └── insulin_model.py
├── simulation/
│   └── simulator.py      # Main simulation loop
├── run_graphs/           # Output graphs from test runs
└── m4_output/            # Sensitivity analysis outputs
```

## Output

- `run_summary.csv` — min, max, and final glucose for each test run
- `run_graphs/` — CGM-style graphs for each scenario
- `m4_output/sensitivity_ratios.csv` — sensitivity ratios per parameter
- `m4_output/replications_ci.csv` — mean, SD, and 95% CI for each run
- `m4_output/scenarios_ci.csv` — distinct scenario results with CI
- `m4_output/fig1_sensitivity_curves.png` — response curves
- `m4_output/fig2_replications_ci.png` — replications with error bars
- `m4_output/fig3_scenarios_ci.png` — scenario comparison
- `m4_output/fig4_ranking.png` — parameter influence ranking
