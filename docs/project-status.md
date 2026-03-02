# Project Status

## What's Implemented So Far
- A glucose simulation model that estimates blood glucose changes over time.
- A meal absorption model that simulates time-to-peak carbohydrate absorption and total absorption duration.
- A insulin action model that estimates glucose reduction based on insulin dosage and timing.
- Added exercise and stress effect models that modify glucose trends using intensity and duration factors.
- Implemented fixed 15-minute time-step forecasting with a 3-hour prediction.
- Designed user input handling to allow real-world scenario simulation (meal size, insulin dosage, exercise timing, stress events, starting glucose level).

## What's Still to Come
- Continue improving the model to make the glucose predictions as accurate and realistic as possible.
- Adjust and fine-tune parameters so the simulation better reflects how blood sugar actually behaves in real-life situations.
- Add current glucose trend direction (rising, falling, or steady) as an input to help improve short-term forecast accuracy.
- Test more extreme and edge-case scenarios to see how the model performs under different conditions.
- Improve how results are displayed and add a nicer looking graph.

## Changes from Original Proposal
- Interactive scenario setup (user input):
  - User selects the simulation start time
  - User enters starting glucose (mg/dL)
  - User enters carb-to-insulin ratio (grams per 1 unit)

- Meal modeling (scenario input + validation):
  - User can add a meal with time and carb amount
  - Meals outside the 3-hour forecast window are automatically ignored

- Insulin dosing (manual or automatic):
  - User can enter insulin manually (time + units), or
  - Auto-calculate bolus insulin from meal carbs using the user’s carb ratio
  - Insulin outside the 3-hour forecast window is automatically ignored

- Exercise and stress events (quantified inputs):
  - Exercise supports start time, duration, and intensity (0.0–1.0)
  - Stress supports start time, duration, and multiplier (1.2)
  - Exercise/stress outside the 3-hour forecast window are automatically ignored

- Output:
  - Warnings are shown for hypoglycemia-range starting glucose < 70 mg/dL
  - Additional warning for low start glucose combined with insulin/exercise without carbs
  - Forecast prints glucose values at each time step and plots a 3-hour curve with target lines at 70 and 180 mg/dL
