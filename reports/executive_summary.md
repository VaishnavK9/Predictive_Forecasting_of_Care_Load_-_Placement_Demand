# Executive Summary: Predictive Forecasting for the HHS UAC Program

**Prepared for HHS program leadership and capacity-planning stakeholders**

---

## The Problem

The Unaccompanied Alien Children (UAC) program cares for children transferred from CBP custody
into HHS shelters pending placement with a vetted sponsor. Intake can spike suddenly -- driven by
border activity, policy changes, or humanitarian events -- and today the program finds out about a
surge only after it is already underway. This reactive posture creates real risk: overcrowded
shelters, overworked medical and casework staff, and longer stays for children waiting on
placement.

This project turns three years of HHS's own reporting data (Jan 2023-Dec 2025) into a validated
forecasting system that gives the program **advance visibility** into two things that matter most:

1. **How many children will be in HHS care** in the coming days (the care-load forecast).
2. **How many children are likely to be discharged to sponsors** in the coming days (the
   placement-demand forecast).

## What We Built

- A cleaned, validated historical dataset with a clear, honest accounting of what is and isn't
  known (see "A Note on the Data," below).
- A suite of forecasting models, rigorously tested the way a forecast should be tested: **only
  against data the model never saw**, simulating exactly how it would perform if deployed today.
- An interactive dashboard where a planner can select a forecast horizon (1, 7, or 14 days out),
  compare models, see confidence ranges, and run "what if intake surges by X%" scenarios.

## Top Findings

- **Care-load forecasts are highly accurate**: the best model is right to within about **1%** of
  the actual number of children in care, 1 to 14 days out. This is accurate enough to drive
  staffing and shelter-capacity planning with real confidence.
- **Placement/discharge forecasts are directionally useful but noisier** (accurate to within
  roughly 30%, day to day). Discharge timing depends heavily on individual sponsor-vetting
  completions, which are inherently harder to predict from historical patterns alone. Treat these
  forecasts as a planning range, not an exact daily count.
- **A simple "pressure" signal works as an early-warning indicator**: when children being
  transferred into HHS care consistently outpace children being discharged, care load has
  historically continued to build in the following days/weeks. The dashboard tracks this signal
  directly.
- **The program's baseline load has fallen roughly 4x** since early 2023 (from ~11,000 to ~2,400
  children in care), punctuated by two sharp surge periods (winter 2023-24 and Jan-Feb 2025) tied
  to intake spikes rather than gradual drift. Capacity thresholds should be revisited periodically
  against the current baseline, not a fixed historical number.

## Why This Matters: Early-Warning Value

Because the forecasting models use only information available *as of today* to predict what
happens next, the same system that was tested on historical data can be run operationally, right
now, to project the coming 1-2 weeks. In the historical test window, the model's relative early-
warning signal for a building surge gave roughly **10 days of advance notice** before load rose
meaningfully above its recent baseline -- time that could be used to pre-position shelter capacity,
schedule additional caseworkers, or accelerate sponsor vetting, rather than reacting once beds are
already full.

## Recommendations

1. **Use the care-load forecast as a standing input to weekly capacity-planning meetings.** Its
   accuracy is strong enough to support staffing and bed-count decisions.
2. **Watch the net-flow "pressure" signal** as a leading indicator of building stress, even before
   the care-load number itself moves.
3. **Use discharge-demand forecasts as a planning range**, not a precise target -- pair the point
   forecast with its uncertainty band when scheduling placement/casework staff.
4. **Revisit capacity thresholds periodically.** The program's typical load has changed
   substantially over the past three years; a "surge" threshold set in 2023 is not meaningful in
   today's operating environment.
5. **Re-validate the models after any future surge event** so the system keeps learning from the
   highest-stakes periods, not just calm ones.

## Expected Impact

- **Earlier action, not reaction**: shifting from finding out about a surge after beds fill up to
  anticipating it roughly a week or more in advance.
- **Better-targeted staffing**: medical and casework surge staff can be scheduled ahead of
  anticipated demand rather than pulled in after the fact.
- **More defensible planning decisions**: every forecast comes with a validated accuracy track
  record and an honest uncertainty range, not just a single number.

## A Note on the Data (Why You Can Trust These Numbers)

HHS's own reporting is not a strict daily feed -- it follows an approximate Sunday-through-Thursday
publishing schedule, so roughly a third of calendar days in the historical record are genuinely
never reported (not lost or corrupted, just not part of the reporting cycle). This project
identified that pattern explicitly and built the entire forecasting pipeline around the program's
*actual* reporting rhythm, rather than guessing at values for days that were never reported. Every
accuracy figure in this summary comes from testing the models only on time periods they had never
seen before -- the same standard a forecast would be held to if deployed for real operational use
today.
