"""Central configuration for the HHS UAC care-load forecasting project."""
from pathlib import Path

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV_PATH = DATA_DIR / "HHS_Unaccompanied_Alien_Children_Program.csv"
CLEAN_CSV_PATH = DATA_DIR / "hhs_uac_cleaned.csv"
BUSINESS_DAY_CSV_PATH = DATA_DIR / "hhs_uac_business_day.csv"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw column names as they appear in the source CSV
COL_DATE = "Date"
COL_APPREHENDED = "Children apprehended and placed in CBP custody*"
COL_CBP_CUSTODY = "Children in CBP custody"
COL_TRANSFERRED = "Children transferred out of CBP custody"
COL_HHS_CARE = "Children in HHS Care"
COL_DISCHARGED = "Children discharged from HHS Care"

NUMERIC_COLS = [
    COL_APPREHENDED,
    COL_CBP_CUSTODY,
    COL_TRANSFERRED,
    COL_HHS_CARE,
    COL_DISCHARGED,
]

# Forecasting targets
TARGET_CARE_LOAD = COL_HHS_CARE          # primary target
TARGET_DISCHARGE = COL_DISCHARGED         # secondary target
NET_FLOW_COL = "net_flow_pressure"        # derived: transferred_in - discharged

# Frequency strategy: empirically the source reports on a Sun-Thu cadence
# (Sunday 130x, Mon/Tue/Wed/Thu ~145-149x each, Friday only 2x, Saturday 0x
# across 720 rows). Standard pandas business-day freq ("B" = Mon-Fri) would
# misalign this -- it would slot in Fridays that are essentially never
# reported and silently drop genuine Sunday observations. We instead define
# a custom reporting-week frequency via CustomBusinessDay with weekmask
# "Sun Mon Tue Wed Thu", which matches the true cadence and reindexes
# without fabricating or discarding real data. Only short internal gaps
# (<= MAX_GAP_INTERPOLATE reporting days, e.g. a holiday closure) are
# interpolated; longer gaps are left as NaN.
REPORTING_WEEKMASK = "Sun Mon Tue Wed Thu"
REINDEX_FREQ = "CUSTOM_REPORTING_DAY"  # constructed at runtime via CustomBusinessDay
MAX_GAP_INTERPOLATE = 3  # reporting days; longer gaps are left as NaN, not invented

# Capacity threshold used for KPI computation (illustrative operational ceiling)
CAPACITY_THRESHOLD = 9000

# Multi-horizon evaluation steps (in reporting-cadence observations)
FORECAST_HORIZONS = [1, 7, 14]

TEST_SIZE_FRACTION = 0.2  # fraction of observations held out for final test
N_WALK_FORWARD_FOLDS = 5
