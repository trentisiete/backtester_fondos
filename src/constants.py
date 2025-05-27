# src/constants.py

import pandas as pd

# --- Constantes y Configuraciones ---
RISK_FREE_RATE_DEFAULT = 0.0
CONFIDENCE_LEVEL_VAR_DEFAULT = 0.99 # 99% para VaR/ES
HISTORICAL_SCENARIOS = {
    "Ninguno": None,
    "Crisis Financiera Global (Sep 2008 - Mar 2009)": (pd.to_datetime("2008-09-01"), pd.to_datetime("2009-03-31")),
    "Lunes Negro (Oct 1987)": (pd.to_datetime("1987-10-01"), pd.to_datetime("1987-10-31")),
    "Burbuja .com (Mar 2000 - Sep 2001)": (pd.to_datetime("2000-03-01"), pd.to_datetime("2001-09-30")),
    "COVID-19 Crash (Feb 2020 - Mar 2020)": (pd.to_datetime("2020-02-15"), pd.to_datetime("2020-03-31")),
    "Crisis Deuda Europea (May 2010 - Oct 2011)": (pd.to_datetime("2010-05-01"), pd.to_datetime("2011-10-31")),
}