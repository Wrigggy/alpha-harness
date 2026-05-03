# Data Directory

This repository does not ship raw data or processed panels.

Expected local structure:

```text
data/
  raw_equity/
  equity_panel/
  governance/
  factors/
```

You can rebuild data locally with the included pipelines:

- China A-shares:
  - `scripts/fetch_csi500_constituents.py`
  - `scripts/fetch_equity_baostock.py`
  - `scripts/fetch_equity_akshare.py`
  - `scripts/build_csi500_baostock_panel.py`
- US equities:
  - `scripts/fetch_us_symbol_universe.py`
  - `scripts/fetch_equity_us_yfinance.py`
  - `scripts/import_equity_panel.py`
  - `scripts/build_us_universe_masks.py`

No local data files are required to inspect the codebase, but they are required to run research workflows.
