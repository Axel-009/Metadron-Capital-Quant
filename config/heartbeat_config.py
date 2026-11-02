# Metadron Capital — Heartbeat Configuration
# Adaptive scan cadence for best execution across 1,044+ securities

# ┌─────────────────────────────────────────────────────────────┐
# │                   MARKET DAY TIMELINE                       │
# │                                                             │
# │  08:00  Pre-market    30-min scans (earnings, news gaps)   │
# │  09:30  Market Open   1-min burst × 5 (catch the open)     │
# │  09:35  Active Scan   3-min cadence (full universe)        │
# │  11:30  Midday        5-min cadence (lower vol)            │
# │  14:00  Afternoon     3-min cadence (closing positioning)  │
# │  15:45  Power Hour    1-min burst × 15 (closing auction)   │
# │  16:00  Market Close  5-min scans (rebalancing)            │
# │  16:30  After Hours   30-min scans (earnings reactions)    │
# │  20:00  Overnight     Run backtest, retrain models         │
# └─────────────────────────────────────────────────────────────┘

# Phase cadences (seconds)
HEARTBEAT_INTERVAL = 180           # Base: 3 minutes during active hours
SIGNAL_CADENCE = 180               # Full signal pipeline every 3 min
INTELLIGENCE_CADENCE = 300         # Alpha optimizer, ML ensemble every 5 min
MONITORING_CADENCE = 300           # P&L, risk, anomaly every 5 min

# Burst modes (seconds)
OPEN_BURST_INTERVAL = 60           # 1-min for first 5 minutes after open
OPEN_BURST_COUNT = 5               # 5 rapid scans at open
CLOSE_BURST_INTERVAL = 60          # 1-min for last 15 minutes
CLOSE_BURST_COUNT = 15             # 15 rapid scans into close

# Pre/after market (seconds)
PRE_MARKET_INTERVAL = 1800         # 30 minutes
AFTER_HOURS_INTERVAL = 1800        # 30 minutes
OVERNIGHT_INTERVAL = 3600          # 1 hour (backtesting + model retrain)

# Market hours (ET)
PRE_MARKET_START = "08:00"
MARKET_OPEN = "09:30"
MIDDAY_SLOWDOWN = "11:30"
AFTERNOON_ACTIVE = "14:00"
POWER_HOUR = "15:45"
MARKET_CLOSE = "16:00"
AFTER_HOURS_END = "20:00"

# Universe scan strategy
# Alpaca batch API: 1,000 symbols per call, ~2-5s response
# Strategy: scan ALL tickers every heartbeat, not staggered
# Rationale: stale signals = missed opportunities
SCAN_ALL_TICKERS = True            # Scan full universe every heartbeat
BATCH_SIZE = 1000                  # Alpaca API batch limit
QUOTE_CACHE_TTL = 5                # Seconds before re-fetching quotes

# Best execution considerations
# - Scan frequency should exceed average holding period / 10
# - If avg hold = 30 min, scan every 3 min minimum
# - If avg hold = 2 hours, scan every 12 min minimum
# - Current target: intraday (avg hold 30-120 min) → 3-5 min cadence
MIN_SCAN_INTERVAL = 60             # Never scan more than 1/min (API limits)
MAX_SCAN_INTERVAL = 900            # Never scan less than 15/min (missed opps)
