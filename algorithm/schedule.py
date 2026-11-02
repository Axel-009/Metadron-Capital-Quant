"""Heartbeat — when each part of the book trades, aligned to the day's liquidity.

US equity volume is U-shaped: heaviest in the first hour and the last 30 minutes, with the closing
auction the single deepest print of the day (≈ 10% of volume). The first 15–30 minutes also have the
widest spreads and opening-auction noise. Listed-option spreads are widest at the open, tighten after
the first 30–60 minutes, and are tightest roughly 10:30–11:30 and 14:30–15:30 ET (avoid the last 10–15
minutes, when market makers widen into the close). MES futures trade deep all session, deepest around
the cash open and close.

    time (ET)   minutes after open   job
    09:31       1                    RISK_OPEN    gap check: hard stops, position protection only (no entries)
    09:50       20                   RESEARCH     regime, forward Rm, sector view, theme rotation plan, decisions
    10:00       30                   CORE         MES beta overlay; equities: OPEN_SLICE of each order
    10:15/10:45 45 / 75              OPTIONS AM   select contracts (subscribe), then price + order (limit near mid)
    every 30 m  30 … 360             RISK         daily-loss breaker, CVaR check, protection, stale-data check
    14:15/14:45 285 / 315            OPTIONS PM   second options window (rolls, hedges, LEAPS)
    15:30       360                  EXPIRY       close options ≤ EXPIRY_CLOSE_DTE, beta true-up
    15:45       375                  CLOSE        equities: remaining delta via market-on-close (closing auction)

Research and fundamentals refresh run in universe selection (before the open). Equity orders are split
OPEN_SLICE / (1 − OPEN_SLICE) between the morning window and the closing auction.
"""

RISK_OPEN = 1
RESEARCH = 20
CORE = 30
OPTIONS_SELECT = (45, 285)
OPTIONS_EXECUTE = (75, 315)
RISK_EVERY = 30
RISK_FROM, RISK_TO = 30, 360
EXPIRY = 360
CLOSE = 375                  # 15:45 — before the 15:50 MOC cut-off

OPEN_SLICE = 0.4             # share of each equity order traded in the morning window
OPTION_LIMIT_FRACTION = 0.3  # limit = mid + 0.3 × half-spread (buy) / mid − 0.3 × half-spread (sell)
EXPIRY_CLOSE_DTE = 2


def risk_minutes():
    return list(range(RISK_FROM, RISK_TO + 1, RISK_EVERY))
