"""Metadron Capital — Decimal-based financial arithmetic.

All monetary values (NAV, cash, prices, P&L, costs, premiums, notional)
MUST use these helpers instead of raw float to prevent precision drift.

Usage:
    from engine.utils.money import D, money, to_float, round_money, safe_div

    nav = D("1000000.00")
    cost = money(qty * price)      # Converts float result to Decimal
    pnl = money(proceeds - cost)
    shares = int(safe_div(dollar_amount, price))
    result = to_float(nav)         # Convert back for ML/numpy interop
"""

from decimal import Decimal, ROUND_HALF_UP, getcontext, InvalidOperation

# Set global precision — 28 significant digits (default) is sufficient
# for portfolio values up to $999,999,999,999.99 with sub-cent precision.
getcontext().prec = 28

# Quantize targets
TWO_PLACES = Decimal("0.01")       # Dollars and cents
FOUR_PLACES = Decimal("0.0001")    # Basis points / percentages
SIX_PLACES = Decimal("0.000001")   # Position sizes (fractional)


def D(value) -> Decimal:
    """Convert any value to Decimal safely.

    Handles: str, int, float, Decimal, None.
    Float is converted via string to avoid float→Decimal precision artifacts.
    """
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def money(value) -> Decimal:
    """Convert to Decimal and round to 2 decimal places (dollars + cents)."""
    return D(value).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def money4(value) -> Decimal:
    """Convert to Decimal and round to 4 decimal places (basis points)."""
    return D(value).quantize(FOUR_PLACES, rounding=ROUND_HALF_UP)


def money6(value) -> Decimal:
    """Convert to Decimal and round to 6 decimal places (position sizes)."""
    return D(value).quantize(SIX_PLACES, rounding=ROUND_HALF_UP)


def round_money(value, places: int = 2) -> Decimal:
    """Round a Decimal to N places."""
    quantizer = Decimal(10) ** -places
    return D(value).quantize(quantizer, rounding=ROUND_HALF_UP)


def to_float(value) -> float:
    """Convert Decimal back to float for numpy/ML interop."""
    if isinstance(value, Decimal):
        return float(value)
    return float(value) if value is not None else 0.0


def safe_div(numerator, denominator, default=Decimal("0")) -> Decimal:
    """Safe division — returns default if denominator is zero."""
    num = D(numerator)
    den = D(denominator)
    if den == 0:
        return D(default)
    return num / den
