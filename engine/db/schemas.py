"""
Metadron Capital — Pydantic schemas for flows, runs, trades, portfolio, and API keys.
"""

from datetime import datetime
from typing import Any, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise RuntimeError("Pydantic is required. Install it with: pip install pydantic")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
class FlowCreate(BaseModel):
    """Payload for creating a new flow."""

    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None


class FlowResponse(BaseModel):
    """Serialised representation of a Flow."""

    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    status: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# FlowRun
# ---------------------------------------------------------------------------
class FlowRunCreate(BaseModel):
    """Payload for triggering a flow run."""

    parameters: Optional[dict[str, Any]] = None


class FlowRunResponse(BaseModel):
    """Serialised representation of a FlowRun."""

    id: int
    flow_id: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    result_json: Optional[str] = None
    status: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------
class TradeResponse(BaseModel):
    """Serialised representation of a Trade."""

    id: int
    flow_run_id: int
    ticker: str
    side: str
    quantity: float
    price: float
    signal_type: Optional[str] = None
    timestamp: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class PortfolioResponse(BaseModel):
    """Serialised representation of a Portfolio snapshot."""

    id: int
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    timestamp: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------
class ApiKeyCreate(BaseModel):
    """Payload for creating a new API key."""

    name: str = Field(..., min_length=1, max_length=128)
