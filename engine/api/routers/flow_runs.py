import logging
from typing import List

logger = logging.getLogger("metadron.api.flow_runs")

try:
    from fastapi import APIRouter, Depends, HTTPException
    from sqlalchemy.orm import Session
except ImportError:
    raise RuntimeError("FastAPI and SQLAlchemy are required.")

from engine.db.database import get_db
from engine.db.tables import FlowRun, Trade
from engine.db.schemas import FlowRunResponse, TradeResponse

router = APIRouter()


@router.get("", response_model=List[FlowRunResponse])
def list_flow_runs(
    skip: int = 0, limit: int = 50, db: Session = Depends(get_db)
):
    """Return a paginated list of flow runs, most recent first."""
    return (
        db.query(FlowRun)
        .order_by(FlowRun.started_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{run_id}", response_model=FlowRunResponse)
def get_flow_run(run_id: int, db: Session = Depends(get_db)):
    """Retrieve a single flow run by ID."""
    run = db.query(FlowRun).filter(FlowRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Flow run not found")
    return run


@router.get("/{run_id}/trades", response_model=List[TradeResponse])
def get_run_trades(run_id: int, db: Session = Depends(get_db)):
    """Return all trades associated with a specific flow run."""
    run = db.query(FlowRun).filter(FlowRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Flow run not found")
    return (
        db.query(Trade)
        .filter(Trade.flow_run_id == run_id)
        .order_by(Trade.timestamp.desc())
        .all()
    )
