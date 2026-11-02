"""
Metadron Capital — API routes for Flow management.

Endpoints
---------
POST /flows          — create a new flow
GET  /flows          — list all flows
GET  /flows/{id}     — retrieve a single flow
POST /flows/{id}/run — trigger an execution run for a flow
"""

import logging
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger("metadron.api.flows")

try:
    from fastapi import APIRouter, Depends, HTTPException
    from sqlalchemy.orm import Session
except ImportError:
    raise RuntimeError("FastAPI and SQLAlchemy are required.")

from engine.db.database import get_db
from engine.db.tables import Flow, FlowRun
from engine.db.schemas import (
    FlowCreate,
    FlowResponse,
    FlowRunCreate,
    FlowRunResponse,
)

router = APIRouter()


@router.post("", response_model=FlowResponse, status_code=201)
def create_flow(payload: FlowCreate, db: Session = Depends(get_db)):
    """Create a new orchestration flow."""
    flow = Flow(name=payload.name, description=payload.description)
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


@router.get("", response_model=List[FlowResponse])
def list_flows(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Return a paginated list of flows."""
    return db.query(Flow).offset(skip).limit(limit).all()


@router.get("/{flow_id}", response_model=FlowResponse)
def get_flow(flow_id: int, db: Session = Depends(get_db)):
    """Retrieve a single flow by ID."""
    flow = db.query(Flow).filter(Flow.id == flow_id).first()
    if flow is None:
        raise HTTPException(status_code=404, detail="Flow not found")
    return flow


@router.post("/{flow_id}/run", response_model=FlowRunResponse, status_code=201)
def execute_flow_run(
    flow_id: int,
    payload: FlowRunCreate = FlowRunCreate(),
    db: Session = Depends(get_db),
):
    """Trigger a new execution run for the specified flow.

    The run is created in *pending* status; a background worker or
    the caller is expected to advance it through its lifecycle.
    """
    flow = db.query(Flow).filter(Flow.id == flow_id).first()
    if flow is None:
        raise HTTPException(status_code=404, detail="Flow not found")

    run = FlowRun(
        flow_id=flow_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info("Flow run %s started for flow %s", run.id, flow_id)
    return run
