"""
PRISM API - Compute interface for ORTHON.

ORTHON commands → PRISM computes → ORTHON SQL

Endpoints:
    POST /compute     - Run computation (synchronous, blocks until complete)
    GET  /health      - Status check
    GET  /files       - List available parquet files
    GET  /read        - Read parquet as JSON (query param: path)
    GET  /disciplines - List available disciplines
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import io
import json
import subprocess
import sys
import os

app = FastAPI(title="PRISM", version="0.3.0", description="Compute engine for ORTHON")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class ComputeRequest(BaseModel):
    """Request from ORTHON to run computation.

    ORTHON sends:
        config: dict with 'discipline' key and computation parameters
        observations_path: path to observations.parquet
    """
    config: Dict[str, Any]
    observations_path: str


class ComputeResponse(BaseModel):
    """Response after computation completes.

    PRISM returns:
        status: 'complete' or 'error'
        results_path: directory containing output parquets
        parquets: list of parquet filenames created
        duration_seconds: computation time
        message: error message if status='error'
        hint: helpful hint for fixing errors
        engine: which engine failed (if error)
    """
    status: str  # 'complete' or 'error'
    results_path: Optional[str] = None
    parquets: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    message: Optional[str] = None
    hint: Optional[str] = None
    engine: Optional[str] = None


class JobStatus(BaseModel):
    """Status of a compute job."""
    job_id: str
    status: str  # pending, running, completed, failed
    discipline: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# State (in-memory for now, could be Redis/DB)
# =============================================================================

_jobs: Dict[str, JobStatus] = {}


# =============================================================================
# Helpers
# =============================================================================

def _get_data_dir() -> Path:
    """Get the PRISM data directory."""
    return Path(os.path.expanduser("~/prism-mac/data"))


def _get_inbox_dir() -> Path:
    """Get the PRISM inbox directory."""
    return Path(os.path.expanduser("~/prism-inbox"))


def _generate_job_id() -> str:
    """Generate a unique job ID."""
    import uuid
    return str(uuid.uuid4())[:8]


def _run_compute_sync(config: Dict, observations_path: str) -> ComputeResponse:
    """Run computation synchronously. Returns when complete."""
    start_time = time.time()

    discipline = config.get("discipline")
    # Discipline is optional - core engines run regardless

    output_dir = _get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write config to YAML for PRISM to read
        import yaml
        config_path = output_dir / "config.yaml"

        # Build config with required fields + user overrides
        prism_config = {
            "discipline": discipline,
            "observations_path": observations_path,
            # Required window config (use defaults if not provided)
            "window": config.get("window", {"size": 100, "stride": 50}),
            "min_samples": config.get("min_samples", 50),
            # Pass through other config
            **{k: v for k, v in config.items() if k not in ["discipline", "window", "min_samples"]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(prism_config, f)

        # Copy observations to data dir if not already there
        obs_path = Path(observations_path)
        target_obs = output_dir / "observations.parquet"
        if obs_path.exists() and obs_path != target_obs:
            import shutil
            shutil.copy(obs_path, target_obs)

        # Run PRISM compute (runs all layers: vector, geometry, dynamics, physics)
        cmd = [
            sys.executable, "-m", "prism.entry_points.compute",
            "--force",  # Always recompute for API calls
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "PRISM_DATA_DIR": str(output_dir)},
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            # Parse error for helpful hints
            error_msg = result.stderr or result.stdout or "Unknown error"
            hint = None
            engine = None

            # Try to extract engine name from error
            if "Missing required" in error_msg:
                hint = "Check config has all required constants for this discipline"
            if "viscosity" in error_msg.lower():
                hint = "Add to config: global_constants.viscosity_Pa_s = 0.001"

            return ComputeResponse(
                status="error",
                message=error_msg[:500],  # Truncate long errors
                hint=hint,
                engine=engine,
                duration_seconds=round(duration, 2),
            )

        # Find created parquet files
        parquets = [f.name for f in output_dir.glob("*.parquet")]

        # Write completion status
        status_file = output_dir / "job_status.json"
        with open(status_file, 'w') as f:
            json.dump({
                "status": "complete",
                "discipline": discipline,
                "timestamp": datetime.now().isoformat(),
            }, f)

        return ComputeResponse(
            status="complete",
            results_path=str(output_dir),
            parquets=parquets,
            duration_seconds=round(duration, 2),
        )

    except Exception as e:
        return ComputeResponse(
            status="error",
            message=str(e),
            duration_seconds=round(time.time() - start_time, 2),
        )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    from prism import __version__
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/disciplines")
async def list_disciplines():
    """List available disciplines."""
    from prism.disciplines import DISCIPLINES
    return {
        "disciplines": list(DISCIPLINES.keys()),
        "details": {k: v.get("name", k) for k, v in DISCIPLINES.items()},
    }


@app.get("/disciplines/{discipline}")
async def get_discipline(discipline: str):
    """Get discipline requirements and engines."""
    from prism.disciplines import DISCIPLINES
    from prism.disciplines.requirements import get_requirements_text, check_requirements

    if discipline not in DISCIPLINES:
        raise HTTPException(404, f"Unknown discipline: {discipline}")

    return {
        "discipline": discipline,
        "info": DISCIPLINES[discipline],
        "requirements_text": get_requirements_text(discipline),
    }


@app.post("/compute", response_model=ComputeResponse)
async def compute(request: ComputeRequest):
    """
    Run PRISM computation (synchronous).

    ORTHON sends:
        config: dict with 'discipline' and parameters
        observations_path: path to observations.parquet

    PRISM returns when complete:
        status: 'complete' or 'error'
        results_path: directory with output parquets
        parquets: list of created files
        duration_seconds: how long it took

    Example request:
        {
            "config": {
                "discipline": "reaction",
                "entities": ["run_1", "run_2"],
                "global_constants": {"reactor_volume_L": 2.5}
            },
            "observations_path": "/path/to/observations.parquet"
        }
    """
    from prism.disciplines import DISCIPLINES

    discipline = request.config.get("discipline")

    # Validate discipline
    if discipline and discipline not in DISCIPLINES:
        return ComputeResponse(
            status="error",
            message=f"Unknown discipline: {discipline}",
            hint=f"Available: {', '.join(DISCIPLINES.keys())}",
        )

    # Run synchronously (blocks until complete)
    return _run_compute_sync(request.config, request.observations_path)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    return _jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": list(_jobs.values())}


@app.get("/files")
async def list_files():
    """List available parquet files."""
    data_dir = _get_data_dir()
    files = {}
    for f in ['observations', 'data', 'vector', 'geometry', 'dynamics', 'physics']:
        path = data_dir / f"{f}.parquet"
        if path.exists():
            files[f] = {
                "exists": True,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        else:
            files[f] = {"exists": False}
    return files


@app.get("/read/{filename}")
async def read_file_by_name(filename: str, limit: int = 100, offset: int = 0):
    """Read a parquet file by name and return as JSON."""
    import polars as pl

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    df = pl.read_parquet(path)
    return {
        "file": filename,
        "total_rows": len(df),
        "columns": df.columns,
        "offset": offset,
        "limit": limit,
        "data": df.slice(offset, limit).to_dicts(),
    }


@app.get("/read")
async def read_file_by_path(path: str = Query(..., description="Path to parquet file"), limit: int = 100, offset: int = 0):
    """Read a parquet file by full path and return as JSON.

    ORTHON uses this to read results after compute completes.
    """
    import polars as pl
    from pathlib import Path as P

    file_path = P(path)
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")

    if not path.endswith('.parquet'):
        raise HTTPException(400, "Only .parquet files supported")

    df = pl.read_parquet(file_path)
    return {
        "file": file_path.name,
        "path": str(file_path),
        "total_rows": len(df),
        "columns": df.columns,
        "offset": offset,
        "limit": limit,
        "data": df.slice(offset, limit).to_dicts(),
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a parquet file."""
    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return StreamingResponse(
        io.BytesIO(path.read_bytes()),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/trigger-github")
async def trigger_github(request: ComputeRequest):
    """
    Trigger GitHub Actions workflow (alternative to local compute).

    Requires GITHUB_TOKEN env var.
    """
    import httpx

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise HTTPException(500, "GITHUB_TOKEN not configured")

    discipline = request.config.get("discipline", "reaction")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.github.com/repos/prism-engines/prism/dispatches",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "event_type": "compute",
                "client_payload": {
                    "discipline": discipline,
                    "config": request.config,
                    "observations_path": request.observations_path,
                }
            }
        )

    if response.status_code != 204:
        raise HTTPException(response.status_code, f"GitHub API error: {response.text}")

    return {"status": "triggered", "message": "GitHub Actions workflow dispatched"}


# =============================================================================
# CLI
# =============================================================================

def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="PRISM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print("=" * 60)
    print("PRISM API - Compute Engine for ORTHON")
    print("=" * 60)
    print(f"Server:  http://{args.host}:{args.port}")
    print(f"Docs:    http://{args.host}:{args.port}/docs")
    print(f"Health:  http://{args.host}:{args.port}/health")
    print("=" * 60)
    print("\nORTHON connects to: http://localhost:8100")
    print("POST /compute with {config, observations_path}\n")

    uvicorn.run(
        "prism.entry_points.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
