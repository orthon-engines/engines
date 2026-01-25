"""
PRISM API - Read-only access to computed results.

Full computation runs via CLI:
    python -m prism.entry_points.compute

This API just serves what's already been computed.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
import io

app = FastAPI(title="PRISM", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_data_dir() -> Path:
    """Get the PRISM data directory."""
    from prism.db.parquet_store import get_path, OBSERVATIONS
    return Path(get_path(OBSERVATIONS)).parent


@app.get("/health")
async def health():
    from prism import __version__
    return {"status": "ok", "version": __version__}


@app.post("/analyze")
async def analyze():
    """Run the PRISM pipeline."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "prism.entry_points.analyze"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {"status": "error", "message": result.stderr}

    return {"status": "ok", "output": result.stdout}


@app.get("/files")
async def list_files():
    """List available parquet files."""
    data_dir = _get_data_dir()
    files = {}
    for f in ['observations', 'vector', 'geometry', 'dynamics', 'physics']:
        path = data_dir / f"{f}.parquet"
        files[f] = path.exists()
    return files


@app.get("/read/{filename}")
async def read_file(filename: str):
    """Read a parquet file and return as JSON."""
    import polars as pl

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    df = pl.read_parquet(path)
    return {
        "rows": len(df),
        "columns": df.columns,
        "data": df.head(100).to_dicts(),  # First 100 rows
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


def main():
    import uvicorn
    uvicorn.run("prism.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
