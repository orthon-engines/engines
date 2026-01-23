"""
Ørthon Server
geometry leads — ørthon

FastAPI backend for the Ørthon Explorer.
- Serves folder browser
- Detects parquet files
- Serves parquets to browser
- Runs pipeline on raw uploads
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
import shutil
import tempfile

app = FastAPI(title="Ørthon Server", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected parquet files
EXPECTED_FILES = ["vector.parquet", "geometry.parquet", "state.parquet", "cohort.parquet", "derivatives.parquet"]

# Default starting directory (user's home)
DEFAULT_ROOT = Path.home()


# =============================================================================
# FOLDER BROWSING
# =============================================================================

@app.get("/api/folders")
async def list_folders(path: str = "~") -> Dict[str, Any]:
    """
    List folders and parquet files in a directory.
    Returns folder tree for GUI navigation.
    """
    # Expand ~ to home directory
    if path == "~" or path == "":
        folder = DEFAULT_ROOT
    else:
        folder = Path(path).expanduser()
    
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
    
    # List contents
    folders = []
    parquet_files = []
    
    try:
        for item in sorted(folder.iterdir()):
            # Skip hidden files
            if item.name.startswith('.'):
                continue
            
            if item.is_dir():
                folders.append({
                    "name": item.name,
                    "path": str(item),
                })
            elif item.suffix == '.parquet':
                parquet_files.append({
                    "name": item.name,
                    "path": str(item),
                    "size": item.stat().st_size,
                    "expected": item.name in EXPECTED_FILES,
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
    
    # Check which expected files are present
    found_files = {f["name"] for f in parquet_files}
    expected_status = {
        name: name in found_files for name in EXPECTED_FILES
    }
    
    # Check for config
    config_path = folder / "orthon_config.json"
    config = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except:
            pass
    
    return {
        "current_path": str(folder),
        "parent_path": str(folder.parent) if folder != folder.parent else None,
        "folders": folders,
        "parquet_files": parquet_files,
        "expected_files": expected_status,
        "ready": len(parquet_files) > 0,  # Enable if ANY parquet file exists in folder
        "config": config,
    }


@app.get("/api/parquet/{file_path:path}")
async def get_parquet(file_path: str):
    """
    Serve a parquet file.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    if path.suffix != '.parquet':
        raise HTTPException(status_code=400, detail="Not a parquet file")
    
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=path.name
    )


@app.get("/api/config/{folder_path:path}")
async def get_config(folder_path: str) -> Dict[str, Any]:
    """
    Get or generate config for a results folder.
    Reads window options from parquet metadata if available.
    """
    folder = Path(folder_path)
    config_path = folder / "orthon_config.json"
    
    # Return existing config if present
    if config_path.exists():
        return json.loads(config_path.read_text())
    
    # Generate default config
    config = {
        "domain": folder.name,
        "windows": [252, 126, 63],  # Default options
        "stride": 21,
        "entity_col": "entity_id",
        "time_col": "timestamp",
        "generated_at": datetime.now().isoformat(),
    }
    
    # Try to read window info from vector.parquet
    vector_path = folder / "vector.parquet"
    if vector_path.exists():
        try:
            import polars as pl
            df = pl.read_parquet(vector_path)
            
            # Detect entity column
            for col in ["entity_id", "unit_id", "unit", "id"]:
                if col in df.columns:
                    config["entity_col"] = col
                    break
            
            # Detect time column
            for col in ["timestamp", "cycle", "time", "window_end"]:
                if col in df.columns:
                    config["time_col"] = col
                    break
            
            # Get unique window sizes if present
            if "window_size" in df.columns:
                windows = df["window_size"].unique().sort().to_list()
                if windows:
                    config["windows"] = windows
        except Exception as e:
            print(f"Warning: Could not read vector.parquet: {e}")
    
    return config


@app.post("/api/config/{folder_path:path}")
async def save_config(folder_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save config to a results folder.
    """
    folder = Path(folder_path)
    config_path = folder / "orthon_config.json"
    
    config["saved_at"] = datetime.now().isoformat()
    config_path.write_text(json.dumps(config, indent=2))
    
    return {"status": "saved", "path": str(config_path)}


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

@app.post("/api/run")
async def run_pipeline(
    file: UploadFile = File(...),
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Run Ørthon pipeline on uploaded file.
    Returns path to results folder.
    """
    # Save uploaded file to temp location
    temp_dir = Path(tempfile.mkdtemp(prefix="orthon_"))
    input_path = temp_dir / file.filename
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    output_dir = temp_dir / "results"
    output_dir.mkdir()
    
    try:
        # Import and run pipeline
        from prism.entry_points.orchestrator import run
        
        result = run(
            input_path=str(input_path),
            output_dir=str(output_dir),
            entity_col=entity_col,
            time_col=time_col,
        )
        
        # Generate config
        config = {
            "domain": input_path.stem,
            "windows": [252, 126, 63],
            "stride": 21,
            "entity_col": entity_col,
            "time_col": time_col,
            "generated_at": datetime.now().isoformat(),
            "input_file": file.filename,
        }
        
        config_path = output_dir / "orthon_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        
        return {
            "status": "success",
            "results_path": str(output_dir),
            "files": [f.name for f in output_dir.glob("*.parquet")],
            "config": config,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCHEMA INFO
# =============================================================================

@app.get("/api/schema/{file_path:path}")
async def get_schema(file_path: str) -> Dict[str, Any]:
    """
    Get schema and row count for a parquet file.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found")
    
    try:
        import polars as pl
        df = pl.read_parquet(path)
        
        return {
            "file": path.name,
            "rows": len(df),
            "columns": [
                {"name": col, "dtype": str(df[col].dtype)}
                for col in df.columns
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STATIC FILES (serve the frontend)
# =============================================================================

# Serve index.html at root
@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {"message": "Ørthon Server running. Frontend not found at ./index.html"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Ørthon Server")
    print("  geometry leads — ørthon")
    print("=" * 60)
    print()
    print("  Open http://localhost:8000 in your browser")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
