"""
Orthon Engines — Compute API
Upload CSV → Run pipeline → Download parquet atlas
"""
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI(title="Orthon Engines", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ─────────────────────────────────────────────
DATA_DIR = Path(os.getenv("ORTHON_DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_MODE = os.getenv("ORTHON_DEV", "true").lower() == "true"

# In-memory stores (swap for Redis/Postgres in production)
API_KEYS = {}  # { key: { email, tier, created, runs } }
PENDING = {}   # { email: { code, expires, attempts } }


# ─── Helpers ────────────────────────────────────────────

def is_academic(email: str) -> bool:
    academic = [".edu", ".ac.uk", ".ac.jp", ".edu.au", ".ac.in",
                ".edu.cn", ".ac.kr", ".edu.br", ".ac.nz", ".edu.sg"]
    return any(email.lower().endswith(d) for d in academic)


def make_key() -> str:
    return f"orth_{uuid.uuid4().hex[:24]}"


def make_code() -> str:
    import random
    return str(random.randint(100000, 999999))


# ─── CSV Loader ─────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """Auto-detect CSV format → long-format observations."""
    df = pd.read_csv(path)

    # Already long format?
    if {"I", "signal_id", "value"}.issubset(df.columns):
        obs = df[["I", "signal_id", "value"]].copy()
        obs["cohort"] = df["cohort"] if "cohort" in df.columns else "default"
        return obs

    # Try common column name variations
    renames = {}
    lower = {c.lower().strip(): c for c in df.columns}
    for c in ["i", "index", "time", "timestamp", "t", "step", "sample"]:
        if c in lower:
            renames[lower[c]] = "I"
            break
    for c in ["signal_id", "signal", "sensor", "channel", "variable"]:
        if c in lower:
            renames[lower[c]] = "signal_id"
            break
    for c in ["value", "val", "measurement", "reading"]:
        if c in lower:
            renames[lower[c]] = "value"
            break

    if renames:
        df2 = df.rename(columns=renames)
        if {"I", "signal_id", "value"}.issubset(df2.columns):
            obs = df2[["I", "signal_id", "value"]].copy()
            obs["cohort"] = df2["cohort"] if "cohort" in df2.columns else "default"
            return obs

    # Assume wide format: col 0 = index, rest = signals
    idx_col = df.columns[0]

    # Find cohort column
    coh_col = None
    for c in df.columns:
        if c.lower().strip() in ("cohort", "group", "unit", "run", "trial"):
            coh_col = c
            break

    sig_cols = [c for c in df.columns
                if c != idx_col and c != coh_col
                and pd.api.types.is_numeric_dtype(df[c])]

    if not sig_cols:
        raise ValueError("No numeric signal columns found in CSV")

    id_vars = [idx_col] + ([coh_col] if coh_col else [])
    long = df.melt(id_vars=id_vars, value_vars=sig_cols,
                   var_name="signal_id", value_name="value")
    long = long.rename(columns={idx_col: "I"})
    if coh_col:
        long = long.rename(columns={coh_col: "cohort"})
    else:
        long["cohort"] = "default"

    long["I"] = pd.to_numeric(long["I"], errors="coerce")
    long = long.dropna(subset=["I", "value"])
    long["I"] = long["I"].astype(int)

    return long[["I", "signal_id", "value", "cohort"]].sort_values(
        ["cohort", "signal_id", "I"]).reset_index(drop=True)


# ─── Pipeline (placeholder — wire real engines here) ────

def run_pipeline(obs: pd.DataFrame, out: Path) -> dict:
    """
    Run Orthon pipeline. Currently placeholder outputs.

    TODO: Replace with actual engine imports:
        from engines.manifold import eigendecomp, lyapunov, granger, breaks
    """
    signals = obs["signal_id"].unique().tolist()
    cohorts = obs["cohort"].unique().tolist()
    n_sig = len(signals)

    # ── observations.parquet ──
    obs.to_parquet(out / "observations.parquet", index=False)

    # ── signal_vector.parquet (windowed means — placeholder) ──
    sv_rows = []
    for coh in cohorts:
        cd = obs[obs["cohort"] == coh]
        max_I = int(cd["I"].max())
        win = max(16, min(64, max_I // 15))
        step = max(1, win // 2)
        for start in range(0, max(1, max_I - win + 1), step):
            row = {"cohort": coh, "I": start + win // 2,
                   "window_start": start, "window_end": start + win}
            for sig in signals:
                sd = cd[(cd["signal_id"] == sig) &
                        (cd["I"] >= start) & (cd["I"] < start + win)]["value"]
                row[sig] = float(sd.mean()) if len(sd) > 0 else 0.0
            sv_rows.append(row)

    if sv_rows:
        sv = pd.DataFrame(sv_rows)
        sv.to_parquet(out / "signal_vector.parquet", index=False)

        # ── state_geometry.parquet (eigendecomp placeholder) ──
        geo_rows = []
        for coh in cohorts:
            cv = sv[sv["cohort"] == coh]
            for _, r in cv.iterrows():
                vals = [r[s] for s in signals if s in r.index]
                if len(vals) >= 2:
                    # Actual eigendecomp would go here
                    # For now: compute on real data when engines are wired
                    n = min(len(vals), 5)
                    fake_eig = sorted(np.random.exponential(1.0, n), reverse=True)
                    total = sum(fake_eig)
                    norm = [e / total for e in fake_eig]
                    eff_dim = float(np.exp(-sum(p * np.log(p + 1e-12) for p in norm)))

                    geo_rows.append({
                        "cohort": coh, "I": int(r["I"]),
                        "window_start": int(r["window_start"]),
                        "window_end": int(r["window_end"]),
                        **{f"eigenvalue_{i+1}": float(fake_eig[i])
                           for i in range(min(5, len(fake_eig)))},
                        "effective_dim": eff_dim,
                        "total_variance": total,
                    })

        if geo_rows:
            pd.DataFrame(geo_rows).to_parquet(out / "state_geometry.parquet", index=False)

    # ── statistics.parquet ──
    stat_rows = []
    for coh in cohorts:
        for sig in signals:
            sd = obs[(obs["cohort"] == coh) & (obs["signal_id"] == sig)]["value"]
            stat_rows.append({
                "cohort": coh, "signal_id": sig,
                "n_samples": len(sd),
                "mean": float(sd.mean()) if len(sd) > 0 else 0,
                "std": float(sd.std()) if len(sd) > 1 else 0,
                "min": float(sd.min()) if len(sd) > 0 else 0,
                "max": float(sd.max()) if len(sd) > 0 else 0,
            })
    pd.DataFrame(stat_rows).to_parquet(out / "statistics.parquet", index=False)

    files = [f.name for f in out.glob("*.parquet")]
    return {
        "n_signals": n_sig,
        "n_cohorts": len(cohorts),
        "n_samples": len(obs),
        "files_generated": len(files),
        "signals": signals[:20],
        "cohorts": cohorts[:10],
        "files": files,
    }


# ─── Routes: Registration ──────────────────────────────

@app.post("/v1/register")
async def register(email: str = Query(...)):
    code = make_code()
    PENDING[email] = {
        "code": code,
        "expires": datetime.now() + timedelta(minutes=10),
        "attempts": 0,
    }
    # TODO: wire SendGrid / SES to actually email the code
    resp = {"message": f"Code sent to {email}"}
    if DEV_MODE:
        resp["dev_code"] = code
    return resp


@app.post("/v1/verify")
async def verify(email: str = Query(...), code: str = Query(...)):
    p = PENDING.get(email)
    if not p:
        raise HTTPException(400, "No pending registration")
    if datetime.now() > p["expires"]:
        del PENDING[email]
        raise HTTPException(400, "Code expired")
    p["attempts"] += 1
    if p["attempts"] > 5:
        del PENDING[email]
        raise HTTPException(429, "Too many attempts")
    if p["code"] != code:
        raise HTTPException(400, f"Wrong code. {5 - p['attempts']} left.")

    tier = "academic" if is_academic(email) else "trial"
    key = make_key()
    API_KEYS[key] = {"email": email, "tier": tier,
                     "created": datetime.now().isoformat(), "runs": 0}
    del PENDING[email]
    return {"api_key": key, "tier": tier,
            "message": "Unlimited. Citation required." if tier == "academic"
            else "Trial: 10 runs."}


# ─── Routes: Atlas ──────────────────────────────────────

@app.post("/v1/atlas")
async def create_atlas(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    # Auth
    if not authorization:
        raise HTTPException(401, "API key required")
    key = authorization.replace("Bearer ", "").strip()
    kd = API_KEYS.get(key)
    if not kd:
        raise HTTPException(401, "Invalid API key")
    if kd["tier"] == "trial" and kd.get("runs", 0) >= 10:
        raise HTTPException(429, "Trial limit reached")

    # Job setup
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save upload
    upload_path = job_dir / "upload.csv"
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    # Run
    try:
        obs = load_csv(upload_path)
        summary = run_pipeline(obs, job_dir)
    except Exception as e:
        shutil.rmtree(job_dir)
        raise HTTPException(400, f"Pipeline error: {e}")

    kd["runs"] = kd.get("runs", 0) + 1

    # Zip parquets
    shutil.make_archive(str(job_dir / "atlas"), "zip", str(job_dir))

    # Cleanup later
    background_tasks.add_task(cleanup_job, job_dir)

    return {
        "job_id": job_id,
        "summary": summary,
        "download": f"/v1/atlas/{job_id}/download",
        "files": {f: f"/v1/atlas/{job_id}/file/{f}"
                  for f in summary["files"]},
    }


@app.get("/v1/atlas/{job_id}/download")
async def download(job_id: str):
    zp = OUTPUT_DIR / job_id / "atlas.zip"
    if not zp.exists():
        raise HTTPException(404, "Expired or not found")
    return FileResponse(zp, media_type="application/zip",
                        filename=f"orthon_{job_id}.zip")


@app.get("/v1/atlas/{job_id}/file/{name}")
async def download_file(job_id: str, name: str):
    fp = OUTPUT_DIR / job_id / name
    if not fp.exists() or not name.endswith(".parquet"):
        raise HTTPException(404, "Not found")
    return FileResponse(fp, media_type="application/octet-stream", filename=name)


async def cleanup_job(job_dir: Path, hours: int = 24):
    await asyncio.sleep(hours * 3600)
    if job_dir.exists():
        shutil.rmtree(job_dir)


@app.get("/health")
async def health():
    return {"status": "ok", "v": "0.1.0"}


# ─── Serve static ──────────────────────────────────────

STATIC = Path("./static")

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
