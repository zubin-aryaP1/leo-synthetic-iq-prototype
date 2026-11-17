"""Browser demo API. Localhost research prototype; intentionally no production auth."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.engine import DemoEngine
from app.simulator import SCENARIOS
from app.storage import SQLiteObservationRepository

ROOT = Path(__file__).resolve().parents[1]


class Control(BaseModel):
    action: str  # start / stop / reset
    scenario: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = SQLiteObservationRepository(os.getenv("LEO_DEMO_DB", str(ROOT / "data" / "demo.sqlite3")))
    engine = DemoEngine()
    repository.create_run(engine.run_id, engine.source.scenario, engine.created_at)
    app.state.engine = engine
    app.state.repository = repository
    app.state.lock = asyncio.Lock()
    app.state.queue = asyncio.Queue(maxsize=256)
    app.state.dropped_batches = 0

    async def persistence():
        while True:
            batch = await app.state.queue.get()
            try:
                await asyncio.to_thread(repository.save_batch, batch)
            finally:
                app.state.queue.task_done()

    async def simulate():
        while True:
            if engine.running:
                async with app.state.lock:
                    _, rows = await asyncio.to_thread(engine.step)
                    if rows:
                        try:
                            app.state.queue.put_nowait(rows)
                        except asyncio.QueueFull:
                            app.state.dropped_batches += 1
            await asyncio.sleep(0.1)

    writer = asyncio.create_task(persistence())
    runner = asyncio.create_task(simulate())
    try:
        yield
    finally:
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
        await app.state.queue.join()
        writer.cancel()
        with suppress(asyncio.CancelledError):
            await writer
        repository.close()


app = FastAPI(title="Synthetic LEO I/Q Prototype", version="0.1.0", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
async def status():
    async with app.state.lock:
        result = dict(app.state.engine.last)
        result["running"] = app.state.engine.running
    result["database"] = await asyncio.to_thread(app.state.repository.counts)
    result["persistence_queue_depth"] = app.state.queue.qsize()
    result["dropped_persistence_batches"] = app.state.dropped_batches
    result["note"] = "All I/Q signals, reference curves and satellite labels are synthetic."
    return result


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": list(SCENARIOS)}


@app.post("/api/control")
async def control(body: Control):
    if body.action not in ("start", "stop", "reset"):
        raise HTTPException(422, "action must be start, stop or reset")
    if body.scenario and body.scenario not in SCENARIOS:
        raise HTTPException(422, f"scenario must be one of {SCENARIOS}")
    async with app.state.lock:
        eng = app.state.engine
        if body.action == "reset":
            eng.running = False
            eng.reset(body.scenario or eng.source.scenario)
            app.state.repository.create_run(eng.run_id, eng.source.scenario, eng.created_at)
        elif body.action == "start":
            eng.running = True
        else:
            eng.running = False
        return {"ok": True, "action": body.action, "run_id": eng.run_id,
                "running": eng.running, "scenario": eng.source.scenario}


@app.get("/api/history")
async def history(limit: int = 300):
    async with app.state.lock:
        run_id = app.state.engine.run_id
    return {"run_id": run_id, "observations": await asyncio.to_thread(app.state.repository.history, run_id, limit)}


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            async with app.state.lock:
                payload = dict(app.state.engine.last)
                payload["running"] = app.state.engine.running
                payload["dropped_persistence_batches"] = app.state.dropped_batches
            await ws.send_json(payload)
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        return

# Recording evaluation is independent of the live simulation session.
from fastapi.responses import StreamingResponse
from app.evaluate import evaluation_events
import csv
import io
import json
import math


def recording_catalog():
    folders = [ROOT / "recordings", Path(os.getenv("LEO_CAPTURE_DIR", str(Path.home() / "Downloads" / "Capture_Scripts")))]
    return {f"{slot}/{p.stem}": p for slot, folder in enumerate(folders)
            for p in sorted(folder.glob("*.sigmf-data")) if p.with_suffix(".sigmf-meta").is_file()}


class EvaluationRequest(BaseModel):
    recording: str
    reference_csv: str | None = None
    tolerance_hz: float = 100.0


@app.get("/recordings")
def recording_page():
    return FileResponse(ROOT / "static" / "recordings.html")


@app.get("/api/recordings")
def list_recordings():
    return [{"id": key, "name": path.name} for key, path in recording_catalog().items()]


@app.post("/api/recordings/evaluate")
def evaluate_recording(body: EvaluationRequest):
    path = recording_catalog().get(body.recording)
    if path is None:
        raise HTTPException(404, "Recording unavailable")
    labels = {}
    if body.reference_csv is not None:
        try:
            for row in csv.DictReader(io.StringIO(body.reference_csv.lstrip("\ufeff"))):
                epoch, freq = int(row["epoch"]), float(row["offset_hz"])
                if epoch < 0 or epoch in labels or not math.isfinite(freq):
                    raise ValueError()
                labels[epoch] = freq
            if not labels:
                raise ValueError()
        except (TypeError, KeyError, ValueError, csv.Error):
            raise HTTPException(422, "Reference CSV needs unique nonnegative epoch values and finite offset_hz values.")

    def stream():
        try:
            for event in evaluation_events(path, labels, body.tolerance_hz):
                yield json.dumps(event) + "\n"
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")
