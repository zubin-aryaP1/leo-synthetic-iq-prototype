# LEO Synthetic I/Q Lab

A standalone, **hardware-free** proof of concept for the UW/Daintree LEO satellite observatory. It runs on Windows, Linux, or macOS and does not need UHD, an NI USRP, a dish, an LNBF, or internet access (after dependencies are installed).

The prototype uses independently implemented signal-processing algorithms for demonstration. All RF samples and satellite candidate labels are synthetic. It does **not** use real CelesTrak/TLE/SGP4 predictions and does not identify real satellites.

## Run on Windows (Python 3.12)

Unzip the project, open Git Bash or PowerShell inside the `leo-synthetic-prototype` folder, then:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8008
```

Open **http://127.0.0.1:8008**, choose a scenario, and click **Start simulation**. Leave the terminal open. The demo runs only when started, and a reset creates a new persisted run.

`start_windows.bat` starts the server once dependencies are installed. Run on **127.0.0.1**, not a public interface: this proof of concept has no authentication or CSRF controls.

### What you should see

1. A moving synthetic carrier (~+8 kHz drifting toward −7 kHz over the pass), a stationary +17 kHz interferer, and complex Gaussian noise enter the same FFT detector.
2. The CA-CFAR-like detector finds strong spectral peaks, including the moving carrier.
3. A simple global assignment / constant-velocity Kalman tracker estimates each confirmed peak's frequency and drift rate.
4. A synthetic Doppler matcher compares track observations with three **made-up** reference curves (`SIM-LEO-A/B/C`). It fits a constant receiver frequency offset and reports RMSE and a qualitative match state. The fixed interferer should not be accepted as a satellite because the offset bound rejects its fit.
5. The dashboard plots the current spectrum, moving waterfall, observed-vs-predicted frequency, and tracked signals.
6. Confirmed track observations are asynchronously written to SQLite at `data/demo.sqlite3`. The API exposes them at `/api/history`.

Choose `dropout` to simulate a short signal disappearance; `interference` for a stronger competing tone; or `noisy` for a more difficult detection setting.

**IMPORTANT:** The frequency reference is **baseband frequency offset (Hz)** from a fictitious IF receiver center at 2.575 GHz. This is not an actual 12.325 GHz downlink, and the Doppler templates are not orbital predictions.

## Optional: write fake SigMF recordings and replay them

```bash
python -m app.cli generate --seconds 12 --scenario clean --out recordings/synthetic
python -m app.cli replay recordings/synthetic.sigmf-data
```

This creates `synthetic.sigmf-data` (complex64 `cf32_le`) and `synthetic.sigmf-meta` with explicit `SYNTHETIC` labels. At 100 kS/s, 12 seconds uses ~9.6 MB. The export tool caps duration at 120 seconds (~96 MB) to avoid filling a small SSD. **Do not mistake the output for a real RF capture.**

The SigMF data may be useful to exercise a compatible offline input path in the UW application, but do not expect the UW production tracker settings (which are tuned for real Starlink signal motion and higher sample rates) to detect this synthetic tone without adapting its detector configuration.

## Architecture and replaceable boundaries

```text
SyntheticIQ (replaces UHD during software testing)
     | complex64 epochs, 100 kS/s
     v
FFT + approximate CA-CFAR (app/processing.py)
     | frequency detections
     v
Global assignment + Kalman (app/processing.py)
     | measured frequency tracks
     +------> synthetic RMSE matcher (app/matching.py)
                    |
                    v
             DemoEngine (app/engine.py)
                    |
          +---------+---------------+
          |                         |
     FastAPI + WebSocket      bounded asyncio.Queue
       (app/main.py)                 |
          |                    SQLite writer
     HTML/JS dashboard        (app/storage.py)
```

- `app/simulator.py`: implements synthetic baseband sample acquisition. A real UHD adapter could replace this source later.
- `app/processing.py`: intentionally simplified detector and tracker for rapid prototyping.
- `app/matching.py`: fabricated Doppler-shaped trajectories, **not** SGP4.
- `app/engine.py`: orchestration independent of API/database.
- `app/storage.py`: `ObservationRepository` protocol and an SQLite adapter. A future PostgreSQL implementation can implement the same contract.
- `app/main.py`: FastAPI app with live WebSocket fanout, API control, and bounded asynchronous persistence queue.
- `static/index.html`: no-build, offline-friendly dashboard using native Canvas.
- `app/cli.py`: create/replay synthetic `cf32_le` SigMF.

### API

| Route | Purpose |
| --- | --- |
| `GET /` | Live dashboard |
| `GET /api/status` | Latest processing snapshot, run info, DB counters |
| `POST /api/control` | `{ "action": "start" \| "stop" \| "reset", "scenario": "clean" }` |
| `GET /api/history?limit=300` | Persisted observations for current run |
| `GET /api/scenarios` | Available scenarios |
| `WS /ws/live` | Live JSON snapshots |
| `GET /docs` | FastAPI interactive docs |

### Why SQLite and not MongoDB for this experiment?

For this exercise, every persistent observation has a stable run ID, track ID, elapsed timestamp, frequency and SNR. Relational rows are straightforward. Raw IQ lives in `.sigmf-data`, not a database BLOB. For a multi-station platform, implement the repository with PostgreSQL and use a separate object-storage interface for recordings.

### Limitations (deliberate)

- No real RF hardware, no actual satellite identification, no SGP4, and no validated detector performance. CA-CFAR thresholds and Kalman parameters are demonstrations, **not** field-calibrated research defaults.
- Simulated epoch time is separate from actual UTC acquisition timestamps. A real backend must retain sample timing, device timestamps, frequency/LO calibration, and overflow gaps.
- Browser broadcast uses per-connection snapshot polling rather than a scaled multi-instance message bus.
- The persistence queue is **in-memory and bounded**: a slow disk can cause dropped batches, which are reported. It does not guarantee crash-safe delivery. Production should use durable local spooling or a broker, retry logic, and idempotent writes.
- Single process / single machine; no user authentication, export controls, station permissions or cloud deployment. **Not production ready.**
- SNR is a relative local-spectrum estimate only; cellular RSRP, RSRQ and SINR are not calculated.

## Test

```bash
python -m pip install pytest httpx
python -m pytest -q
```

The automated tests exercise signal generation, detector/tracker/matcher behavior, SQLite persistence and the FastAPI control endpoints. No network or USRP is needed.

## Suggested next integration

Keep UW's `capture/service.py`, `satellite_tracker_api.py`, `processing/service.py`, prediction modules, and React dashboard as the production baseline. Use this independent sandbox to test interfaces and data shapes, then integrate the **persistent observation repository** and source/replay abstractions into a feature branch of UW's existing application. Avoid copying this prototype's research-grade-unverified tracker over UW's working implementation.

### Dashboard connection recovery

The dashboard uses WebSockets when available and automatically falls back to HTTP updates when the live connection is unavailable or stale. Controls time out after eight seconds and report failures beside the buttons. If upgrading an existing installation, rerun `python -m pip install -r requirements.txt` to install WebSocket support, restart the server, and refresh the browser.
