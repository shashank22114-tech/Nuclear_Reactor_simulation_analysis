# ==============================================================================
# FASTAPI BACKEND: NUCLEAR REACTOR CONTROL & PREDICTIVE MAINTENANCE
# ==============================================================================

import asyncio
import io
from contextlib import asynccontextmanager
from collections import deque

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 1. CORE PHYSICS & NEURAL NETWORK DEFINITIONS
# ============================================================

class Config:
    DT = 0.01
    POWER_TARGET = 1.00
    MIN_POWER = 0.50
    HARD_LIMIT = 2.00
    DEVICE = torch.device("cpu")

cfg = Config()

# DL preprocessing is loaded from reactor_lstm_scaler.pkl,
# which is the exact StandardScaler used during training.

class ReactorPhysics:
    """6-Group Point Kinetics Engine for live RL control."""
    def __init__(self):
        self.beta = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
        self.lam = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
        self.beta_total = np.sum(self.beta)
        self.Lambda = 1e-4
        self.alpha_fuel = -1.5e-5
        self.alpha_cool = -3.0e-5
        self.ha_nominal = 3.0
        self.Cp_f = 20.0
        self.Cp_c = 40.0
        self.reset()

    def reset(self):
        self.n = 1.0
        self.C = (self.beta / (self.lam * self.Lambda)) * self.n
        self.Tf = self.n
        self.Tc = 0.8 * self.n
        self.scram = False
        return self.get_state()

    def step(self, action_rod, action_flow):
        if self.scram:
            action_rod = 1.0

        rho_control = action_rod * self.beta_total * 0.10
        rho_feedback = self.alpha_fuel * (self.Tf - 1.0) + self.alpha_cool * (self.Tc - 0.8)
        rho = rho_control + rho_feedback

        dndt = ((rho - self.beta_total) / self.Lambda) * self.n + np.sum(self.lam * self.C)
        dCdt = (self.beta / self.Lambda) * self.n - self.lam * self.C

        flow = np.clip(action_flow, 0.0, 1.5)
        hA = self.ha_nominal * (flow ** 0.8)

        P_gen = self.n
        Q_transfer = hA * (self.Tf - self.Tc)
        Q_sink = max(0.1, 2.0 * flow * (self.Tc - 0.5))

        self.n = np.clip(self.n + dndt * cfg.DT, 0.0, 1e5)
        self.C += dCdt * cfg.DT
        self.Tf += (P_gen - Q_transfer) / self.Cp_f * cfg.DT
        self.Tc += (Q_transfer - Q_sink) / self.Cp_c * cfg.DT

        if self.n > cfg.HARD_LIMIT or self.Tf > 2.2 or self.n < cfg.MIN_POWER:
            self.scram = True

        return self.get_state()

    def get_state(self):
        obs = np.array([self.n, np.mean(self.C), self.Tf, self.Tc], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

class RunningMeanStd:
    """Online normalization strictly for the PPO RL Agent."""
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        if np.max(np.abs(x)) > 50:
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / tot_count
        ) / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.actor_mu = nn.Linear(128, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 1.0)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return torch.tanh(self.actor_mu(features))

class ReactorLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ============================================================
# 2. FASTAPI LIFESPAN & STATE MANAGEMENT
# ============================================================

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading PyTorch Models...")

    # --------------------------------------------------------
    # RL MODEL - UNCHANGED
    # --------------------------------------------------------
    rl_agent = ActorCritic().to(cfg.DEVICE)
    try:
        rl_state_dict = torch.load(
            "ppo_reactor_agent.pth",
            map_location=cfg.DEVICE,
            weights_only=True
        )
        rl_agent.load_state_dict(
            rl_state_dict,
            strict=False
        )
        rl_agent.eval()
        models["rl_agent"] = rl_agent
        print("[+] RL Agent Loaded Successfully.")
    except Exception as e:
        print(f"[!] Warning: Failed to load RL Agent: {e}")

    # --------------------------------------------------------
    # DL MODEL
    # --------------------------------------------------------
    dl_model = ReactorLSTM().to(cfg.DEVICE)
    try:
        dl_state_dict = torch.load(
            "reactor_lstm_model.pth",
            map_location=cfg.DEVICE,
            weights_only=True
        )
        dl_model.load_state_dict(dl_state_dict)
        dl_model.eval()
        models["dl_model"] = dl_model
        print("[+] DL Model Loaded Successfully.")
    except Exception as e:
        print(f"[!] Warning: Failed to load DL Model: {e}")

    # --------------------------------------------------------
    # EXACT TRAINING SCALER
    # --------------------------------------------------------
    try:
        dl_scaler = joblib.load(
            "reactor_lstm_scaler.pkl"
        )

        if not hasattr(dl_scaler, "mean_") or not hasattr(dl_scaler, "scale_"):
            raise ValueError(
                "Loaded scaler is not a valid StandardScaler object."
            )

        if len(dl_scaler.mean_) != 6 or len(dl_scaler.scale_) != 6:
            raise ValueError(
                f"Expected scaler with 6 features, got {len(dl_scaler.mean_)}."
            )

        models["dl_scaler"] = dl_scaler
        print("[+] DL Scaler Loaded Successfully.")
    except Exception as e:
        print(f"[!] Warning: Failed to load DL Scaler: {e}")

    yield

    models.clear()
    print("Server shutting down, models cleared.")

app = FastAPI(
    title="Nuclear Reactor Core API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3. DL HISTORICAL TELEMETRY ENDPOINT
# ============================================================

@app.post("/api/simulate-dl")
async def simulate_dl(file: UploadFile = File(...)):
    if "dl_model" not in models:
        raise HTTPException(
            status_code=503,
            detail="DL Model not loaded in backend."
        )

    if "dl_scaler" not in models:
        raise HTTPException(
            status_code=503,
            detail="DL Scaler not loaded in backend."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(
            io.BytesIO(contents)
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file uploaded."
        )

    req_cols = [
        "timestamp_sec",
        "power_n",
        "avg_precursors",
        "temp_fuel",
        "temp_coolant",
        "action_rod",
        "action_flow"
    ]

    missing_cols = [
        col
        for col in req_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing_cols}"
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=req_cols)
        .reset_index(drop=True)
    )

    feature_cols = [
        "power_n",
        "avg_precursors",
        "temp_fuel",
        "temp_coolant",
        "action_rod",
        "action_flow"
    ]

    seq_len = 50
    step_size = 10

    if len(df) < seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset too short. Minimum {seq_len} rows required."
        )

    raw_features = df[
        feature_cols
    ].values.astype(np.float32)

    # Use the exact StandardScaler fitted during training.
    try:
        scaled_features = models[
            "dl_scaler"
        ].transform(
            raw_features
        ).astype(np.float32)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to scale telemetry using "
                f"the trained DL scaler: {e}"
            )
        )

    # Match the training pipeline exactly:
    # range(0, len(df) - seq_len, step_size)
    windows = []
    endpoint_indices = []

    for start_idx in range(
        0,
        len(scaled_features) - seq_len,
        step_size
    ):
        end_idx = start_idx + seq_len

        windows.append(
            scaled_features[
                start_idx:end_idx
            ]
        )

        endpoint_indices.append(
            end_idx - 1
        )

    if not windows:
        raise HTTPException(
            status_code=400,
            detail="No valid sequence windows could be generated."
        )

    X_array = np.asarray(
        windows,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Batched inference
    # --------------------------------------------------------
    probs_list = []
    batch_size = 256
    dl_model = models["dl_model"]

    with torch.no_grad():
        for batch_start in range(
            0,
            len(X_array),
            batch_size
        ):
            batch_end = min(
                batch_start + batch_size,
                len(X_array)
            )

            X_tensor = torch.from_numpy(
                X_array[
                    batch_start:batch_end
                ]
            ).to(cfg.DEVICE)

            outputs = dl_model(
                X_tensor
            )

            batch_probs = (
                torch.softmax(
                    outputs,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

            probs_list.extend(
                batch_probs.tolist()
            )

    probs = np.asarray(
        probs_list,
        dtype=np.float32
    )

    preds = (
        probs > 0.5
    ).astype(np.int64)

    # --------------------------------------------------------
    # Prepare frontend response
    # --------------------------------------------------------
    frontend_data = []

    for prediction_idx, orig_idx in enumerate(
        endpoint_indices
    ):
        row = df.iloc[orig_idx]

        actual_scram = None
        actual_physical_scram = None

        if "scram_status" in df.columns:
            try:
                actual_scram = int(
                    row["scram_status"]
                )
            except (TypeError, ValueError):
                actual_scram = None

        if "scram_active" in df.columns:
            try:
                actual_physical_scram = int(
                    row["scram_active"]
                )
            except (TypeError, ValueError):
                actual_physical_scram = None

        frontend_data.append(
            {
                "time": float(
                    row["timestamp_sec"]
                ),
                "power": float(
                    row["power_n"]
                ),
                "avg_precursors": float(
                    row["avg_precursors"]
                ),
                "temp_fuel": float(
                    row["temp_fuel"]
                ),
                "temp_coolant": float(
                    row["temp_coolant"]
                ),
                "rod_position": float(
                    row["action_rod"]
                ),
                "coolant_flow": float(
                    row["action_flow"]
                ),
                "scram_prob": float(
                    probs[prediction_idx]
                ),
                "prediction": int(
                    preds[prediction_idx]
                ),
                "alert": (
                    "CRITICAL"
                    if preds[prediction_idx] == 1
                    else "NORMAL"
                ),
                "actual_scram": actual_scram,
                "actual_physical_scram": actual_physical_scram
            }
        )

    return {
        "status": "success",
        "data": frontend_data,
        "metadata": {
            "total_rows": int(
                len(df)
            ),
            "sequence_length": seq_len,
            "step_size": step_size,
            "predictions_generated": int(
                len(frontend_data)
            ),
            "normal_predictions": int(
                np.sum(preds == 0)
            ),
            "scram_predictions": int(
                np.sum(preds == 1)
            )
        }
    }

# ============================================================
# 4. LIVE RL SIMULATION WEBSOCKET
# ============================================================
# RL MODEL / PHYSICS / CONTROL LOGIC PRESERVED.
# Only the DL scaler used for the 50-step DL prediction buffer
# has been changed from hard-coded statistics to the exact scaler.
# ============================================================

@app.websocket("/ws/simulate")
async def websocket_simulator(websocket: WebSocket):
    await websocket.accept()

    if (
        "rl_agent" not in models
        or "dl_model" not in models
        or "dl_scaler" not in models
    ):
        await websocket.send_json(
            {
                "error": (
                    "Models or DL scaler not loaded. "
                    "Ensure ppo_reactor_agent.pth, "
                    "reactor_lstm_model.pth, and "
                    "reactor_lstm_scaler.pkl "
                    "are in the directory."
                )
            }
        )
        await websocket.close()
        return

    physics = ReactorPhysics()
    normalizer = RunningMeanStd(
        shape=(4,)
    )

    obs_raw = physics.reset()

    normalizer.update(
        obs_raw
    )

    obs = normalizer.normalize(
        obs_raw
    )

    time_step = 0.0

    control_mode = "AUTO"
    manual_rods = 0.0
    manual_flow = 0.2
    target_power_demand = 1.0

    feature_buffer = deque(
        maxlen=50
    )

    async def receive_commands():
        nonlocal control_mode, manual_rods, manual_flow, target_power_demand

        try:
            while True:
                data = await websocket.receive_json()

                if data.get(
                    "type"
                ) == "RESUME_AUTO":
                    control_mode = "AUTO"

                elif data.get(
                    "type"
                ) == "MANUAL_OVERRIDE":
                    param = data.get(
                        "parameter"
                    )
                    val = data.get(
                        "value"
                    )

                    control_mode = "MANUAL"

                    if param == "control_rods":
                        manual_rods = (
                            1.0 - (val / 50.0)
                        )

                    elif param == "coolant_flow":
                        manual_flow = float(
                            val
                        )

                    elif param == "target_power":
                        target_power_demand = (
                            float(val) / 100.0
                        )

        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    listener_task = asyncio.create_task(
        receive_commands()
    )

    try:
        while True:
            if control_mode == "AUTO":
                obs_t = (
                    torch.FloatTensor(obs)
                    .to(cfg.DEVICE)
                    .unsqueeze(0)
                )

                with torch.no_grad():
                    action_mu = models[
                        "rl_agent"
                    ](
                        obs_t
                    )

                action_np = (
                    action_mu
                    .cpu()
                    .numpy()[0]
                )

                act_rod = np.clip(
                    action_np[0],
                    -1.0,
                    1.0
                )

                act_flow = (
                    np.clip(
                        action_np[1],
                        -1.0,
                        1.0
                    )
                    + 0.5
                )

            else:
                act_rod = manual_rods
                act_flow = manual_flow

            current_features = np.array(
                [
                    physics.n,
                    np.mean(physics.C),
                    physics.Tf,
                    physics.Tc,
                    act_rod,
                    act_flow
                ],
                dtype=np.float32
            )

            feature_buffer.append(
                current_features
            )

            scram_prob = 0.0
            ai_alert = "NORMAL"

            if len(feature_buffer) == 50:
                raw_window = np.array(
                    feature_buffer
                )

                # Exact scaler from LSTM training.
                scaled_window = (
                    models["dl_scaler"]
                    .transform(
                        raw_window
                    )
                    .astype(np.float32)
                )

                X_tensor = (
                    torch.FloatTensor(
                        scaled_window
                    )
                    .unsqueeze(0)
                    .to(cfg.DEVICE)
                )

                with torch.no_grad():
                    dl_outputs = models[
                        "dl_model"
                    ](
                        X_tensor
                    )

                    scram_prob = float(
                        torch.softmax(
                            dl_outputs,
                            dim=1
                        )[0, 1]
                        .cpu()
                        .numpy()
                    )

                if scram_prob > 0.85:
                    ai_alert = (
                        "WARNING: HIGH SCRAM RISK"
                    )
                elif scram_prob > 0.50:
                    ai_alert = (
                        "CAUTION: ANOMALY DETECTED"
                    )

            manual_alert = None

            if control_mode == "MANUAL":
                if (
                    physics.n > 1.8
                    or physics.Tf > 2.0
                ):
                    manual_alert = (
                        "MANUAL OVERRIDE WARNING: "
                        "APPROACHING MELTDOWN!"
                    )
                elif physics.n < 0.6:
                    manual_alert = (
                        "MANUAL OVERRIDE WARNING: "
                        "REACTOR STALLING!"
                    )

            next_obs_raw = physics.step(
                act_rod,
                act_flow
            )

            if (
                np.isfinite(
                    next_obs_raw
                ).all()
                and np.max(
                    np.abs(
                        next_obs_raw
                    )
                ) < 100
            ):
                normalizer.update(
                    next_obs_raw
                )

            obs = normalizer.normalize(
                next_obs_raw
            )

            time_step += cfg.DT

            telemetry = {
                "time": round(
                    time_step,
                    2
                ),
                "power": float(
                    physics.n
                ),
                "temp_fuel": float(
                    physics.Tf
                ),
                "temp_coolant": float(
                    physics.Tc
                ),
                "rod_position": float(
                    act_rod
                ),
                "coolant_flow": float(
                    act_flow
                ),
                "scram_prob": scram_prob,
                "alert": (
                    "CRITICAL MELTDOWN"
                    if physics.scram
                    else ai_alert
                ),
                "manual_alert": manual_alert,
                "control_mode": control_mode
            }

            await websocket.send_json(
                telemetry
            )

            await asyncio.sleep(
                0.05
            )

            if physics.scram:
                telemetry["alert"] = (
                    "CRITICAL MELTDOWN"
                )

                await websocket.send_json(
                    telemetry
                )

                break

    except WebSocketDisconnect:
        print(
            "Frontend Client Disconnected."
        )
    finally:
        listener_task.cancel()