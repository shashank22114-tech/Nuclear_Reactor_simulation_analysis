# Nuclear Reactor Simulation, Reinforcement Learning & Predictive SCRAM Analysis

A physics-based nuclear reactor simulation and AI analysis platform integrating **point-kinetics modeling, PPO reinforcement learning, LSTM-based predictive SCRAM diagnostics, FastAPI, and React + Three.js visualization**.

> **Academic / Simulation Project:** This project is intended for simulation, experimentation, and AI research. It is not designed for deployment on real nuclear reactor control systems.

## Overview

The project provides two integrated AI-driven modes.

### Live Simulation — PPO Reinforcement Learning

A simulated reactor environment uses a six-group point-kinetics model and a trained **Proximal Policy Optimization (PPO)** agent for autonomous control.

The RL system observes reactor state and controls:

* Control rod position
* Primary coolant flow

The simulation continuously models reactor power, delayed neutron precursors, fuel temperature, coolant temperature, reactivity feedback, heat transfer, and SCRAM conditions.

### DL Diagnostics — LSTM Predictive SCRAM Detection

The DL subsystem analyzes historical reactor telemetry stored in CSV files.

A trained **Long Short-Term Memory (LSTM)** network processes 50-timestep sequences and predicts the probability of an impending SCRAM condition.

The six model inputs are:

```text
power_n
avg_precursors
temp_fuel
temp_coolant
action_rod
action_flow
```

The results are visualized through an interactive reactor playback interface.

## System Architecture

```text
                    NUCLEAR REACTOR AI PLATFORM
                               |
          +--------------------+--------------------+
          |                                         |
     LIVE SIMULATION                           DL DIAGNOSTICS
          |                                         |
   Reactor Physics Engine                     Historical CSV
          |                                         |
      PPO RL Agent                            50-Step LSTM
          |                                         |
 Control Rod / Coolant Flow                 SCRAM Probability
          |                                         |
          +--------------------+--------------------+
                               |
                         FastAPI Backend
                               |
                     REST API + WebSocket
                               |
                        React + Three.js
                               |
                    Interactive 3D Dashboard
```

## Reactor Physics Model

The simulation uses a reduced-order **six-group point-kinetics model** containing:

* Neutron population / normalized reactor power
* Six delayed neutron precursor groups
* Fuel temperature
* Coolant temperature
* Reactivity feedback
* Heat transfer and coolant heat removal
* SCRAM safety limits

The model is intended as a computational simulation environment for AI experimentation.

## Reinforcement Learning

The PPO controller observes:

```text
[n, avg_precursors, Tf, Tc]
```

and produces continuous control actions for:

```text
Control rod position
Coolant flow
```

The frontend provides both:

* **AUTO (PPO AGENT)**
* **MANUAL OVERRIDE**

Real-time communication is handled through a FastAPI WebSocket endpoint.

## Deep Learning Model

The predictive diagnostics model uses a two-layer LSTM:

```text
Input: 50 timesteps × 6 features
        ↓
LSTM
Hidden Size: 64
Layers: 2
Dropout: 0.3
        ↓
Linear: 64 → 32
        ↓
ReLU
        ↓
Linear: 32 → 2
        ↓
Normal / SCRAM
```

The final LSTM timestep is used for classification.

## Synthetic Dataset

The dataset is generated from the reactor simulation and contains four operating scenarios:

```text
steady_state
rod_transient
startup
scram_event
```

Telemetry includes:

```text
timestamp_sec
power_n
avg_precursors
temp_fuel
temp_coolant
action_rod
action_flow
scram_active
scram_status
scenario_label
```

The predictive target is designed to identify an impending SCRAM condition rather than only classify an already-triggered event.

## Training Methodology

To reduce validation leakage, the data is split at the **CSV-file level before sequence generation**.

```text
Total CSV files:        60
Training files:         48
Validation files:       12
```

Each scenario is represented in both partitions.

### Sequence Configuration

```text
Sequence length: 50
Step size:       10
Features:        6
```

### Feature Scaling

A `StandardScaler` is fitted using training data only and saved as:

```text
reactor_lstm_scaler.pkl
```

The same scaler is used during backend inference.

## Model Performance

Evaluation was performed on held-out synthetic telemetry files.

| Metric              |   Result |
| ------------------- | -------: |
| Validation Accuracy |   99.81% |
| ROC-AUC             | 0.999976 |
| SCRAM Precision     |   99.91% |
| SCRAM Recall        |   99.07% |
| SCRAM F1 Score      |   99.49% |

### Confusion Matrix

```text
                 Predicted
               Normal   SCRAM

Actual Normal    9670      2
Actual SCRAM       21   2247
```

The model detected **2247 of 2268 SCRAM validation windows**.

> These results are based on synthetic simulation data and should not be interpreted as real-world nuclear reactor safety performance.

## Backend

The backend is implemented using **FastAPI**.

### Historical DL Analysis

```text
POST /api/simulate-dl
```

The endpoint:

1. Accepts a telemetry CSV
2. Validates the input
3. Applies the saved training scaler
4. Generates LSTM sequence windows
5. Performs batched inference
6. Returns telemetry and SCRAM predictions

### Live Simulation

```text
WS /ws/simulate
```

The WebSocket provides:

* Reactor telemetry
* PPO control actions
* DL SCRAM probability
* Safety status
* Manual control interaction

## Frontend

The frontend uses:

* React
* Three.js
* Vite
* OrbitControls
* CSS2DRenderer
* Lucide React

The interface provides:

### Live Simulation

* Interactive 3D reactor model
* PPO autonomous control
* Manual control mode
* Real-time telemetry
* SCRAM probability
* Reactor safety status

### DL Diagnostics

* CSV upload
* LSTM inference
* Historical playback
* SCRAM probability visualization
* Normal / SCRAM classification
* Telemetry snapshots

## Project Structure

```text
Nuclear_Reactor_simulation_analysis/
│
├── main.py
├── ppo_reactor_agent.pth
├── reactor_lstm_model.pth
├── reactor_lstm_scaler.pkl
├── reactor_lstm_metadata.json
├── package.json
├── package-lock.json
├── .gitignore
│
└── reactor-frontend/
    ├── package.json
    ├── package-lock.json
    ├── index.html
    ├── vite.config.js
    ├── public/
    └── src/
        ├── App.jsx
        ├── App.css
        ├── index.css
        └── main.jsx
```

## Installation

### Backend

Install the required Python packages:

```bash
pip install fastapi uvicorn torch numpy pandas scikit-learn joblib python-multipart
```

### Frontend

```bash
cd reactor-frontend
npm install
```

## Running the Project

### Start the Backend

From the project root:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Start the Frontend

In another terminal:

```bash
cd reactor-frontend
npm run dev
```

Open the Vite URL shown in the terminal, typically:

```text
http://localhost:5173
```

## Usage

### Live Simulation

1. Start the FastAPI backend.
2. Start the React frontend.
3. Select **LIVE SIMULATION**.
4. Click **INITIALIZE SIMULATION**.
5. Run the PPO agent in AUTO mode or use MANUAL OVERRIDE.

### DL Diagnostics

1. Start the backend and frontend.
2. Select **DL DIAGNOSTICS**.
3. Upload a compatible telemetry CSV.
4. Click **ANALYZE TELEMETRY**.
5. Use the playback timeline to inspect predictions and reactor state.

## Limitations

This project uses a simplified reactor model and synthetic data.

It does not represent:

* A full spatial reactor-core model
* Fuel depletion or burnup
* Xenon / iodine poisoning
* Detailed neutron transport
* Structural mechanics
* Radiation transport
* Real reactor instrumentation
* Certified reactor protection systems
* Real-world nuclear plant control protocols

High validation performance on synthetic telemetry does not establish performance on real reactor data.

## Future Work

Potential extensions include:

* More detailed reactor dynamics
* Additional transient and failure scenarios
* Detection lead-time evaluation
* Model uncertainty estimation
* Explainable AI for SCRAM predictions
* Comparison of PPO with other RL algorithms
* Advanced telemetry analytics
* Automated experiment tracking
* Containerized deployment

## Technologies

```text
Python
PyTorch
LSTM
PPO
NumPy
Pandas
Scikit-learn
Joblib
FastAPI
WebSockets
React
Vite
Three.js
Lucide React
```

## Disclaimer

This software is intended for **educational, research, and simulation purposes only**. It must not be used as a substitute for certified nuclear reactor safety systems, engineering analysis, operational procedures, or regulatory processes.

## Author

**N Shashank Reddy**

Computer Science Engineering Capstone Project
