import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Thermometer, Zap, AlertTriangle, Sliders,
  UploadCloud, ArrowLeft, ShieldAlert, FileText, Database, CheckCircle
} from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export default function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home', 'rl', 'dl'

  // ==========================================
  // RL MODE STATES & LOGIC
  // ==========================================
  const [telemetry, setTelemetry] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [rlError, setRlError] = useState(null);
  const [mode, setMode] = useState('AUTO');
  const [controlRods, setControlRods] = useState(50);
  const [coolantFlow, setCoolantFlow] = useState(0.2);
  const [targetPower, setTargetPower] = useState(100);
  const wsRef = useRef(null);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/simulate');
    wsRef.current.onopen = () => { setIsConnected(true); setRlError(null); };
    wsRef.current.onmessage = (event) => setTelemetry(JSON.parse(event.data));
    wsRef.current.onerror = () => { setRlError("WebSocket connection failed."); setIsConnected(false); };
    wsRef.current.onclose = () => setIsConnected(false);
  };

  const disconnectWebSocket = () => { if (wsRef.current) wsRef.current.close(); };

  useEffect(() => {
    if (currentView !== 'rl') disconnectWebSocket();
    return () => disconnectWebSocket();
  }, [currentView]);

  // --- NEW AI SYNC LOGIC ---
  // Sync the frontend UI and 3D rods with the AI's movements in AUTO mode
  useEffect(() => {
    if (mode === 'AUTO' && telemetry && telemetry.rod_position !== undefined) {
      // The backend agent outputs rod positions from -1.0 to 1.0
      // We must reverse-map this back to our slider's 0 to 100% scale
      const aiRodPosition = (1.0 - telemetry.rod_position) * 50.0;
      setControlRods(aiRodPosition);

      // Sync the coolant flow slider too so you can watch both move!
      if (telemetry.coolant_flow !== undefined) {
        setCoolantFlow(telemetry.coolant_flow);
      }
    }
  }, [telemetry, mode]);
  // -------------------------

  const sendControlCommand = (param, value) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && mode === 'MANUAL') {
      wsRef.current.send(JSON.stringify({ type: 'MANUAL_OVERRIDE', parameter: param, value: value }));
    }
  };

  // ==========================================
  // DL MODE STATES & LOGIC
  // ==========================================
  const [dlFile, setDlFile] = useState(null);
  const [dlResults, setDlResults] = useState(null);
  const [dlLoading, setDlLoading] = useState(false);
  const [dlError, setDlError] = useState(null);
  const [dlPlaybackIndex, setDlPlaybackIndex] = useState(0);
  const [dlMetadata, setDlMetadata] = useState(null);

  const handleDlUpload = async () => {
    if (!dlFile) return;
    setDlLoading(true);
    setDlError(null);
    setDlResults(null);
    setDlMetadata(null);
    setDlPlaybackIndex(0);

    const formData = new FormData();
    formData.append("file", dlFile);

    try {
      const res = await fetch("http://localhost:8000/api/simulate-dl", {
        method: "POST",
        body: formData
      });

      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Failed to process telemetry data.");

      if (!Array.isArray(json.data) || json.data.length === 0) {
        throw new Error("The backend returned no valid prediction windows.");
      }
      setDlResults(json.data);
      setDlMetadata(json.metadata || null);
    } catch (err) {
      setDlError(err.message || "Failed to process telemetry data.");
    } finally {
      setDlLoading(false);
    }
  };

  const currentDlTelemetry = dlResults?.[Math.min(dlPlaybackIndex, Math.max(dlResults.length - 1, 0))] || null;

  // ==========================================
  // RENDER: HOME HUB
  // ==========================================
  if (currentView === 'home') {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-full bg-slate-950 text-slate-100 font-sans p-8">
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-black tracking-widest text-cyan-400 mb-2 uppercase">Nuclear Reactor Simulation and Analysis</h1>
          <p className="text-slate-400 uppercase tracking-widest text-sm">Next-Generation Reactor Operations & Predictive Diagnostics</p>
        </div>

        <div className="flex gap-8 max-w-4xl w-full">
          {/* RL Entry Card */}
          <button
            onClick={() => setCurrentView('rl')}
            className="flex-1 group flex flex-col items-center p-10 bg-slate-900 border border-slate-800 rounded-2xl hover:border-cyan-500 hover:bg-slate-800 transition-all shadow-xl"
          >
            <div className="p-5 bg-cyan-950 rounded-full mb-6 group-hover:scale-110 transition-transform">
              <Activity size={48} className="text-cyan-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2 tracking-wide">LIVE SIMULATION</h2>
            <p className="text-slate-400 text-center text-sm">
              Connect to the live Point-Kinetics physics engine. Monitor real-time telemetry and engage the PPO Reinforcement Learning Agent for automated stability control.
            </p>
          </button>

          {/* DL Entry Card */}
          <button
            onClick={() => setCurrentView('dl')}
            className="flex-1 group flex flex-col items-center p-10 bg-slate-900 border border-slate-800 rounded-2xl hover:border-amber-500 hover:bg-slate-800 transition-all shadow-xl"
          >
            <div className="p-5 bg-amber-950 rounded-full mb-6 group-hover:scale-110 transition-transform">
              <Database size={48} className="text-amber-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2 tracking-wide">DL DIAGNOSTICS</h2>
            <p className="text-slate-400 text-center text-sm">
              Upload historical telemetry logs (CSV). Utilize the trained Deep Learning LSTM sequence model to predict SCRAM probabilities and identify impending critical failures.
            </p>
          </button>
        </div>
      </div>
    );
  }

  // ==========================================
  // RENDER: DL DIAGNOSTICS MODE
  // ==========================================
  if (currentView === 'dl') {
    return (
      <div className="flex h-screen w-full bg-slate-950 text-slate-100 font-sans overflow-hidden">

        {/* LEFT PANEL: 3D INDUSTRIAL SCHEMATIC (Reused for DL Playback) */}
        <div className="flex-1 relative border-r border-slate-800 bg-slate-900">
          <ResearchPlantCanvas
            telemetry={currentDlTelemetry}
            controlRods={currentDlTelemetry ? (1.0 - currentDlTelemetry.rod_position) * 50 : 50}
          />

          <div className="absolute top-4 left-4 z-10 flex flex-col gap-3">
            <button onClick={() => setCurrentView('home')} className="flex items-center gap-2 px-3 py-2 bg-slate-800/80 backdrop-blur hover:bg-slate-700 rounded-lg transition-colors border border-slate-700 w-max shadow-lg">
              <ArrowLeft size={16} className="text-slate-300" />
              <span className="text-xs font-bold text-slate-300">BACK TO MENU</span>
            </button>
          </div>

          {/* LARGE STATUS NOTIFICATION OVERLAY */}
          {currentDlTelemetry && (
            <div className="absolute top-8 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
              {currentDlTelemetry.prediction === 1 || currentDlTelemetry.scram_prob > 0.85 ? (
                <div className="px-8 py-3 bg-red-600/90 backdrop-blur border-2 border-red-400 rounded-full shadow-[0_0_50px_rgba(220,38,38,0.6)] flex items-center gap-3 animate-pulse">
                  <ShieldAlert size={28} className="text-white" />
                  <span className="text-xl font-black tracking-widest text-white">CRITICAL SCRAM / EXPLOSION DETECTED</span>
                </div>
              ) : (
                <div className="px-8 py-3 bg-emerald-600/90 backdrop-blur border-2 border-emerald-400 rounded-full shadow-[0_0_30px_rgba(5,150,105,0.4)] flex items-center gap-3">
                  <CheckCircle size={28} className="text-white" />
                  <span className="text-xl font-black tracking-widest text-white">SYSTEM HEALTHY</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT PANEL: DL UPLOAD & PLAYBACK UI */}
        <div className="w-[440px] flex flex-col bg-slate-900/90 backdrop-blur-md shadow-2xl z-10 h-full overflow-y-auto border-l border-slate-800">

          <div className="p-6 border-b border-slate-800 bg-slate-900 sticky top-0 z-20">
            <div className="flex justify-between items-center mb-2">
              <h1 className="text-xl font-black tracking-widest text-amber-400">DL DIAGNOSTICS</h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800">LSTM SEQUENCE PREDICTOR</span>
            </div>
            <p className="text-xs text-slate-400">Upload a telemetry CSV to visualize predictive failure states.</p>
          </div>

          {!dlResults ? (
            <div className="p-6 flex flex-col gap-4">
              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl shadow-inner">
                <h2 className="text-xs font-bold text-slate-300 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <FileText size={14} className="text-amber-400" /> Required CSV Parameters
                </h2>
                <div className="grid grid-cols-2 gap-2 text-[10px] font-mono text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> timestamp_sec</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> power_n</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> avg_precursors</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> temp_fuel</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> temp_coolant</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> action_rod</div>
                  <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div> action_flow</div>
                </div>
              </div>

              <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-xl p-8 bg-slate-900/50 hover:bg-slate-800/50 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setDlFile(e.target.files[0])}
                  className="mb-6 w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-bold file:bg-amber-900/50 file:text-amber-300 hover:file:bg-amber-800/80 cursor-pointer"
                />
                <button
                  onClick={handleDlUpload}
                  disabled={!dlFile || dlLoading}
                  className="w-full flex justify-center items-center gap-2 px-6 py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold rounded-lg transition-colors shadow-lg text-sm tracking-wide"
                >
                  <UploadCloud size={18} />
                  {dlLoading ? 'PROCESSING SEQUENCE...' : 'ANALYZE TELEMETRY'}
                </button>
              </div>
              {dlError && <div className="p-3 bg-red-950 border border-red-800 text-red-300 text-xs rounded-lg">{dlError}</div>}
            </div>
          ) : (
            <div className="p-6 flex flex-col gap-5">
              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
                <div className="flex justify-between items-end mb-2">
                  <label className="text-xs font-semibold text-slate-300">SEQUENCE TIMELINE PLAYBACK</label>
                  <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-950/50 border border-cyan-900/50 px-2 py-0.5 rounded">
                    T+ {currentDlTelemetry.time.toFixed(1)}s
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={dlResults.length - 1}
                  step="1"
                  value={dlPlaybackIndex}
                  onChange={(e) => setDlPlaybackIndex(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-900 rounded-lg appearance-none cursor-pointer accent-cyan-500 border border-slate-700 mt-2"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                  <span>START</span>
                  <span>END</span>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-1"><Activity size={13} className="text-amber-400" /> Snapshot Telemetry</h2>
                <StatCard icon={<Zap className="text-cyan-400" />} title="Recorded Thermal Power" value={`${(currentDlTelemetry.power * 100).toFixed(1)}%`} />
                <StatCard icon={<Thermometer className="text-orange-400" />} title="Fuel Temperature" value={currentDlTelemetry.temp_fuel.toFixed(2)} />
                <StatCard icon={<Thermometer className="text-blue-400" />} title="Coolant Temperature" value={currentDlTelemetry.temp_coolant.toFixed(2)} />
                <StatCard icon={<Activity className="text-cyan-400" />} title="Coolant Flow Action" value={currentDlTelemetry.coolant_flow.toFixed(3)} />
                <StatCard icon={<Sliders className="text-amber-400" />} title="Rod Action" value={currentDlTelemetry.rod_position.toFixed(3)} />
                <StatCard icon={<ShieldAlert className={currentDlTelemetry.scram_prob > 0.5 ? "text-red-400 animate-pulse" : "text-emerald-400"} />} title="LSTM Predicted SCRAM Risk" value={`${(currentDlTelemetry.scram_prob * 100).toFixed(1)}%`} alert={currentDlTelemetry.scram_prob > 0.5} />
                <StatCard icon={<AlertTriangle className={currentDlTelemetry.prediction === 1 ? "text-red-400 animate-pulse" : "text-emerald-400"} />} title="LSTM Classification" value={currentDlTelemetry.prediction === 1 ? "SCRAM / CRITICAL" : "NORMAL"} alert={currentDlTelemetry.prediction === 1} />
                {currentDlTelemetry.actual_scram !== null && currentDlTelemetry.actual_scram !== undefined && <StatCard icon={<CheckCircle className={currentDlTelemetry.actual_scram === 1 ? "text-red-400" : "text-emerald-400"} />} title="Dataset Ground Truth" value={currentDlTelemetry.actual_scram === 1 ? "SCRAM" : "NORMAL"} />}
              </div>
              {dlMetadata && <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-[10px] font-mono text-slate-400 space-y-1">
                <div className="flex justify-between"><span>Rows</span><span>{dlMetadata.total_rows}</span></div>
                <div className="flex justify-between"><span>Windows</span><span>{dlMetadata.predictions_generated}</span></div>
                <div className="flex justify-between"><span>Sequence Length</span><span>{dlMetadata.sequence_length}</span></div>
                <div className="flex justify-between"><span>Step Size</span><span>{dlMetadata.step_size}</span></div>
                <div className="flex justify-between"><span>SCRAM Predictions</span><span className="text-red-400">{dlMetadata.scram_predictions}</span></div>
              </div>}

              <button onClick={() => { setDlResults(null); setDlMetadata(null); setDlFile(null); setDlPlaybackIndex(0); setDlError(null); }} className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold rounded-lg transition-colors">
                UPLOAD NEW DATASET
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ==========================================
  // RENDER: RL LIVE SIMULATION MODE
  // ==========================================
  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-100 font-sans overflow-hidden">

      {/* LEFT PANEL: 3D INDUSTRIAL SCHEMATIC */}
      <div className="flex-1 relative border-r border-slate-800 bg-slate-900">
        <ResearchPlantCanvas telemetry={telemetry} controlRods={controlRods} />

        <div className="absolute top-4 left-4 z-10 flex flex-col gap-3">
          <button onClick={() => setCurrentView('home')} className="flex items-center gap-2 px-3 py-2 bg-slate-800/80 backdrop-blur hover:bg-slate-700 rounded-lg transition-colors border border-slate-700 w-max shadow-lg">
            <ArrowLeft size={16} className="text-slate-300" />
            <span className="text-xs font-bold text-slate-300">BACK TO MENU</span>
          </button>

          <button
            onClick={isConnected ? disconnectWebSocket : connectWebSocket}
            className={`px-4 py-2 rounded-lg font-bold shadow-lg transition-all text-sm tracking-wide border ${
              isConnected ? 'bg-red-900/80 border-red-500 hover:bg-red-800 text-white' : 'bg-emerald-900/80 border-emerald-500 hover:bg-emerald-800 text-white'
            }`}
          >
            {isConnected ? 'TERMINATE SIMULATION' : 'INITIALIZE SIMULATION'}
          </button>
        </div>
      </div>

      {/* RIGHT PANEL: TECHNICAL TELEMETRY & HUD */}
      <div className="w-[440px] flex flex-col bg-slate-900/90 backdrop-blur-md shadow-2xl z-10 h-full overflow-y-auto border-l border-slate-800">

        <div className="p-6 border-b border-slate-800 bg-slate-900 sticky top-0 z-20">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-xl font-black tracking-widest text-cyan-400">NUCLEAR CORE HUD</h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
              PHYSICS v2.4
            </span>
          </div>
          <div className="flex gap-2 p-1 bg-slate-950 rounded-lg border border-slate-800">
            <button
              onClick={() => {
                setMode('AUTO');
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'RESUME_AUTO' }));
                }
              }}
              className={`flex-1 py-2 text-xs font-bold rounded transition-all ${
                mode === 'AUTO' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              AUTO (PPO AGENT)
            </button>
            <button
              onClick={() => setMode('MANUAL')}
              className={`flex-1 py-2 text-xs font-bold rounded transition-all ${
                mode === 'MANUAL' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              MANUAL OVERRIDE
            </button>
          </div>
        </div>

        {rlError && (
          <div className="mx-6 mt-6 bg-red-950 border border-red-800 p-3 rounded-lg text-xs text-red-300">
            {rlError}
          </div>
        )}

        {/* Dynamic Warning for Manual Override Danger */}
        {telemetry?.manual_alert && (
          <div className="mx-6 mt-6 bg-orange-950/80 border border-orange-600 p-4 rounded-xl flex items-center gap-4 animate-pulse shadow-[0_0_15px_rgba(234,88,12,0.4)]">
            <AlertTriangle className="text-orange-500" size={24} />
            <span className="text-sm text-orange-200 font-bold tracking-wider">{telemetry.manual_alert}</span>
          </div>
        )}

        <div className="p-6 flex flex-col gap-3 border-b border-slate-800">
          <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-1">
            <Activity size={13} className="text-cyan-400" /> Real-Time Telemetry Matrix
          </h2>
          <StatCard
            icon={<Zap className="text-amber-400" />}
            title="Core Thermal Power"
            value={telemetry ? `${(telemetry.power * 100).toFixed(1)}%` : '---'}
          />
          <StatCard
            icon={<Zap className="text-yellow-400" />}
            title="Grid EMF / Electrical Output"
            value={telemetry ? `${(telemetry.power * 400).toFixed(1)} kV` : '---'}
          />
          <StatCard
            icon={<Thermometer className="text-orange-400" />}
            title="Fuel Cladding Temp"
            value={telemetry ? `${telemetry.temp_fuel.toFixed(2)} °C` : '---'}
          />
          <StatCard
            icon={<Activity className="text-cyan-400" />}
            title="Primary Coolant Mass Flow"
            value={telemetry ? `${telemetry.coolant_flow.toFixed(3)} m³/s` : '---'}
          />

          <StatCard
            icon={<ShieldAlert className={telemetry?.scram_prob > 0.5 ? "text-red-400 animate-pulse" : "text-emerald-400"} />}
            title="AI SCRAM Warning Probability"
            value={telemetry && telemetry.scram_prob !== undefined ? `${(telemetry.scram_prob * 100).toFixed(1)}%` : (telemetry ? 'BUFFERING...' : '---')}
            alert={telemetry?.scram_prob > 0.5}
          />

          <StatCard
            icon={<AlertTriangle className={telemetry?.alert === "CRITICAL MELTDOWN" || telemetry?.alert === "WARNING: HIGH SCRAM RISK" || telemetry?.alert === "CRITICAL" ? "text-red-500 animate-pulse" : "text-slate-500"} />}
            title="System Safety Status"
            value={telemetry?.alert ? telemetry.alert : (telemetry ? "NOMINAL" : "STANDBY")}
            alert={telemetry?.alert === "CRITICAL MELTDOWN" || telemetry?.alert === "WARNING: HIGH SCRAM RISK" || telemetry?.alert === "CRITICAL"}
          />
        </div>

        <div className={`p-6 flex flex-col gap-5 transition-all duration-300 ${
          mode === 'AUTO' ? 'opacity-30 pointer-events-none grayscale' : 'opacity-100'
        }`}>
          <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <Sliders size={13} className="text-amber-500" /> Operator Control Interface
          </h2>

          <ControlSlider
            label="Control Rod Assembly Insertion"
            value={controlRods}
            min="0" max="100" step="1" unit="%"
            onChange={(v) => { setControlRods(v); sendControlCommand('control_rods', v); }}
          />

          <ControlSlider
            label="Primary Coolant Circulation Pump"
            value={coolantFlow}
            min="0" max="0.5" step="0.01" unit="m³/s"
            onChange={(v) => { setCoolantFlow(v); sendControlCommand('coolant_flow', v); }}
          />

          <ControlSlider
            label="Grid Electrical Power Demand"
            value={targetPower}
            min="0" max="120" step="1" unit="%"
            onChange={(v) => { setTargetPower(v); sendControlCommand('target_power', v); }}
          />
        </div>

      </div>
    </div>
  );
}

// ============================================================
// INDUSTRIAL GRADE THREE.JS PLANT SCHEMATIC
// ============================================================
function ResearchPlantCanvas({ telemetry, controlRods }) {
  const mountRef = useRef(null);
  const telemetryRef = useRef(telemetry);
  const refsData = useRef({
    core: null, rodGroup: null, turbineBlades: null, generator: null,
    coolantLiquid: null, steamChamberLiquid: null, steamParticles: null,
    explosionMesh: null, explosionMat: null, isExploding: false
  });

  useEffect(() => {
    telemetryRef.current = telemetry;
  }, [telemetry]);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060913);
    scene.fog = new THREE.FogExp2(0x060913, 0.01);

    const camera = new THREE.PerspectiveCamera(
      45,
      currentMount.clientWidth / currentMount.clientHeight,
      0.1,
      1000
    );
    camera.position.set(-1, 4.5, 16);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    currentMount.appendChild(renderer.domElement);

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none';
    currentMount.appendChild(labelRenderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x38bdf8, 3.5);
    dirLight.position.set(6, 12, 8);
    scene.add(dirLight);

    const gridHelper = new THREE.GridHelper(35, 35, 0x0284c7, 0x1e293b);
    gridHelper.position.y = -2.5;
    scene.add(gridHelper);

    const createLabel = (text, pos, colorClass = "text-cyan-400 border-cyan-500/40") => {
      const p = document.createElement('div');
      p.className = `font-mono text-[10px] font-bold px-3 py-1.5 rounded-md bg-slate-900/95 border shadow-2xl backdrop-blur whitespace-nowrap ${colorClass}`;
      p.textContent = text;
      const labelObj = new CSS2DObject(p);
      labelObj.position.set(pos[0], pos[1], pos[2]);
      scene.add(labelObj);
      return labelObj;
    };

    createLabel("📍 CRDM MOTOR DECK", [-6.0, 3.2, 0], "text-amber-300 border-amber-500/60");
    createLabel("📍 REACTOR PRESSURE VESSEL", [-6.0, 1.2, 0], "text-cyan-300 border-cyan-500/60");
    createLabel("📍 PRESSURIZER", [-3.8, 2.2, 0], "text-slate-300 border-slate-600");
    createLabel("📍 COOLANT PUMP", [-4.0, -1.2, 0.6], "text-cyan-400 border-cyan-700/60");
    createLabel("📍 STEAM GENERATOR", [-1.5, 2.6, 0], "text-sky-200 border-sky-500/60");
    createLabel("📍 TURBINE - GENERATOR", [2.0, 1.8, 0], "text-amber-300 border-amber-500/60");
    createLabel("📍 STEP-UP TRANSFORMER", [5.2, 1.8, 1.5], "text-yellow-300 border-yellow-500/60");
    createLabel("📍 TRANSMISSION GRID TOWER", [8.2, 3.2, 0], "text-red-300 border-red-500/60");
    createLabel("📍 CONDENSER", [2.0, -1.5, 0.7], "text-blue-300 border-blue-500/60");

    const reactorGroup = new THREE.Group();
    reactorGroup.position.set(-6.0, 0, 0);

    const vesselGeo = new THREE.CylinderGeometry(1.2, 1.2, 3.6, 32, 1, true);
    const vesselMat = new THREE.MeshPhysicalMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.25,
      roughness: 0.1,
      metalness: 0.9,
      side: THREE.DoubleSide,
    });
    const vessel = new THREE.Mesh(vesselGeo, vesselMat);
    reactorGroup.add(vessel);

    const coolantGeo = new THREE.CylinderGeometry(1.15, 1.15, 3.4, 32);
    const coolantMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.4,
      roughness: 0.1,
      transmission: 0.6,
    });
    const coolantLiquid = new THREE.Mesh(coolantGeo, coolantMat);
    reactorGroup.add(coolantLiquid);
    refsData.current.coolantLiquid = coolantLiquid;

    const coreGeo = new THREE.CylinderGeometry(0.75, 0.75, 2.2, 32);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      emissive: 0x0369a1,
      emissiveIntensity: 0.6,
      roughness: 0.3,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    reactorGroup.add(core);
    refsData.current.core = core;

    const expGeo = new THREE.SphereGeometry(2.0, 32, 32);
    const expMat = new THREE.MeshBasicMaterial({
      color: 0xff3300,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending
    });
    const explosion = new THREE.Mesh(expGeo, expMat);
    reactorGroup.add(explosion);
    refsData.current.explosionMesh = explosion;
    refsData.current.explosionMat = expMat;

    const rodGroup = new THREE.Group();
    const crdmBaseGeo = new THREE.CylinderGeometry(1.1, 1.1, 0.4, 32);
    const crdmBaseMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.2 });
    const crdmBase = new THREE.Mesh(crdmBaseGeo, crdmBaseMat);
    crdmBase.position.set(0, 2.0, 0);
    rodGroup.add(crdmBase);

    for (let m = 0; m < 5; m++) {
      const angle = (m / 5) * Math.PI * 2;
      const r = m === 0 ? 0 : 0.4;
      const motorGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.5, 16);
      const motorMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 0.8, roughness: 0.2 });
      const motor = new THREE.Mesh(motorGeo, motorMat);
      motor.position.set(Math.cos(angle) * r, 2.35, Math.sin(angle) * r);
      rodGroup.add(motor);
    }

    const rodGeo = new THREE.CylinderGeometry(0.08, 0.08, 1.5, 16);
    const rodMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.95, roughness: 0.1 });
    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI * 2;
      const radius = 0.4;
      const rod = new THREE.Mesh(rodGeo, rodMat);
      if (i === 0) rod.position.set(0, 0.9, 0);
      else rod.position.set(Math.cos(angle) * radius, 0.9, Math.sin(angle) * radius);
      rodGroup.add(rod);
    }

    reactorGroup.add(rodGroup);
    refsData.current.rodGroup = rodGroup;
    scene.add(reactorGroup);

    const pumpGroup = new THREE.Group();
    pumpGroup.position.set(-4.0, -1.2, 0);
    const pumpBodyGeo = new THREE.SphereGeometry(0.38, 24, 24);
    const pumpMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.9, roughness: 0.2 });
    const pumpBody = new THREE.Mesh(pumpBodyGeo, pumpMat);
    pumpGroup.add(pumpBody);
    scene.add(pumpGroup);

    const steamGroup = new THREE.Group();
    steamGroup.position.set(-1.5, 0, 0);

    const sgShellGeo = new THREE.CylinderGeometry(0.75, 0.75, 3.4, 32, 1, true);
    const sgShellMat = new THREE.MeshPhysicalMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.25,
      roughness: 0.1,
      metalness: 0.9,
      side: THREE.DoubleSide,
    });
    const sgShell = new THREE.Mesh(sgShellGeo, sgShellMat);
    steamGroup.add(sgShell);

    const waterChamberGeo = new THREE.CylinderGeometry(0.7, 0.7, 1.8, 32);
    const waterChamberMat = new THREE.MeshPhysicalMaterial({
      color: 0x0284c7,
      transparent: true,
      opacity: 0.6,
      roughness: 0.1,
    });
    const steamChamberLiquid = new THREE.Mesh(waterChamberGeo, waterChamberMat);
    steamChamberLiquid.position.set(0, -0.7, 0);
    steamGroup.add(steamChamberLiquid);
    refsData.current.steamChamberLiquid = steamChamberLiquid;

    const particleCount = 40;
    const particleGeo = new THREE.SphereGeometry(0.06, 8, 8);
    const particleMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.5,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.8,
    });
    const particlesGroup = new THREE.Group();
    for (let i = 0; i < particleCount; i++) {
      const p = new THREE.Mesh(particleGeo, particleMat);
      p.position.set(
        (Math.random() - 0.5) * 0.8,
        (Math.random() - 0.2) * 1.2,
        (Math.random() - 0.5) * 0.8
      );
      particlesGroup.add(p);
    }
    steamGroup.add(particlesGroup);
    refsData.current.steamParticles = particlesGroup;
    scene.add(steamGroup);

    const presGroup = new THREE.Group();
    presGroup.position.set(-3.8, 1.1, 0);
    const presGeo = new THREE.CylinderGeometry(0.3, 0.3, 1.2, 24);
    const presMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8, roughness: 0.3 });
    const pressurizer = new THREE.Mesh(presGeo, presMat);
    presGroup.add(pressurizer);
    scene.add(presGroup);

    const turbineGroup = new THREE.Group();
    turbineGroup.position.set(2.0, 0, 0);

    const turbineShellGeo = new THREE.CylinderGeometry(0.85, 0.95, 2.2, 32, 1, true);
    const turbineShellMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.25,
      roughness: 0.1,
      metalness: 0.9,
      side: THREE.DoubleSide,
    });
    const turbineShell = new THREE.Mesh(turbineShellGeo, turbineShellMat);
    turbineShell.rotation.z = Math.PI / 2;
    turbineGroup.add(turbineShell);

    const bladesGroup = new THREE.Group();
    bladesGroup.rotation.z = Math.PI / 2;
    const bladeMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.95, roughness: 0.1, side: THREE.DoubleSide });
    for (let b = 0; b < 6; b++) {
      const bladeGeo = new THREE.BoxGeometry(0.05, 1.5, 0.35);
      const blade = new THREE.Mesh(bladeGeo, bladeMat);
      blade.rotation.y = (b / 6) * Math.PI;
      bladesGroup.add(blade);
    }
    turbineGroup.add(bladesGroup);
    refsData.current.turbineBlades = bladesGroup;

    const genGeo = new THREE.CylinderGeometry(1.0, 1.0, 1.6, 32);
    const genMat = new THREE.MeshStandardMaterial({
      color: 0xfbbf24,
      metalness: 0.9,
      roughness: 0.2,
    });
    const generator = new THREE.Mesh(genGeo, genMat);
    generator.rotation.z = Math.PI / 2;
    generator.position.set(1.9, 0, 0);
    turbineGroup.add(generator);
    refsData.current.generator = generator;
    scene.add(turbineGroup);

    const condenserGroup = new THREE.Group();
    condenserGroup.position.set(2.0, -1.5, 0);
    const condGeo = new THREE.BoxGeometry(2.2, 0.5, 1.0);
    const condMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.7, roughness: 0.4 });
    const condenser = new THREE.Mesh(condGeo, condMat);
    condenserGroup.add(condenser);
    scene.add(condenserGroup);

    const transformerGroup = new THREE.Group();
    transformerGroup.position.set(5.2, -0.4, 1.5);

    const transBoxGeo = new THREE.BoxGeometry(1.2, 1.4, 1.0);
    const transBoxMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });
    const transBox = new THREE.Mesh(transBoxGeo, transBoxMat);
    transformerGroup.add(transBox);

    for (let f = -0.4; f <= 0.4; f += 0.2) {
      const finGeo = new THREE.BoxGeometry(0.05, 1.2, 1.1);
      const finMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9 });
      const fin = new THREE.Mesh(finGeo, finMat);
      fin.position.set(f, 0, 0);
      transformerGroup.add(fin);
    }

    for (let b = -0.3; b <= 0.3; b += 0.6) {
      const bushGeo = new THREE.CylinderGeometry(0.06, 0.08, 0.7, 16);
      const bushMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.1 });
      const bushing = new THREE.Mesh(bushGeo, bushMat);
      bushing.position.set(b, 1.05, 0);
      transformerGroup.add(bushing);
    }
    scene.add(transformerGroup);

    const towerGroup = new THREE.Group();
    towerGroup.position.set(8.2, -0.5, 0);
    const towerMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.9, roughness: 0.2 });

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, 5.5, 16), towerMat);
    mast.position.set(0, 2.2, 0);
    towerGroup.add(mast);

    for (let yPos of [2.5, 3.5, 4.5]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.1, 0.1), towerMat);
      arm.position.set(0, yPos, 0);
      towerGroup.add(arm);
    }
    scene.add(towerGroup);

    const pipeMat = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.95, roughness: 0.15 });
    const flangeMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 });

    const createFlangedPipe = (length, x, y, z, rotZ = Math.PI / 2) => {
      const group = new THREE.Group();
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, length, 16), pipeMat);
      pipe.rotation.z = rotZ;
      group.add(pipe);

      const f1 = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.08, 16), flangeMat);
      f1.rotation.z = rotZ;
      f1.position.set(-length / 2 + 0.04, 0, 0);
      group.add(f1);

      const f2 = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.08, 16), flangeMat);
      f2.rotation.z = rotZ;
      f2.position.set(length / 2 - 0.04, 0, 0);
      group.add(f2);

      group.position.set(x, y, z);
      scene.add(group);
    };

    createFlangedPipe(2.2, -3.7, 0.8, 0);
    createFlangedPipe(1.3, -0.35, 0.4, 0);
    createFlangedPipe(1.1, 2.0, -0.85, 0, 0);
    createFlangedPipe(3.8, -2.0, -1.5, 0);

    const cableMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xf59e0b, emissiveIntensity: 0.8, metalness: 0.9 });

    const cable1 = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.8, 8), cableMat);
    cable1.rotation.x = Math.PI / 2;
    cable1.position.set(3.9, 0.6, 0.75);
    scene.add(cable1);

    const cable2 = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 3.2, 8), cableMat);
    cable2.rotation.z = Math.PI / 2;
    cable2.position.set(6.7, 1.5, 0.75);
    scene.add(cable2);

    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const liveTelemetry = telemetryRef.current;
      const powerFactor = liveTelemetry ? liveTelemetry.power : 1.0;

      if (refsData.current.turbineBlades) refsData.current.turbineBlades.rotation.x += 0.15 * powerFactor;
      if (refsData.current.generator) refsData.current.generator.rotation.x += 0.05 * powerFactor;

      if (refsData.current.steamParticles) {
        refsData.current.steamParticles.children.forEach((p) => {
          p.position.y += 0.015 * powerFactor;
          if (p.position.y > 1.0) p.position.y = -0.7;
        });
      }

      if (refsData.current.isExploding && refsData.current.explosionMesh && refsData.current.explosionMat) {
        const time = Date.now() * 0.005;
        const scale = 1.0 + Math.sin(time) * 0.15;
        refsData.current.explosionMesh.scale.set(scale, scale, scale);
        refsData.current.explosionMat.opacity = 0.6 + Math.sin(time * 2) * 0.2;
      }

      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!currentMount) return;
      camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
      labelRenderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      controls.dispose();
      if (currentMount) currentMount.innerHTML = '';
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    const { core, rodGroup, coolantLiquid, steamChamberLiquid, explosionMesh, explosionMat } = refsData.current;
    if (!rodGroup || !core) return;

    const targetY = 0.5 - (controlRods / 100) * 1.0;
    rodGroup.position.y = targetY;

    if (telemetry) {
      const isCritical = telemetry.alert === 'CRITICAL' || telemetry.alert === 'CRITICAL MELTDOWN' || telemetry.prediction === 1 || telemetry.scram_prob > 0.85;

      if (isCritical) {
        core.material.color.setHex(0xe11d48);
        core.material.emissive.setHex(0x991b1b);
        core.material.emissiveIntensity = 0.95;
        if (coolantLiquid) coolantLiquid.material.color.setHex(0xf97316);
        if (steamChamberLiquid) steamChamberLiquid.material.color.setHex(0xf97316);

        refsData.current.isExploding = true;
      } else {
        core.material.color.setHex(0x0284c7);
        core.material.emissive.setHex(0x0369a1);
        core.material.emissiveIntensity = 0.5;
        if (coolantLiquid) coolantLiquid.material.color.setHex(0x38bdf8);
        if (steamChamberLiquid) steamChamberLiquid.material.color.setHex(0x0284c7);

        refsData.current.isExploding = false;
        if (explosionMat) explosionMat.opacity = 0;
      }
    }
  }, [controlRods, telemetry]);

  return <div ref={mountRef} className="w-full h-full relative" />;
}

function StatCard({ icon, title, value, alert }) {
  return (
    <div className={`p-3 rounded-xl flex items-center gap-4 border ${alert ? 'bg-red-950/60 border-red-700/80 shadow-lg shadow-red-950/50' : 'bg-slate-900/50 border-slate-800'}`}>
      <div className="p-2.5 bg-slate-950 rounded-lg shadow-inner border border-slate-800">
        {icon}
      </div>
      <div>
        <p className="text-[11px] text-slate-400 font-medium tracking-wide">{title}</p>
        <p className={`text-lg font-mono font-bold ${alert ? 'text-red-400 animate-pulse' : 'text-slate-100'}`}>
          {value}
        </p>
      </div>
    </div>
  );
}

function ControlSlider({ label, value, min, max, step, unit, onChange }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-end mb-1">
        <label className="text-xs font-semibold text-slate-300">{label}</label>
        <span className="text-xs font-mono font-bold text-amber-400 bg-amber-950/50 border border-amber-900/50 px-2 py-0.5 rounded">
          {value} {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-amber-500 border border-slate-800"
      />
      <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-0.5">
        <span>MIN: {min}</span>
        <span>MAX: {max}</span>
      </div>
    </div>
  );
}