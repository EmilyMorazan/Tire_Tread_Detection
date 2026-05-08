import { useState, useRef, useCallback, useEffect } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

// ── Data (classes)────────────────────────────────────────────────────────────────────
const CLASS_DATA = [
  { depth: '2/32"', count: 18, mae: 0.1434, safety: "critical" },
  { depth: '3/32"', count: 50, mae: 0.2699, safety: "critical" },
  { depth: '4/32"', count: 106, mae: 0.2674, safety: "warning" },
  { depth: '5/32"', count: 172, mae: 0.2722, safety: "ok" },
  { depth: '6/32"', count: 127, mae: 0.2847, safety: "ok" },
  { depth: '7/32"', count: 60, mae: 0.3324, safety: "ok" },
  { depth: '8/32"', count: 50, mae: 0.2187, safety: "ok" },
  { depth: '9/32"', count: 50, mae: 0.1956, safety: "ok" },
  { depth: '10/32"', count: 26, mae: 0.1831, safety: "ok" },
];

const TRAINING_DATA = (() => {
  const pts = [];
  // Phase 1: 30 epochs, plateau ~0.94→0.75
  for (let i = 0; i < 30; i++)
    pts.push({
      epoch: i + 1,
      loss: +(0.94 - i * 0.006 + Math.sin(i) * 0.04).toFixed(3),
      phase: 1,
    });
  // Phase 2: 50 epochs, 0.75→0.51
  for (let i = 0; i < 50; i++)
    pts.push({
      epoch: 31 + i,
      loss: +(0.75 - i * 0.005 + Math.sin(i * 1.3) * 0.03).toFixed(3),
      phase: 2,
    });
  // Phase 3: 75 epochs, 0.51→0.19
  for (let i = 0; i < 75; i++)
    pts.push({
      epoch: 81 + i,
      loss: +(0.51 - i * 0.0045 + Math.sin(i * 0.8) * 0.025).toFixed(3),
      phase: 3,
    });
  return pts;
})();

// ── Helpers ──────────────────────────────────────────────────────────────────
function simulatePredict(depthIdx, lighting, angle) {
  const base = depthIdx + 2; // 2..10
  const noise = (Math.random() - 0.5) * 0.6;
  const penalty = (100 - lighting) * 0.003 + (100 - angle) * 0.002;
  const pred = Math.max(2, Math.min(10, base + noise + penalty));
  const mae = CLASS_DATA[depthIdx].mae;
  const conf = Math.round(
    Math.max(42, Math.min(96, (lighting + angle) / 2 - 5 + Math.random() * 8)),
  );
  return { pred: +pred.toFixed(1), mae, conf };
}

function safetyInfo(depth) {
  if (depth <= 2)
    return { label: "Replace immediately", color: "#dc2626", bg: "#fef2f2" };
  if (depth <= 3)
    return { label: "Replace soon", color: "#ea580c", bg: "#fff7ed" };
  if (depth <= 4)
    return { label: "Monitor closely", color: "#d97706", bg: "#fffbeb" };
  return { label: "Safe to drive", color: "#16a34a", bg: "#f0fdf4" };
}

function TireIcon({ depth, size = 120 }) {
  const pct = (depth - 2) / 8;
  const grooveColor =
    pct < 0.15 ? "#dc2626" : pct < 0.3 ? "#ea580c" : "#16a34a";
  const grooves = [0.28, 0.42, 0.58, 0.72];
  return (
    <svg width={size} height={size} viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="54" fill="#1a1a1a" />
      <circle cx="60" cy="60" r="40" fill="#2a2a2a" />
      {grooves.map((r, i) => {
        const radius = r * 108;
        const gh = 4 + pct * 10;
        return (
          <g key={i}>
            <circle
              cx="60"
              cy="60"
              r={radius / 2}
              fill="none"
              stroke={grooveColor}
              strokeWidth={gh}
              opacity={0.85}
            />
          </g>
        );
      })}
      <circle cx="60" cy="60" r="22" fill="#111" />
      <circle cx="60" cy="60" r="14" fill="#333" />
      <text
        x="60"
        y="64"
        textAnchor="middle"
        fontSize="10"
        fontWeight="600"
        fill="#fff"
        fontFamily="monospace"
      >
        {depth}/32"
      </text>
    </svg>
  );
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
const TABS = ["Predictor", "Upload", "Distribution", "Errors", "Training"];

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f0f0f",
        color: "#f0ede8",
        fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          borderBottom: "1px solid #2a2a2a",
          padding: "20px 32px",
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        <TireIcon depth={6} size={44} />
        <div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              letterSpacing: "-0.5px",
              color: "#f0ede8",
            }}
          >
            Tire Tread Depth Detection
          </div>
          <div style={{ fontSize: 12, color: "#888", marginTop: 2 }}>
            ResNet-18 · ML Regression · Computer Vision &nbsp;·&nbsp; Mark
            Youssef, Spencer Caillat, Emily Morazan
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {[
            { label: "MAE", val: "0.26" },
            { label: "Val Loss", val: "0.19" },
            { label: "Images", val: "660" },
          ].map((m) => (
            <div
              key={m.label}
              style={{
                background: "#1a1a1a",
                border: "1px solid #2a2a2a",
                borderRadius: 8,
                padding: "6px 14px",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 16, fontWeight: 700, color: "#e85d20" }}>
                {m.val}
              </div>
              <div style={{ fontSize: 10, color: "#666" }}>{m.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: "flex",
          gap: 0,
          borderBottom: "1px solid #2a2a2a",
          padding: "0 32px",
        }}
      >
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setActiveTab(i)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "14px 20px",
              fontSize: 13,
              fontWeight: activeTab === i ? 600 : 400,
              color: activeTab === i ? "#e85d20" : "#666",
              borderBottom:
                activeTab === i ? "2px solid #e85d20" : "2px solid transparent",
              marginBottom: -1,
              transition: "all 0.15s",
              fontFamily: "inherit",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "32px" }}>
        {activeTab === 0 && <PredictorTab />}
        {activeTab === 1 && <UploadTab />}
        {activeTab === 2 && <DistributionTab />}
        {activeTab === 3 && <ErrorsTab />}
        {activeTab === 4 && <TrainingTab />}
      </div>
    </div>
  );
}

// ── Predictor Tab ─────────────────────────────────────────────────────────────
function PredictorTab() {
  const [depthIdx, setDepthIdx] = useState(3);
  const [lighting, setLighting] = useState(80);
  const [angle, setAngle] = useState(75);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);

  const run = () => {
    setRunning(true);
    setResult(null);
    setTimeout(() => {
      setResult(simulatePredict(depthIdx, lighting, angle));
      setRunning(false);
    }, 700);
  };
}
