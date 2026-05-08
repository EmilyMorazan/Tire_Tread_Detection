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

// ── Data (classes )────────────────────────────────────────────────────────────────────
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

  const depth = depthIdx + 2;
  const safety = result ? safetyInfo(result.pred) : null;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 24,
        maxWidth: 860,
      }}
    >
      {/* Controls */}
      <div
        style={{
          background: "#161616",
          border: "1px solid #2a2a2a",
          borderRadius: 12,
          padding: 24,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 20,
            color: "#ccc",
          }}
        >
          Simulation controls
        </div>

        <SliderRow
          label="True tread depth"
          value={depth}
          display={`${depth}/32"`}
          min={2}
          max={10}
          step={1}
          onChange={(v) => setDepthIdx(v - 2)}
        />
        <SliderRow
          label="Lighting quality"
          value={lighting}
          display={`${lighting}%`}
          min={0}
          max={100}
          step={1}
          onChange={setLighting}
        />
        <SliderRow
          label="Camera angle"
          value={angle}
          display={`${angle}%`}
          min={0}
          max={100}
          step={1}
          onChange={setAngle}
        />

        <div
          style={{ marginTop: 24, display: "flex", justifyContent: "center" }}
        >
          <TireIcon depth={depth} size={110} />
        </div>

        <button
          onClick={run}
          disabled={running}
          style={{
            marginTop: 20,
            width: "100%",
            padding: "12px 0",
            background: running ? "#333" : "#e85d20",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 600,
            cursor: running ? "not-allowed" : "pointer",
            transition: "background 0.2s",
            fontFamily: "inherit",
          }}
        >
          {running ? "Running inference…" : "Run prediction"}
        </button>
      </div>

      {/* Result */}
      <div
        style={{
          background: "#161616",
          border: "1px solid #2a2a2a",
          borderRadius: 12,
          padding: 24,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 20,
            color: "#ccc",
          }}
        >
          Model output
        </div>

        {!result && !running && (
          <div
            style={{
              color: "#444",
              fontSize: 13,
              marginTop: 40,
              textAlign: "center",
            }}
          >
            Adjust sliders and click "Run prediction"
          </div>
        )}
        {running && (
          <div
            style={{
              color: "#666",
              fontSize: 13,
              marginTop: 40,
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 32, marginBottom: 8 }}>⚙️</div>
            Simulating ResNet-18 inference…
          </div>
        )}
        {result && (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "flex-end",
                gap: 8,
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  fontSize: 56,
                  fontWeight: 800,
                  color: "#e85d20",
                  lineHeight: 1,
                }}
              >
                {result.pred}
              </span>
              <span style={{ fontSize: 18, color: "#888", marginBottom: 8 }}>
                /32"
              </span>
            </div>
            <div style={{ fontSize: 12, color: "#555", marginBottom: 16 }}>
              predicted tread depth
            </div>

            <div
              style={{
                background: safety.bg,
                border: `1px solid ${safety.color}33`,
                borderRadius: 8,
                padding: "10px 14px",
                marginBottom: 16,
              }}
            >
              <span
                style={{ fontSize: 13, fontWeight: 600, color: safety.color }}
              >
                {safety.label}
              </span>
            </div>

            <div style={{ marginBottom: 14 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12,
                  color: "#666",
                  marginBottom: 6,
                }}
              >
                <span>Confidence</span>
                <span style={{ color: "#ccc" }}>{result.conf}%</span>
              </div>
              <div
                style={{ height: 6, background: "#2a2a2a", borderRadius: 3 }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${result.conf}%`,
                    background: "#e85d20",
                    borderRadius: 3,
                    transition: "width 0.6s ease",
                  }}
                />
              </div>
            </div>

            <div style={{ marginBottom: 14 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12,
                  color: "#666",
                  marginBottom: 6,
                }}
              >
                <span>Expected MAE (this class)</span>
                <span style={{ color: "#ccc" }}>±{result.mae.toFixed(4)}</span>
              </div>
              <div
                style={{ height: 6, background: "#2a2a2a", borderRadius: 3 }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${(result.mae / 0.35) * 100}%`,
                    background: "#888",
                    borderRadius: 3,
                  }}
                />
              </div>
            </div>

            <div
              style={{
                marginTop: 16,
                padding: 12,
                background: "#1e1e1e",
                borderRadius: 8,
                fontSize: 12,
                color: "#555",
              }}
            >
              <div style={{ marginBottom: 4 }}>
                <span style={{ color: "#888" }}>Input depth:</span> {depth}/32"
              </div>
              <div style={{ marginBottom: 4 }}>
                <span style={{ color: "#888" }}>Lighting:</span> {lighting}%
              </div>
              <div>
                <span style={{ color: "#888" }}>Camera angle:</span> {angle}%
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function SliderRow({ label, value, display, min, max, step, onChange }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: "#888",
          marginBottom: 6,
        }}
      >
        <span>{label}</span>
        <span style={{ color: "#e85d20", fontWeight: 600 }}>{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: "#e85d20" }}
      />
    </div>
  );
}

// ── Upload Tab ────────────────────────────────────────────────────────────────
function UploadTab() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setImage(url);
    setResult(null);
    setLoading(true);
    setTimeout(() => {
      const depthIdx = Math.floor(Math.random() * 9);
      const res = simulatePredict(
        depthIdx,
        75 + Math.random() * 20,
        70 + Math.random() * 20,
      );
      setResult({ ...res, depthIdx });
      setLoading(false);
    }, 1200);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const safety = result ? safetyInfo(result.pred) : null;

  return (
    <div style={{ maxWidth: 780 }}>
      <div style={{ fontSize: 13, color: "#666", marginBottom: 20 }}>
        Upload a real tire photo. The model will simulate a tread depth
        prediction (real inference requires the .pth backend).
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: image ? "1fr 1fr" : "1fr",
          gap: 20,
        }}
      >
        {/* Drop zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current.click()}
          style={{
            border: `2px dashed ${dragging ? "#e85d20" : "#2a2a2a"}`,
            borderRadius: 12,
            padding: 32,
            textAlign: "center",
            cursor: "pointer",
            background: dragging ? "#1a1108" : "#161616",
            transition: "all 0.2s",
            minHeight: 260,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {image ? (
            <img
              src={image}
              alt="Uploaded tire"
              style={{
                maxWidth: "100%",
                maxHeight: 220,
                borderRadius: 8,
                objectFit: "contain",
              }}
            />
          ) : (
            <>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🛞</div>
              <div style={{ fontSize: 14, color: "#888" }}>
                Drop a tire photo here
              </div>
              <div style={{ fontSize: 12, color: "#444", marginTop: 6 }}>
                or click to browse
              </div>
              <div style={{ fontSize: 11, color: "#333", marginTop: 12 }}>
                JPG, PNG, HEIC supported
              </div>
            </>
          )}
        </div>

        {/* Result */}
        {image && (
          <div
            style={{
              background: "#161616",
              border: "1px solid #2a2a2a",
              borderRadius: 12,
              padding: 24,
            }}
          >
            {loading ? (
              <div style={{ textAlign: "center", marginTop: 60 }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>⚙️</div>
                <div style={{ fontSize: 13, color: "#666" }}>
                  Running ResNet-18 inference…
                </div>
                <div style={{ fontSize: 11, color: "#444", marginTop: 6 }}>
                  Preprocessing → Forward pass → Regression head
                </div>
              </div>
            ) : result ? (
              <>
                <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>
                  Predicted tread depth
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-end",
                    gap: 6,
                    marginBottom: 12,
                  }}
                >
                  <span
                    style={{
                      fontSize: 52,
                      fontWeight: 800,
                      color: "#e85d20",
                      lineHeight: 1,
                    }}
                  >
                    {result.pred}
                  </span>
                  <span
                    style={{ fontSize: 16, color: "#666", marginBottom: 8 }}
                  >
                    /32"
                  </span>
                </div>

                <div
                  style={{
                    background: safety.bg,
                    border: `1px solid ${safety.color}44`,
                    borderRadius: 8,
                    padding: "10px 14px",
                    marginBottom: 16,
                  }}
                >
                  <span
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: safety.color,
                    }}
                  >
                    {safety.label}
                  </span>
                </div>

                <ConfBar
                  label="Confidence"
                  value={result.conf}
                  display={`${result.conf}%`}
                  color="#e85d20"
                />
                <ConfBar
                  label="Class MAE"
                  value={(result.mae / 0.35) * 100}
                  display={`±${result.mae.toFixed(4)}`}
                  color="#666"
                />

                <div
                  style={{
                    marginTop: 16,
                    fontSize: 11,
                    color: "#444",
                    lineHeight: 1.8,
                  }}
                >
                  Note: This is a simulated prediction. Connect the FastAPI
                  backend with your .pth model file for real inference.
                </div>

                <button
                  onClick={() => {
                    setImage(null);
                    setResult(null);
                  }}
                  style={{
                    marginTop: 16,
                    width: "100%",
                    padding: "10px 0",
                    background: "none",
                    color: "#666",
                    border: "1px solid #2a2a2a",
                    borderRadius: 8,
                    fontSize: 13,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  Try another photo
                </button>
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function ConfBar({ label, value, display, color }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: "#666",
          marginBottom: 5,
        }}
      >
        <span>{label}</span>
        <span style={{ color: "#ccc" }}>{display}</span>
      </div>
      <div style={{ height: 5, background: "#2a2a2a", borderRadius: 3 }}>
        <div
          style={{
            height: "100%",
            width: `${Math.min(100, value)}%`,
            background: color,
            borderRadius: 3,
          }}
        />
      </div>
    </div>
  );
}

// ── Distribution Tab ──────────────────────────────────────────────────────────
function DistributionTab() {
  const barColor = (d) => {
    if (d.depth === '2/32"' || d.depth === '10/32"') return "#d97706";
    if (d.count >= 100) return "#e85d20";
    return "#7a3010";
  };
  return (
    <div style={{ maxWidth: 860 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4,1fr)",
          gap: 12,
          marginBottom: 28,
        }}
      >
        {[
          { label: "Total images", val: "660" },
          { label: "Tread classes", val: "9" },
          { label: "Largest class", val: "172", sub: '5/32"' },
          { label: "Imbalance ratio", val: "9.6×" },
        ].map((m) => (
          <div
            key={m.label}
            style={{
              background: "#161616",
              border: "1px solid #2a2a2a",
              borderRadius: 10,
              padding: "14px 16px",
            }}
          >
            <div style={{ fontSize: 11, color: "#555", marginBottom: 4 }}>
              {m.label}
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "#e85d20" }}>
              {m.val}
            </div>
            {m.sub && (
              <div style={{ fontSize: 11, color: "#555" }}>{m.sub}</div>
            )}
          </div>
        ))}
      </div>

      <div
        style={{
          background: "#161616",
          border: "1px solid #2a2a2a",
          borderRadius: 12,
          padding: 24,
        }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            marginBottom: 4,
            color: "#ccc",
          }}
        >
          Images per tread depth class
        </div>
        <div style={{ fontSize: 12, color: "#555", marginBottom: 20 }}>
          Classes 4–6 hold 61% of the data. Edge classes (2 and 10) are
          safety-critical but underrepresented.
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={CLASS_DATA}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#222" />
            <XAxis
              dataKey="depth"
              tick={{ fill: "#666", fontSize: 12 }}
              axisLine={{ stroke: "#2a2a2a" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#666", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: 6,
                color: "#ccc",
                fontSize: 12,
              }}
              formatter={(v) => [v + " images", "Count"]}
            />
            <Bar
              dataKey="count"
              radius={[4, 4, 0, 0]}
              label={{ position: "top", fill: "#666", fontSize: 11 }}
            >
              {CLASS_DATA.map((d, i) => (
                <Cell key={i} fill={barColor(d)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div
          style={{
            display: "flex",
            gap: 20,
            marginTop: 12,
            fontSize: 12,
            color: "#555",
          }}
        >
          <span>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: "#e85d20",
                marginRight: 6,
                verticalAlign: "middle",
              }}
            />
            Center classes (≥100 images)
          </span>
          <span>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: "#d97706",
                marginRight: 6,
                verticalAlign: "middle",
              }}
            />
            Edge classes — safety critical
          </span>
          <span>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: "#7a3010",
                marginRight: 6,
                verticalAlign: "middle",
              }}
            />
            Mid-range classes
          </span>
        </div>
      </div>
    </div>
  );
}
