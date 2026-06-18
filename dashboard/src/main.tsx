import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Box,
  Camera,
  Check,
  CircleGauge,
  Cpu,
  Database,
  Download,
  Focus,
  Gauge,
  Grid3X3,
  Maximize2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ScanLine,
  Server,
  Square,
  Thermometer,
  Triangle,
  Video,
  Wifi,
} from "lucide-react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./styles.css";

type ViewName = "reconstruction" | "evaluation" | "system" | "field";

type FieldStatus = {
  camera: {
    connected?: boolean;
    captures?: number;
    uptime_s?: number;
    error?: string;
    video?: {
      recording?: boolean;
      frames?: number;
      fps?: number;
      started_at?: number | null;
      duration_s?: number;
    };
  };
  pi: {
    temperature_c?: number | null;
    load_1m?: number | null;
    memory_total_bytes?: number | null;
    memory_used_bytes?: number | null;
    disk_free_bytes?: number;
    process_uptime_s?: number;
  };
  session: null | {
    session_id: string;
    active: boolean;
    attempts: number;
    accepted: number;
    rejected: number;
    arc_direction: string;
    last_latency_ms?: number | null;
    last_guidance?: {
      code?: string;
      recommendation?: string;
      match_count?: number;
      geometric_inlier_ratio?: number;
      median_motion_px?: number;
    } | null;
    last_quality?: {
      blur_score?: number;
      mean_intensity?: number;
      dark_fraction?: number;
      bright_fraction?: number;
    } | null;
    latest_image?: string | null;
    recording: boolean;
    video_started_at?: number | null;
    latest_video?: string | null;
    video_metadata?: {
      duration_s?: number;
      frames?: number;
      fps?: number;
      width?: number;
      height?: number;
      stored_bytes?: number;
    } | null;
  };
};

const sessions = {
  baseline: {
    label: "Session 002",
    registered: 59,
    extracted: 81,
    sparsePoints: 4095,
    observations: 24332,
    reprojection: 1.414,
    densePoints: 232999,
    meshFaces: 114243,
  },
  refined: {
    label: "Session 003",
    registered: 173,
    extracted: 173,
    sparsePoints: 31685,
    observations: 220309,
    reprojection: 0.856,
    densePoints: 4060810,
    meshFaces: 1198446,
  },
};

const stages = [
  { name: "Capture", detail: "89.17 s / 2,675 frames", icon: Video },
  { name: "Quality gate", detail: "173 accepted / 6 rejected", icon: ScanLine },
  { name: "Sparse SfM", detail: "173 cameras / 31,685 points", icon: Camera },
  { name: "Dense MVS", detail: "173 depth maps / 4.06 M points", icon: Database },
  { name: "Mesh", detail: "1.20 M faces", icon: Triangle },
  { name: "Metric export", detail: "84.14 mm reference / Z-up", icon: Gauge },
];

function formatCompact(value: number) {
  return new Intl.NumberFormat("en-US", {
    notation: value >= 100000 ? "compact" : "standard",
    maximumFractionDigits: value >= 100000 ? 2 : 0,
  }).format(value);
}

function ModelViewer() {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    controls?: OrbitControls;
    model?: THREE.Object3D;
    frame?: number;
    reset?: () => void;
  }>({});
  const [autoRotate, setAutoRotate] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [grid, setGrid] = useState(true);
  const [status, setStatus] = useState("Loading metric mesh");

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#c8cec7");
    scene.fog = new THREE.Fog("#c8cec7", 2.4, 4.5);

    const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 20);
    camera.position.set(0.62, 0.42, 0.68);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.autoRotateSpeed = 0.8;
    controls.minDistance = 0.35;
    controls.maxDistance = 2.4;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.set(0, 0.035, 0);

    scene.add(new THREE.HemisphereLight("#f3f6ee", "#697166", 2.1));
    const key = new THREE.DirectionalLight("#fff7e2", 2.4);
    key.position.set(1.5, 2.2, 1.4);
    key.castShadow = true;
    scene.add(key);
    const fill = new THREE.DirectionalLight("#b9d9df", 1.2);
    fill.position.set(-1.4, 0.7, -1.2);
    scene.add(fill);

    const gridHelper = new THREE.GridHelper(0.6, 12, "#65716a", "#9ba39d");
    gridHelper.name = "metric-grid";
    gridHelper.position.y = -0.002;
    scene.add(gridHelper);

    const loader = new GLTFLoader();
    loader.load(
      "/assets/reconstruction.glb?v=session-003-tape",
      (gltf) => {
        const model = gltf.scene;
        model.rotation.x = -Math.PI / 2;
        model.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            child.material.side = THREE.DoubleSide;
          }
        });
        scene.add(model);
        stateRef.current.model = model;

        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        model.position.sub(center);
        model.position.y += box.getSize(new THREE.Vector3()).y / 2;
        controls.target.set(0, 0.035, 0);
        setStatus("Metric model ready");
      },
      undefined,
      () => setStatus("Model unavailable"),
    );

    const reset = () => {
      camera.position.set(0.62, 0.42, 0.68);
      controls.target.set(0, 0.035, 0);
      controls.update();
    };

    const resize = () => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      stateRef.current.frame = requestAnimationFrame(animate);
    };
    animate();
    stateRef.current = { renderer, scene, camera, controls, frame: stateRef.current.frame, reset };

    return () => {
      observer.disconnect();
      if (stateRef.current.frame) cancelAnimationFrame(stateRef.current.frame);
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    if (stateRef.current.controls) stateRef.current.controls.autoRotate = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    stateRef.current.scene?.getObjectByName("metric-grid")?.setVisible?.(grid);
    const helper = stateRef.current.scene?.getObjectByName("metric-grid");
    if (helper) helper.visible = grid;
  }, [grid]);

  useEffect(() => {
    stateRef.current.model?.traverse((child) => {
      if (child instanceof THREE.Mesh) child.material.wireframe = wireframe;
    });
  }, [wireframe]);

  const fullscreen = async () => {
    if (mountRef.current?.requestFullscreen) await mountRef.current.requestFullscreen();
  };

  return (
    <section className="viewer-shell" aria-label="Interactive metric reconstruction">
      <div className="viewer-toolbar">
        <div className="viewer-status">
          <span className="status-dot" />
          {status}
        </div>
        <div className="tool-actions">
          <button
            className={autoRotate ? "icon-button active" : "icon-button"}
            onClick={() => setAutoRotate((value) => !value)}
            title={autoRotate ? "Pause rotation" : "Auto rotate"}
          >
            {autoRotate ? <Pause size={17} /> : <Play size={17} />}
          </button>
          <button
            className={grid ? "icon-button active" : "icon-button"}
            onClick={() => setGrid((value) => !value)}
            title="Toggle metric grid"
          >
            <Grid3X3 size={17} />
          </button>
          <button
            className={wireframe ? "icon-button active" : "icon-button"}
            onClick={() => setWireframe((value) => !value)}
            title="Toggle wireframe"
          >
            <Triangle size={17} />
          </button>
          <button
            className="icon-button"
            onClick={() => stateRef.current.reset?.()}
            title="Reset view"
          >
            <RotateCcw size={17} />
          </button>
          <button className="icon-button" onClick={fullscreen} title="Fullscreen">
            <Maximize2 size={17} />
          </button>
        </div>
      </div>
      <div className="viewer-canvas" ref={mountRef} />
      <div className="viewer-scale">
        <span>100 mm</span>
        <i />
      </div>
      <div className="axis-badge">Z-up · meters</div>
    </section>
  );
}

function Metric({ label, value, detail, tone = "neutral" }: {
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "green" | "amber";
}) {
  return (
    <div className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function ReconstructionView() {
  return (
    <>
      <div className="workspace">
        <ModelViewer />
        <aside className="inspector">
          <div className="inspector-heading">
            <div>
              <span className="eyebrow">Active reconstruction</span>
              <h2>Session 003</h2>
            </div>
            <span className="run-state"><Check size={14} /> Complete</span>
          </div>

          <div className="inspector-section">
            <h3>Geometry</h3>
            <dl className="property-list">
              <div><dt>Dense points</dt><dd>4,060,810</dd></div>
              <div><dt>Full mesh</dt><dd>1,198,446 faces</dd></div>
              <div><dt>Portfolio mesh</dt><dd>63,605 faces</dd></div>
              <div><dt>Scene footprint</dt><dd>400 x 400 mm</dd></div>
              <div><dt>Reference height</dt><dd>84.14 mm</dd></div>
            </dl>
          </div>

          <div className="inspector-section">
            <h3>Estimation quality</h3>
            <div className="quality-row">
              <CircleGauge size={22} />
              <div>
                <strong>0.856 px</strong>
                <span>Mean reprojection error</span>
              </div>
            </div>
            <div className="quality-row">
              <Focus size={22} />
              <div>
                <strong>100%</strong>
                <span>Registered image ratio</span>
              </div>
            </div>
          </div>

          <div className="inspector-section">
            <h3>Compute profile</h3>
            <dl className="property-list">
              <div><dt>Dense peak memory</dt><dd>4.11 GB</dd></div>
              <div><dt>Mesh peak memory</dt><dd>2.63 GB</dd></div>
              <div><dt>Dense backend</dt><dd>OpenMVS CPU</dd></div>
            </dl>
          </div>
        </aside>
      </div>

      <div className="metrics-band">
        <Metric label="Registered views" value="173 / 173" detail="100% of accepted frames" tone="green" />
        <Metric label="Sparse landmarks" value="31,685" detail="7.74x baseline" />
        <Metric label="Dense points" value="4.06 M" detail="17.4x baseline" tone="green" />
        <Metric label="Metric accuracy" value="0.13%" detail="Validation against casing height" tone="amber" />
      </div>
    </>
  );
}

function ComparisonBar({ label, baseline, refined, lowerIsBetter = false, unit = "" }: {
  label: string;
  baseline: number;
  refined: number;
  lowerIsBetter?: boolean;
  unit?: string;
}) {
  const max = Math.max(baseline, refined);
  const displayValue = (value: number) =>
    value < 10 ? value.toFixed(3) : formatCompact(value);
  return (
    <div className="comparison-row">
      <div className="comparison-label">
        <strong>{label}</strong>
        <span>{lowerIsBetter ? "Lower is better" : "Higher is better"}</span>
      </div>
      <div className="bar-pair">
        <div className="bar-line">
          <span>002</span>
          <div className="bar-track"><i className="bar-baseline" style={{ width: `${(baseline / max) * 100}%` }} /></div>
          <b>{displayValue(baseline)}{unit}</b>
        </div>
        <div className="bar-line">
          <span>003</span>
          <div className="bar-track"><i className="bar-refined" style={{ width: `${(refined / max) * 100}%` }} /></div>
          <b>{displayValue(refined)}{unit}</b>
        </div>
      </div>
    </div>
  );
}

function EvaluationView() {
  const registrationGain = Math.round(
    (sessions.refined.registered / sessions.refined.extracted -
      sessions.baseline.registered / sessions.baseline.extracted) *
      100,
  );
  return (
    <div className="evaluation-layout">
      <section className="evaluation-main">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Controlled experiment</span>
            <h2>Capture quality changed the map</h2>
          </div>
          <div className="legend">
            <span><i className="legend-baseline" /> Session 002</span>
            <span><i className="legend-refined" /> Session 003</span>
          </div>
        </div>
        <div className="comparison-list">
          <ComparisonBar label="Registered images" baseline={59} refined={173} />
          <ComparisonBar label="Sparse points" baseline={4095} refined={31685} />
          <ComparisonBar label="Observations" baseline={24332} refined={220309} />
          <ComparisonBar label="Dense points" baseline={232999} refined={4060810} />
          <ComparisonBar label="Mesh faces" baseline={114243} refined={1198446} />
          <ComparisonBar label="Reprojection error" baseline={1.414} refined={0.856} lowerIsBetter unit=" px" />
        </div>
      </section>

      <aside className="findings-panel">
        <h3>Measured outcome</h3>
        <div className="finding">
          <strong>+{registrationGain} pp</strong>
          <span>Registration improvement</span>
        </div>
        <div className="finding">
          <strong>9.05×</strong>
          <span>More feature observations</span>
        </div>
        <div className="finding">
          <strong>−39.4%</strong>
          <span>Lower reprojection error</span>
        </div>
        <div className="finding-note">
          <Activity size={18} />
          <p>Slow motion, consistent distance, stable lighting, and continuous overlap produced one connected 173-camera model.</p>
        </div>
      </aside>
    </div>
  );
}

function SystemView() {
  return (
    <div className="system-layout">
      <section className="pipeline-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Field-to-laptop workflow</span>
            <h2>System execution trace</h2>
          </div>
          <span className="run-state"><Check size={14} /> 6 / 6 stages</span>
        </div>
        <div className="stage-list">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            return (
              <div className="stage" key={stage.name}>
                <span className="stage-index">{String(index + 1).padStart(2, "0")}</span>
                <Icon size={20} />
                <div><strong>{stage.name}</strong><span>{stage.detail}</span></div>
                <Check size={16} className="stage-check" />
              </div>
            );
          })}
        </div>
      </section>

      <section className="video-panel">
        <div className="video-heading">
          <div>
            <span className="eyebrow">Deployment evidence</span>
            <h2>Portfolio rotation</h2>
          </div>
          <span>30.3 s</span>
        </div>
        <video controls loop muted playsInline preload="metadata">
          <source src="/media/portfolio-clip.mp4" type="video/mp4" />
        </video>
      </section>

      <section className="deployment-strip">
        <div><Cpu size={21} /><strong>Intel i7 CPU</strong><span>Dense MVS fallback</span></div>
        <div><Server size={21} /><strong>Raspberry Pi 5</strong><span>Field capture target</span></div>
        <div><Wifi size={21} /><strong>Portable link</strong><span>Wi-Fi result serving</span></div>
        <div><Box size={21} /><strong>Metric PLY / GLB</strong><span>Deployment artifacts</span></div>
      </section>
    </div>
  );
}

function bytes(value?: number | null) {
  if (value == null) return "Unavailable";
  const units = ["B", "KB", "MB", "GB"];
  let current = value;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  return `${current.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function duration(value?: number | null) {
  if (value == null) return "N/A";
  const seconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function FieldView() {
  const [status, setStatus] = useState<FieldStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(
    `field_${new Date().toISOString().slice(0, 10).replaceAll("-", "")}`,
  );
  const [direction, setDirection] = useState("right");
  const [imageVersion, setImageVersion] = useState(0);
  const [clock, setClock] = useState(Date.now());

  const refresh = async () => {
    try {
      const response = await fetch("/api/field/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Coordinator returned ${response.status}`);
      const payload = (await response.json()) as FieldStatus;
      setStatus(payload);
      setError("");
    } catch {
      setStatus(null);
      setError("Pi coordinator is offline. Start the coordinator to enable field controls.");
    }
  };

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setImageVersion((value) => value + 1);
      setClock(Date.now());
    }, 750);
    return () => window.clearInterval(timer);
  }, []);

  const command = async (path: string, body?: object) => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Command failed");
      await refresh();
      setImageVersion((value) => value + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Command failed");
    } finally {
      setBusy(false);
    }
  };

  const session = status?.session;
  const recordingDuration =
    session?.recording && session.video_started_at
      ? clock / 1000 - session.video_started_at
      : session?.video_metadata?.duration_s;
  const memoryPercent =
    status?.pi.memory_total_bytes && status.pi.memory_used_bytes
      ? (status.pi.memory_used_bytes / status.pi.memory_total_bytes) * 100
      : null;

  return (
    <div className="field-layout">
      <section className="field-preview-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Video acquisition</span>
            <h2>Field recording console</h2>
          </div>
          <span className={status?.camera.connected ? "run-state" : "run-state offline"}>
            <Radio size={14} />
            {status?.camera.connected ? "Camera online" : "Offline"}
          </span>
        </div>

        <div className="live-frame">
          {session?.latest_video && !session.recording ? (
            <video
              key={session.latest_video}
              src="/api/field/video.mp4"
              controls
              playsInline
            />
          ) : status?.camera.connected ? (
            <img
              src={`/api/field/preview.jpg?v=${imageVersion}`}
              alt="Live laptop webcam preview"
            />
          ) : (
            <div className="frame-placeholder">
              <Video size={34} />
              <strong>Camera preview unavailable</strong>
              <span>Start the laptop camera node and check the network link.</span>
            </div>
          )}
          <div className="frame-badges">
            <span>{session ? session.session_id : "No active session"}</span>
            <span className={session?.recording ? "recording-badge" : ""}>
              {session?.recording
                ? `REC ${duration(recordingDuration)}`
                : session?.latest_video
                  ? "Recording ready"
                  : "Live preview"}
            </span>
          </div>
        </div>

        <div className={`guidance-banner ${session?.recording ? "recording-state" : ""}`}>
          <Video size={22} />
          <div>
            <span>{session?.recording ? "Recording in progress" : "Video workflow"}</span>
            <strong>
              {session?.recording
                ? `Carry the camera slowly ${direction} around the stationary target. Keep 60-80% overlap.`
                : session?.latest_video
                  ? "Video saved on the Pi. Download it or finish the session for reconstruction."
                  : "Keep the target still, then physically carry the camera around it."}
            </strong>
          </div>
        </div>

        <div className="capture-controls">
          <label>
            <span>Session ID</span>
            <input
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value)}
              disabled={Boolean(session?.active)}
            />
          </label>
          <label>
            <span>Arc direction</span>
            <select
              value={direction}
              onChange={(event) => setDirection(event.target.value)}
              disabled={Boolean(session?.active)}
            >
              <option value="right">Move right</option>
              <option value="left">Move left</option>
            </select>
          </label>
          <div className="command-buttons">
            {!session?.active ? (
              <button
                className="command primary"
                disabled={busy || !status?.camera.connected}
                onClick={() =>
                  command("/api/field/session/start", {
                    session_id: sessionId,
                    arc_direction: direction,
                  })
                }
              >
                <Play size={17} /> Start session
              </button>
            ) : session.recording ? (
              <button
                className="command danger"
                disabled={busy}
                onClick={() => command("/api/field/video/stop")}
              >
                <Square size={15} /> Stop recording
              </button>
            ) : !session.latest_video ? (
              <button
                className="command record"
                disabled={busy}
                onClick={() => command("/api/field/video/start", { fps: 15 })}
              >
                <Video size={17} /> Start recording
              </button>
            ) : (
              <>
                <a
                  className="command primary"
                  href="/api/field/video.mp4"
                  download={`${session.session_id}.mp4`}
                >
                  <Download size={17} /> Download video
                </a>
                <button
                  className="command"
                  disabled={busy}
                  onClick={() => command("/api/field/session/stop")}
                >
                  <Check size={15} /> Finish session
                </button>
              </>
            )}
          </div>
        </div>
        {error && <div className="field-error">{error}</div>}
      </section>

      <aside className="field-telemetry">
        <div className="telemetry-heading">
          <span className="eyebrow">Raspberry Pi 5</span>
          <h2>Node telemetry</h2>
        </div>
        <div className="telemetry-grid">
          <Metric
            label="Temperature"
            value={
              status?.pi.temperature_c == null
                ? "N/A"
                : `${status.pi.temperature_c.toFixed(1)} C`
            }
            detail="SoC thermal sensor"
            tone="amber"
          />
          <Metric
            label="Load"
            value={status?.pi.load_1m == null ? "N/A" : status.pi.load_1m.toFixed(2)}
            detail="One-minute average"
          />
          <Metric
            label="Memory"
            value={memoryPercent == null ? "N/A" : `${memoryPercent.toFixed(0)}%`}
            detail={
              memoryPercent == null
                ? "Unavailable"
                : `${bytes(status?.pi.memory_used_bytes)} used`
            }
          />
          <Metric
            label="Disk free"
            value={bytes(status?.pi.disk_free_bytes)}
            detail="Session storage"
            tone="green"
          />
        </div>

        <div className="field-stats">
          <h3>Recording state</h3>
          <dl className="property-list">
            <div><dt>Status</dt><dd>{session?.recording ? "Recording" : session?.latest_video ? "Ready" : "Idle"}</dd></div>
            <div><dt>Duration</dt><dd>{duration(recordingDuration)}</dd></div>
            <div><dt>Frames</dt><dd>{status?.camera.video?.frames ?? session?.video_metadata?.frames ?? 0}</dd></div>
            <div><dt>Frame rate</dt><dd>{session?.video_metadata?.fps ?? status?.camera.video?.fps ?? 15} fps</dd></div>
            <div><dt>Resolution</dt><dd>{session?.video_metadata?.width ? `${session.video_metadata.width} x ${session.video_metadata.height}` : "Live camera"}</dd></div>
            <div><dt>Video size</dt><dd>{bytes(session?.video_metadata?.stored_bytes)}</dd></div>
            <div><dt>Laptop link</dt><dd>{status?.camera.connected ? "Online" : "Offline"}</dd></div>
            <div><dt>Target overlap</dt><dd>60-80%</dd></div>
          </dl>
        </div>

        <div className="field-rule">
          <Thermometer size={20} />
          <p>Move the camera for 45-90 seconds. A fixed camera with a rotating person or object cannot be reconstructed.</p>
        </div>
      </aside>
    </div>
  );
}

function App() {
  const [view, setView] = useState<ViewName>("reconstruction");
  const viewContent = useMemo(() => {
    if (view === "evaluation") return <EvaluationView />;
    if (view === "system") return <SystemView />;
    if (view === "field") return <FieldView />;
    return <ReconstructionView />;
  }, [view]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Box size={19} /></div>
          <div><strong>ReconBot</strong><span>Portable monocular mapping</span></div>
        </div>
        <nav className="view-tabs" aria-label="Dashboard views">
          <button className={view === "reconstruction" ? "selected" : ""} onClick={() => setView("reconstruction")}>
            <Box size={16} /> Reconstruction
          </button>
          <button className={view === "evaluation" ? "selected" : ""} onClick={() => setView("evaluation")}>
            <Activity size={16} /> Evaluation
          </button>
          <button className={view === "system" ? "selected" : ""} onClick={() => setView("system")}>
            <Cpu size={16} /> System
          </button>
          <button className={view === "field" ? "selected" : ""} onClick={() => setView("field")}>
            <Radio size={16} /> Field
          </button>
        </nav>
        <div className="header-status">
          <span className="status-dot" />
          Session 003
        </div>
      </header>
      <div className="page-content">{viewContent}</div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
