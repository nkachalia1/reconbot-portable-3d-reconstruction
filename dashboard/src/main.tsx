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
  Film,
  Focus,
  Gauge,
  Grid3X3,
  LoaderCircle,
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
  Trash2,
  Video,
  Wifi,
} from "lucide-react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./styles.css";

type ViewName = "reconstruction" | "evaluation" | "system" | "field";

type ReconstructionMetrics = {
  video_duration_s?: number | null;
  video_frames?: number | null;
  sampled_frames?: number;
  extracted_frames?: number;
  blur_rejected_samples?: number;
  registered_images?: number;
  registration_ratio?: number;
  sparse_points?: number;
  observations?: number;
  mean_track_length?: number;
  mean_reprojection_error_px?: number;
  dense_depth_maps?: number;
  dense_points?: number;
  mesh_vertices?: number;
  mesh_faces?: number;
  full_mesh_vertices?: number;
  full_mesh_faces?: number;
  processing_time_s?: number;
  metric_accuracy_percent?: number;
  reference_height_mm?: number;
};

type ReconstructionRecord = {
  id: string;
  title: string;
  created_at: string;
  status: "complete";
  source: "field" | "portfolio";
  model_url: string;
  video_url?: string | null;
  metrics: ReconstructionMetrics;
  viewer?: {
    rotation_x?: number;
    up_axis?: string;
    scale_label?: string;
    metric?: boolean;
  };
};

type ReconstructionCatalog = {
  active_id: string | null;
  items: ReconstructionRecord[];
};

type ReconstructionJob = {
  id: string;
  reconstruction_id: string;
  status: "queued" | "running" | "complete" | "failed";
  stage: string;
  progress: number;
  message: string;
  error?: string | null;
  result?: ReconstructionRecord | null;
};

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
  reconstruction?: {
    connected?: boolean;
    error?: string;
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
    reconstruction_job_id?: string | null;
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

function formatCompact(value: number) {
  return new Intl.NumberFormat("en-US", {
    notation: value >= 100000 ? "compact" : "standard",
    maximumFractionDigits: value >= 100000 ? 2 : 0,
  }).format(value);
}

function ModelViewer({ reconstruction }: { reconstruction: ReconstructionRecord }) {
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
  const [status, setStatus] = useState("Loading reconstruction");

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
    setStatus("Loading reconstruction");
    loader.load(
      `${reconstruction.model_url}?v=${encodeURIComponent(reconstruction.id)}`,
      (gltf) => {
        const model = gltf.scene;
        model.rotation.x = reconstruction.viewer?.rotation_x ?? 0;
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
        const size = box.getSize(new THREE.Vector3());
        const largestDimension = Math.max(size.x, size.y, size.z) || 1;
        const presentationScale = 0.62 / largestDimension;
        model.scale.setScalar(presentationScale);
        model.position.set(
          -center.x * presentationScale,
          -box.min.y * presentationScale,
          -center.z * presentationScale,
        );
        controls.target.set(0, size.y * presentationScale * 0.36, 0);
        setStatus("Reconstruction ready");
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
  }, [reconstruction.id, reconstruction.model_url, reconstruction.viewer?.rotation_x]);

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
        <span>{reconstruction.viewer?.scale_label ?? "Auto-fit"}</span>
        <i />
      </div>
      <div className="axis-badge">
        {reconstruction.viewer?.up_axis ?? "Y-up"} ·{" "}
        {reconstruction.viewer?.metric ? "meters" : "SfM units"}
      </div>
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

function metricValue(value?: number | null, suffix = "") {
  if (value == null) return "N/A";
  const formatted =
    Math.abs(value) < 1000 && !Number.isInteger(value)
      ? value.toFixed(2).replace(/\.?0+$/, "")
      : formatCompact(value);
  return `${formatted}${suffix}`;
}

function ReconstructionLibrary({
  records,
  activeId,
  workerOnline,
  onActivate,
  onDelete,
}: {
  records: ReconstructionRecord[];
  activeId: string;
  workerOnline: boolean;
  onActivate: (identifier: string) => Promise<void>;
  onDelete: (identifier: string) => Promise<void>;
}) {
  const active = records.find((record) => record.id === activeId);
  return (
    <section className="library-layout" aria-label="Past reconstructions">
      <div className="history-panel">
        <div className="section-heading compact-heading">
          <div>
            <span className="eyebrow">Saved runs</span>
            <h2>Reconstruction history</h2>
          </div>
          <span>{records.length} saved</span>
        </div>
        <div className="history-list">
          {records.map((record) => (
            <article
              className={record.id === activeId ? "history-item selected" : "history-item"}
              key={record.id}
            >
              <button className="history-select" onClick={() => onActivate(record.id)}>
                <span className="history-icon"><Box size={19} /></span>
                <span>
                  <strong>{record.title}</strong>
                  <small>
                    {new Date(record.created_at).toLocaleDateString()} ·{" "}
                    {metricValue(record.metrics.registered_images)} registered views
                  </small>
                </span>
                <span className="history-state">
                  {record.id === activeId ? "Viewing" : "Open model"}
                </span>
              </button>
              <button
                className="icon-button history-delete"
                title={`Delete ${record.title}`}
                disabled={!workerOnline}
                onClick={() => onDelete(record.id)}
              >
                <Trash2 size={16} />
              </button>
            </article>
          ))}
        </div>
      </div>
      <div className="history-video">
        <div className="video-heading">
          <div>
            <span className="eyebrow">Source recording</span>
            <h2>{active?.title ?? "No active reconstruction"}</h2>
          </div>
          <Film size={18} />
        </div>
        {active?.video_url ? (
          <video key={active.id} src={active.video_url} controls playsInline preload="metadata" />
        ) : (
          <div className="video-unavailable">
            <Video size={28} />
            <span>No saved source video</span>
          </div>
        )}
      </div>
    </section>
  );
}

function ReconstructionView({
  reconstruction,
  records,
  workerOnline,
  onActivate,
  onDelete,
}: {
  reconstruction: ReconstructionRecord;
  records: ReconstructionRecord[];
  workerOnline: boolean;
  onActivate: (identifier: string) => Promise<void>;
  onDelete: (identifier: string) => Promise<void>;
}) {
  const metrics = reconstruction.metrics;
  const registered = metrics.registered_images ?? 0;
  const extracted = metrics.extracted_frames ?? registered;
  const registrationRatio =
    metrics.registration_ratio ?? (extracted ? registered / extracted : 0);
  return (
    <>
      <div className="workspace">
        <ModelViewer reconstruction={reconstruction} />
        <aside className="inspector">
          <div className="inspector-heading">
            <div>
              <span className="eyebrow">Active reconstruction</span>
              <h2>{reconstruction.title}</h2>
            </div>
            <span className="run-state"><Check size={14} /> Complete</span>
          </div>

          <div className="inspector-section">
            <h3>Geometry</h3>
            <dl className="property-list">
              <div><dt>Dense points</dt><dd>{metricValue(metrics.dense_points)}</dd></div>
              <div><dt>Mesh vertices</dt><dd>{metricValue(metrics.mesh_vertices)}</dd></div>
              <div><dt>Mesh faces</dt><dd>{metricValue(metrics.mesh_faces)}</dd></div>
              <div><dt>Sparse landmarks</dt><dd>{metricValue(metrics.sparse_points)}</dd></div>
              <div>
                <dt>Reference height</dt>
                <dd>{metricValue(metrics.reference_height_mm, " mm")}</dd>
              </div>
            </dl>
          </div>

          <div className="inspector-section">
            <h3>Estimation quality</h3>
            <div className="quality-row">
              <CircleGauge size={22} />
              <div>
                <strong>
                  {metrics.mean_reprojection_error_px == null
                    ? "N/A"
                    : `${metrics.mean_reprojection_error_px.toFixed(3)} px`}
                </strong>
                <span>Mean reprojection error</span>
              </div>
            </div>
            <div className="quality-row">
              <Focus size={22} />
              <div>
                <strong>{(registrationRatio * 100).toFixed(1)}%</strong>
                <span>Registered image ratio</span>
              </div>
            </div>
          </div>

          <div className="inspector-section">
            <h3>Compute profile</h3>
            <dl className="property-list">
              <div><dt>Source duration</dt><dd>{duration(metrics.video_duration_s)}</dd></div>
              <div><dt>Processing time</dt><dd>{duration(metrics.processing_time_s)}</dd></div>
              <div><dt>Dense backend</dt><dd>OpenMVS CPU</dd></div>
            </dl>
          </div>
        </aside>
      </div>

      <div className="metrics-band">
        <Metric
          label="Registered views"
          value={`${registered} / ${extracted}`}
          detail={`${(registrationRatio * 100).toFixed(1)}% of accepted frames`}
          tone="green"
        />
        <Metric
          label="Sparse landmarks"
          value={metricValue(metrics.sparse_points)}
          detail={metricValue(metrics.observations) + " feature observations"}
        />
        <Metric
          label="Dense points"
          value={metricValue(metrics.dense_points)}
          detail={metricValue(metrics.mesh_faces) + " mesh faces"}
          tone="green"
        />
        <Metric
          label={metrics.metric_accuracy_percent == null ? "Mean track length" : "Metric accuracy"}
          value={
            metrics.metric_accuracy_percent == null
              ? metricValue(metrics.mean_track_length)
              : `${metrics.metric_accuracy_percent.toFixed(2)}%`
          }
          detail={
            metrics.metric_accuracy_percent == null
              ? "Views observing each sparse point"
              : "Validation against physical reference"
          }
          tone="amber"
        />
      </div>
      <ReconstructionLibrary
        records={records}
        activeId={reconstruction.id}
        workerOnline={workerOnline}
        onActivate={onActivate}
        onDelete={onDelete}
      />
    </>
  );
}

function ComparisonBar({ label, reference, selected, lowerIsBetter = false, unit = "" }: {
  label: string;
  reference: number;
  selected: number;
  lowerIsBetter?: boolean;
  unit?: string;
}) {
  const max = Math.max(reference, selected, Number.EPSILON);
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
          <span>REF</span>
          <div className="bar-track"><i className="bar-baseline" style={{ width: `${(reference / max) * 100}%` }} /></div>
          <b>{displayValue(reference)}{unit}</b>
        </div>
        <div className="bar-line">
          <span>ACTIVE</span>
          <div className="bar-track"><i className="bar-refined" style={{ width: `${(selected / max) * 100}%` }} /></div>
          <b>{displayValue(selected)}{unit}</b>
        </div>
      </div>
    </div>
  );
}

function EvaluationView({
  reconstruction,
  records,
}: {
  reconstruction: ReconstructionRecord;
  records: ReconstructionRecord[];
}) {
  const reference =
    records.find((record) => record.id !== reconstruction.id) ?? reconstruction;
  const selectedMetrics = reconstruction.metrics;
  const referenceMetrics = reference.metrics;
  const selectedRegistration =
    selectedMetrics.registration_ratio ??
    (selectedMetrics.extracted_frames
      ? (selectedMetrics.registered_images ?? 0) / selectedMetrics.extracted_frames
      : 0);
  const referenceRegistration =
    referenceMetrics.registration_ratio ??
    (referenceMetrics.extracted_frames
      ? (referenceMetrics.registered_images ?? 0) / referenceMetrics.extracted_frames
      : 0);
  const registrationDelta = (selectedRegistration - referenceRegistration) * 100;
  const observationRatio =
    (selectedMetrics.observations ?? 0) /
    Math.max(referenceMetrics.observations ?? 0, 1);
  const reprojectionChange =
    (((selectedMetrics.mean_reprojection_error_px ?? 0) -
      (referenceMetrics.mean_reprojection_error_px ?? 0)) /
      Math.max(
        referenceMetrics.mean_reprojection_error_px ?? 0,
        Number.EPSILON,
      )) *
    100;
  return (
    <div className="evaluation-layout">
      <section className="evaluation-main">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Selected reconstruction comparison</span>
            <h2>{reconstruction.title}</h2>
          </div>
          <div className="legend">
            <span><i className="legend-baseline" /> {reference.title}</span>
            <span><i className="legend-refined" /> {reconstruction.title}</span>
          </div>
        </div>
        <div className="comparison-list">
          <ComparisonBar
            label="Registered images"
            reference={referenceMetrics.registered_images ?? 0}
            selected={selectedMetrics.registered_images ?? 0}
          />
          <ComparisonBar
            label="Sparse points"
            reference={referenceMetrics.sparse_points ?? 0}
            selected={selectedMetrics.sparse_points ?? 0}
          />
          <ComparisonBar
            label="Observations"
            reference={referenceMetrics.observations ?? 0}
            selected={selectedMetrics.observations ?? 0}
          />
          <ComparisonBar
            label="Dense points"
            reference={referenceMetrics.dense_points ?? 0}
            selected={selectedMetrics.dense_points ?? 0}
          />
          <ComparisonBar
            label="Published mesh faces"
            reference={referenceMetrics.mesh_faces ?? 0}
            selected={selectedMetrics.mesh_faces ?? 0}
          />
          <ComparisonBar
            label="Reprojection error"
            reference={referenceMetrics.mean_reprojection_error_px ?? 0}
            selected={selectedMetrics.mean_reprojection_error_px ?? 0}
            lowerIsBetter
            unit=" px"
          />
        </div>
      </section>

      <aside className="findings-panel">
        <h3>Selected run outcome</h3>
        <div className="finding">
          <strong>
            {registrationDelta > 0 ? "+" : ""}
            {registrationDelta.toFixed(1)} pp
          </strong>
          <span>Registration rate versus saved reference</span>
        </div>
        <div className="finding">
          <strong>{observationRatio.toFixed(2)}x</strong>
          <span>Feature observations versus saved reference</span>
        </div>
        <div className="finding">
          <strong>{Math.abs(reprojectionChange).toFixed(1)}%</strong>
          <span>
            {reprojectionChange <= 0 ? "Lower" : "Higher"} reprojection error
            versus reference
          </span>
        </div>
        <div className="finding-note">
          <Activity size={18} />
          <p>
            {reconstruction.title} registered{" "}
            {selectedMetrics.registered_images ?? 0} of{" "}
            {selectedMetrics.extracted_frames ?? 0} accepted frames and produced{" "}
            {metricValue(selectedMetrics.dense_points)} dense points.
          </p>
        </div>
      </aside>
    </div>
  );
}

function SystemView({ reconstruction }: { reconstruction: ReconstructionRecord | null }) {
  if (!reconstruction) return null;
  const metrics = reconstruction.metrics;
  const rejected = metrics.blur_rejected_samples;
  const fullFaces = metrics.full_mesh_faces;
  const publishedFaces = metrics.mesh_faces;
  const stages = [
    {
      name: "Capture",
      detail: `${duration(metrics.video_duration_s)} / ${
        metrics.video_frames == null
          ? "frame count unavailable"
          : `${formatCompact(metrics.video_frames)} frames`
      }`,
      icon: Video,
    },
    {
      name: "Quality gate",
      detail: `${metricValue(metrics.extracted_frames)} accepted${
        rejected == null ? "" : ` / ${metricValue(rejected)} rejected`
      }`,
      icon: ScanLine,
    },
    {
      name: "Sparse SfM",
      detail: `${metricValue(metrics.registered_images)} cameras / ${metricValue(metrics.sparse_points)} points`,
      icon: Camera,
    },
    {
      name: "Dense MVS",
      detail: `${metricValue(metrics.dense_depth_maps)} depth maps / ${metricValue(metrics.dense_points)} points`,
      icon: Database,
    },
    {
      name: "Mesh",
      detail:
        fullFaces != null && fullFaces !== publishedFaces
          ? `${metricValue(fullFaces)} full / ${metricValue(publishedFaces)} published faces`
          : `${metricValue(publishedFaces)} published faces`,
      icon: Triangle,
    },
    {
      name: reconstruction.viewer?.metric ? "Metric export" : "GLB export",
      detail: reconstruction.viewer?.metric
        ? `${metricValue(metrics.reference_height_mm, " mm")} reference / ${reconstruction.viewer?.up_axis ?? "Z-up"}`
        : `Textured mesh / ${reconstruction.viewer?.up_axis ?? "Y-up"} SfM units`,
      icon: Gauge,
    },
  ];
  return (
    <div className="system-layout">
      <section className="pipeline-panel">
        <div className="section-heading">
          <div>
            <span className="eyebrow">{reconstruction.title}</span>
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
            <h2>{reconstruction.title}</h2>
          </div>
          <span>{duration(metrics.video_duration_s)}</span>
        </div>
        {reconstruction.video_url ? (
          <video
            key={reconstruction.id}
            src={reconstruction.video_url}
            controls
            loop
            muted
            playsInline
            preload="metadata"
          />
        ) : (
          <div className="video-unavailable">
            <Video size={28} />
            <span>No saved source video</span>
          </div>
        )}
      </section>

      <section className="deployment-strip">
        <div><Cpu size={21} /><strong>Intel i7 CPU</strong><span>Dense MVS fallback</span></div>
        <div>
          <Server size={21} />
          <strong>{reconstruction.source === "field" ? "Raspberry Pi 5" : "Laptop webcam"}</strong>
          <span>{reconstruction.source === "field" ? "Field coordinator" : "Portfolio capture"}</span>
        </div>
        <div>
          <Wifi size={21} />
          <strong>{reconstruction.source === "field" ? "Portable Wi-Fi link" : "Local workflow"}</strong>
          <span>{reconstruction.source === "field" ? "Remote capture and transfer" : "Laptop capture and processing"}</span>
        </div>
        <div>
          <Box size={21} />
          <strong>{reconstruction.viewer?.metric ? "Metric GLB" : "Textured GLB"}</strong>
          <span>{metricValue(metrics.mesh_faces)} published faces</span>
        </div>
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

function FieldView({
  onReconstructionReady,
}: {
  onReconstructionReady: (identifier: string) => Promise<void>;
}) {
  const [status, setStatus] = useState<FieldStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(
    `field_${new Date().toISOString().slice(0, 10).replaceAll("-", "")}`,
  );
  const [direction, setDirection] = useState("right");
  const [imageVersion, setImageVersion] = useState(0);
  const [clock, setClock] = useState(Date.now());
  const [job, setJob] = useState<ReconstructionJob | null>(null);
  const completedJobRef = useRef<string | null>(null);

  const refresh = async () => {
    try {
      const response = await fetch("/api/field/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Coordinator returned ${response.status}`);
      const payload = (await response.json()) as FieldStatus;
      setStatus(payload);
      if (payload.session?.reconstruction_job_id && !job) {
        const jobResponse = await fetch(
          `/api/reconstruction/jobs/${payload.session.reconstruction_job_id}`,
          { cache: "no-store" },
        );
        if (jobResponse.ok) {
          const jobPayload = await jobResponse.json();
          setJob(jobPayload.job as ReconstructionJob);
        }
      }
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
    if (!job || job.status === "complete" || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/reconstruction/jobs/${job.id}`, {
          cache: "no-store",
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not read job status");
        setJob(payload.job as ReconstructionJob);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not read job status");
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (
      job?.status !== "complete" ||
      !job.result?.id ||
      completedJobRef.current === job.id
    ) {
      return;
    }
    completedJobRef.current = job.id;
    void onReconstructionReady(job.result.id);
  }, [job, onReconstructionReady]);

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

  const reconstruct = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/field/reconstruct", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start reconstruction");
      setJob(payload.job as ReconstructionJob);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start reconstruction");
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
                : job?.status === "running" || job?.status === "queued"
                  ? job.message
                : session?.latest_video
                  ? "Video saved on the Pi. Start reconstruction when the orbit looks usable."
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
                <button
                  className="command primary"
                  disabled={
                    busy ||
                    !status?.reconstruction?.connected ||
                    job?.status === "queued" ||
                    job?.status === "running"
                  }
                  onClick={reconstruct}
                >
                  {job?.status === "queued" || job?.status === "running" ? (
                    <LoaderCircle size={17} className="spin" />
                  ) : (
                    <Box size={17} />
                  )}
                  {job?.status === "queued" || job?.status === "running"
                    ? "Reconstructing"
                    : "Reconstruct video"}
                </button>
                <a
                  className="command"
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
        {job && (
          <div className={`job-status job-${job.status}`}>
            <div>
              <span>{job.stage.replaceAll("_", " ")}</span>
              <strong>{job.message}</strong>
            </div>
            <b>{Math.round(job.progress)}%</b>
            <div className="job-progress">
              <i style={{ width: `${Math.max(2, job.progress)}%` }} />
            </div>
          </div>
        )}
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
            <div><dt>Reconstruction worker</dt><dd>{status?.reconstruction?.connected ? "Online" : "Offline"}</dd></div>
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
  const [catalog, setCatalog] = useState<ReconstructionCatalog>({
    active_id: null,
    items: [],
  });
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [workerOnline, setWorkerOnline] = useState(false);

  const loadCatalog = async (preferredId?: string) => {
    try {
      const response = await fetch("/api/reconstructions", { cache: "no-store" });
      if (!response.ok || !response.headers.get("content-type")?.includes("json")) {
        throw new Error("Dynamic reconstruction API is unavailable");
      }
      const payload = (await response.json()) as ReconstructionCatalog;
      setCatalog({
        ...payload,
        active_id:
          preferredId && payload.items.some((item) => item.id === preferredId)
            ? preferredId
            : payload.active_id,
      });
      setWorkerOnline(true);
    } catch {
      const response = await fetch("/reconstructions/catalog.json", { cache: "no-store" });
      const payload = (await response.json()) as ReconstructionCatalog;
      setCatalog({
        ...payload,
        active_id:
          preferredId && payload.items.some((item) => item.id === preferredId)
            ? preferredId
            : payload.active_id,
      });
      setWorkerOnline(false);
    } finally {
      setCatalogLoading(false);
    }
  };

  useEffect(() => {
    void loadCatalog();
  }, []);

  const activate = async (identifier: string) => {
    if (workerOnline) {
      const response = await fetch(`/api/reconstructions/${identifier}/activate`, {
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not activate reconstruction");
      setCatalog({ active_id: payload.active_id, items: payload.items });
    } else {
      setCatalog((current) => ({ ...current, active_id: identifier }));
    }
  };

  const remove = async (identifier: string) => {
    const record = catalog.items.find((item) => item.id === identifier);
    if (!record || !window.confirm(`Delete ${record.title} and its saved video?`)) return;
    const response = await fetch(`/api/reconstructions/${identifier}`, { method: "DELETE" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not delete reconstruction");
    setCatalog({ active_id: payload.active_id, items: payload.items });
  };

  const reconstructionReady = async (identifier: string) => {
    await loadCatalog(identifier);
    setView("reconstruction");
  };

  const activeReconstruction =
    catalog.items.find((item) => item.id === catalog.active_id) ?? catalog.items[0] ?? null;
  const viewContent = useMemo(() => {
    if (view === "field") {
      return <FieldView onReconstructionReady={reconstructionReady} />;
    }
    if (catalogLoading) {
      return (
        <div className="empty-state">
          <LoaderCircle size={24} className="spin" />
          <strong>Loading reconstruction library</strong>
        </div>
      );
    }
    if (!activeReconstruction) {
      return (
        <div className="empty-state">
          <Box size={28} />
          <strong>No saved reconstructions</strong>
          <span>Record a field video and reconstruct it to create the first entry.</span>
        </div>
      );
    }
    if (view === "evaluation") {
      return (
        <EvaluationView
          reconstruction={activeReconstruction}
          records={catalog.items}
        />
      );
    }
    if (view === "system") {
      return <SystemView reconstruction={activeReconstruction} />;
    }
    return (
      <ReconstructionView
        reconstruction={activeReconstruction}
        records={catalog.items}
        workerOnline={workerOnline}
        onActivate={activate}
        onDelete={remove}
      />
    );
  }, [view, catalog, catalogLoading, workerOnline, activeReconstruction]);

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
          {activeReconstruction?.title ?? "No active reconstruction"}
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
