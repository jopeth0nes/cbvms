import { useCallback, useEffect, useRef, useState } from "react";
import { Camera } from "lucide-react";
import { detectionsWsUrl, fetchLatestDetections, formatViolation } from "../api/detections";
import { streamUrl } from "../api/cameras";
import type { ActiveCamera } from "../types/camera";
import type {
  DetectionAlertMessage,
  DetectionBatchMessage,
  DetectionPayload,
  LiveAlert,
} from "../types/detection";
import "./LiveMonitor.css";

interface LiveMonitorProps {
  activeCamera: ActiveCamera | null;
  streamKey: number;
}

const ALERT_DEDUP_MS = 15_000;

function gradeLine(det: DetectionPayload | null): string {
  if (!det) return "—";
  const parts = [det.year_level, det.course].filter(Boolean);
  return parts.length ? parts.join(" - ") : "—";
}

function alertFromDetection(data: DetectionAlertMessage): LiveAlert {
  return {
    id: `${Date.now()}-${data.student_id}`,
    name: data.name,
    time: data.time,
    face_violations: data.face_violations ?? [],
    torso_violations: data.torso_violations ?? [],
  };
}

function payloadFromAlert(data: DetectionAlertMessage): DetectionPayload {
  const id = data.identity?.id ?? data.student_id;
  const name = data.identity?.name ?? data.name;
  return {
    identity: { id, name },
    face_box: [0, 0, 0, 0],
    torso_box: [0, 0, 0, 0],
    face_violations: data.face_violations ?? [],
    torso_violations: data.torso_violations ?? [],
    all_violations: data.all_violations ?? [],
    year_level: data.grade,
    course: data.section,
  };
}

export function LiveMonitor({ activeCamera, streamKey }: LiveMonitorProps) {
  const [now, setNow] = useState(() => new Date());
  const [primary, setPrimary] = useState<DetectionPayload | null>(null);
  const [alerts, setAlerts] = useState<LiveAlert[]>([]);
  const alertDedupRef = useRef<Map<string, number>>(new Map());

  const addAlert = useCallback((data: DetectionAlertMessage) => {
    const key = data.student_id || data.name;
    const last = alertDedupRef.current.get(key) ?? 0;
    const ts = Date.now();
    if (ts - last < ALERT_DEDUP_MS) {
      return;
    }
    alertDedupRef.current.set(key, ts);

    const alert = alertFromDetection(data);
    setAlerts((prev) => [alert, ...prev].slice(0, 50));
  }, []);

  const applyDetectionBatch = useCallback((list: DetectionPayload[]) => {
    const first = list[0] ?? null;
    setPrimary(first);
  }, []);

  const handleWsMessage = useCallback(
    (raw: string) => {
      try {
        const data = JSON.parse(raw) as DetectionBatchMessage | DetectionAlertMessage;
        if (data.type === "detection") {
          setPrimary(payloadFromAlert(data));
          addAlert(data);
          return;
        }
        if (data.type === "detections" || Array.isArray((data as DetectionBatchMessage).detections)) {
          const batch = data as DetectionBatchMessage;
          applyDetectionBatch(batch.detections ?? []);
        }
      } catch {
        /* ignore malformed payloads */
      }
    },
    [addAlert, applyDetectionBatch],
  );

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    alertDedupRef.current.clear();
    setAlerts([]);
    setPrimary(null);

    if (!activeCamera) return;

    void fetchLatestDetections().then((list) => {
      applyDetectionBatch(list);
      if (list[0]) {
        const det = list[0];
        addAlert({
          type: "detection",
          name: det.identity.name,
          student_id: det.identity.id,
          grade: det.year_level,
          section: det.course,
          face_violations: det.face_violations,
          torso_violations: det.torso_violations,
          all_violations: det.all_violations,
          time: new Date().toLocaleTimeString(undefined, {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          }),
        });
      }
    });

    const ws = new WebSocket(detectionsWsUrl());
    ws.onopen = () => console.log("[CBVMS] WebSocket connected");
    ws.onerror = (e) => console.error("[CBVMS] WebSocket error", e);
    ws.onmessage = (event) => handleWsMessage(event.data as string);

    return () => {
      ws.close();
    };
  }, [activeCamera, streamKey, addAlert, applyDetectionBatch, handleWsMessage]);

  const dateTimeLabel = now.toLocaleString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const hasStream = Boolean(activeCamera);
  const torsoDetected = Boolean(primary);

  return (
    <div className="live-monitor">
      <header className="live-header">
        <h1>Live Monitor</h1>
        <time className="live-datetime" dateTime={now.toISOString()}>
          {dateTimeLabel}
        </time>
      </header>

      <div className="live-body">
        <main className="live-main">
          <div className="live-stack">
            <div className="feed-wrap">
              <div className="feed-card">
                {hasStream ? (
                  <img
                    key={streamKey}
                    className="feed-stream"
                    src={streamUrl(streamKey)}
                    alt={`Live feed from ${activeCamera?.label ?? "camera"}`}
                  />
                ) : (
                  <div className="feed-placeholder">
                    <Camera size={48} strokeWidth={1.25} color="#3b4558" />
                    <p>No camera connected</p>
                    <p>Go to Settings → Camera Configuration to scan and connect a camera.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="live-status-bar">
              {hasStream ? (
                <span className="status-live">
                  <span className="status-dot status-dot--available" />
                  {activeCamera?.label}
                </span>
              ) : (
                <span className="status-muted">Camera offline</span>
              )}
              <span className="status-muted">
                {activeCamera?.type === "rj45"
                  ? "IP / RJ45"
                  : activeCamera?.type === "usb"
                    ? "USB"
                    : "—"}
              </span>
            </div>

            <section className="detection-panel" aria-label="Detection info">
              <div className="detection-grid">
                <div className="zone-card zone-card--face">
                  <div className="zone-card-header">
                    <span className="zone-icon zone-icon--face" aria-hidden />
                    <h2 className="zone-title">Face Detection</h2>
                  </div>
                  {primary ? (
                    <div className="zone-card-body">
                      <p className="zone-identity">{primary.identity.name}</p>
                      <p className="zone-meta">{gradeLine(primary)}</p>
                      <p className="zone-meta">ID: {primary.identity.id}</p>
                      <p className="zone-violations-label">Face Violations:</p>
                      <div className="zone-violations">
                        {primary.face_violations.length === 0 ? (
                          <span className="violation-chip violation-chip--ok">✓ No violations</span>
                        ) : (
                          primary.face_violations.map((v) => (
                            <span key={v} className="violation-chip violation-chip--face">
                              ✗ {formatViolation(v)}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="zone-meta">No person in frame</p>
                  )}
                </div>

                <div className="zone-card zone-card--torso">
                  <div className="zone-card-header">
                    <span className="zone-icon zone-icon--torso" aria-hidden />
                    <h2 className="zone-title">Torso Detection</h2>
                  </div>
                  {primary ? (
                    <div className="zone-card-body">
                      <p className={`zone-status ${torsoDetected ? "zone-status--detected" : "zone-status--idle"}`}>
                        {torsoDetected ? "● Upper Body Detected" : "○ Not Detected"}
                      </p>
                      <p className="zone-violations-label">Torso Violations:</p>
                      <div className="zone-violations">
                        {primary.torso_violations.length === 0 ? (
                          <span className="violation-chip violation-chip--ok">✓ No violations</span>
                        ) : (
                          primary.torso_violations.map((v) => (
                            <span key={v} className="violation-chip violation-chip--torso">
                              ✗ {formatViolation(v)}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="zone-status zone-status--idle">○ Not Detected</p>
                  )}
                </div>
              </div>
            </section>
          </div>
        </main>

        <aside className="live-alerts" aria-label="Live alerts">
          <h2>Live Alerts</h2>
          <div className="live-alerts-list">
            {alerts.length === 0 ? (
              <p className="detection-meta">No alerts yet</p>
            ) : (
              alerts.map((alert) => (
                <article key={alert.id} className="alert-card">
                  <div className="alert-card-header">
                    <span className="alert-name">{alert.name}</span>
                    <time className="alert-time">{alert.time}</time>
                  </div>
                  <div className="alert-pills">
                    {alert.face_violations?.map((v) => (
                      <span key={`f-${v}`} className="alert-pill alert-pill--face">
                        face · {formatViolation(v)}
                      </span>
                    ))}
                    {alert.torso_violations?.map((v) => (
                      <span key={`t-${v}`} className="alert-pill alert-pill--torso">
                        torso · {formatViolation(v)}
                      </span>
                    ))}
                    {!alert.face_violations?.length && !alert.torso_violations?.length && (
                      <span className="alert-pill alert-pill--ok">Person detected</span>
                    )}
                  </div>
                </article>
              ))
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
