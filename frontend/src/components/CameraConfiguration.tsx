import { useCallback, useEffect, useState } from "react";
import {
  Camera,
  CheckCircle,
  Network,
  Plus,
  RefreshCw,
  Trash2,
  Usb,
  XCircle,
} from "lucide-react";
import {
  addIpCamera,
  deleteIpCamera,
  getActiveCamera,
  scanCameras,
  selectCamera,
  testCamera,
} from "../api/cameras";
import type { ActiveCamera, CameraDevice } from "../types/camera";
import "./CameraConfiguration.css";

type Tab = "usb" | "ip";

interface CameraConfigurationProps {
  onCameraConnected?: (camera: ActiveCamera) => void;
}

export function CameraConfiguration({ onCameraConnected }: CameraConfigurationProps) {
  const [tab, setTab] = useState<Tab>("usb");
  const [scanning, setScanning] = useState(false);
  const [connectingId, setConnectingId] = useState<string | null>(null);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [activeCamera, setActiveCamera] = useState<ActiveCamera | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [ipLabel, setIpLabel] = useState("");
  const [ipUrl, setIpUrl] = useState("");
  const [testing, setTesting] = useState(false);
  const [testPassed, setTestPassed] = useState<boolean | null>(null);
  const [savingIp, setSavingIp] = useState(false);

  const refreshActive = useCallback(async () => {
    try {
      const { active } = await getActiveCamera();
      setActiveCamera(active);
    } catch {
      setActiveCamera(null);
    }
  }, []);

  const runScan = useCallback(async () => {
    setScanning(true);
    try {
      const data = await scanCameras();
      setCameras(data.cameras);
      if (data.active) {
        setActiveCamera(data.active);
      }
    } catch {
      setCameras([]);
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    void refreshActive();
    void runScan();
  }, [refreshActive, runScan]);

  const isActive = (cam: CameraDevice) =>
    activeCamera?.id === cam.id || cam.is_active === true;

  const handleConnect = async (cam: CameraDevice) => {
    if (isActive(cam) || cam.status === "unreachable") return;
    setConnectingId(cam.id);
    try {
      const result = await selectCamera(cam);
      if (result.success && result.active) {
        setActiveCamera(result.active);
        onCameraConnected?.(result.active);
      }
    } finally {
      setConnectingId(null);
    }
  };

  const handleDeleteIp = async (cam: CameraDevice) => {
    try {
      await deleteIpCamera(cam.id);
      await runScan();
      await refreshActive();
    } catch {
      /* ignore */
    }
  };

  const handleTestConnection = async () => {
    if (!ipUrl.trim()) return;
    setTesting(true);
    setTestPassed(null);
    try {
      const result = await testCamera(ipUrl.trim());
      setTestPassed(result.success);
    } catch {
      setTestPassed(false);
    } finally {
      setTesting(false);
    }
  };

  const handleSaveIp = async () => {
    if (!testPassed || !ipLabel.trim() || !ipUrl.trim()) return;
    setSavingIp(true);
    try {
      await addIpCamera(ipLabel.trim(), ipUrl.trim());
      setIpLabel("");
      setIpUrl("");
      setTestPassed(null);
      setShowAddForm(false);
      setTab("ip");
      await runScan();
    } finally {
      setSavingIp(false);
    }
  };

  const usbCameras = cameras.filter((c) => c.type === "usb");
  const ipCameras = cameras.filter((c) => c.type === "rj45");
  const connected = Boolean(activeCamera);

  return (
    <section className="camera-config-section" aria-labelledby="camera-config-title">
      <div className="camera-config-header">
        <div className="camera-config-header-text">
          <h2 id="camera-config-title">Camera Configuration</h2>
          <p>Manage and connect camera sources for the live monitor</p>
        </div>
        <button
          type="button"
          className="btn-scan"
          onClick={() => void runScan()}
          disabled={scanning}
        >
          <RefreshCw size={16} className={scanning ? "spin" : ""} />
          {scanning ? "Scanning…" : "Scan for Cameras"}
        </button>
      </div>

      <div
        className={`active-camera-card${connected ? " active-camera-card--connected" : ""}`}
      >
        <div className="active-camera-icon">
          <Camera size={22} />
        </div>
        <div className="active-camera-meta">
          <h3>{connected ? activeCamera!.label : "No camera selected"}</h3>
          <p>
            {connected
              ? activeCamera!.type === "rj45"
                ? activeCamera!.url ?? "IP / RJ45 stream"
                : `USB device index ${activeCamera!.index ?? 0}`
              : "Scan for cameras below and connect a source for the live monitor"}
          </p>
        </div>
        {connected ? (
          <span className="badge-connected">
            <span className="status-dot status-dot--available" style={{ display: "inline-block" }} />
            Connected
          </span>
        ) : (
          <span className="badge-offline">Not connected</span>
        )}
      </div>

      <div className="camera-config-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "usb"}
          className={`camera-config-tab${tab === "usb" ? " camera-config-tab--active" : ""}`}
          onClick={() => setTab("usb")}
        >
          USB Cameras
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "ip"}
          className={`camera-config-tab${tab === "ip" ? " camera-config-tab--active" : ""}`}
          onClick={() => setTab("ip")}
        >
          IP / RJ45 Cameras
        </button>
      </div>

      {scanning ? (
        <div className="skeleton-grid" aria-busy="true">
          {[0, 1, 2].map((i) => (
            <div key={i} className="skeleton-card" />
          ))}
        </div>
      ) : tab === "usb" ? (
        usbCameras.length === 0 ? (
          <p className="empty-state">No USB cameras detected. Try scanning again.</p>
        ) : (
          <div className="camera-grid">
            {usbCameras.map((cam) => (
              <UsbCameraCard
                key={cam.id}
                cam={cam}
                active={isActive(cam)}
                connecting={connectingId === cam.id}
                onConnect={() => void handleConnect(cam)}
              />
            ))}
          </div>
        )
      ) : (
        <>
          {ipCameras.length === 0 ? (
            <p className="empty-state">No IP cameras added yet.</p>
          ) : (
            <div className="camera-grid">
              {ipCameras.map((cam) => (
                <IpCameraCard
                  key={cam.id}
                  cam={cam}
                  active={isActive(cam)}
                  connecting={connectingId === cam.id}
                  onConnect={() => void handleConnect(cam)}
                  onDelete={() => void handleDeleteIp(cam)}
                />
              ))}
            </div>
          )}
          <div className="add-ip-wrap">
            {!showAddForm ? (
              <button
                type="button"
                className="btn-add-ip"
                onClick={() => {
                  setShowAddForm(true);
                  setTestPassed(null);
                }}
              >
                <Plus size={18} />
                Add IP Camera
              </button>
            ) : (
              <div className="ip-form">
                <label>
                  Label
                  <input
                    value={ipLabel}
                    onChange={(e) => setIpLabel(e.target.value)}
                    placeholder="Entrance Camera 1"
                  />
                </label>
                <label>
                  Stream URL
                  <input
                    value={ipUrl}
                    onChange={(e) => {
                      setIpUrl(e.target.value);
                      setTestPassed(null);
                    }}
                    placeholder="rtsp://192.168.1.x:554/stream"
                  />
                </label>
                <div className="ip-form-actions">
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() => void handleTestConnection()}
                    disabled={testing || !ipUrl.trim()}
                  >
                    <RefreshCw size={14} className={testing ? "spin" : ""} />
                    Test Connection
                  </button>
                  {testPassed === true && (
                    <span className="test-result test-result--ok">
                      <CheckCircle size={16} />
                      Reachable
                    </span>
                  )}
                  {testPassed === false && (
                    <span className="test-result test-result--fail">
                      <XCircle size={16} />
                      Unreachable
                    </span>
                  )}
                  <button
                    type="button"
                    className="btn-connect"
                    style={{ flex: "none" }}
                    disabled={!testPassed || savingIp}
                    onClick={() => void handleSaveIp()}
                  >
                    Save Camera
                  </button>
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() => {
                      setShowAddForm(false);
                      setTestPassed(null);
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function UsbCameraCard({
  cam,
  active,
  connecting,
  onConnect,
}: {
  cam: CameraDevice;
  active: boolean;
  connecting: boolean;
  onConnect: () => void;
}) {
  const available = cam.status === "available";
  return (
    <article className={`camera-card${active ? " camera-card--active" : ""}`}>
      <div className="camera-card-top">
        <div className="camera-card-icon">
          <Usb size={20} />
        </div>
        <span
          className={`status-dot ${available ? "status-dot--available" : "status-dot--muted"}`}
          title={cam.status}
        />
      </div>
      <div className="camera-card-meta">
        <h3>{cam.label}</h3>
      </div>
      <div className="camera-card-actions" style={{ marginTop: "0.75rem" }}>
        {active ? (
          <span className="chip-active">Active</span>
        ) : (
          <button
            type="button"
            className="btn-connect"
            disabled={!available || connecting}
            onClick={onConnect}
          >
            {connecting ? "Connecting…" : "Connect"}
          </button>
        )}
      </div>
    </article>
  );
}

function IpCameraCard({
  cam,
  active,
  connecting,
  onConnect,
  onDelete,
}: {
  cam: CameraDevice;
  active: boolean;
  connecting: boolean;
  onConnect: () => void;
  onDelete: () => void;
}) {
  const reachable = cam.status === "available";
  const url = cam.url ?? "";
  const truncated = url.length > 48 ? `${url.slice(0, 45)}…` : url;

  return (
    <article className={`camera-card camera-card--ip${active ? " camera-card--active" : ""}`}>
      <div className="camera-card-icon">
        <Network size={20} />
      </div>
      <div className="camera-card-meta">
        <h3>{cam.label}</h3>
        <p title={url}>{truncated}</p>
      </div>
      <div className="camera-card-actions">
        <span
          className={`status-dot ${
            reachable ? "status-dot--available" : "status-dot--unreachable"
          }`}
          title={cam.status}
        />
        {active ? (
          <span className="chip-active">Active</span>
        ) : (
          <button
            type="button"
            className="btn-connect"
            disabled={!reachable || connecting}
            onClick={onConnect}
          >
            {connecting ? "Connecting…" : "Connect"}
          </button>
        )}
        <button type="button" className="btn-ghost btn-ghost--danger" onClick={onDelete}>
          <Trash2 size={14} />
          Delete
        </button>
      </div>
    </article>
  );
}
