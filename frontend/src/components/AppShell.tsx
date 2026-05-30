import { useCallback, useEffect, useState } from "react";
import { LayoutDashboard, Settings as SettingsIcon } from "lucide-react";
import { getActiveCamera } from "../api/cameras";
import type { ActiveCamera } from "../types/camera";
import { LiveMonitor } from "../pages/LiveMonitor";
import { Settings } from "../pages/Settings";
import "./AppShell.css";

type View = "live" | "settings";

export function AppShell() {
  const [view, setView] = useState<View>("live");
  const [activeCamera, setActiveCamera] = useState<ActiveCamera | null>(null);
  const [streamKey, setStreamKey] = useState(0);

  const refreshActive = useCallback(async () => {
    try {
      const { active } = await getActiveCamera();
      setActiveCamera(active);
      if (active) {
        setStreamKey((k) => k + 1);
      }
    } catch {
      setActiveCamera(null);
    }
  }, []);

  useEffect(() => {
    void refreshActive();
  }, [refreshActive]);

  const handleCameraConnected = useCallback((camera: ActiveCamera) => {
    setActiveCamera(camera);
    setStreamKey((k) => k + 1);
  }, []);

  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Main">
        <span className="app-brand">CBVMS</span>
        <button
          type="button"
          className={`app-nav-btn${view === "live" ? " app-nav-btn--active" : ""}`}
          onClick={() => setView("live")}
        >
          <LayoutDashboard size={18} />
          Live Monitor
        </button>
        <button
          type="button"
          className={`app-nav-btn${view === "settings" ? " app-nav-btn--active" : ""}`}
          onClick={() => setView("settings")}
        >
          <SettingsIcon size={18} />
          Settings
        </button>
      </nav>

      <div className="app-content">
        {view === "live" ? (
          <LiveMonitor activeCamera={activeCamera} streamKey={streamKey} />
        ) : (
          <Settings onCameraConnected={handleCameraConnected} />
        )}
      </div>
    </div>
  );
}
