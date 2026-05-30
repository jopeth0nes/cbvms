import { CameraConfiguration } from "../components/CameraConfiguration";
import { ToastStack, type ToastItem } from "../components/Toast";
import type { ActiveCamera } from "../types/camera";
import { useCallback, useState } from "react";
import "./Settings.css";

interface SettingsProps {
  onCameraConnected: (camera: ActiveCamera) => void;
}

export function Settings({ onCameraConnected }: SettingsProps) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const handleConnected = useCallback(
    (camera: ActiveCamera) => {
      onCameraConnected(camera);
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), message: `Connected to ${camera.label}`, type: "success" },
      ]);
    },
    [onCameraConnected],
  );

  return (
    <div className="settings-page">
      <header className="settings-header">
        <h1>Settings</h1>
      </header>

      <main className="settings-main">
        <CameraConfiguration onCameraConnected={handleConnected} />
      </main>

      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
