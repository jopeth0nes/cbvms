import { useEffect } from "react";
import "./Toast.css";

export type ToastType = "success" | "error" | "info";

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastStackProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

export function ToastStack({ toasts, onDismiss }: ToastStackProps) {
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map((t) => (
        <Toast key={t.id} item={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function Toast({ item, onDismiss }: { item: ToastItem; onDismiss: (id: string) => void }) {
  useEffect(() => {
    const timer = window.setTimeout(() => onDismiss(item.id), 3200);
    return () => window.clearTimeout(timer);
  }, [item.id, onDismiss]);

  return (
    <div className={`toast toast--${item.type}`} role="status">
      {item.message}
    </div>
  );
}
