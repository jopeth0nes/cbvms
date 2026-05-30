import type { DetectionBatchMessage, DetectionPayload } from "../types/detection";

const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export function detectionsWsUrl(): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/api/ws/detections`;
}

export async function fetchLatestDetections(): Promise<DetectionPayload[]> {
  const res = await fetch(`${API_BASE}/api/detections/latest`);
  if (!res.ok) {
    return [];
  }
  const data = (await res.json()) as DetectionBatchMessage & { detections?: DetectionPayload[] };
  return data.detections ?? [];
}

export function formatViolation(code: string): string {
  const labels: Record<string, string> = {
    earrings_male: "Earrings (male)",
    dress_code: "Dress Code",
    no_id_badge: "No ID Badge",
    prohibited_items: "Prohibited Items",
  };
  return labels[code] ?? code.replace(/_/g, " ");
}
