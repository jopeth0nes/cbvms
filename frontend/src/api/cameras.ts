import type { ActiveCamera, CameraDevice } from "../types/camera";

const jsonHeaders = { "Content-Type": "application/json" };

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function scanCameras(): Promise<{
  cameras: CameraDevice[];
  active: ActiveCamera | null;
}> {
  return parseJson(await fetch("/api/cameras/scan"));
}

export async function getActiveCamera(): Promise<{ active: ActiveCamera | null }> {
  return parseJson(await fetch("/api/cameras/active"));
}

export async function testCamera(url: string): Promise<{ success: boolean; message: string }> {
  return parseJson(
    await fetch("/api/cameras/test", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ url }),
    }),
  );
}

export async function selectCamera(camera: CameraDevice): Promise<{
  success: boolean;
  active: ActiveCamera | null;
  message?: string;
}> {
  return parseJson(
    await fetch("/api/cameras/select", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({
        id: camera.id,
        type: camera.type,
        index: camera.index,
        label: camera.label,
        url: camera.url,
      }),
    }),
  );
}

export async function addIpCamera(
  label: string,
  url: string,
): Promise<{ camera: CameraDevice }> {
  return parseJson(
    await fetch("/api/cameras/ip/add", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ label, url }),
    }),
  );
}

export async function deleteIpCamera(id: string): Promise<void> {
  const raw = id.startsWith("ip_") ? id.slice(3) : id;
  await parseJson(
    await fetch(`/api/cameras/ip/${encodeURIComponent(raw)}`, { method: "DELETE" }),
  );
}

export function streamUrl(cacheKey: number): string {
  return `/api/cameras/stream?k=${cacheKey}`;
}
