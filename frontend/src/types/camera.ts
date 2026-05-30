export type CameraType = "usb" | "rj45";

export type CameraStatus = "available" | "unreachable" | "connected";

export interface CameraDevice {
  id: string;
  type: CameraType;
  label: string;
  index?: number;
  url?: string;
  status: CameraStatus | string;
  is_active?: boolean;
}

export interface ActiveCamera {
  id: string;
  type: CameraType;
  label: string;
  index?: number;
  url?: string;
  status?: string;
}
