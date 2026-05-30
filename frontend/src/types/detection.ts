export interface DetectionIdentity {
  id: string;
  name: string;
}

export interface DetectionPayload {
  identity: DetectionIdentity;
  face_box: [number, number, number, number];
  torso_box: [number, number, number, number];
  face_violations: string[];
  torso_violations: string[];
  all_violations: string[];
  course?: string;
  year_level?: string;
  status?: string;
  confidence?: number;
}

export interface DetectionBatchMessage {
  type: "detections";
  detections: DetectionPayload[];
}

export interface DetectionAlertMessage {
  type: "detection";
  name: string;
  student_id: string;
  grade?: string;
  section?: string;
  face_violations: string[];
  torso_violations: string[];
  all_violations: string[];
  time: string;
  identity?: DetectionIdentity;
}

export interface LiveAlert {
  id: string;
  name: string;
  time: string;
  face_violations: string[];
  torso_violations: string[];
}
