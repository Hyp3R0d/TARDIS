export interface GenerationSettings {
  quality: string;
  size: string;
  fps: number;
  duration: number;
  withAudio: boolean;
}

export interface CreateGenerationRequest {
  prompt: string;
  settings: GenerationSettings;
  imageData?: string;
}

export class GenerationTask {
  id: string = '';
  requestId: string = '';
  taskStatus: string = '';
  model: string = '';
  videoUrl: string = '';
  coverUrl: string = '';
  message: string = '';
  prompt: string = '';
  createdAt: number = 0;

  static fromRecord(raw: Record<string, Object>): GenerationTask {
    const task: GenerationTask = new GenerationTask();
    task.id = readString(raw, 'id');
    task.requestId = readString(raw, 'requestId');
    task.taskStatus = readString(raw, 'taskStatus');
    if (task.taskStatus.length === 0) {
      task.taskStatus = readString(raw, 'status');
    }
    task.model = readString(raw, 'model');
    task.videoUrl = readString(raw, 'videoUrl');
    task.coverUrl = readString(raw, 'coverUrl');
    task.message = readString(raw, 'error');
    if (task.message.length === 0) {
      task.message = readString(raw, 'message');
    }
    task.prompt = readString(raw, 'prompt');
    task.createdAt = readNumber(raw, 'createdAt');
    return task;
  }
}

function readString(raw: Record<string, Object>, key: string): string {
  const value: Object = raw[key];
  if (value === undefined || value === null) {
    return '';
  }
  return `${value}`;
}

function readNumber(raw: Record<string, Object>, key: string): number {
  const value: Object = raw[key];
  if (value === undefined || value === null) {
    return 0;
  }
  return Number(`${value}`);
}
