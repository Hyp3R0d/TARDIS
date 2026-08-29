import { http } from '@kit.NetworkKit';
import { CreateGenerationRequest, GenerationTask } from '../model/Types';

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = trimTrailingSlash(baseUrl);
  }

  createGeneration(request: CreateGenerationRequest): Promise<GenerationTask> {
    const body: string = JSON.stringify(request);
    return this.requestJson(http.RequestMethod.POST, '/api/generations', body)
      .then((raw: Record<string, Object>) => {
        const task: GenerationTask = GenerationTask.fromRecord(raw);
        task.prompt = request.prompt;
        task.createdAt = Date.now();
        return task;
      });
  }

  fetchGeneration(taskId: string): Promise<GenerationTask> {
    const path: string = `/api/generations/${taskId}`;
    return this.requestJson(http.RequestMethod.GET, path, '')
      .then((raw: Record<string, Object>) => {
        return GenerationTask.fromRecord(raw);
      });
  }

  private requestJson(method: http.RequestMethod, path: string, body: string): Promise<Record<string, Object>> {
    return new Promise<Record<string, Object>>((resolve, reject) => {
      const client: http.HttpRequest = http.createHttp();
      const header: Record<string, string> = {
        'Content-Type': 'application/json'
      };
      const options: http.HttpRequestOptions = {
        method: method,
        header: header,
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 10000,
        readTimeout: 60000
      };
      if (method === http.RequestMethod.POST) {
        options.extraData = body;
      }
      client.request(`${this.baseUrl}${path}`, options)
        .then((resp: http.HttpResponse) => {
          client.destroy();
          if (resp.responseCode >= 200 && resp.responseCode < 300) {
            const text: string = resp.result as string;
            if (text.length === 0) {
              const empty: Record<string, Object> = {};
              resolve(empty);
              return;
            }
            resolve(JSON.parse(text) as Record<string, Object>);
          } else {
            reject(new Error(`HTTP ${resp.responseCode}`));
          }
        })
        .catch((err: Error) => {
          client.destroy();
          reject(new Error(`${err.message}`));
        });
    });
  }
}

function trimTrailingSlash(url: string): string {
  if (url.length > 0 && url.endsWith('/')) {
    return url.substring(0, url.length - 1);
  }
  return url;
}
