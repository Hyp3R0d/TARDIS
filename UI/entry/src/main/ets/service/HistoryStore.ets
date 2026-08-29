import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';
import { GenerationTask } from '../model/Types';

const STORE_NAME: string = 'tardis_studio';
const HISTORY_KEY: string = 'generation_history';

export class HistoryStore {
  private store: preferences.Preferences | null = null;

  async init(context: common.Context): Promise<void> {
    const options: preferences.Options = {
      name: STORE_NAME
    };
    this.store = await preferences.getPreferences(context, options);
  }

  load(): GenerationTask[] {
    if (this.store === null) {
      return [];
    }
    const raw: preferences.ValueType = this.store.getSync(HISTORY_KEY, '[]');
    try {
      const items: Array<Record<string, Object>> = JSON.parse(raw as string) as Array<Record<string, Object>>;
      const tasks: GenerationTask[] = [];
      for (let i = 0; i < items.length; i++) {
        tasks.push(GenerationTask.fromRecord(items[i]));
      }
      return tasks;
    } catch (err) {
      return [];
    }
  }

  save(tasks: GenerationTask[]): void {
    if (this.store === null) {
      return;
    }
    const text: string = JSON.stringify(tasks);
    this.store.putSync(HISTORY_KEY, text);
    this.store.flush();
  }
}
