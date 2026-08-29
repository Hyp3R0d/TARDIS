import { common } from '@kit.AbilityKit';

export class AppContext {
  private static context: common.Context | null = null;

  static set(context: common.Context): void {
    AppContext.context = context;
  }

  static get(): common.Context | null {
    return AppContext.context;
  }
}
