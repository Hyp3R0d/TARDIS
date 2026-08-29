import { mediaquery } from '@kit.ArkUI';
import { CreateGenerationRequest, GenerationSettings, GenerationTask } from '../model/Types';
import { ApiClient } from '../service/ApiClient';
import { HistoryStore } from '../service/HistoryStore';
import { PickedImage, pickReferenceImage } from '../service/ImagePicker';
import { AppContext } from '../common/AppContext';

const POLL_INTERVAL_MS: number = 3000;
const MAX_POLL_SECONDS: number = 600;
const PROMPT_MAX_LENGTH: number = 512;

@Entry
@Component
struct Index {
  @State prompt: string = '';
  @State referenceUri: string = '';
  @State referenceDataUrl: string = '';
  @State qualityIndex: number = 0;
  @State sizeIndex: number = 0;
  @State fpsIndex: number = 0;
  @State durationIndex: number = 0;
  @State withAudio: boolean = false;
  @State endpoint: string = 'http://127.0.0.1:8787';
  @State endpointEditing: boolean = false;
  @State submitting: boolean = false;
  @State currentTask: GenerationTask | null = null;
  @State history: GenerationTask[] = [];
  @State elapsedSeconds: number = 0;
  @State isWide: boolean = false;
  private api: ApiClient = new ApiClient('http://127.0.0.1:8787');
  private pollTimer: number = -1;
  private historyStore: HistoryStore = new HistoryStore();
  private mediaListener: mediaquery.MediaQueryListener | null = null;
  private videoController: VideoController = new VideoController();
  private qualityLabels: string[] = ['极速 speed', '画质 quality'];
  private qualityValues: string[] = ['speed', 'quality'];
  private sizeLabels: string[] = ['1280 × 720 横屏', '720 × 1280 竖屏', '1024 × 1024 方形', '1920 × 1080 全高清'];
  private sizeValues: string[] = ['1280x720', '720x1280', '1024x1024', '1920x1080'];
  private fpsLabels: string[] = ['30 FPS', '60 FPS'];
  private fpsValues: number[] = [30, 60];
  private durationLabels: string[] = ['5 秒', '10 秒'];
  private durationValues: number[] = [5, 10];

  aboutToAppear(): void {
    const listener: mediaquery.MediaQueryListener = mediaquery.matchMediaSync('(min-width: 840vp)');
    this.isWide = listener.matches;
    listener.on('change', (result: mediaquery.MediaQueryResult) => {
      this.isWide = result.matches;
    });
    this.mediaListener = listener;
    const context = AppContext.get();
    if (context !== null) {
      this.historyStore.init(context).then(() => {
        this.history = this.historyStore.load();
      });
    }
  }

  aboutToDisappear(): void {
    this.stopPolling();
    if (this.mediaListener !== null) {
      this.mediaListener.off('change');
      this.mediaListener = null;
    }
  }

  private toast(message: string): void {
    this.getUIContext().getPromptAction().showToast({
      message: message,
      duration: 2500
    });
  }

  private chooseReference(): void {
    pickReferenceImage()
      .then((image: PickedImage | null) => {
        if (image === null) {
          return;
        }
        this.referenceUri = image.uri;
        this.referenceDataUrl = image.dataUrl;
        this.toast('参考图已就绪');
      })
      .catch((err: Error) => {
        this.toast(err.message);
      });
  }

  private submitGeneration(): void {
    const trimmed: string = this.prompt.trim();
    if (trimmed.length === 0) {
      this.toast('请先输入创作提示词');
      return;
    }
    if (this.submitting) {
      return;
    }
    this.stopPolling();
    this.submitting = true;
    this.elapsedSeconds = 0;
    const settings: GenerationSettings = {
      quality: this.qualityValues[this.qualityIndex],
      size: this.sizeValues[this.sizeIndex],
      fps: this.fpsValues[this.fpsIndex],
      duration: this.durationValues[this.durationIndex],
      withAudio: this.withAudio
    };
    const request: CreateGenerationRequest = {
      prompt: trimmed,
      settings: settings
    };
    if (this.referenceDataUrl.length > 0) {
      request.imageData = this.referenceDataUrl;
    }
    this.api = new ApiClient(this.endpoint);
    this.api.createGeneration(request)
      .then((task: GenerationTask) => {
        this.submitting = false;
        this.currentTask = task;
        this.startPolling();
        this.toast(`任务已创建：${task.id}`);
      })
      .catch((err: Error) => {
        this.submitting = false;
        this.toast(`创建失败：${err.message}`);
      });
  }

  private startPolling(): void {
    this.stopPolling();
    this.pollTimer = setInterval(() => {
      this.elapsedSeconds += POLL_INTERVAL_MS / 1000;
      if (this.elapsedSeconds >= MAX_POLL_SECONDS) {
        this.stopPolling();
        this.toast('等待超时，请稍后从创作记录查看结果');
        return;
      }
      this.refreshTask();
    }, POLL_INTERVAL_MS);
  }

  private stopPolling(): void {
    if (this.pollTimer >= 0) {
      clearInterval(this.pollTimer);
      this.pollTimer = -1;
    }
  }

  private refreshTask(): void {
    const task: GenerationTask | null = this.currentTask;
    if (task === null) {
      return;
    }
    if (task.taskStatus === 'SUCCESS' || task.taskStatus === 'FAIL') {
      this.stopPolling();
      return;
    }
    this.api.fetchGeneration(task.id)
      .then((fresh: GenerationTask) => {
        fresh.prompt = task.prompt;
        fresh.createdAt = task.createdAt;
        this.currentTask = fresh;
        if (fresh.taskStatus === 'SUCCESS') {
          this.stopPolling();
          this.pushHistory(fresh);
          this.toast('生成完成，已存入创作记录');
        } else if (fresh.taskStatus === 'FAIL') {
          this.stopPolling();
          this.pushHistory(fresh);
          this.toast('生成失败，请查看任务信息');
        }
      })
      .catch((err: Error) => {
        this.toast(`状态查询失败：${err.message}`);
      });
  }

  private pushHistory(task: GenerationTask): void {
    this.history = [task].concat(this.history).slice(0, 50);
    this.historyStore.save(this.history);
  }

  private replayTask(task: GenerationTask): void {
    this.stopPolling();
    this.elapsedSeconds = 0;
    this.currentTask = task;
    this.toast('已从创作记录载入任务');
  }

  private hasTask(): boolean {
    return this.currentTask !== null;
  }

  private isProcessing(): boolean {
    const task: GenerationTask | null = this.currentTask;
    return task !== null && task.taskStatus === 'PROCESSING';
  }

  private isSucceeded(): boolean {
    const task: GenerationTask | null = this.currentTask;
    return task !== null && task.taskStatus === 'SUCCESS';
  }

  private isFailed(): boolean {
    const task: GenerationTask | null = this.currentTask;
    return task !== null && task.taskStatus === 'FAIL';
  }

  private statusLabel(): string {
    if (this.isSucceeded()) {
      return '已完成';
    }
    if (this.isFailed()) {
      return '失败';
    }
    if (this.isProcessing()) {
      return '生成中';
    }
    return '待创建';
  }

  private statusBadgeColor(): string {
    if (this.isSucceeded()) {
      return '#1F7A4D';
    }
    if (this.isFailed()) {
      return '#8A3242';
    }
    if (this.isProcessing()) {
      return '#4A5AD4';
    }
    return '#3A4560';
  }

  private taskIdText(): string {
    const task: GenerationTask | null = this.currentTask;
    return task === null ? '—' : task.id;
  }

  private modelText(): string {
    const task: GenerationTask | null = this.currentTask;
    if (task === null || task.model.length === 0) {
      return 'tardis-v1';
    }
    return task.model;
  }

  private failureText(): string {
    const task: GenerationTask | null = this.currentTask;
    if (task === null) {
      return '';
    }
    if (task.message.length > 0) {
      return task.message;
    }
    return '云端生成失败，请调整提示词后重试';
  }

  private hasVideo(): boolean {
    const task: GenerationTask | null = this.currentTask;
    return task !== null && task.taskStatus === 'SUCCESS' && task.videoUrl.length > 0;
  }

  private videoSource(): string {
    const task: GenerationTask | null = this.currentTask;
    return task === null ? '' : task.videoUrl;
  }

  private coverSource(): string {
    const task: GenerationTask | null = this.currentTask;
    return task === null ? '' : task.coverUrl;
  }

  private resultPlaceholder(): string {
    if (this.isFailed()) {
      return '生成失败，请调整提示词后重试';
    }
    if (this.isProcessing()) {
      return `正在生成视频，已等待 ${this.elapsedSeconds} 秒`;
    }
    if (this.isSucceeded()) {
      return '生成完成，视频地址暂不可用';
    }
    return '生成完成后，视频将在这里播放';
  }

  private formatTime(timestamp: number): string {
    if (timestamp <= 0) {
      return '刚刚';
    }
    const date: Date = new Date(timestamp);
    return `${date.getFullYear()}-${this.pad2(date.getMonth() + 1)}-${this.pad2(date.getDate())} ${this.pad2(date.getHours())}:${this.pad2(date.getMinutes())}`;
  }

  private pad2(value: number): string {
    if (value < 10) {
      return `0${value}`;
    }
    return `${value}`;
  }

  @Builder
  private titleBar() {
    Row({ space: 12 }) {
      Column({ space: 3 }) {
        Text('TARDIS Studio')
          .fontSize(21)
          .fontWeight(FontWeight.Bold)
          .fontColor('#F2F5FC')
        Text('华为 ArkUI Engine · HarmonyOS 客户端')
          .fontSize(11)
          .fontColor('#8A93A6')
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)

      Column({ space: 4 }) {
        if (this.endpointEditing) {
          TextInput({ text: this.endpoint, placeholder: 'http://127.0.0.1:8787' })
            .fontSize(12)
            .width(230)
            .height(36)
            .backgroundColor('#141B2E')
            .fontColor('#E8ECF4')
            .placeholderColor('#5A6478')
            .borderRadius(8)
            .onChange((value: string) => {
              this.endpoint = value;
            })
            .onSubmit(() => {
              this.endpointEditing = false;
            })
          Text('按回车保存代理地址')
            .fontSize(10)
            .fontColor('#5A6478')
        } else {
          Text(`接口 ${this.endpoint}`)
            .fontSize(11)
            .fontColor('#8A93A6')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .constraintSize({ maxWidth: 230 })
            .onClick(() => {
              this.endpointEditing = true;
            })
          Text('点击修改本地代理地址')
            .fontSize(10)
            .fontColor('#5A6478')
        }
      }
      .alignItems(HorizontalAlign.End)
    }
    .width('100%')
    .padding({ left: 20, right: 20, top: 16, bottom: 14 })
  }

  @Builder
  private promptCard() {
    Column({ space: 10 }) {
      Text('创意描述')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor('#E8ECF4')
      TextArea({ text: this.prompt, placeholder: '例如：A cinematic robot walks through a rainy neon street' })
        .placeholderColor('#5A6478')
        .fontColor('#E8ECF4')
        .fontSize(14)
        .backgroundColor('#141B2E')
        .borderRadius(10)
        .height(110)
        .maxLength(PROMPT_MAX_LENGTH)
        .showCounter(true)
        .onChange((value: string) => {
          this.prompt = value;
        })
        .width('100%')
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private referenceCard() {
    Column({ space: 10 }) {
      Text('参考图片（可选 · PNG/JPG · 不超过 5 MB）')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor('#E8ECF4')
      if (this.referenceUri.length > 0) {
        Stack({ alignContent: Alignment.TopEnd }) {
          Image(this.referenceUri)
            .width('100%')
            .height(170)
            .objectFit(ImageFit.Cover)
            .borderRadius(12)
          Text('移除')
            .fontSize(12)
            .fontColor('#FFFFFF')
            .padding({ left: 10, right: 10, top: 5, bottom: 5 })
            .borderRadius(10)
            .backgroundColor('#99203040')
            .margin(8)
            .onClick(() => {
              this.referenceUri = '';
              this.referenceDataUrl = '';
            })
        }
        .width('100%')
      } else {
        Row({ space: 8 }) {
          Text('＋')
            .fontSize(22)
            .fontColor('#6C7A99')
          Text('点击选择参考画面，生成时作为首帧参考')
            .fontSize(13)
            .fontColor('#8A93A6')
        }
        .width('100%')
        .height(110)
        .borderRadius(12)
        .backgroundColor('#141B2E')
        .border({ width: 1, color: '#26304A', style: BorderStyle.Dashed })
        .justifyContent(FlexAlign.Center)
        .onClick(() => this.chooseReference())
      }
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private settingsCard() {
    Column({ space: 14 }) {
      Text('生成设置')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor('#E8ECF4')
      Row({ space: 12 }) {
        Column({ space: 6 }) {
          Text('画质')
            .fontSize(12)
            .fontColor('#8A93A6')
          Select([
            { value: '极速 speed' },
            { value: '画质 quality' }
          ])
            .selected(this.qualityIndex)
            .value(this.qualityLabels[this.qualityIndex])
            .fontColor('#E8ECF4')
            .fontSize(13)
            .backgroundColor('#141B2E')
            .borderRadius(8)
            .width('100%')
            .onSelect((index: number) => {
              this.qualityIndex = index;
            })
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)

        Column({ space: 6 }) {
          Text('画幅')
            .fontSize(12)
            .fontColor('#8A93A6')
          Select([
            { value: '1280 × 720 横屏' },
            { value: '720 × 1280 竖屏' },
            { value: '1024 × 1024 方形' },
            { value: '1920 × 1080 全高清' }
          ])
            .selected(this.sizeIndex)
            .value(this.sizeLabels[this.sizeIndex])
            .fontColor('#E8ECF4')
            .fontSize(13)
            .backgroundColor('#141B2E')
            .borderRadius(8)
            .width('100%')
            .onSelect((index: number) => {
              this.sizeIndex = index;
            })
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)
      }
      .width('100%')

      Row({ space: 12 }) {
        Column({ space: 6 }) {
          Text('帧率')
            .fontSize(12)
            .fontColor('#8A93A6')
          Select([
            { value: '30 FPS' },
            { value: '60 FPS' }
          ])
            .selected(this.fpsIndex)
            .value(this.fpsLabels[this.fpsIndex])
            .fontColor('#E8ECF4')
            .fontSize(13)
            .backgroundColor('#141B2E')
            .borderRadius(8)
            .width('100%')
            .onSelect((index: number) => {
              this.fpsIndex = index;
            })
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)

        Column({ space: 6 }) {
          Text('时长')
            .fontSize(12)
            .fontColor('#8A93A6')
          Select([
            { value: '5 秒' },
            { value: '10 秒' }
          ])
            .selected(this.durationIndex)
            .value(this.durationLabels[this.durationIndex])
            .fontColor('#E8ECF4')
            .fontSize(13)
            .backgroundColor('#141B2E')
            .borderRadius(8)
            .width('100%')
            .onSelect((index: number) => {
              this.durationIndex = index;
            })
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)
      }
      .width('100%')

      Row() {
        Column({ space: 4 }) {
          Text('生成音效')
            .fontSize(13)
            .fontColor('#E8ECF4')
          Text('开启后为生成视频附带音轨')
            .fontSize(11)
            .fontColor('#5A6478')
        }
        .alignItems(HorizontalAlign.Start)

        Toggle({ type: ToggleType.Switch, isOn: this.withAudio })
          .onChange((isOn: boolean) => {
            this.withAudio = isOn;
          })
      }
      .width('100%')
      .justifyContent(FlexAlign.SpaceBetween)
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private submitButton() {
    Button(this.submitting ? '正在提交任务…' : '开始生成')
      .width('100%')
      .height(48)
      .fontSize(16)
      .fontWeight(FontWeight.Medium)
      .backgroundColor(this.submitting ? '#3A4560' : '#5B5FE9')
      .borderRadius(12)
      .enabled(!this.submitting)
      .onClick(() => this.submitGeneration())
  }

  @Builder
  private statusCard() {
    Column({ space: 12 }) {
      Row({ space: 10 }) {
        Text('任务状态')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor('#E8ECF4')
          .layoutWeight(1)
        Text(this.statusLabel())
          .fontSize(11)
          .fontColor('#FFFFFF')
          .padding({ left: 10, right: 10, top: 4, bottom: 4 })
          .borderRadius(10)
          .backgroundColor(this.statusBadgeColor())
      }
      .width('100%')

      if (!this.hasTask()) {
        Text('输入提示词并点击「开始生成」后，这里会展示任务状态与进度。')
          .fontSize(13)
          .fontColor('#5A6478')
      } else {
        Column({ space: 6 }) {
          Row() {
            Text('任务 ID')
              .fontSize(12)
              .fontColor('#5A6478')
            Blank()
            Text(this.taskIdText())
              .fontSize(12)
              .fontColor('#B8C0D4')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .constraintSize({ maxWidth: 180 })
          }
          .width('100%')
          Row() {
            Text('模型')
              .fontSize(12)
              .fontColor('#5A6478')
            Blank()
            Text(this.modelText())
              .fontSize(12)
              .fontColor('#B8C0D4')
          }
          .width('100%')
        }
        .width('100%')

        if (this.isProcessing()) {
          Progress({ value: Math.min(this.elapsedSeconds, 120), total: 120, type: ProgressType.Linear })
            .color('#6E74F2')
            .width('100%')
          Text(`正在生成，已等待 ${this.elapsedSeconds} 秒，客户端每 3 秒轮询一次任务状态`)
            .fontSize(12)
            .fontColor('#5A6478')
        } else if (this.isFailed()) {
          Text(this.failureText())
            .fontSize(13)
            .fontColor('#FF8095')
        } else if (this.isSucceeded()) {
          Text('生成完成，可在下方播放生成结果。')
            .fontSize(13)
            .fontColor('#69D29A')
        }
      }
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private resultCard() {
    Column({ space: 12 }) {
      Text('生成结果')
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor('#E8ECF4')
      if (this.hasVideo()) {
        Video({
          src: this.videoSource(),
          controller: this.videoController
        })
          .width('100%')
          .height(230)
          .borderRadius(12)
          .autoPlay(true)
          .controls(true)
          .loop(false)
          .objectFit(ImageFit.Contain)
          .poster(this.coverSource())
      } else {
        Column({ space: 10 }) {
          Text(this.resultPlaceholder())
            .fontSize(13)
            .fontColor('#5A6478')
            .textAlign(TextAlign.Center)
            .padding({ left: 24, right: 24 })
        }
        .width('100%')
        .height(200)
        .borderRadius(12)
        .backgroundColor('#0E1424')
        .justifyContent(FlexAlign.Center)
      }
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private historyCard() {
    Column({ space: 12 }) {
      Row({ space: 10 }) {
        Text('创作记录')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor('#E8ECF4')
          .layoutWeight(1)
        Text(`${this.history.length} 条`)
          .fontSize(12)
          .fontColor('#8A93A6')
      }
      .width('100%')

      if (this.history.length === 0) {
        Text('暂无记录，完成一次生成后会自动归档。')
          .fontSize(13)
          .fontColor('#5A6478')
      } else {
        ForEach(this.history, (task: GenerationTask) => {
          this.historyItem(task)
        }, (task: GenerationTask) => `${task.id}-${task.createdAt}`)
      }
    }
    .padding(16)
    .borderRadius(14)
    .backgroundColor('#111827')
    .width('100%')
  }

  @Builder
  private historyItem(task: GenerationTask) {
    Row({ space: 12 }) {
      Text('')
        .width(8)
        .height(8)
        .borderRadius(4)
        .backgroundColor(task.taskStatus === 'SUCCESS' ? '#69D29A' : (task.taskStatus === 'FAIL' ? '#FF8095' : '#6E74F2'))
      Column({ space: 4 }) {
        Text(task.prompt.length > 0 ? task.prompt : '未命名创作')
          .fontSize(13)
          .fontColor('#D6DCE8')
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
        Text(`${task.id} · ${this.formatTime(task.createdAt)}`)
          .fontSize(11)
          .fontColor('#5A6478')
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
      }
      .layoutWeight(1)
      .alignItems(HorizontalAlign.Start)

      if (task.taskStatus === 'SUCCESS') {
        Text('查看')
          .fontSize(12)
          .fontColor('#8F9BFF')
          .padding({ left: 10, right: 10, top: 5, bottom: 5 })
          .borderRadius(8)
          .backgroundColor('#1A2140')
          .onClick(() => {
            this.replayTask(task);
          })
      }
    }
    .width('100%')
    .padding({ top: 10, bottom: 10 })
  }

  build() {
    Column() {
      this.titleBar()
      Divider()
        .color('#1E2740')
        .strokeWidth(1)
      Scroll() {
        if (this.isWide) {
          Row({ space: 16 }) {
            Column({ space: 16 }) {
              this.promptCard()
              this.referenceCard()
              this.settingsCard()
              this.submitButton()
            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1.1)

            Column({ space: 16 }) {
              this.statusCard()
              this.resultCard()
              this.historyCard()
            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1)
          }
          .alignItems(VerticalAlign.Top)
          .padding(20)
          .width('100%')
        } else {
          Column({ space: 16 }) {
            this.promptCard()
            this.referenceCard()
            this.settingsCard()
            this.submitButton()
            this.statusCard()
            this.resultCard()
            this.historyCard()
          }
          .alignItems(HorizontalAlign.Start)
          .padding(16)
          .width('100%')
        }
      }
      .scrollBar(BarState.Off)
      .layoutWeight(1)
      .width('100%')
      .align(Alignment.Top)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#0B0F1A')
  }
}
