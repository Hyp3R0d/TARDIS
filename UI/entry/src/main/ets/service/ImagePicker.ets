import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';

const MAX_IMAGE_BYTES: number = 5 * 1024 * 1024;

export interface PickedImage {
  uri: string;
  dataUrl: string;
}

export function pickReferenceImage(): Promise<PickedImage | null> {
  const picker: photoAccessHelper.PhotoViewPicker = new photoAccessHelper.PhotoViewPicker();
  const options: photoAccessHelper.PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
  options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
  options.maxSelectNumber = 1;
  return picker.select(options)
    .then((result: photoAccessHelper.PhotoSelectResult) => {
      if (result.photoUris.length === 0) {
        return null;
      }
      const uri: string = result.photoUris[0];
      const dataUrl: string = readAsDataUrl(uri);
      const image: PickedImage = {
        uri: uri,
        dataUrl: dataUrl
      };
      return image;
    });
}

function readAsDataUrl(uri: string): string {
  const file = fileIo.openSync(uri, fileIo.OpenMode.READ_ONLY);
  try {
    const stat = fileIo.statSync(file.fd);
    if (stat.size > MAX_IMAGE_BYTES) {
      throw new Error('参考图片超过 5 MB 限制，请压缩后重试');
    }
    const buffer: ArrayBuffer = new ArrayBuffer(stat.size);
    fileIo.readSync(file.fd, buffer);
    const helper: util.Base64Helper = new util.Base64Helper();
    const encoded: string = helper.encodeToStringSync(new Uint8Array(buffer));
    return `data:${resolveMime(uri)};base64,${encoded}`;
  } finally {
    fileIo.closeSync(file);
  }
}

function resolveMime(uri: string): string {
  const lower: string = uri.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
    return 'image/jpeg';
  }
  if (lower.endsWith('.webp')) {
    return 'image/webp';
  }
  return 'image/png';
}
