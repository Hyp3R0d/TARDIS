import crypto from 'node:crypto'
import dotenv from 'dotenv'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '..')
const ZHIPU_BASE = 'https://open.bigmodel.cn/api/paas/v4'
const DATA_IMAGE_RE = /^data:(image\/(?:png|jpe?g));base64,([A-Za-z0-9+/=\r\n]+)$/i
const API_KEY_RE = /^[A-Za-z0-9_-]{8,128}\.[A-Za-z0-9_-]{8,128}$/
const MAX_IMAGE_BYTES = 5 * 1024 * 1024
const MAX_DOWNLOAD_BYTES = 768 * 1024 * 1024

// The packaged application intentionally does not ship .env.local. For local
// development, load it from the project directory; deployed clients should
// provide ZHIPU_API_KEY through the process environment or a secure config UI.
dotenv.config({ path: path.join(PROJECT_ROOT, '.env.local') })
dotenv.config()

let localApiKey = ''
let settingsPath = null
let settingsDocument = {}
let settingsIssue = null
let encryptionAvailable = false
let encryptString = null
let decryptString = null

export class TardisError extends Error {
  constructor(message, status = 400, details) {
    super(message)
    this.name = 'TardisError'
    this.status = status
    this.details = details
  }
}

const cleanPrompt = (value) => typeof value === 'string' ? value.trim() : ''

function validateApiKey(value, label = 'API Key') {
  const key = String(value || '').trim()
  if (!key || key === 'replace-with-your-key') {
    throw new TardisError(`${label} 不能为空。`)
  }
  if (!API_KEY_RE.test(key)) {
    throw new TardisError(`${label} 格式无效，应为“标识.密钥”的形式。`)
  }
  return key
}

function environmentKeyState() {
  const raw = String(process.env.ZHIPU_API_KEY || '').trim()
  if (!raw || raw === 'replace-with-your-key') return { present: false, key: '' }
  try {
    return { present: true, key: validateApiKey(raw, '环境变量 ZHIPU_API_KEY') }
  } catch (error) {
    return { present: true, key: '', error: error.message }
  }
}

function publicApiKeyStatus() {
  const environment = environmentKeyState()
  const configured = environment.present ? Boolean(environment.key) : Boolean(localApiKey)
  return {
    configured,
    apiKeySource: environment.present ? 'environment' : (localApiKey ? 'local' : 'none'),
    environmentConfigured: Boolean(environment.key),
    localConfigured: Boolean(localApiKey),
    canStoreApiKey: Boolean(settingsPath && encryptionAvailable),
    storageProtected: Boolean(encryptionAvailable),
    configurationIssue: environment.error || settingsIssue || null,
  }
}

async function readSettingsFile() {
  try {
    const parsed = JSON.parse(await fsp.readFile(settingsPath, 'utf8'))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch (error) {
    if (error?.code === 'ENOENT') return {}
    settingsIssue = '本机密钥设置无法读取，请重新保存 API Key。'
    return {}
  }
}

async function writeSettingsFile(document) {
  if (!settingsPath) throw new TardisError('本机设置目录尚未初始化。', 503)
  await fsp.mkdir(path.dirname(settingsPath), { recursive: true })
  await fsp.writeFile(settingsPath, JSON.stringify(document, null, 2), { encoding: 'utf8', mode: 0o600 })
}

export async function initializeApiKeySettings({ userDataPath, safeStorage } = {}) {
  settingsPath = userDataPath ? path.join(path.resolve(userDataPath), 'settings.json') : null
  encryptionAvailable = Boolean(safeStorage?.isEncryptionAvailable?.())
  encryptString = encryptionAvailable ? (value) => safeStorage.encryptString(value) : null
  decryptString = encryptionAvailable ? (value) => safeStorage.decryptString(value) : null
  localApiKey = ''
  settingsIssue = null

  if (!settingsPath) {
    settingsIssue = '本机设置目录不可用。'
    return publicApiKeyStatus()
  }

  settingsDocument = await readSettingsFile()
  const encrypted = settingsDocument?.apiKey?.encrypted
  if (typeof encrypted === 'string' && encrypted) {
    if (!encryptionAvailable) {
      settingsIssue = '系统凭据保护暂不可用，本机密钥未载入。'
    } else {
      try {
        localApiKey = validateApiKey(decryptString(Buffer.from(encrypted, 'base64')), '已保存的 API Key')
      } catch {
        localApiKey = ''
        settingsIssue = '已保存的 API Key 无法解密，请重新配置。'
      }
    }
  }
  return publicApiKeyStatus()
}

export async function saveApiKey(value) {
  if (!encryptionAvailable || !encryptString) {
    throw new TardisError('当前系统无法安全保存 API Key，请改用 ZHIPU_API_KEY 环境变量。', 503)
  }
  const key = validateApiKey(value)
  let encrypted
  try {
    encrypted = encryptString(key).toString('base64')
  } catch {
    throw new TardisError('系统凭据保护失败，API Key 未保存。', 500)
  }
  const nextDocument = {
    ...settingsDocument,
    version: 1,
    apiKey: { scheme: 'electron-safe-storage', encrypted },
    updatedAt: new Date().toISOString(),
  }
  await writeSettingsFile(nextDocument)
  settingsDocument = nextDocument
  localApiKey = key
  settingsIssue = null
  return publicApiKeyStatus()
}

export async function clearApiKey() {
  const { apiKey: _apiKey, ...rest } = settingsDocument
  const nextDocument = { ...rest, version: 1, updatedAt: new Date().toISOString() }
  await writeSettingsFile(nextDocument)
  settingsDocument = nextDocument
  localApiKey = ''
  settingsIssue = null
  return publicApiKeyStatus()
}

function decodeImageData(value) {
  if (typeof value !== 'string') return null
  const match = value.match(DATA_IMAGE_RE)
  if (!match) return null
  const base64 = match[2].replace(/\s/g, '')
  let bytes
  try { bytes = Buffer.from(base64, 'base64') } catch { return null }
  const normalized = base64.replace(/=+$/, '')
  const roundTrip = bytes.toString('base64').replace(/=+$/, '')
  if (!normalized || normalized !== roundTrip) return null
  if (bytes.byteLength > MAX_IMAGE_BYTES) {
    throw new TardisError('参考图解码后不能超过 5 MB，请压缩后重试。')
  }
  return { mimeType: match[1].toLowerCase(), bytes }
}

function makeRequestId() {
  return `tardis-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`
}

function apiKey() {
  const environment = environmentKeyState()
  if (environment.present) {
    if (environment.error) throw new TardisError(environment.error, 503)
    return environment.key
  }
  if (localApiKey) return localApiKey
  throw new TardisError('尚未配置 API Key。请打开客户端右上角的密钥设置后重试。', 503)
}

async function zhipuFetch(url, options = {}) {
  const headers = new Headers(options.headers || {})
  headers.set('Authorization', `Bearer ${apiKey()}`)
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let response
  try {
    response = await fetch(url, { ...options, headers })
  } catch (error) {
    throw new TardisError(`无法连接视频生成服务：${error.message || '网络错误'}`, 502)
  }
  const text = await response.text()
  let data
  try { data = text ? JSON.parse(text) : {} } catch { data = { raw: text } }
  if (!response.ok) {
    const message = data?.error?.message || data?.message || `云端请求失败（HTTP ${response.status}）`
    throw new TardisError(message, response.status, data)
  }
  return data
}

function normalizeSettings(settings = {}) {
  const sizes = ['1280x720', '720x1280', '1024x1024', '1920x1080', '1080x1920', '2048x1080', '3840x2160']
  return {
    quality: settings.quality === 'quality' ? 'quality' : 'speed',
    withAudio: Boolean(settings.withAudio),
    size: sizes.includes(settings.size) ? settings.size : '1280x720',
    fps: Number(settings.fps) === 60 ? 60 : 30,
    duration: Number(settings.duration) === 10 ? 10 : 5,
  }
}

export async function createGeneration(input = {}) {
  const prompt = cleanPrompt(input.prompt)
  if (!prompt) throw new TardisError('请输入视频描述后再发送。')
  if (prompt.length > 512) throw new TardisError('视频描述不能超过 512 个字符。')

  const imageData = input.imageData || null
  if (imageData && !decodeImageData(imageData)) {
    throw new TardisError('参考图仅支持 PNG、JPG 或 JPEG 的 Base64 Data URL。')
  }

  const settings = normalizeSettings(input.settings)
  const requestId = makeRequestId()
  const payload = {
    model: 'cogvideox-3',
    prompt,
    quality: settings.quality,
    with_audio: settings.withAudio,
    watermark_enabled: true,
    size: settings.size,
    fps: settings.fps,
    duration: settings.duration,
    request_id: requestId,
    user_id: 'tardis-local-user',
  }
  if (imageData) payload.image_url = imageData

  const result = await zhipuFetch(`${ZHIPU_BASE}/videos/generations`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  return {
    id: result.id,
    requestId: result.request_id || requestId,
    taskStatus: result.task_status || 'PROCESSING',
    model: result.model || payload.model,
    createdAt: new Date().toISOString(),
  }
}

export async function pollGeneration(id) {
  if (!id || !/^[A-Za-z0-9._:-]{1,256}$/.test(String(id))) {
    throw new TardisError('任务 ID 无效。')
  }
  const result = await zhipuFetch(`${ZHIPU_BASE}/async-result/${encodeURIComponent(id)}`, { method: 'GET' })
  const video = Array.isArray(result.video_result) ? result.video_result[0] : null
  const providerStatus = String(result.task_status || '').toUpperCase()
  const status = video?.url ? 'SUCCESS' : (['PROCESSING', 'SUCCESS', 'FAIL'].includes(providerStatus) ? providerStatus : 'PROCESSING')
  return {
    id: result.id || id,
    status,
    created: result.created,
    model: result.model,
    videoUrl: video?.url || null,
    coverUrl: video?.cover_image_url || null,
    error: status === 'FAIL' ? (result.error?.message || result.message || '云端生成失败。') : null,
    contentFilter: result.content_filter || [],
  }
}

function safeSegment(value) {
  return String(value || '').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 100) || 'generation'
}

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate)
  return relative === '' || (relative && !relative.startsWith('..') && !path.isAbsolute(relative))
}

async function downloadTo(url, destination) {
  if (!url || !/^https?:\/\//i.test(url)) throw new TardisError('云端返回了无效的视频地址。', 502)
  let response
  try { response = await fetch(url) } catch (error) {
    throw new TardisError(`下载生成结果失败：${error.message || '网络错误'}`, 502)
  }
  if (!response.ok) throw new TardisError(`下载生成结果失败（HTTP ${response.status}）。`, response.status)
  const contentLength = Number(response.headers.get('content-length') || 0)
  if (contentLength > MAX_DOWNLOAD_BYTES) throw new TardisError('生成结果超过本地归档大小限制。', 413)
  const buffer = Buffer.from(await response.arrayBuffer())
  if (buffer.byteLength > MAX_DOWNLOAD_BYTES) throw new TardisError('生成结果超过本地归档大小限制。', 413)
  await fsp.mkdir(path.dirname(destination), { recursive: true })
  const temporary = `${destination}.${process.pid}.${Date.now()}.part`
  await fsp.writeFile(temporary, buffer)
  await fsp.rename(temporary, destination)
  return destination
}

function assetUrl(archiveRoot, filePath) {
  const relative = path.relative(archiveRoot, filePath)
  if (!isWithin(archiveRoot, filePath)) throw new TardisError('归档路径越界。', 400)
  const encoded = relative.split(path.sep).map((part) => encodeURIComponent(part)).join('/')
  return `tardis-asset://asset/${encoded}`
}

export async function archiveGeneration({ archiveRoot, id, prompt, settings, referenceName, videoUrl, coverUrl, createdAt } = {}) {
  if (!archiveRoot) throw new TardisError('本地归档目录不可用。', 500)
  if (!id) throw new TardisError('缺少任务 ID。')
  if (!videoUrl) throw new TardisError('云端尚未返回可下载的视频地址。')

  await fsp.mkdir(archiveRoot, { recursive: true })
  const existingManifest = await findManifest(archiveRoot, id)
  if (existingManifest?.videoPath && fs.existsSync(existingManifest.videoPath)) {
    return manifestWithAssetUrls(archiveRoot, existingManifest)
  }

  const folderName = `${new Date().toISOString().replace(/[:.]/g, '-')}-${safeSegment(id).slice(-48)}`
  const directory = path.join(archiveRoot, folderName)
  await fsp.mkdir(directory, { recursive: true })
  const videoPath = path.join(directory, 'video.mp4')
  const coverPath = path.join(directory, 'cover.jpg')
  await downloadTo(videoUrl, videoPath)
  let savedCoverPath = null
  if (coverUrl) {
    try { await downloadTo(coverUrl, coverPath); savedCoverPath = coverPath } catch {
      // A cover is optional; the video itself remains a valid archive.
    }
  }
  const manifest = {
    version: 1,
    id: String(id),
    prompt: cleanPrompt(prompt),
    settings: normalizeSettings(settings),
    referenceName: referenceName || null,
    videoPath,
    coverPath: savedCoverPath,
    sourceVideoUrl: videoUrl,
    sourceCoverUrl: coverUrl || null,
    createdAt: createdAt || new Date().toISOString(),
    archivedAt: new Date().toISOString(),
  }
  await fsp.writeFile(path.join(directory, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8')
  return manifestWithAssetUrls(archiveRoot, manifest)
}

async function findManifest(archiveRoot, id) {
  let entries
  try { entries = await fsp.readdir(archiveRoot, { withFileTypes: true }) } catch { return null }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const manifestPath = path.join(archiveRoot, entry.name, 'manifest.json')
    try {
      const manifest = JSON.parse(await fsp.readFile(manifestPath, 'utf8'))
      if (String(manifest.id) === String(id)) return manifest
    } catch { /* incomplete archive is ignored */ }
  }
  return null
}

function manifestWithAssetUrls(archiveRoot, manifest) {
  const result = {
    archived: Boolean(manifest.videoPath && fs.existsSync(manifest.videoPath)),
    archiveDir: path.dirname(manifest.videoPath),
    localVideoPath: manifest.videoPath,
    localCoverPath: manifest.coverPath || null,
    videoUrl: assetUrl(archiveRoot, manifest.videoPath),
    coverUrl: manifest.coverPath ? assetUrl(archiveRoot, manifest.coverPath) : null,
  }
  return { ...manifest, ...result }
}

export async function openArchivePath(archiveRoot, target, { show = false, shell } = {}) {
  if (!target) throw new TardisError('缺少归档目标。')
  let resolved = String(target)
  if (!path.isAbsolute(resolved)) {
    const manifest = await findManifest(archiveRoot, resolved)
    if (!manifest) throw new TardisError('找不到对应的本地归档。', 404)
    resolved = path.dirname(manifest.videoPath)
  }
  resolved = path.resolve(resolved)
  if (!isWithin(archiveRoot, resolved)) throw new TardisError('归档路径越界。', 400)
  if (!fs.existsSync(resolved)) throw new TardisError('本地归档不存在。', 404)
  if (show && shell?.showItemInFolder) shell.showItemInFolder(resolved)
  else if (shell?.openPath) await shell.openPath(resolved)
  return { path: resolved }
}

export function getArchiveRoot(app) {
  const override = String(process.env.TARDIS_ARCHIVE_DIR || '').trim()
  return override ? path.resolve(override) : path.join(app.getPath('userData'), 'archives')
}

export function getStatus() {
  return publicApiKeyStatus()
}
