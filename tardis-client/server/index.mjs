import dotenv from 'dotenv'
import express from 'express'
import crypto from 'node:crypto'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
dotenv.config({ path: path.resolve(__dirname, '../.env.local') })
dotenv.config()
const app = express()
const PORT = Number(process.env.PORT || 8787)
const API_KEY = process.env.ZHIPU_API_KEY
const ZHIPU_BASE = 'https://open.bigmodel.cn/api/paas/v4'

app.use(express.json({ limit: '10mb' }))

const cleanPrompt = (value) => typeof value === 'string' ? value.trim() : ''
const DATA_IMAGE_RE = /^data:(image\/(?:png|jpe?g));base64,([A-Za-z0-9+/=\r\n]+)$/i
const MAX_IMAGE_BYTES = 5 * 1024 * 1024

function decodeImageData(value) {
  if (typeof value !== 'string') return null
  const match = value.match(DATA_IMAGE_RE)
  if (!match) return null
  const base64 = match[2].replace(/\s/g, '')
  let bytes
  try { bytes = Buffer.from(base64, 'base64') } catch { return null }
  // Buffer.from is intentionally followed by a round-trip check: Node is
  // permissive about malformed Base64 and would otherwise silently truncate it.
  const normalized = base64.replace(/=+$/, '')
  const roundTrip = bytes.toString('base64').replace(/=+$/, '')
  if (!normalized || normalized !== roundTrip) return null
  if (bytes.byteLength > MAX_IMAGE_BYTES) throw apiError('参考图解码后不能超过 5 MB，请压缩后重试。')
  return { mimeType: match[1].toLowerCase(), bytes }
}

function makeRequestId() {
  return `tardis-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`
}

function apiError(message, status = 400, details = undefined) {
  const error = new Error(message)
  error.status = status
  error.details = details
  return error
}

async function zhipuFetch(url, options = {}) {
  if (!API_KEY) throw apiError('服务端尚未配置 ZHIPU_API_KEY。请在 .env.local 中设置 API Key。', 503)
  const headers = new Headers(options.headers || {})
  headers.set('Authorization', `Bearer ${API_KEY}`)
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(url, {
    ...options,
    headers,
  })
  const text = await response.text()
  let data
  try { data = text ? JSON.parse(text) : {} } catch { data = { raw: text } }
  if (!response.ok) {
    const message = data?.error?.message || data?.message || `云端请求失败（HTTP ${response.status}）`
    throw apiError(message, response.status, data)
  }
  return data
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, configured: Boolean(API_KEY), service: 'tardis-studio' })
})

app.post('/api/generations', async (req, res, next) => {
  try {
    const prompt = cleanPrompt(req.body?.prompt)
    if (!prompt) throw apiError('请输入视频描述后再发送。')
    if (prompt.length > 512) throw apiError('视频描述不能超过 512 个字符。')

    const imageData = req.body?.imageData
    if (imageData) {
      const decodedImage = decodeImageData(imageData)
      if (!decodedImage) throw apiError('参考图仅支持 PNG、JPG 或 JPEG 的 Base64 Data URL。')
    }

    const settings = req.body?.settings || {}
    const requestId = makeRequestId()
    const payload = {
      model: 'cogvideox-3',
      prompt,
      quality: settings.quality === 'quality' ? 'quality' : 'speed',
      with_audio: Boolean(settings.withAudio),
      watermark_enabled: true,
      size: ['1280x720', '720x1280', '1024x1024', '1920x1080', '1080x1920', '2048x1080', '3840x2160'].includes(settings.size)
        ? settings.size : '1280x720',
      fps: Number(settings.fps) === 60 ? 60 : 30,
      duration: Number(settings.duration) === 10 ? 10 : 5,
      request_id: requestId,
      user_id: 'tardis-local-user',
    }
    if (imageData) payload.image_url = imageData

    const result = await zhipuFetch(`${ZHIPU_BASE}/videos/generations`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    res.json({
      id: result.id,
      requestId: result.request_id || requestId,
      taskStatus: result.task_status || 'PROCESSING',
      model: result.model || payload.model,
      createdAt: new Date().toISOString(),
    })
  } catch (error) {
    next(error)
  }
})

app.get('/api/generations/:id', async (req, res, next) => {
  try {
    if (!req.params.id || !/^[A-Za-z0-9._:-]{1,256}$/.test(req.params.id)) throw apiError('任务 ID 无效。')
    const result = await zhipuFetch(`${ZHIPU_BASE}/async-result/${encodeURIComponent(req.params.id)}`, {
      method: 'GET',
    })
    const video = Array.isArray(result.video_result) ? result.video_result[0] : null
    const providerStatus = String(result.task_status || '').toUpperCase()
    const status = video?.url ? 'SUCCESS' : (['PROCESSING', 'SUCCESS', 'FAIL'].includes(providerStatus) ? providerStatus : 'PROCESSING')
    res.json({
      id: result.id || req.params.id,
      status,
      created: result.created,
      model: result.model,
      videoUrl: video?.url || null,
      coverUrl: video?.cover_image_url || null,
      error: status === 'FAIL' ? (result.error?.message || result.message || '云端生成失败。') : null,
      contentFilter: result.content_filter || [],
    })
  } catch (error) {
    next(error)
  }
})

app.use((error, _req, res, _next) => {
  const status = Number(error.status) || 500
  res.status(status).json({
    error: error.message || '本地服务发生未知错误。',
    details: process.env.NODE_ENV === 'development' ? error.details : undefined,
  })
})

app.listen(PORT, '127.0.0.1', () => {
  console.log(`TARDIS Studio API listening at http://127.0.0.1:${PORT}`)
})
