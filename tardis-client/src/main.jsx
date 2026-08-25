import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  ArrowUp,
  Check,
  ChevronDown,
  Clock3,
  Clapperboard,
  Download,
  FileImage,
  FolderOpen,
  GalleryHorizontalEnd,
  ImagePlus,
  LoaderCircle,
  Menu,
  Minimize2,
  Maximize2,
  MoreHorizontal,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from 'lucide-react'
import './styles.css'

const STORAGE_KEY = 'tardis-studio-creations-v1'
const DEFAULT_SETTINGS = { quality: 'speed', size: '1280x720', fps: 30, duration: 5, withAudio: false }
const desktopBridge = typeof window !== 'undefined' && window.tardis?.isDesktop ? window.tardis : null
const isDesktop = Boolean(desktopBridge)

// The desktop build keeps the API key and archive filesystem behind the
// context-isolated preload bridge. The browser fallback is intentionally kept
// for local UI development and does not change the desktop protocol.
async function createGenerationRequest(payload) {
  if (desktopBridge) return desktopBridge.createGeneration(payload)
  const response = await fetch('/api/generations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.error || '任务提交失败。')
  return result
}

async function pollGenerationRequest(id) {
  if (desktopBridge) return desktopBridge.pollGeneration(id)
  const response = await fetch(`/api/generations/${encodeURIComponent(id)}`)
  const result = await response.json()
  if (!response.ok) throw new Error(result.error || '查询任务状态失败。')
  return result
}

function readStoredCreations() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    return Array.isArray(value) ? value : []
  } catch {
    return []
  }
}

function writeStoredCreations(items) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, 24))) } catch { /* local history is best effort */ }
}

function formatDate(value) {
  if (!value) return '刚刚'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚'
  return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date)
}

function statusLabel(status) {
  if (status === 'SUCCESS') return '已完成'
  if (status === 'FAIL') return '生成失败'
  return '生成中'
}

function App() {
  const [creations, setCreations] = useState(readStoredCreations)
  const [activeId, setActiveId] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [reference, setReference] = useState(null)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [showSettings, setShowSettings] = useState(false)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [maximized, setMaximized] = useState(false)
  const [archiveRoot, setArchiveRoot] = useState(null)
  const [apiStatus, setApiStatus] = useState(null)
  const [showKeySettings, setShowKeySettings] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [keyBusy, setKeyBusy] = useState(false)
  const fileRef = useRef(null)
  const timers = useRef(new Map())

  useEffect(() => {
    writeStoredCreations(creations)
  }, [creations])

  useEffect(() => () => {
    for (const timer of timers.current.values()) clearTimeout(timer)
  }, [])

  useEffect(() => {
    if (!desktopBridge) return undefined
    let active = true
    desktopBridge.getAppStatus?.().then((status) => {
      if (active) {
        setMaximized(Boolean(status?.maximized))
        setArchiveRoot(status?.archiveRoot || null)
        setApiStatus(status || null)
      }
    }).catch(() => {})
    const unsubscribe = desktopBridge.window?.onMaximizedChange?.((value) => setMaximized(value))
    return () => {
      active = false
      unsubscribe?.()
    }
  }, [])

  useEffect(() => {
    // Resume visible processing jobs after reopening the desktop client. The
    // provider task remains valid even if the renderer was closed meanwhile.
    for (const item of creations) {
      if (item.status === 'PROCESSING' && !timers.current.has(item.id)) poll(item)
    }
  }, [])

  const active = useMemo(() => creations.find((item) => item.id === activeId) || null, [creations, activeId])

  const updateCreation = (id, patch) => {
    setCreations((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item))
  }

  const chooseReference = (file) => {
    if (!file) return
    if (!/^image\/(png|jpe?g)$/i.test(file.type)) {
      setNotice({ type: 'error', text: '请上传 PNG、JPG 或 JPEG 图片。' })
      return
    }
    if (file.size > 5 * 1024 * 1024) {
      setNotice({ type: 'error', text: '参考图不能超过 5 MB。' })
      return
    }
    const reader = new FileReader()
    reader.onload = () => setReference({ name: file.name, dataUrl: reader.result, size: file.size })
    reader.readAsDataURL(file)
  }

  const poll = async (creation) => {
    const started = Date.now()
    const tick = async () => {
      try {
        const result = await pollGenerationRequest(creation.id)
        const elapsed = Date.now() - started
        const simulatedProgress = Math.min(94, Math.max(12, Math.round(12 + elapsed / 950)))
        if (result.status === 'SUCCESS' && result.videoUrl) {
          let archive = null
          if (desktopBridge) {
            try {
              updateCreation(creation.id, { progress: 97, archivePending: true })
              archive = await desktopBridge.archiveGeneration({
                id: creation.id,
                prompt: creation.prompt,
                settings: creation.settings,
                referenceName: creation.referenceName,
                videoUrl: result.videoUrl,
                coverUrl: result.coverUrl,
                createdAt: creation.createdAt,
              })
            } catch (archiveError) {
              updateCreation(creation.id, { archiveError: archiveError.message || '本地归档失败。' })
            }
          }
          updateCreation(creation.id, {
            status: 'SUCCESS',
            progress: 100,
            videoUrl: archive?.videoUrl || result.videoUrl,
            coverUrl: archive?.coverUrl || result.coverUrl,
            localVideoPath: archive?.localVideoPath || null,
            localCoverPath: archive?.localCoverPath || null,
            archiveDir: archive?.archiveDir || null,
            archived: Boolean(archive?.archived),
            archivePending: false,
            finishedAt: new Date().toISOString(),
          })
          setNotice({ type: archive?.archived === false ? 'info' : 'success', text: archive?.archived === false ? '视频已生成，但本地归档未完成。' : '视频已生成并归档，可以直接播放。' })
          return
        }
        if (result.status === 'FAIL') {
          updateCreation(creation.id, { status: 'FAIL', progress: 0, error: result.error || '云端生成失败，请调整描述后重试。' })
          setNotice({ type: 'error', text: result.error || '云端生成失败，请查看创作记录。' })
          return
        }
        updateCreation(creation.id, { progress: simulatedProgress })
        const timer = setTimeout(tick, 3500)
        timers.current.set(creation.id, timer)
      } catch (error) {
        updateCreation(creation.id, { status: 'FAIL', progress: 0, error: error.message })
        setNotice({ type: 'error', text: error.message })
      }
    }
    await tick()
  }

  const submit = async () => {
    const value = prompt.trim()
    if (!value) {
      setNotice({ type: 'error', text: '请先写下你想生成的画面。' })
      return
    }
    if (value.length > 512) {
      setNotice({ type: 'error', text: '描述不能超过 512 个字符。' })
      return
    }
    setBusy(true)
    setNotice(null)
    try {
      const result = await createGenerationRequest({ prompt: value, imageData: reference?.dataUrl || null, settings })
      const item = {
        id: result.id,
        requestId: result.requestId,
        prompt: value,
        status: 'PROCESSING',
        progress: 8,
        reference: reference?.dataUrl || null,
        referenceName: reference?.name || null,
        settings: { ...settings },
        createdAt: new Date().toISOString(),
        videoUrl: null,
        coverUrl: null,
        archived: false,
        archivePending: false,
      }
      setCreations((items) => [item, ...items.filter((entry) => entry.id !== item.id)])
      setActiveId(item.id)
      setPrompt('')
      setReference(null)
      setShowSettings(false)
      setNotice({ type: 'info', text: '任务已提交，正在等待云端生成。' })
      poll(item)
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setBusy(false)
    }
  }

  const removeCreation = (id) => {
    const timer = timers.current.get(id)
    if (timer) clearTimeout(timer)
    timers.current.delete(id)
    setCreations((items) => items.filter((item) => item.id !== id))
    if (activeId === id) setActiveId(null)
  }

  const newCreation = () => {
    setActiveId(null)
    setPrompt('')
    setReference(null)
    setNotice(null)
    setSidebarOpen(false)
  }

  const openArchive = (target) => {
    if (!desktopBridge) return
    desktopBridge.showArchive?.(target || archiveRoot).catch((error) => setNotice({ type: 'error', text: error.message || '无法打开本地归档。' }))
  }

  const saveApiKey = async () => {
    if (!desktopBridge) return
    setKeyBusy(true)
    try {
      const status = await desktopBridge.saveApiKey(apiKeyInput)
      setApiStatus(status)
      setApiKeyInput('')
      setShowKeySettings(false)
      setNotice({ type: 'success', text: 'API Key 已安全保存，可以开始生成视频。' })
    } catch (error) {
      setNotice({ type: 'error', text: error.message || 'API Key 保存失败。' })
    } finally {
      setKeyBusy(false)
    }
  }

  const clearStoredApiKey = async () => {
    if (!desktopBridge || !window.confirm('清除本机保存的 API Key？')) return
    setKeyBusy(true)
    try {
      const status = await desktopBridge.clearApiKey()
      setApiStatus(status)
      setNotice({ type: 'info', text: '本机 API Key 已清除。' })
    } catch (error) {
      setNotice({ type: 'error', text: error.message || 'API Key 清除失败。' })
    } finally {
      setKeyBusy(false)
    }
  }

  return (
    <div className="studio-shell">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><span /><span /><span /></div>
          <div><strong>TARDIS</strong><small>VIDEO STUDIO</small></div>
        </div>
        <button className="new-work" onClick={newCreation}><Plus size={17} strokeWidth={2.5} /> 新建创作 <span>⌘ K</span></button>
        <div className="sidebar-heading"><span>创作记录</span><span className="record-count">{creations.length}</span></div>
        <div className="history-list">
          {creations.length === 0 ? (
            <div className="history-empty"><GalleryHorizontalEnd size={22} /><span>你的创作会显示在这里</span></div>
          ) : creations.map((item) => (
            <button key={item.id} className={`history-item ${activeId === item.id ? 'is-active' : ''}`} onClick={() => { setActiveId(item.id); setSidebarOpen(false) }}>
              <div className="history-thumb">
                {item.coverUrl || item.reference ? <img src={item.coverUrl || item.reference} alt="" /> : <Clapperboard size={18} />}
                {item.status === 'PROCESSING' && <span className="thumb-spinner"><LoaderCircle size={15} /></span>}
                {item.status === 'SUCCESS' && <span className="thumb-play"><Play size={11} fill="currentColor" /></span>}
              </div>
              <div className="history-copy"><strong>{item.prompt}</strong><small>{item.archived ? '已归档' : statusLabel(item.status)} · {formatDate(item.createdAt)}</small></div>
              <span className="history-menu" onClick={(event) => { event.stopPropagation(); removeCreation(item.id) }}><Trash2 size={14} /></span>
            </button>
          ))}
        </div>
        <div className="sidebar-footer"><div className="status-dot" /><span>{isDesktop ? '桌面服务已连接' : '云端服务已连接'}</span><button title="打开归档目录" onClick={() => openArchive()} disabled={!isDesktop || !archiveRoot}><FolderOpen size={15} /></button></div>
      </aside>

      <main className="workspace">
          <header className="topbar app-titlebar">
            <button className="icon-button mobile-menu" onClick={() => setSidebarOpen((value) => !value)} title="打开创作记录"><Menu size={19} /></button>
          <div className="crumb titlebar-drag"><span className="desktop-app-name">TARDIS Studio</span><ChevronDown size={14} /><span className="muted">工作台 / 视频生成</span></div>
          <div className="topbar-actions"><span className={`model-pill ${apiStatus && !apiStatus.configured ? 'model-pill--needs-key' : ''}`}><Sparkles size={14} /> TARDIS-v1</span><button className="icon-button" onClick={() => setShowKeySettings(true)} title="设置 API Key"><MoreHorizontal size={19} /></button>{isDesktop && <div className="window-controls"><button className="window-control" onClick={() => desktopBridge.window.minimize()} title="最小化"><Minimize2 size={15} /></button><button className="window-control" onClick={() => desktopBridge.window.maximize()} title={maximized ? '还原' : '最大化'}>{maximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}</button><button className="window-control window-control--close" onClick={() => desktopBridge.window.close()} title="关闭"><X size={15} /></button></div>}</div>
        </header>

        <section className="canvas-area">
          {!active ? (
            <div className="welcome-block">
              <div className="welcome-orbit"><div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" /><WandSparkles size={28} /></div>
              <div className="eyebrow">TARDIS · VIDEO GENERATION</div>
              <h1>把你的想象，变成<br /><em>一段会呼吸的画面。</em></h1>
              <p>用一句描述开始创作。上传参考图，让角色、构图和氛围沿着你的想法延续。</p>
            </div>
          ) : (
            <div className="active-generation">
              <div className="active-meta"><span className={`state-badge state-${active.status.toLowerCase()}`}><span /> {statusLabel(active.status)}</span><span>{formatDate(active.createdAt)}</span></div>
              <p className="active-prompt">{active.prompt}</p>
              {active.status === 'SUCCESS' && active.videoUrl ? (
                <div className="video-result">
                  <video controls playsInline poster={active.coverUrl || undefined} src={active.videoUrl} />
                  <div className="video-toolbar"><span><Check size={15} /> {active.archived ? '已归档' : '已完成'} · {active.settings?.duration || 5}s</span><div className="video-actions">{active.archiveDir && <button type="button" className="download-button" onClick={() => openArchive(active.archiveDir)}><FolderOpen size={15} /> 打开归档</button>}<a href={active.videoUrl} target="_blank" rel="noreferrer" className="download-button"><Download size={15} /> 打开视频</a></div></div>
                </div>
              ) : active.status === 'FAIL' ? (
                <div className="result-error"><X size={18} /><div><strong>这次生成没有完成</strong><span>{active.error || '请稍后重试。'}</span></div></div>
              ) : (
                <div className="processing-summary"><LoaderCircle size={22} /><div><strong>正在生成你的画面</strong><span>进度会显示在描述框下方，云端完成后这里会自动切换为播放器。</span></div></div>
              )}
            </div>
          )}
        </section>

        <section className="composer-zone">
          {reference && (
            <div className="reference-strip"><div className="reference-label"><ImagePlus size={14} /><span>参考图</span></div><div className="reference-preview"><img src={reference.dataUrl} alt={reference.name} /><button onClick={() => setReference(null)} title="移除参考图"><X size={13} /></button></div><span className="reference-name">{reference.name}</span><span className="reference-hint">将作为首帧构图参考</span></div>
          )}
          {notice && <div className={`notice notice-${notice.type}`}><span>{notice.text}</span><button onClick={() => setNotice(null)}><X size={14} /></button></div>}
          <div className="composer-card">
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') submit() }} placeholder="描述你想生成的视频，例如：雨夜的霓虹小巷中，一位撑伞的人缓慢走过，镜头平稳跟随……" maxLength={512} />
            <div className="composer-bottom"><div className="composer-tools"><button className={`tool-button ${reference ? 'has-file' : ''}`} onClick={() => fileRef.current?.click()} title="上传参考图"><ImagePlus size={18} /><span>{reference ? '更换参考图' : '添加参考图'}</span></button><input ref={fileRef} type="file" accept="image/png,image/jpeg" hidden onChange={(event) => chooseReference(event.target.files?.[0])} /><button className={`tool-button ${showSettings ? 'is-on' : ''}`} onClick={() => setShowSettings((value) => !value)} title="生成参数"><Settings2 size={17} /><span>参数</span></button>{prompt.length > 0 && <span className="char-count">{prompt.length}/512</span>}</div><button className="send-button" onClick={submit} disabled={busy || !prompt.trim()}>{busy ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={19} strokeWidth={2.6} />}<span>{busy ? '提交中' : '生成视频'}</span></button></div>
          </div>
          {showSettings && <div className="settings-panel"><div className="setting-block"><label>生成质量</label><div className="segmented"><button className={settings.quality === 'speed' ? 'selected' : ''} onClick={() => setSettings({ ...settings, quality: 'speed' })}>速度优先</button><button className={settings.quality === 'quality' ? 'selected' : ''} onClick={() => setSettings({ ...settings, quality: 'quality' })}>质量优先</button></div></div><div className="setting-block"><label>画面比例</label><select value={settings.size} onChange={(event) => setSettings({ ...settings, size: event.target.value })}><option value="1280x720">横屏 16:9</option><option value="720x1280">竖屏 9:16</option><option value="1024x1024">方形 1:1</option><option value="1920x1080">横屏高清</option></select></div><div className="setting-block"><label>时长</label><div className="segmented"><button className={settings.duration === 5 ? 'selected' : ''} onClick={() => setSettings({ ...settings, duration: 5 })}>5 秒</button><button className={settings.duration === 10 ? 'selected' : ''} onClick={() => setSettings({ ...settings, duration: 10 })}>10 秒</button></div></div><div className="setting-block"><label>音效</label><button className={`toggle ${settings.withAudio ? 'on' : ''}`} onClick={() => setSettings({ ...settings, withAudio: !settings.withAudio })}><span />{settings.withAudio ? '开启' : '关闭'}</button></div></div>}
          {active?.status === 'PROCESSING' && <div className="composer-progress"><div className="composer-progress-head"><span><LoaderCircle size={14} /> 云端生成进度</span><strong>{active.progress}%</strong></div><div className="composer-progress-track"><i style={{ width: `${active.progress}%` }} /></div><small>任务正在处理中，完成后会自动显示可播放视频。</small></div>}
          <div className="composer-footnote"><span><Upload size={12} /> 支持 PNG / JPG，单张不超过 5 MB</span><span>⌘ Enter 快捷生成</span></div>
        </section>
      </main>
      {showKeySettings && isDesktop && <div className="modal-backdrop" onMouseDown={() => setShowKeySettings(false)}><section className="key-modal" role="dialog" aria-modal="true" aria-labelledby="key-modal-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="key-modal-head"><div><span className="eyebrow">TARDIS · CONNECTION</span><h2 id="key-modal-title">连接云端生成</h2></div><button className="icon-button" onClick={() => setShowKeySettings(false)} title="关闭设置"><X size={18} /></button></div>
        <p className="key-modal-copy">使用你的智谱 API Key 调用视频生成服务。密钥仅通过系统凭据保护保存在本机，不会写入创作记录。</p>
        <label className="key-field"><span>智谱 API Key</span><input type="password" value={apiKeyInput} onChange={(event) => setApiKeyInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && apiKeyInput.trim()) saveApiKey() }} placeholder="标识.密钥" autoFocus /></label>
        {apiStatus?.configurationIssue && <div className="key-warning">{apiStatus.configurationIssue}</div>}
        <div className="key-modal-meta"><span className={apiStatus?.configured ? 'key-state key-state--ok' : 'key-state'}><i />{apiStatus?.configured ? '已连接' : '尚未配置'}</span><span>{apiStatus?.environmentConfigured ? '使用环境变量' : apiStatus?.localConfigured ? '使用本机安全存储' : '等待密钥'}</span></div>
        <div className="key-modal-actions">{apiStatus?.localConfigured && <button className="secondary-button" onClick={clearStoredApiKey} disabled={keyBusy}>清除本机密钥</button>}<button className="send-button" onClick={saveApiKey} disabled={keyBusy || !apiKeyInput.trim()}>{keyBusy ? <LoaderCircle size={17} className="spin" /> : <Check size={17} />}<span>{keyBusy ? '处理中' : '保存并连接'}</span></button></div>
      </section></div>}
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
