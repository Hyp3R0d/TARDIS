import { app, BrowserWindow, ipcMain, net, protocol, safeStorage, shell } from 'electron'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'
import {
  archiveGeneration,
  clearApiKey,
  createGeneration,
  getArchiveRoot,
  getStatus,
  initializeApiKeySettings,
  openArchivePath,
  pollGeneration,
  saveApiKey,
} from './api.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const isDevelopment = process.argv.includes('--dev') || Boolean(process.env.ELECTRON_START_URL)
if (process.env.TARDIS_REMOTE_DEBUG_PORT) {
  app.commandLine.appendSwitch('remote-debugging-port', String(process.env.TARDIS_REMOTE_DEBUG_PORT))
}
const gotSingleInstanceLock = app.requestSingleInstanceLock()

protocol.registerSchemesAsPrivileged([{
  scheme: 'tardis-asset',
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
}])

let mainWindow = null
let archiveRoot = null

function serializeError(error) {
  const result = new Error(error?.message || 'TARDIS 操作失败。')
  if (error?.status) result.status = error.status
  return result
}

function registerIpc() {
  const invoke = (channel, handler) => {
    ipcMain.removeHandler(channel)
    ipcMain.handle(channel, async (...args) => {
      try { return await handler(...args) } catch (error) { throw serializeError(error) }
    })
  }

  invoke('generation:create', (_event, payload) => createGeneration(payload))
  invoke('generation:poll', (_event, id) => pollGeneration(id))
  invoke('generation:archive', (_event, payload) => archiveGeneration({ archiveRoot, ...(payload || {}) }))
  invoke('settings:save-api-key', (_event, value) => saveApiKey(value))
  invoke('settings:clear-api-key', () => clearApiKey())
  invoke('archive:open', (_event, payload) => openArchivePath(archiveRoot, payload?.target, { show: Boolean(payload?.show), shell }))
  invoke('app:status', (event) => ({
    ...getStatus(),
    archiveRoot,
    maximized: Boolean(BrowserWindow.fromWebContents(event.sender)?.isMaximized()),
  }))

  invoke('window:minimize', (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    window?.minimize()
    return true
  })
  invoke('window:maximize', (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    if (!window) return false
    if (window.isMaximized()) window.unmaximize()
    else window.maximize()
    return window.isMaximized()
  })
  invoke('window:is-maximized', (event) => BrowserWindow.fromWebContents(event.sender)?.isMaximized() || false)
  invoke('window:close', (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close()
    return true
  })
}

function emitMaximizedState(window) {
  if (!window || window.isDestroyed()) return
  window.webContents.send('window:maximized-change', window.isMaximized())
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 980,
    minHeight: 640,
    frame: false,
    show: false,
    backgroundColor: '#0c0f12',
    title: 'TARDIS Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      spellcheck: false,
    },
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())
  mainWindow.on('maximize', () => emitMaximizedState(mainWindow))
  mainWindow.on('unmaximize', () => emitMaximizedState(mainWindow))
  mainWindow.on('closed', () => { mainWindow = null })

  if (isDevelopment) {
    const devUrl = process.env.ELECTRON_START_URL || 'http://127.0.0.1:5173'
    mainWindow.loadURL(devUrl)
  } else {
    mainWindow.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'))
  }
}

function registerAssetProtocol() {
  protocol.handle('tardis-asset', async (request) => {
    try {
      const parsed = new URL(request.url)
      if (parsed.hostname !== 'asset' || !archiveRoot) return new Response('Not found', { status: 404 })
      const parts = parsed.pathname.split('/').filter(Boolean).map((part) => decodeURIComponent(part))
      const filePath = path.resolve(archiveRoot, ...parts)
      const relative = path.relative(archiveRoot, filePath)
      if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        return new Response('Not found', { status: 404 })
      }
      return net.fetch(pathToFileURL(filePath).toString())
    } catch {
      return new Response('Bad request', { status: 400 })
    }
  })
}

if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show()
    mainWindow.focus()
  })

  app.whenReady().then(async () => {
    await initializeApiKeySettings({ userDataPath: app.getPath('userData'), safeStorage })
    archiveRoot = getArchiveRoot(app)
    registerAssetProtocol()
    registerIpc()
    createWindow()
    app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
  })
}
