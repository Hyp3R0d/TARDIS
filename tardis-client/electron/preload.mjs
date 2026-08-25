import { contextBridge, ipcRenderer } from 'electron'

// Keep the renderer surface deliberately small. In particular, the API key and
// Node.js primitives never cross the context-isolation boundary.
contextBridge.exposeInMainWorld('tardis', {
  isDesktop: true,
  platform: process.platform,
  createGeneration: (payload) => ipcRenderer.invoke('generation:create', payload),
  pollGeneration: (id) => ipcRenderer.invoke('generation:poll', id),
  archiveGeneration: (payload) => ipcRenderer.invoke('generation:archive', payload),
  saveApiKey: (value) => ipcRenderer.invoke('settings:save-api-key', value),
  clearApiKey: () => ipcRenderer.invoke('settings:clear-api-key'),
  getAppStatus: () => ipcRenderer.invoke('app:status'),
  openArchive: (target) => ipcRenderer.invoke('archive:open', { target, show: false }),
  showArchive: (target) => ipcRenderer.invoke('archive:open', { target, show: true }),
  window: {
    minimize: () => ipcRenderer.invoke('window:minimize'),
    maximize: () => ipcRenderer.invoke('window:maximize'),
    close: () => ipcRenderer.invoke('window:close'),
    isMaximized: () => ipcRenderer.invoke('window:is-maximized'),
    onMaximizedChange: (listener) => {
      if (typeof listener !== 'function') return () => {}
      const wrapped = (_event, value) => listener(Boolean(value))
      ipcRenderer.on('window:maximized-change', wrapped)
      return () => ipcRenderer.removeListener('window:maximized-change', wrapped)
    },
  },
})
