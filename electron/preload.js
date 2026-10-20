/*
 * Preload: only the bridge the renderer needs. Nothing else exposed.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("pocketrag", {
  getBackendUrl: () => ipcRenderer.invoke("get-backend-url"),
  pickFiles: () => ipcRenderer.invoke("pick-files"),
  readFileBytes: (path) => ipcRenderer.invoke("read-file-bytes", path),
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
});
