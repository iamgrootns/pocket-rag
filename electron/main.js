/*
 * Electron main process.
 * One concern: window + backend lifecycle only - UI logic lives in src/app.js.
 */

const { app, BrowserWindow, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const BACKEND_URL = "http://127.0.0.1:8765";

let mainWindow = null;
let backendProcess = null;

function pythonCommand() {
  // Windows users can drop a venv python at backend/venv/Scripts/python.exe
  const winCandidate = path.join(__dirname, "..", "backend", "venv", "Scripts", "python.exe");
  const nixCandidate = path.join(__dirname, "..", "backend", "venv", "bin", "python");
  if (require("fs").existsSync(winCandidate)) return winCandidate;
  if (require("fs").existsSync(nixCandidate)) return nixCandidate;
  return "python3";
}

function startBackend() {
  const backendDir = path.join(__dirname, "..", "backend");
  backendProcess = spawn(pythonCommand(), ["main.py"], {
    cwd: backendDir,
    stdio: "inherit",
  });
  backendProcess.on("error", (err) => {
    console.error("backend failed to start:", err.message);
  });
  backendProcess.on("exit", (code) => console.log("backend exited", code));
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: "Pocket RAG",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "src", "index.html"));
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", stopBackend);
app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});

// ---- tiny IPC surface ----

ipcMain.handle("get-backend-url", () => BACKEND_URL);

// user picks files; renderer cannot see the filesystem (contextIsolation)
ipcMain.handle("pick-files", async () => {
  const { dialog } = require("electron");
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "Documents", extensions: ["pdf", "docx", "txt", "md"] },
    ],
  });
  return result.filePaths;
});

// read one file for upload (renderer has no fs access - contextIsolation)
ipcMain.handle("read-file-bytes", async (_event, filePath) => {
  const fs = require("fs");
  return fs.readFileSync(filePath).buffer;
});

ipcMain.handle("open-external", async (_event, url) => {
  shell.openExternal(url);
});
