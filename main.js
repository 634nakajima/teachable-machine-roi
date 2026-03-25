const { app, BrowserWindow, systemPreferences, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');
}

app.whenReady().then(async () => {
  // macOS camera permission
  if (process.platform === 'darwin') {
    const status = systemPreferences.getMediaAccessStatus('camera');
    console.log('[Main] Camera permission status:', status);
    if (status !== 'granted') {
      const granted = await systemPreferences.askForMediaAccess('camera');
      console.log('[Main] Camera permission granted:', granted);
    }
  }
  createWindow();
});

// IPC: clear HTTP cache
ipcMain.handle('clear-cache', async () => {
  if (mainWindow) {
    await mainWindow.webContents.session.clearCache();
  }
});

// IPC: select and extract ZIP model
ipcMain.handle('select-model-zip', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    title: 'Select Teachable Machine model ZIP',
    filters: [{ name: 'ZIP', extensions: ['zip'] }],
  });
  if (result.canceled) return null;

  const zipPath = result.filePaths[0];
  const AdmZip = require('adm-zip');
  const zip = new AdmZip(zipPath);

  // Extract to a temp folder inside app data
  const extractDir = path.join(app.getPath('userData'), 'models', path.basename(zipPath, '.zip') + '_' + Date.now());
  fs.mkdirSync(extractDir, { recursive: true });
  zip.extractAllTo(extractDir, true);

  return extractDir;
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
