// Pickups W1/T11 test. No WorkflowIntegration.node: this checks whether a plugin that only opens
// windows loads at all (T11), and whether a second, large window receives clicks and keys (W1).
const { app, BrowserWindow } = require('electron');
const path = require('path');

const PAGE = path.join(__dirname, 'test.html');

function open(kind, width, height) {
  const win = new BrowserWindow({
    width, height,
    title: `Pickups ${kind}`,
    backgroundColor: '#141417',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  win.loadFile(PAGE, { query: { kind } });
  return win;
}

app.whenReady().then(() => {
  open('panel', 380, 720);
  open('shot', 1400, 860);
});

app.on('window-all-closed', () => app.quit());
