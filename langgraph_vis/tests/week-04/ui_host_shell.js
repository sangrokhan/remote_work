const DEFAULT_HOST_KEY_PREFIX = "ui-host-shell";
const DEFAULT_MODE_KEY_PREFIX = "ui-host-mode";
const MODE_STORAGE_VERSION = "v1";

const ALLOWED_MODES = new Set(["step-card", "dag", "log"]);

export class MemoryStorage {
  constructor(initial = null) {
    this.data = new Map(Object.entries(initial || {}));
  }

  getItem(key) {
    return this.data.get(key) ?? null;
  }

  setItem(key, value) {
    this.data.set(key, String(value));
  }

  removeItem(key) {
    this.data.delete(key);
  }

  clear() {
    this.data.clear();
  }
}

function _safeInt(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    return fallback;
  }
  return Math.floor(num);
}

function _safeMode(value) {
  return ALLOWED_MODES.has(value) ? value : "step-card";
}

function _hostStateKey(baseKey) {
  return `${DEFAULT_HOST_KEY_PREFIX}:${baseKey}`;
}

function _modeStateKey(baseKey) {
  return `${DEFAULT_MODE_KEY_PREFIX}:${MODE_STORAGE_VERSION}:${baseKey}`;
}

export class ViewModeController {
  constructor({queryKey, storage, isOpen = true, mode = "step-card", width = 320} = {}) {
    this.queryKey = String(queryKey || "");
    this.storage = storage;
    this.hostWidth = _safeInt(width, 320);
    this.isOpen = Boolean(isOpen);
    this.mode = _safeMode(mode);
    this._loadMode();
  }

  _loadMode() {
    if (!this.storage || !this.queryKey) {
      return;
    }
    const payload = this.storage.getItem(_modeStateKey(this.queryKey));
    if (!payload) {
      return;
    }
    try {
      const parsed = JSON.parse(payload);
      if (typeof parsed === "object" && parsed !== null) {
        if (typeof parsed.mode === "string") {
          this.mode = _safeMode(parsed.mode);
        }
        if (typeof parsed.isOpen === "boolean") {
          this.isOpen = parsed.isOpen;
        }
        if (typeof parsed.width === "number") {
          this.hostWidth = _safeInt(parsed.width, this.hostWidth);
        }
      }
    } catch {
      // Ignore malformed storage values.
    }
  }

  _persistMode() {
    if (!this.storage || !this.queryKey) {
      return;
    }
    this.storage.setItem(
      _modeStateKey(this.queryKey),
      JSON.stringify({
        mode: this.mode,
        isOpen: this.isOpen,
        width: this.hostWidth,
      })
    );
  }

  setQueryKey(nextQueryKey) {
    if (this.queryKey === String(nextQueryKey || "")) {
      return;
    }
    this.queryKey = String(nextQueryKey || "");
    this._loadMode();
  }

  setMode(mode) {
    this.mode = _safeMode(mode);
    this._persistMode();
  }

  open() {
    this.isOpen = true;
    this._persistMode();
  }

  close() {
    this.isOpen = false;
    this._persistMode();
  }

  toggleOpen() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  resize(width) {
    this.hostWidth = _safeInt(width, this.hostWidth);
    this._persistMode();
  }

  getState() {
    return {
      queryKey: this.queryKey,
      mode: this.mode,
      isOpen: this.isOpen,
      width: this.hostWidth,
    };
  }
}

export class UiHostShell {
  constructor({runId, threadId, storage = null, defaultWidth = 340} = {}) {
    this.runId = String(runId || "");
    this.threadId = String(threadId || "");
    this.storage = storage || new MemoryStorage();
    this.queryKey = `${this.runId}:${this.threadId}`;
    this.controller = new ViewModeController({
      queryKey: this.queryKey,
      storage: this.storage,
      isOpen: true,
      width: defaultWidth,
      mode: "step-card",
    });
    this._loadHostShell();
  }

  _loadHostShell() {
    if (!this.storage || !this.queryKey) {
      return;
    }
    const raw = this.storage.getItem(_hostStateKey(this.queryKey));
    if (!raw) {
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        if (typeof parsed.width === "number") {
          this.controller.hostWidth = _safeInt(parsed.width, this.controller.hostWidth);
        }
        if (typeof parsed.defaultMode === "string") {
          this.controller.mode = _safeMode(parsed.defaultMode);
        }
      }
    } catch {
      // Ignore malformed data.
    }
  }

  _persistHostShell() {
    if (!this.storage || !this.queryKey) {
      return;
    }
    this.storage.setItem(
      _hostStateKey(this.queryKey),
      JSON.stringify({
        width: this.controller.hostWidth,
        defaultMode: this.controller.mode,
      })
    );
  }

  setQueryKey(nextQueryKey) {
    this.controller.setQueryKey(nextQueryKey);
    this.queryKey = String(nextQueryKey || "");
    this._persistHostShell();
  }

  setMode(mode) {
    this.controller.setMode(mode);
    this._persistHostShell();
  }

  open() {
    this.controller.open();
    this._persistHostShell();
  }

  close() {
    this.controller.close();
    this._persistHostShell();
  }

  toggle() {
    this.controller.toggleOpen();
    this._persistHostShell();
  }

  resize(width) {
    this.controller.resize(width);
    this._persistHostShell();
  }

  getState() {
    return this.controller.getState();
  }
}
