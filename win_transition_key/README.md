# Win Transition Key

Sends a configurable key/combo when the mouse cursor **pushes a screen edge**
for a set number of seconds. Built to automate KVM / multi-PC switch hotkeys:
shove the cursor into the far edge, and the switch key fires by itself.

- 4 directions (Up / Down / Left / Right), each independently configurable
- Per-direction **key or combo** (e.g. `Ctrl+Alt+F13`)
- Per-direction **enable/disable** and **dwell time** (seconds)
- **Apply/Start** and **Stop** buttons
- Settings persist to `config.ini` next to the exe, auto-loaded on launch
- Minimizes to the system tray, keeps running in the background

## How it works

A 50 ms timer polls the cursor position. When it sits against an enabled
**outer** edge of the virtual desktop for that direction's dwell time, the key
is sent **once**. The cursor must leave the edge before it can fire again (no
repeat spam).

> **Multi-monitor note:** only the *outer* boundary of your whole desktop is a
> wall. The seams *between* monitors let the cursor pass through, so they can't
> trigger. This matches KVM use — push the far outer edge to switch machines.

## Requirements

| To do this | You need |
|------------|----------|
| Run `win_transition_key.ahk` directly | AutoHotkey v2 installed |
| Build `win_transition_key.exe` | AutoHotkey v2 (ships with Ahk2Exe) |
| **Run the built `.exe`** | **Nothing — standalone, portable** |

Download AutoHotkey v2: https://www.autohotkey.com/

## Build the .exe (one time)

1. Install **AutoHotkey v2**. It installs the compiler **Ahk2Exe** too.
2. Start menu -> run **"Ahk2Exe"** (Compile AHK to EXE).
3. **Source (in):** select `win_transition_key.ahk`.
4. **Destination (out):** leave default -> `win_transition_key.exe`.
5. **Base file:** pick the **v2** 64-bit base.
6. Click **Convert**. Done — `win_transition_key.exe` appears next to the script.

The `.exe` is self-contained: copy it anywhere, double-click, no install needed.

## Usage

1. Launch the exe (or the `.ahk`). The window opens.
2. For each direction you want:
   - tick **Enable**
   - click the **Set** button, then press the keys — e.g. hold `Ctrl+Alt` and
     tap `F13`. `Esc` cancels capture. (Or type the combo into the field
     directly, e.g. `Ctrl+Alt+F13`.)
   - set **Dwell(s)** — how long the cursor must push the edge (e.g. `1.0`).
3. Click **Apply / Start**. Status turns green **RUNNING**.
4. Click **Stop** (or tray -> Stop Monitoring) to pause.
5. Closing the window **hides to tray**; it keeps running. Tray -> **Exit** to quit.

Settings are saved on Apply and reloaded next launch; if any direction was
enabled, it **auto-starts** on launch.

## Config file

`config.ini` (created next to the exe on first Apply):

```ini
[Up]
Enabled=1
Key=Ctrl+Alt+F13
Dwell=1.5
```

You can hand-edit it; changes load on next launch.

## Notes / limits

- Windows only (AutoHotkey).
- The key is sent to whatever window has focus — which is exactly what a
  global KVM-switch hotkey wants.
- Function keys `F13`–`F24` are ideal switch keys: real keys, nothing else uses
  them.
- If a game's anti-cheat blocks synthetic input, that's unrelated to this use.
