# Win Transition Key — Design

Date: 2026-07-02

## Purpose

Automate KVM / multi-PC switch hotkeys. The user runs 2 PCs + an Android DeX
phone through one keyboard/mouse forwarded by a Bluetooth dongle, and today
presses function keys / Pause to switch machines. This tool fires that switch
key automatically when the cursor pushes a screen edge, so no manual key press
is needed.

## Tech

- **AutoHotkey v2**, single script `win_transition_key.ahk`.
- Compiled to a standalone `.exe` via Ahk2Exe (one-time, on the user's Windows
  PC). The compiled exe needs no runtime installed.
- No third-party dependencies.

Chosen over Python+PyInstaller: smaller exe, purpose-built for cursor polling
and key injection, one-click compile. Neither route can be runtime-tested on the
Linux build host — the user tests on Windows.

## Features

- 4 directions: Up / Down / Left / Right.
- Per direction: enable checkbox, key/combo, dwell seconds.
- Combo capture (Ctrl/Alt/Shift/Win + base key) via one-shot `InputHook`.
- Apply/Start and Stop buttons.
- Persistence to `config.ini` beside the exe; auto-load + auto-start on launch.
- Minimize-to-tray; tray menu Show / Stop / Exit.

## Detection logic

- 50 ms `SetTimer` polls `MouseGetPos` (screen coords).
- Edges = **virtual-desktop outer bounds** via `SysGet(76..79)` — the only true
  walls. Inner monitor seams are pass-through and cannot trigger.
- Per direction, in-zone test:
  - Up `my <= vy`, Down `my >= vy+vh-1`, Left `mx <= vx`, Right `mx >= vx+vw-1`.
- Dwell: on first in-zone tick record `A_TickCount`; when elapsed ≥ dwellMs and
  armed, `Send` the combo once, then disarm.
- Re-arm only after the cursor leaves the zone — prevents repeat spam while the
  cursor sits parked at the edge (expected after a switch).

## Key model

- Capture writes a human display string ("Ctrl+Alt+F13") into the field and INI.
- `DisplayToSend` converts display → AHK Send syntax (`^!{F13}`) at Apply time;
  multi-char base keys are wrapped in braces.
- Field is also hand-editable / INI hand-editable.

## Files

- `win_transition_key.ahk` — source
- `README.md` — install AHK v2, Ahk2Exe build steps, usage
- `.gitignore` — ignores `config.ini` (per-machine) and `*.exe` (build artifact)
- `config.ini` — generated at runtime on Windows

## Out of scope / limits

- Windows only.
- Android DeX cannot be a target of these KVM tools; the dongle still handles the
  phone. This tool only automates the existing switch keypress.
- No runtime verification on the build host; final testing is on Windows.
