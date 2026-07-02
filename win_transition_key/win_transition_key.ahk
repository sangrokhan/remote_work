#Requires AutoHotkey v2.0
#SingleInstance Force
Persistent
CoordMode "Mouse", "Screen"

; ============================================================================
;  Win Transition Key
;  Sends a configurable key/combo when the mouse cursor pushes a screen edge
;  for a per-direction dwell time. Built for KVM/PC-switch hotkey automation.
; ============================================================================

configFile := A_ScriptDir "\config.ini"
dirs := ["Up", "Down", "Left", "Right"]

; runtime state
running := false
states := Map()          ; d -> {enabled, send, dwellMs, enterTick, armed}
ctrls  := Map()          ; d -> {en, key, dwell}

; ------------------------------------------------------------------ GUI build
g := Gui("+Resize", "Win Transition Key")
g.SetFont("s9", "Segoe UI")
g.Add("Text", "xm w400", "Push the cursor against a screen edge for N seconds -> sends that edge's key.")

; column headers
g.Add("Text", "xm y+12 w55", "Dir")
g.Add("Text", "x+8 yp w48", "Enable")
g.Add("Text", "x+8 yp w150", "Key / Combo")
g.Add("Text", "x+52 yp w55", "Dwell(s)")

for i, d in dirs {
    g.Add("Text", "xm y+8 w55", d)
    en  := g.Add("Checkbox", "x+8 yp+2 w48 vEn" d)
    key := g.Add("Edit", "x+8 yp-2 w150 vKey" d, "")  ; hand-editable
    setb := g.Add("Button", "x+4 yp w40", "Set")       ; or capture via Set
    setb.OnEvent("Click", CaptureKey.Bind(d))
    dw  := g.Add("Edit", "x+8 yp w55 vDwell" d, "1.0")
    testb := g.Add("Button", "x+8 yp w45", "Test")     ; send this key now
    testb.OnEvent("Click", TestSend.Bind(d))
    ctrls[d] := {en: en, key: key, dwell: dw}
}

statusTxt := g.Add("Text", "xm y+16 w260 cRed", "Status: STOPPED")
applyBtn  := g.Add("Button", "xm y+10 w130 h32", "Apply / Start")
stopBtn   := g.Add("Button", "x+14 yp w130 h32", "Stop")
applyBtn.OnEvent("Click", (*) => ApplyStart())
stopBtn.OnEvent("Click", (*) => StopMon())
g.OnEvent("Close", (*) => g.Hide())                   ; X hides to tray

; live diagnostic readout (updated by the poll timer while RUNNING)
g.Add("Text", "xm y+14 w400", "Debug (push an edge and watch):")
dbgTxt := g.Add("Edit", "xm y+4 w400 r7 ReadOnly -Wrap")

; ------------------------------------------------------------------ tray menu
tray := A_TrayMenu
tray.Delete()
tray.Add("Show", (*) => g.Show())
tray.Add("Stop Monitoring", (*) => StopMon())
tray.Add("Exit", (*) => ExitApp())
tray.Default := "Show"

; ------------------------------------------------------------------ startup
LoadConfig()
g.Show()
autoStart := false
for d in dirs
    if (ctrls[d].en.Value)
        autoStart := true
if autoStart
    ApplyStart()
return

; ============================================================================
;  Functions
; ============================================================================

; Capture a key combo into the direction's field via a one-shot InputHook.
CaptureKey(d, *) {
    global ctrls
    ToolTip "Press key combo for '" d "'  (Esc = cancel)"
    ih := InputHook()
    ih.VisibleText := false
    ih.KeyOpt("{All}", "E")
    ih.KeyOpt("{LCtrl}{RCtrl}{LAlt}{RAlt}{LShift}{RShift}{LWin}{RWin}", "-E")
    ih.Start()
    ih.Wait()
    ToolTip
    ek := ih.EndKey
    if (ek = "" || ek = "Escape")
        return
    disp := ModsToDisplay(ih.EndMods) . ek
    ctrls[d].key.Value := disp
}

; Modifier symbol string (e.g. "<^>!") -> "Ctrl+Alt+" display prefix.
ModsToDisplay(mods) {
    s := ""
    if InStr(mods, "^")
        s .= "Ctrl+"
    if InStr(mods, "!")
        s .= "Alt+"
    if InStr(mods, "+")
        s .= "Shift+"
    if InStr(mods, "#")
        s .= "Win+"
    return s
}

; "Ctrl+Alt+F13" -> AHK Send string "^!{F13}".
DisplayToSend(disp) {
    disp := Trim(disp)
    if (disp = "")
        return ""
    parts := StrSplit(disp, "+")
    base := Trim(parts[parts.Length])
    prefix := ""
    Loop parts.Length - 1 {
        switch StrLower(Trim(parts[A_Index])) {
            case "ctrl", "control": prefix .= "^"
            case "alt":             prefix .= "!"
            case "shift":           prefix .= "+"
            case "win", "windows", "lwin", "rwin": prefix .= "#"
        }
    }
    if (base = "")                       ; combo ended with literal '+'
        base := "+"
    if (StrLen(base) > 1)
        base := "{" base "}"
    return prefix . base
}

; Read GUI, validate, save config, start the poll timer.
ApplyStart() {
    global ctrls, states, running, dirs, statusTxt
    newStates := Map()
    anyEnabled := false
    for d in dirs {
        en := ctrls[d].en.Value ? true : false
        keyDisp := Trim(ctrls[d].key.Value)
        dwellRaw := Trim(ctrls[d].dwell.Value)
        dwell := IsNumber(dwellRaw) ? dwellRaw + 0.0 : 0.0
        if (en && (keyDisp = "" || dwell <= 0)) {
            MsgBox "Direction '" d "': set a key and a dwell time > 0.", "Invalid input", "Iconx"
            return
        }
        newStates[d] := {enabled: en, send: DisplayToSend(keyDisp), dwellMs: Round(dwell * 1000), enterTick: 0, armed: true}
        if en
            anyEnabled := true
    }
    states := newStates
    SaveConfig()
    if (!anyEnabled)
        MsgBox "No direction enabled - nothing will trigger.", "Note", "Iconi"
    running := true
    statusTxt.Value := "Status: RUNNING"
    statusTxt.SetFont("cGreen")
    SetTimer CheckEdges, 50
}

StopMon() {
    global running, statusTxt
    SetTimer CheckEdges, 0
    running := false
    statusTxt.Value := "Status: STOPPED"
    statusTxt.SetFont("cRed")
}

; Poll cursor against the virtual-desktop outer edges (true physical walls).
CheckEdges() {
    global states, dirs, dbgTxt
    vx := SysGet(76)        ; SM_XVIRTUALSCREEN
    vy := SysGet(77)        ; SM_YVIRTUALSCREEN
    vw := SysGet(78)        ; SM_CXVIRTUALSCREEN
    vh := SysGet(79)        ; SM_CYVIRTUALSCREEN
    MouseGetPos(&mx, &my)
    info := "Virtual desktop: left=" vx " top=" vy " right=" (vx + vw - 1) " bottom=" (vy + vh - 1) "`r`n"
    info .= "Cursor: x=" mx "  y=" my "`r`n"
    for d in dirs {
        if (!states.Has(d)) {
            info .= d ": (not applied yet)`r`n"
            continue
        }
        st := states[d]
        if (!st.enabled) {
            info .= d ": disabled`r`n"
            continue
        }
        if (st.send = "") {
            info .= d ": no key set`r`n"
            continue
        }
        inZone := false
        switch d {
            case "Up":    inZone := (my <= vy)
            case "Down":  inZone := (my >= vy + vh - 1)
            case "Left":  inZone := (mx <= vx)
            case "Right": inZone := (mx >= vx + vw - 1)
        }
        remain := 0
        if (inZone) {
            if (st.armed) {
                if (st.enterTick = 0)
                    st.enterTick := A_TickCount
                remain := st.dwellMs - (A_TickCount - st.enterTick)
                if (remain <= 0) {
                    Send st.send
                    st.armed := false
                    st.enterTick := 0
                    SoundBeep 800, 120
                    ToolTip "FIRED  " d "  ->  " st.send
                    SetTimer ClearTip, -1500
                }
            }
        } else {
            st.enterTick := 0
            st.armed := true
        }
        info .= d ": " (inZone ? "IN-ZONE" : "outside")
             . "  armed=" (st.armed ? "Y" : "N")
             . (inZone && st.armed ? "  fires in " (remain > 0 ? remain : 0) "ms" : "")
             . "  send=" st.send "`r`n"
    }
    dbgTxt.Value := info
}

; Send a direction's key immediately (isolates the send path / dongle from
; edge detection). Also proves whether the target reacts to synthetic keys.
TestSend(d, *) {
    global ctrls
    s := DisplayToSend(Trim(ctrls[d].key.Value))
    if (s = "") {
        MsgBox "Set a key for '" d "' first.", "Nothing to test", "Iconx"
        return
    }
    SoundBeep 600, 120
    ToolTip "TEST send  " d "  ->  " s
    SetTimer ClearTip, -1500
    Send s
}

ClearTip() {
    ToolTip
}

SaveConfig() {
    global ctrls, dirs, configFile
    for d in dirs {
        IniWrite ctrls[d].en.Value, configFile, d, "Enabled"
        IniWrite ctrls[d].key.Value, configFile, d, "Key"
        IniWrite ctrls[d].dwell.Value, configFile, d, "Dwell"
    }
}

LoadConfig() {
    global ctrls, dirs, configFile
    if (!FileExist(configFile))
        return
    for d in dirs {
        ctrls[d].en.Value    := IniRead(configFile, d, "Enabled", "0") + 0
        ctrls[d].key.Value   := IniRead(configFile, d, "Key", "")
        ctrls[d].dwell.Value := IniRead(configFile, d, "Dwell", "1.0")
    }
}
