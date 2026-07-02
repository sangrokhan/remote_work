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
    ctrls[d] := {en: en, key: key, dwell: dw}
}

statusTxt := g.Add("Text", "xm y+16 w260 cRed", "Status: STOPPED")
applyBtn  := g.Add("Button", "xm y+10 w130 h32", "Apply / Start")
stopBtn   := g.Add("Button", "x+14 yp w130 h32", "Stop")
applyBtn.OnEvent("Click", (*) => ApplyStart())
stopBtn.OnEvent("Click", (*) => StopMon())
g.OnEvent("Close", (*) => g.Hide())                   ; X hides to tray

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
    global states, dirs
    vx := SysGet(76)        ; SM_XVIRTUALSCREEN
    vy := SysGet(77)        ; SM_YVIRTUALSCREEN
    vw := SysGet(78)        ; SM_CXVIRTUALSCREEN
    vh := SysGet(79)        ; SM_CYVIRTUALSCREEN
    MouseGetPos(&mx, &my)
    for d in dirs {
        if (!states.Has(d))
            continue
        st := states[d]
        if (!st.enabled || st.send = "")
            continue
        inZone := false
        switch d {
            case "Up":    inZone := (my <= vy)
            case "Down":  inZone := (my >= vy + vh - 1)
            case "Left":  inZone := (mx <= vx)
            case "Right": inZone := (mx >= vx + vw - 1)
        }
        if (inZone) {
            if (st.armed) {
                if (st.enterTick = 0)
                    st.enterTick := A_TickCount
                else if (A_TickCount - st.enterTick >= st.dwellMs) {
                    Send st.send
                    st.armed := false
                    st.enterTick := 0
                }
            }
        } else {
            st.enterTick := 0
            st.armed := true
        }
    }
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
