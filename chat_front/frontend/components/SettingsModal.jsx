export function SettingsModal({ responseMode, onResponseModeChange, maxTokens, onMaxTokensChange, onClose }) {
  return (
    <>
      <button type="button" className="settings-backdrop" aria-label="설정 모달 닫기" onClick={onClose} />
      <section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header className="settings-modal-head">
          <h2 id="settings-title">설정</h2>
          <button className="panel-close" aria-label="설정 닫기" onClick={onClose}>×</button>
        </header>
        <div className="settings-form">
          <label htmlFor="response-mode">응답 모드</label>
          <select
            id="response-mode"
            value={responseMode}
            onChange={(e) => onResponseModeChange(e.target.value)}
          >
            <option value="fast">빠른 응답</option>
            <option value="normal">표준 응답</option>
            <option value="precise">정확도 우선</option>
          </select>

          <label htmlFor="max-tokens">최대 토큰</label>
          <input
            id="max-tokens"
            type="number"
            value={maxTokens}
            min={64}
            max={8192}
            step={64}
            onChange={(e) => onMaxTokensChange(Number(e.target.value))}
          />

          <button type="button" className="settings-save" onClick={onClose}>적용</button>
        </div>
      </section>
    </>
  )
}
