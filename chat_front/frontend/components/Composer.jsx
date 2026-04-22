import { useEffect, useRef } from 'react'

export function Composer({
  text,
  onTextChange,
  onSubmit,
  selectedModel,
  onModelChange,
  isModelMenuOpen,
  onModelMenuToggle,
  isSplitMode,
  models = [],
}) {
  const composerRef = useRef(null)
  const modelMenuRef = useRef(null)

  useEffect(() => {
    const textarea = composerRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    const lineHeight = parseFloat(getComputedStyle(textarea).lineHeight || '20')
    const minHeight = lineHeight * 3
    const maxHeight = lineHeight * 5
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight)
    textarea.style.height = `${nextHeight}px`
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [text])

  useEffect(() => {
    const onDocMouseDown = (e) => {
      if (!modelMenuRef.current?.contains(e.target)) onModelMenuToggle(false)
    }
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onModelMenuToggle(false)
    }
    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [onModelMenuToggle])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <section className="chat-composer-area">
      <form className="composer" onSubmit={(e) => { e.preventDefault(); onSubmit() }}>
        <textarea
          ref={composerRef}
          className="composer-textarea"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="메시지를 입력하세요"
          rows={3}
        />
        <div className="composer-controls" ref={modelMenuRef}>
          {!isSplitMode && (
            <button
              type="button"
              className="model-menu-btn"
              aria-label="모델 선택"
              aria-haspopup="listbox"
              aria-expanded={isModelMenuOpen}
              onClick={() => onModelMenuToggle(!isModelMenuOpen)}
            >
              <span>{selectedModel}</span>
              <span className="model-menu-caret" aria-hidden>
                <svg viewBox="0 0 14 14"><path d="M2 5l5 5 5-5" /></svg>
              </span>
            </button>
          )}

          {isModelMenuOpen && !isSplitMode && (
            <div className="model-menu" role="listbox" aria-label="모델 목록">
              {models.map((model) => (
                <button
                  key={model}
                  type="button"
                  className={`model-item ${model === selectedModel ? 'active' : ''}`}
                  role="option"
                  aria-selected={model === selectedModel}
                  onClick={() => { onModelChange(model); onModelMenuToggle(false) }}
                >
                  {model}
                </button>
              ))}
            </div>
          )}

          <button type="submit" className="composer-send-btn" aria-label="메시지 전송" style={{ marginLeft: 'auto' }}>
            <svg viewBox="0 0 18 18" aria-hidden>
              <path
                d="M9 16V3M9 3L5 7M9 3L13 7"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </form>
    </section>
  )
}
