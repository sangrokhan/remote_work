import { useEffect, useRef, useState } from 'react'
import { Settings } from 'lucide-react'

const PANEL_WIDTH = '25%'

const initialMessages = [
  {
    id: 1,
    role: 'assistant',
    text: '안녕하세요. 무엇을 도와드릴까요?'
  },
]

const dummyLogs = [
  { time: '18:21:01', level: 'info', text: '채팅 엔진 준비 완료' },
  { time: '18:21:05', level: 'info', text: '웹소켓 연결이 수립되었습니다' },
  { time: '18:21:07', level: 'warn', text: '임시 캐시 TTL 5분으로 갱신됨' },
  { time: '18:21:12', level: 'ok', text: '사용자 메시지 수신 및 라우팅 완료' },
  { time: '18:21:17', level: 'info', text: '응답 스트림이 시작되었습니다' }
  ] 
const MODEL_LIST = ['gpt-4.1', 'gpt-4o-mini', 'gpt-4o']

function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [messages, setMessages] = useState(initialMessages)
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState(MODEL_LIST[0])
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  const [maxTokens, setMaxTokens] = useState(1024)
  const [responseMode, setResponseMode] = useState('normal')
  const composerRef = useRef(null)
  const modelMenuRef = useRef(null)

  useEffect(() => {
    const textarea = composerRef.current
    if (!textarea) return

    textarea.style.height = 'auto'
    const lineHeight = parseFloat(getComputedStyle(textarea).lineHeight || '20')
    const minRows = 3
    const maxRows = 5
    const minHeight = lineHeight * minRows
    const maxHeight = lineHeight * maxRows
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight)

    textarea.style.height = `${nextHeight}px`
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [text])

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key !== 'Escape') return
      if (isSettingsOpen) setIsSettingsOpen(false)
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isSettingsOpen])

  useEffect(() => {
    const closeModelMenu = () => setIsModelMenuOpen(false)
    const onDocMouseDown = (event) => {
      if (!modelMenuRef.current?.contains(event.target)) {
        closeModelMenu()
      }
    }
    const onKeyDown = (event) => {
      if (event.key === 'Escape') closeModelMenu()
    }

    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onKeyDown)

    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  const sendMessage = () => {
    const trimmed = text.trim()
    if (!trimmed) return

    const userMessage = { id: Date.now(), role: 'user', text: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setText('')

    window.setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          text: '좋아요. 이 메시지는 패널 토글이 제대로 동작하는지 확인용 샘플 답변입니다.'
        }
      ])
    }, 240)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  return (
    <div
      className={`app-shell ${isPanelOpen ? 'panel-open' : ''}`}
      style={{ '--panel-width': PANEL_WIDTH }}
    >
      <main className="chat-shell">
        <header className="chat-header">
          <div className="chat-title-wrap">
            <h1>Chat Interface</h1>
          </div>
          <div className="header-actions">
            <button
              className="settings-btn"
              aria-label="설정 열기"
              onClick={() => setIsSettingsOpen(true)}
            >
              <Settings size={26} strokeWidth={2.4} aria-hidden="true" />
            </button>
          <button
            className="log-toggle-btn"
            aria-label={isPanelOpen ? '로그 패널 닫기' : '로그 패널 열기'}
            aria-expanded={isPanelOpen}
            aria-controls="log-panel"
            onClick={() => setIsPanelOpen((v) => !v)}
          >
            <span className="log-toggle-icon" aria-hidden>
              {isPanelOpen ? (
                <svg viewBox="0 0 16 16" aria-hidden>
                  <path
                    d="M6 3l5 5-5 5"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : (
                <svg viewBox="0 0 16 16" aria-hidden>
                  <path
                    d="M10 3l-5 5 5 5"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </span>
          </button>
          </div>
        </header>

        <section className="chat-body">
          <div className="messages" role="log" aria-live="polite">
            {messages.map((message) => (
              <article key={message.id} className={`bubble ${message.role}`}>
                <p>{message.text}</p>
              </article>
            ))}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={composerRef}
              className="composer-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="메시지를 입력하세요"
              rows={3}
            />
            <div className="composer-controls" ref={modelMenuRef}>
              <button
                type="button"
                className="model-menu-btn"
                aria-label="모델 선택"
                aria-haspopup="listbox"
                aria-expanded={isModelMenuOpen}
                onClick={() => setIsModelMenuOpen((v) => !v)}
              >
                <span>{selectedModel}</span>
                <span className="model-menu-caret" aria-hidden>
                  <svg viewBox="0 0 14 14">
                    <path d="M2 5l5 5 5-5" />
                  </svg>
                </span>
              </button>

              {isModelMenuOpen && (
                <div className="model-menu" role="listbox" aria-label="모델 목록">
                  {MODEL_LIST.map((model) => (
                    <button
                      key={model}
                      type="button"
                      className={`model-item ${model === selectedModel ? 'active' : ''}`}
                      role="option"
                      aria-selected={model === selectedModel}
                      onClick={() => {
                        setSelectedModel(model)
                        setIsModelMenuOpen(false)
                      }}
                    >
                      {model}
                    </button>
                  ))}
                </div>
              )}

              <button type="submit" className="composer-send-btn" aria-label="메시지 전송">
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
      </main>

      <aside
        className="log-panel"
        id="log-panel"
        aria-label="채팅 시스템 로그 패널"
        aria-hidden={!isPanelOpen}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="log-panel-head">
          <div>
            <p className="log-label">System</p>
            <h2>실시간 로그</h2>
          </div>
        </div>

        <div className="log-content" role="log" aria-live="polite">
          {dummyLogs.map((entry) => (
            <div key={`${entry.time}-${entry.text}`} className={`log-line ${entry.level}`}>
              <span className="log-time">{entry.time}</span>
              <span className={`log-badge ${entry.level}`}>{entry.level}</span>
              <p>{entry.text}</p>
            </div>
          ))}
        </div>
      </aside>

      {isPanelOpen && (
        <button
          className="panel-close log-panel-close-floating"
          aria-label="로그 패널 닫기"
          onClick={() => setIsPanelOpen(false)}
        >
          ×
        </button>
      )}

      {isSettingsOpen && (
        <>
          <button
            type="button"
            className="settings-backdrop"
            aria-label="설정 모달 닫기"
            onClick={() => setIsSettingsOpen(false)}
          />

          <section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
            <header className="settings-modal-head">
              <h2 id="settings-title">설정</h2>
              <button
                className="panel-close"
                aria-label="설정 닫기"
                onClick={() => setIsSettingsOpen(false)}
              >
                ×
              </button>
            </header>

            <div className="settings-form">
              <label htmlFor="response-mode">응답 모드</label>
              <select
                id="response-mode"
                value={responseMode}
                onChange={(e) => setResponseMode(e.target.value)}
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
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />

              <button type="button" className="settings-save" onClick={() => setIsSettingsOpen(false)}>
                적용
              </button>
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default App
