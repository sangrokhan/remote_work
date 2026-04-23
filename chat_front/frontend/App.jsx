import { useEffect, useRef, useState } from 'react'
import { Settings, SquareSplitHorizontal } from 'lucide-react'

import { PANEL_WIDTH, INITIAL_LEFT_MESSAGES, INITIAL_RIGHT_MESSAGES } from './constants'
import { useWorkflowSocket } from './hooks/useWorkflowSocket'
import { useWorkflowSSE } from './hooks/useWorkflowSSE'
import { useWorkflowGraph } from './hooks/useWorkflowGraph'
import { useModels } from './hooks/useModels'
import { ChatPane } from './components/ChatPane'
import { Composer } from './components/Composer'
import { WorkflowPanel } from './components/WorkflowPanel'
import { SettingsModal } from './components/SettingsModal'

function buildRunId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [isResultPanelOpen, setIsResultPanelOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)

  const [messages, setMessages] = useState(INITIAL_LEFT_MESSAGES)
  const [rightMessages, setRightMessages] = useState(INITIAL_RIGHT_MESSAGES)
  const [text, setText] = useState('')

  const { models } = useModels()

  const [selectedModel, setSelectedModel] = useState('')
  const [rightSelectedModel, setRightSelectedModel] = useState('')

  useEffect(() => {
    if (models.length > 0 && !selectedModel) setSelectedModel(models[0])
    if (models.length > 0 && !rightSelectedModel) setRightSelectedModel(models[0])
  }, [models])
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)

  const [maxTokens, setMaxTokens] = useState(1024)
  const [responseMode, setResponseMode] = useState('normal')

  const [leftAgenticRag, setLeftAgenticRag] = useState(true)
  const [rightAgenticRag, setRightAgenticRag] = useState(true)

  const [workflowGraph, setWorkflowGraph] = useState(null)
  const [isWorkflowLoading, setIsWorkflowLoading] = useState(false)
  const [workflowError, setWorkflowError] = useState('')
  const [workflowRenderError, setWorkflowRenderError] = useState('')
  const [workflowConnectionState, setWorkflowConnectionState] = useState('idle')

  const isPanelOpenRef = useRef(isPanelOpen)
  const cyRef = useRef(null)

  const {
    workflowExecutionRef,
    applyWorkflowNodeHighlight,
    clearWorkflowNodeHighlight,
    loadWorkflowGraph,
  } = useWorkflowSocket({
    isPanelOpenRef,
    cyRef,
    setWorkflowGraph,
    setWorkflowError,
    setIsWorkflowLoading,
    setWorkflowRenderError,
    setWorkflowConnectionState,
  })

  const { streamWorkflow } = useWorkflowSSE()

  const { graphContainerRef } = useWorkflowGraph({
    isPanelOpen,
    workflowGraph,
    workflowExecutionRef,
    cyRef,
    applyWorkflowNodeHighlight,
    clearWorkflowNodeHighlight,
    setWorkflowRenderError,
  })

  useEffect(() => {
    isPanelOpenRef.current = isPanelOpen
  }, [isPanelOpen])

  useEffect(() => {
    if (isResultPanelOpen && isPanelOpen) setIsPanelOpen(false)
  }, [isResultPanelOpen])

  useEffect(() => {
    if (!isPanelOpen) {
      clearWorkflowNodeHighlight()
      return
    }
    const activeNode = workflowExecutionRef.current.activeNode
    if (activeNode) {
      const id = window.requestAnimationFrame(() => applyWorkflowNodeHighlight(activeNode))
      return () => window.cancelAnimationFrame(id)
    }
  }, [isPanelOpen])

  useEffect(() => {
    if (!isPanelOpen) return
    const controller = new AbortController()
    loadWorkflowGraph({ signal: controller.signal })
    return () => controller.abort()
  }, [isPanelOpen])

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (isSettingsOpen) setIsSettingsOpen(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isSettingsOpen])

  const sendMessage = () => {
    const trimmed = text.trim()
    if (!trimmed) return

    const leftRunId = buildRunId()
    const now = Date.now()
    setText('')

    setMessages((prev) => [
      ...prev,
      { id: now, role: 'user', text: trimmed },
      { id: now + 1, role: 'assistant', text: '워크플로우 실행됨', runId: leftRunId },
    ])

    const makeReplace = (setter, runId) => (line) =>
      setter((prev) => prev.map((m) =>
        m.runId === runId ? { ...m, text: line } : m
      ))
    const makeUpdate = (setter, runId) => (updates) =>
      setter((prev) => prev.map((m) => (m.runId === runId ? { ...m, ...updates } : m)))

    streamWorkflow({
      params: { run_id: leftRunId, input: trimmed, model: selectedModel, response_mode: responseMode, max_tokens: maxTokens, agentic_rag: leftAgenticRag },
      replaceLine: makeReplace(setMessages, leftRunId),
      updateMeta: makeUpdate(setMessages, leftRunId),
      onNodeEvent: (eventType, data) => {
        const nodeId = data.node || data.name
        if (nodeId && (eventType === 'node_started' || data.stage === 'start')) {
          workflowExecutionRef.current = { runId: leftRunId, isRunning: true, activeNode: nodeId }
          applyWorkflowNodeHighlight(nodeId)
        }
        if (eventType === 'workflow_complete' || eventType === 'workflow_error') {
          workflowExecutionRef.current = { ...workflowExecutionRef.current, isRunning: false }
          clearWorkflowNodeHighlight()
        }
      },
    })

    if (isResultPanelOpen) {
      const rightRunId = buildRunId()
      setRightMessages((prev) => [
        ...prev,
        { id: now + 2, role: 'user', text: trimmed },
        { id: now + 3, role: 'assistant', text: '워크플로우 실행됨', runId: rightRunId },
      ])
      streamWorkflow({
        params: { run_id: rightRunId, input: trimmed, model: rightSelectedModel, response_mode: responseMode, max_tokens: maxTokens, agentic_rag: rightAgenticRag },
        replaceLine: makeReplace(setRightMessages, rightRunId),
        updateMeta: makeUpdate(setRightMessages, rightRunId),
        onNodeEvent: () => {},
      })
    }
  }

  return (
    <div
      className={`app-shell${isPanelOpen ? ' panel-open' : ''}${isResultPanelOpen ? ' split-mode' : ''}`}
      style={{ '--panel-width': PANEL_WIDTH }}
    >
      <main className="chat-shell">
        <header className="chat-header">
          <div className="chat-title-wrap">
            <img src="/logo.png" alt="logo" className="chat-logo" />
            <h1>Chat Interface</h1>
          </div>
          <div className="header-actions">
            <button
              className={`result-panel-btn${isResultPanelOpen ? ' active' : ''}`}
              aria-label={isResultPanelOpen ? '분할 모드 끄기' : '분할 모드 켜기'}
              aria-pressed={isResultPanelOpen}
              onClick={() => setIsResultPanelOpen((v) => !v)}
            >
              <SquareSplitHorizontal size={26} strokeWidth={2.2} aria-hidden="true" />
            </button>
            <button className="settings-btn" aria-label="설정 열기" onClick={() => setIsSettingsOpen(true)}>
              <Settings size={26} strokeWidth={2.4} aria-hidden="true" />
            </button>
            {!isResultPanelOpen && (
              <button
                className="log-toggle-btn"
                aria-label={isPanelOpen ? '워크플로우 패널 닫기' : '워크플로우 패널 열기'}
                aria-expanded={isPanelOpen}
                aria-controls="log-panel"
                onClick={() => setIsPanelOpen((v) => !v)}
              >
                <span className="log-toggle-icon" aria-hidden>
                  {isPanelOpen ? (
                    <svg viewBox="0 0 16 16" aria-hidden>
                      <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 16 16" aria-hidden>
                      <path d="M10 3l-5 5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                </span>
              </button>
            )}
          </div>
        </header>

        <div className="chat-split">
          <ChatPane
            messages={messages}
            showHeader={isResultPanelOpen}
            title="System A"
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            agenticRag={leftAgenticRag}
            onAgenticRagToggle={() => setLeftAgenticRag((v) => !v)}
            isRight={false}
            models={models}
          />
          {isResultPanelOpen && (
            <ChatPane
              messages={rightMessages}
              showHeader
              title="System B"
              selectedModel={rightSelectedModel}
              onModelChange={setRightSelectedModel}
              agenticRag={rightAgenticRag}
              onAgenticRagToggle={() => setRightAgenticRag((v) => !v)}
              isRight
              models={models}
            />
          )}
        </div>

        <Composer
          text={text}
          onTextChange={setText}
          onSubmit={sendMessage}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          isModelMenuOpen={isModelMenuOpen}
          onModelMenuToggle={setIsModelMenuOpen}
          isSplitMode={isResultPanelOpen}
          models={models}
        />
      </main>

      {!isResultPanelOpen && (
        <aside
          className="log-panel"
          id="log-panel"
          aria-label="워크플로우 패널"
          aria-hidden={!isPanelOpen}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="log-panel-head">
            <div>
              <p className="log-label">System</p>
              <h2>워크플로우 구성</h2>
            </div>
          </div>
          <WorkflowPanel
            isWorkflowLoading={isWorkflowLoading}
            workflowError={workflowError}
            workflowRenderError={workflowRenderError}
            workflowGraph={workflowGraph}
            graphContainerRef={graphContainerRef}
          />
        </aside>
      )}

      {isPanelOpen && !isResultPanelOpen && (
        <button
          className="panel-close log-panel-close-floating"
          aria-label="워크플로우 패널 닫기"
          onClick={() => setIsPanelOpen(false)}
        >
          ×
        </button>
      )}

      {isSettingsOpen && (
        <SettingsModal
          responseMode={responseMode}
          onResponseModeChange={setResponseMode}
          maxTokens={maxTokens}
          onMaxTokensChange={setMaxTokens}
          onClose={() => setIsSettingsOpen(false)}
        />
      )}
    </div>
  )
}

export default App
