import { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'
import { Settings } from 'lucide-react'

const PANEL_WIDTH = '25%'
const GRAPH_API_URL =
  import.meta.env.VITE_WORKFLOW_GRAPH_URL ?? 'http://localhost:10001/graph'
const GRAPH_WS_URL = (() => {
  const wsEnv = import.meta.env.VITE_WORKFLOW_WS_URL
  if (wsEnv) return wsEnv
  try {
    const apiUrl = new URL(GRAPH_API_URL)
    apiUrl.protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:'
    apiUrl.pathname = '/ws/connect'
    apiUrl.search = ''
    apiUrl.hash = ''
    return apiUrl.toString()
  } catch {
    return 'ws://localhost:10001/ws/connect'
  }
})()

const initialMessages = [
  {
    id: 1,
    role: 'assistant',
    text: '안녕하세요. 무엇을 도와드릴까요?'
  },
]

const MODEL_LIST = ['gpt-4.1', 'gpt-4o-mini', 'gpt-4o']

const WORKFLOW_NODE_PALETTE = {
  planner: {
    kind: 'planner',
    bg: '#d7ecff',
    border: '#68a8ee',
    text: '#12365f',
  },
  executor: {
    kind: 'executor',
    bg: '#d7f4dd',
    border: '#7dcf90',
    text: '#1a4f2f',
  },
  refiner: {
    kind: 'refiner',
    bg: '#fff1c7',
    border: '#e2be5e',
    text: '#5e4b17',
  },
  synthesizer: {
    kind: 'synthesizer',
    bg: '#f5d9fc',
    border: '#c78ce0',
    text: '#5d2e69',
  },
  start: {
    kind: 'start',
    bg: '#e7e7f8',
    border: '#9ca2df',
    text: '#32366c',
  },
  end: {
    kind: 'end',
    bg: '#dceeff',
    border: '#5f8bb0',
    text: '#263246',
  },
  default: {
    kind: 'default',
    bg: '#edf2ff',
    border: '#a9b6d7',
    text: '#2f3c56',
  },
}

const WORKFLOW_NODE_ALIASES = {
  '__start__': 'start',
  '__end__': 'end',
}

const normalizeNodeId = (value) => {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (typeof value === 'object') {
    if (typeof value.id === 'string' || typeof value.id === 'number') {
      return String(value.id)
    }
    if (typeof value.value === 'string' || typeof value.value === 'number') {
      return String(value.value)
    }
    if (typeof value.name === 'string' || typeof value.name === 'number') {
      return String(value.name)
    }
  }
  return String(value)
}

const normalizeNodeLabel = (nodeId) => {
  const normalized = normalizeNodeId(nodeId)
  if (!normalized) return ''
  const cleaned = normalized.replace(/^__+|__+$/g, '')
  if (!cleaned) return ''
  if (cleaned.toUpperCase() === 'START') return 'START'
  if (cleaned.toUpperCase() === 'END') return 'END'
  return cleaned
}

const normalizeNodeClass = (nodeId) => {
  const normalized = normalizeNodeId(nodeId).toUpperCase()
  if (normalized === '__START__' || normalized === 'START') return 'start'
  if (normalized === '__END__' || normalized === 'END') return 'end'
  return 'default'
}

const resolveNodeVisual = (nodeId) => {
  const rawId = normalizeNodeId(nodeId).trim()
  const alias = WORKFLOW_NODE_ALIASES[rawId] || WORKFLOW_NODE_ALIASES[rawId.toUpperCase()]
  const candidate = (alias || rawId).toLowerCase()
  return WORKFLOW_NODE_PALETTE[candidate] || WORKFLOW_NODE_PALETTE.default
}

cytoscape.use(dagre)

function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [messages, setMessages] = useState(initialMessages)
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState(MODEL_LIST[0])
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  const [maxTokens, setMaxTokens] = useState(1024)
  const [responseMode, setResponseMode] = useState('normal')
  const [workflowGraph, setWorkflowGraph] = useState(null)
  const [isWorkflowLoading, setIsWorkflowLoading] = useState(false)
  const [workflowError, setWorkflowError] = useState('')
  const [workflowRenderError, setWorkflowRenderError] = useState('')
  const [workflowConnectionState, setWorkflowConnectionState] = useState('idle')
  const composerRef = useRef(null)
  const modelMenuRef = useRef(null)
  const graphContainerRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const cyRef = useRef(null)
  const workflowSocketRef = useRef(null)
  const messagesAutoScrollRef = useRef(false)
  const hasMessageOverflowRef = useRef(false)
  const messagesScrollbarTimerRef = useRef(null)
  const workflowExecutionRef = useRef({
    runId: null,
    isRunning: false,
    activeNode: '',
  })
  const isPanelOpenRef = useRef(isPanelOpen)
  const [isMessagesScrollable, setIsMessagesScrollable] = useState(false)
  const [hasMessageOverflow, setHasMessageOverflow] = useState(false)

  const setWorkflowFromPayload = (payload) => {
    setWorkflowGraph(payload)
    setWorkflowError('')
    setIsWorkflowLoading(false)
  }

  const loadWorkflowGraph = async (opts = {}) => {
    const { signal } = opts
    if (!signal?.aborted) {
      setIsWorkflowLoading(true)
      setWorkflowError('')
      setWorkflowRenderError('')
    }

    try {
      const response = await fetch(GRAPH_API_URL, { signal })
      if (!response.ok) {
        throw new Error(`워크플로우 API 응답 오류 (${response.status})`)
      }
      const payload = await response.json()
      if (!signal?.aborted) {
        setWorkflowFromPayload(payload)
      }
    } catch (error) {
      if (error.name === 'AbortError') return
      if (!signal?.aborted) {
        setWorkflowError(error.message || '워크플로우 구성 조회 실패')
      }
    } finally {
      if (!signal?.aborted) {
        setIsWorkflowLoading(false)
      }
    }
  }

  const buildRunId = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID()
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`
  }

  const appendMessageLineByRunId = (runId, nextText) => {
    if (!runId || !nextText) return
    const safeText = String(nextText)
    setMessages((prev) =>
      prev.map((message) => {
        if (message.role !== 'assistant' || message.runId !== runId) return message
        return {
          ...message,
          text: message.text ? `${message.text}\n${safeText}` : safeText,
        }
      })
    )
  }

  const updateRunMessageMetaByRunId = (runId, updates) => {
    if (!runId) return
    setMessages((prev) =>
      prev.map((message) => {
        if (message.role !== 'assistant' || message.runId !== runId) return message
        return { ...message, ...updates }
      })
    )
  }

  const normalizeWorkflowNodeId = (nodeId) => {
    const normalized = normalizeNodeId(nodeId).trim()
    if (!normalized) return ''
    const upper = normalized.toUpperCase()
    if (upper === 'START') return '__start__'
    if (upper === 'END') return '__end__'
    if (WORKFLOW_NODE_ALIASES[normalized]) {
      return `__${WORKFLOW_NODE_ALIASES[normalized]}__`
    }
    if (WORKFLOW_NODE_ALIASES[upper]) {
      return `__${WORKFLOW_NODE_ALIASES[upper]}__`
    }
    return normalized
  }

  const clearWorkflowNodeHighlight = () => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().removeClass('wf-active')
  }

  const applyWorkflowNodeHighlight = (nodeId) => {
    if (!isPanelOpenRef.current) return
    const cy = cyRef.current
    const targetNodeId = normalizeWorkflowNodeId(nodeId)
    if (!cy || !targetNodeId) return

    const target = cy.getElementById(targetNodeId)
    if (!target || target.length === 0) return

    workflowExecutionRef.current.activeNode = targetNodeId
    cy.nodes().removeClass('wf-active')
    target.addClass('wf-active')
  }

  const isActiveRun = (runId) => {
    const state = workflowExecutionRef.current
    if (!runId) return false
    return !state.runId || !state.isRunning || state.runId === runId
  }

  const startWorkflowRun = (runId) => {
    workflowExecutionRef.current = {
      runId,
      isRunning: true,
      activeNode: '',
    }
    clearWorkflowNodeHighlight()
  }

  const finishWorkflowRun = (runId) => {
    const state = workflowExecutionRef.current
    if (state.runId && state.runId !== runId) return
    clearWorkflowNodeHighlight()
    workflowExecutionRef.current = {
      ...state,
      runId,
      isRunning: false,
      activeNode: '',
    }
  }

  const updateActiveWorkflowNode = (runId, nodeId, shouldHighlight = true) => {
    if (!runId || !nodeId) return
    if (!isActiveRun(runId)) return
    workflowExecutionRef.current = {
      ...workflowExecutionRef.current,
      runId,
      isRunning: true,
      activeNode: normalizeWorkflowNodeId(nodeId),
    }
    if (shouldHighlight) {
      applyWorkflowNodeHighlight(nodeId)
    }
  }

  useEffect(() => {
    isPanelOpenRef.current = isPanelOpen
  }, [isPanelOpen])

  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const recalc = () => {
      const overflow = container.scrollHeight > container.clientHeight + 1
      hasMessageOverflowRef.current = overflow
      setHasMessageOverflow(overflow)
      if (!overflow) {
        setIsMessagesScrollable(false)
      }
    }
    recalc()
    const onResize = () => recalc()
    const onScroll = () => {
      if (messagesAutoScrollRef.current || !hasMessageOverflowRef.current) return
      setIsMessagesScrollable(true)
      if (messagesScrollbarTimerRef.current) {
        window.clearTimeout(messagesScrollbarTimerRef.current)
      }
      messagesScrollbarTimerRef.current = window.setTimeout(() => {
        setIsMessagesScrollable(false)
      }, 1200)
    }
    container.addEventListener('scroll', onScroll)
    window.addEventListener('resize', onResize)
    return () => {
      container.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onResize)
      if (messagesScrollbarTimerRef.current) {
        window.clearTimeout(messagesScrollbarTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const id = requestAnimationFrame(() => {
      messagesAutoScrollRef.current = true
      container.scrollTop = container.scrollHeight
      setHasMessageOverflow(container.scrollHeight > container.clientHeight + 1)
      requestAnimationFrame(() => {
        messagesAutoScrollRef.current = false
      })
    })
    return () => cancelAnimationFrame(id)
  }, [messages])

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
    let reconnectOnClose = false
    let aborted = false
    const controller = new AbortController()
    let pingTimerId = null
    setWorkflowConnectionState('connecting')

    const socket = new WebSocket(GRAPH_WS_URL)
    workflowSocketRef.current = socket

    socket.addEventListener('open', () => {
      if (aborted) return
      reconnectOnClose = true
      setWorkflowConnectionState('open')
      if (isPanelOpenRef.current) {
        socket.send('get_graph')
      }
      pingTimerId = window.setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send('ping')
        }
      }, 20000)
    })

    socket.addEventListener('message', async (event) => {
      if (aborted) return
      try {
        const parsed = JSON.parse(event.data)
        if (parsed && parsed.type === 'connected') {
          setWorkflowConnectionState('ready')
          setWorkflowError('')
          return
        }
        if (parsed && parsed.type === 'graph' && parsed.payload) {
          setWorkflowFromPayload(parsed.payload)
          return
        }
        if (parsed && parsed.type === 'workflow_started') {
          startWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 실행됨')
          return
        }
        if (parsed && parsed.type === 'workflow_event' && parsed.run_id) {
          const nodeId = parsed.node || parsed.name
          const eventType = String(parsed.event || '').trim()
          const eventStage = String(parsed.stage || '').trim()

          if (
            isActiveRun(parsed.run_id) &&
            nodeId &&
            (eventType === 'node_started' || eventStage === 'start')
          ) {
            updateActiveWorkflowNode(parsed.run_id, nodeId)
          }

          if (parsed.message) {
            appendMessageLineByRunId(parsed.run_id, parsed.message)
          }
          if (parsed.payload?.final_output) {
            appendMessageLineByRunId(parsed.run_id, parsed.payload.final_output)
          }
          return
        }
        if (parsed && parsed.type === 'workflow_complete' && parsed.run_id) {
          finishWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 완료')
          updateRunMessageMetaByRunId(parsed.run_id, {
            status: 'done',
          })
          return
        }
        if (parsed && parsed.type === 'workflow_error' && parsed.run_id) {
          finishWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 오류')
          updateRunMessageMetaByRunId(parsed.run_id, {
            status: 'error',
          })
          return
        }
      } catch (_error) {
        if (typeof event.data === 'string' && event.data.startsWith('echo:')) return
      }
    })

    socket.addEventListener('error', () => {
      if (aborted) return
      setWorkflowError('워크플로우 WebSocket 연결 오류')
      setWorkflowConnectionState('error')
      loadWorkflowGraph({ signal: controller.signal })
    })

    socket.addEventListener('close', (event) => {
      if (pingTimerId) {
        window.clearInterval(pingTimerId)
        pingTimerId = null
      }
      const currentRunId = workflowExecutionRef.current.runId
      if (currentRunId && workflowExecutionRef.current.isRunning) {
        finishWorkflowRun(currentRunId)
      }
      if (aborted) return
      setWorkflowConnectionState('closed')
      workflowSocketRef.current = null
      if (reconnectOnClose && !event.wasClean) {
        setWorkflowError('워크플로우 WebSocket 연결이 종료되었습니다.')
        loadWorkflowGraph({ signal: controller.signal })
      }
    })

    return () => {
      aborted = true
      if (pingTimerId) window.clearInterval(pingTimerId)
      controller.abort()
      reconnectOnClose = false
      if (workflowSocketRef.current && workflowSocketRef.current.readyState <= 1) {
        workflowSocketRef.current.close(1000)
      }
      workflowSocketRef.current = null
      setWorkflowConnectionState('closed')
    }
  }, [])

  useEffect(() => {
    if (!isPanelOpen) return
    const socket = workflowSocketRef.current
    if (!socket) {
      loadWorkflowGraph()
      return
    }

    if (socket.readyState === WebSocket.OPEN) {
      socket.send('get_graph')
      return
    }

    const controller = new AbortController()
    loadWorkflowGraph({ signal: controller.signal })
    return () => {
      controller.abort()
    }
  }, [isPanelOpen])

  useEffect(() => {
    if (!isPanelOpen) {
      clearWorkflowNodeHighlight()
      return
    }

    const activeNode = workflowExecutionRef.current.activeNode
    if (activeNode) {
      const task = window.requestAnimationFrame(() => {
        applyWorkflowNodeHighlight(activeNode)
      })
      return () => window.cancelAnimationFrame(task)
    }
    return undefined
  }, [isPanelOpen])

  useEffect(() => {
    if (!isPanelOpen || !workflowGraph || !graphContainerRef.current) {
      if (!isPanelOpen && cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }
      return
    }

    try {
      const nodeSource = Array.isArray(workflowGraph.nodes) ? workflowGraph.nodes : []
      const edgeSource = Array.isArray(workflowGraph.edges) ? workflowGraph.edges : []

      const existingNodeIds = new Set()
      const nodeElements = []
      const edgeElements = []
      const seenEdges = new Set()

      for (const item of nodeSource) {
        const id = normalizeNodeId(item?.id ?? item)
        if (!id || existingNodeIds.has(id)) continue
        existingNodeIds.add(id)
        const normalized = normalizeNodeLabel(id)
        const visual = resolveNodeVisual(id)
        nodeElements.push({
          data: {
            id,
            label: normalized || id,
            bg: visual.bg,
            border: visual.border,
            text: visual.text,
          },
          classes: `${normalizeNodeClass(id)} wf-${visual.kind}`,
        })
      }

      for (let index = 0; index < edgeSource.length; index += 1) {
        const edge = edgeSource[index] || {}
        const source = normalizeNodeId(edge.from ?? edge.source)
        const target = normalizeNodeId(edge.to ?? edge.target)
        if (!source || !target) continue

        if (!existingNodeIds.has(source)) {
          existingNodeIds.add(source)
          const sourceVisual = resolveNodeVisual(source)
          nodeElements.push({
            data: {
              id: source,
              label: normalizeNodeLabel(source),
              bg: sourceVisual.bg,
              border: sourceVisual.border,
              text: sourceVisual.text,
            },
            classes: `${normalizeNodeClass(source)} wf-${sourceVisual.kind}`,
          })
        }

        if (!existingNodeIds.has(target)) {
          existingNodeIds.add(target)
          const targetVisual = resolveNodeVisual(target)
          nodeElements.push({
            data: {
              id: target,
              label: normalizeNodeLabel(target),
              bg: targetVisual.bg,
              border: targetVisual.border,
              text: targetVisual.text,
            },
            classes: `${normalizeNodeClass(target)} wf-${targetVisual.kind}`,
          })
        }

        const condition = edge.condition
        const conditionText = typeof condition === 'string'
          ? condition
          : condition == null
            ? ''
            : String(condition)
        const edgeId = `${index}-${source}->${target}${conditionText ? `:${conditionText}` : ''}`

        if (seenEdges.has(edgeId)) continue
        seenEdges.add(edgeId)

        edgeElements.push({
          data: {
            id: `e-${edgeId}`,
            source,
            target,
          },
          classes: conditionText ? 'has-condition' : 'default-edge',
        })
      }

      if (nodeElements.length === 0 && edgeElements.length === 0) {
        setWorkflowRenderError('워크플로우 노드/엣지 정보를 읽을 수 없습니다.')
        if (cyRef.current) {
          cyRef.current.destroy()
          cyRef.current = null
        }
        return
      }

      if (cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }

        cyRef.current = cytoscape({
        container: graphContainerRef.current,
        elements: [...nodeElements, ...edgeElements],
        boxSelectionEnabled: false,
        autounselectify: true,
        minZoom: 1,
        maxZoom: 2,
        zoom: 1,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': 'data(bg)',
              color: 'data(text)',
              content: (node) => node.data('label'),
              'text-wrap': 'wrap',
              'text-max-width': '130px',
              padding: 8,
              'font-size': 16,
              'min-zoomed-font-size': 16,
              'text-outline-width': 0,
              'font-weight': 600,
              'text-valign': 'center',
              'text-halign': 'center',
              'border-width': 1.2,
              'border-color': 'data(border)',
              'border-style': 'solid',
              width: 140,
              height: 40,
              'min-width': 140,
              'min-height': 40,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
              'shape': 'round-rectangle',
            },
          },
          {
            selector: 'node.wf-active',
            style: {
              'background-color': '#ffe8c7',
              'border-color': '#ff9e2c',
              'border-width': 3.8,
            },
          },
          {
            selector: 'node.start',
            style: {
              'font-weight': 800,
            },
          },
          {
            selector: 'node.end',
            style: {
              'font-style': 'italic',
            },
          },
          {
            selector: 'edge',
            style: {
              width: 2,
              'line-color': '#97a7bf',
              'target-arrow-color': '#97a7bf',
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'arrow-scale': 1.2,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
            },
          },
        ],
        layout: {
          name: 'dagre',
          nodeDimensionsIncludeLabels: true,
          rankDir: 'TB',
          ranker: 'network-simplex',
          nodeSep: 16,
          edgeSep: 10,
          rankSep: 30,
          spacingFactor: 0.95,
          fit: false,
          padding: 20,
        },
      })

      if (typeof cyRef.current.batch === 'function') {
        cyRef.current.batch(() => {
          cyRef.current.nodes().forEach((node) => {
            node.style({
              content: (node) => node.data('label'),
              'font-size': 16,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
              'text-wrap': 'wrap',
              'text-max-width': '130px',
              'text-outline-width': 0,
            })
          })
        })
      }

      cyRef.current.center()

      const currentActive = workflowExecutionRef.current.activeNode
      if (currentActive) {
        applyWorkflowNodeHighlight(currentActive)
      }

      setWorkflowRenderError('')
    } catch (error) {
      setWorkflowRenderError(error instanceof Error ? error.message : '워크플로우 그래프 렌더링 실패')
      if (cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }
    }

    return () => {
      if (cyRef.current) {
        clearWorkflowNodeHighlight()
        cyRef.current.destroy()
        cyRef.current = null
      }
    }
  }, [isPanelOpen, workflowGraph])

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
    const runId = buildRunId()
    const assistantMessageId = Date.now() + 1
    setMessages((prev) => [...prev, userMessage])
    setText('')

    setMessages((prev) => [
      ...prev,
      {
        id: assistantMessageId,
        role: 'assistant',
        text: '워크플로우 실행 요청 전송됨',
        runId,
      },
    ])

    const socket = workflowSocketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: 'run_workflow',
          run_id: runId,
          input: trimmed,
          model: selectedModel,
          response_mode: responseMode,
          max_tokens: maxTokens,
        })
      )
      return
    }

    appendMessageLineByRunId(runId, '워크플로우 WebSocket이 현재 닫혀 있어 실행 요청을 전송할 수 없습니다.')
    updateRunMessageMetaByRunId(runId, { status: 'error' })
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
            aria-label={isPanelOpen ? '워크플로우 패널 닫기' : '워크플로우 패널 열기'}
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
          <div
            ref={messagesContainerRef}
            className={`messages ${hasMessageOverflow && isMessagesScrollable ? 'messages-scrollbar-visible' : ''}`}
            role="log"
            aria-live="polite"
          >
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

        <div className="log-content" role="log" aria-live="polite">
          {isWorkflowLoading && (
            <p className="panel-state">워크플로우 구성을 조회하는 중입니다...</p>
          )}
          {workflowError && (
            <p className="panel-state panel-state-error">{workflowError}</p>
          )}
          {workflowRenderError && (
            <p className="panel-state panel-state-error">{workflowRenderError}</p>
          )}
          {!isWorkflowLoading && !workflowError && !workflowRenderError && !workflowGraph && (
            <p className="panel-state">패널을 열면 workflow 구성을 조회합니다.</p>
          )}
          {!isWorkflowLoading && !workflowError && !workflowRenderError && workflowGraph && (
            <div className="graph-view">
              <div ref={graphContainerRef} className="workflow-graph" aria-label="workflow graph" />
            </div>
          )}
        </div>
      </aside>

      {isPanelOpen && (
        <button
          className="panel-close log-panel-close-floating"
          aria-label="워크플로우 패널 닫기"
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
