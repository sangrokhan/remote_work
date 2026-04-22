import { useEffect, useRef } from 'react'

const BACKEND_PORT = 10001
const _host = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const _isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:'
const GRAPH_API_URL = `${_isSecure ? 'https' : 'http'}://${_host}:${BACKEND_PORT}/graph`
const GRAPH_WS_URL = `${_isSecure ? 'wss' : 'ws'}://${_host}:${BACKEND_PORT}/ws/connect`

const normalizeNodeId = (value) => {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (typeof value === 'object') {
    if (typeof value.id === 'string' || typeof value.id === 'number') return String(value.id)
    if (typeof value.value === 'string' || typeof value.value === 'number') return String(value.value)
    if (typeof value.name === 'string' || typeof value.name === 'number') return String(value.name)
  }
  return String(value)
}

const WORKFLOW_NODE_ALIASES = {
  '__start__': 'start',
  '__end__': 'end',
}

const normalizeWorkflowNodeId = (nodeId) => {
  const normalized = normalizeNodeId(nodeId).trim()
  if (!normalized) return ''
  const upper = normalized.toUpperCase()
  if (upper === 'START') return '__start__'
  if (upper === 'END') return '__end__'
  if (WORKFLOW_NODE_ALIASES[normalized]) return `__${WORKFLOW_NODE_ALIASES[normalized]}__`
  if (WORKFLOW_NODE_ALIASES[upper]) return `__${WORKFLOW_NODE_ALIASES[upper]}__`
  return normalized
}

export function useWorkflowSocket({
  isPanelOpenRef,
  cyRef,
  setMessages,
  setWorkflowGraph,
  setWorkflowError,
  setIsWorkflowLoading,
  setWorkflowRenderError,
  setWorkflowConnectionState,
}) {
  const workflowSocketRef = useRef(null)
  const workflowExecutionRef = useRef({ runId: null, isRunning: false, activeNode: '' })

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
      if (!response.ok) throw new Error(`워크플로우 API 응답 오류 (${response.status})`)
      const payload = await response.json()
      if (!signal?.aborted) setWorkflowFromPayload(payload)
    } catch (error) {
      if (error.name === 'AbortError') return
      if (!signal?.aborted) setWorkflowError(error.message || '워크플로우 구성 조회 실패')
    } finally {
      if (!signal?.aborted) setIsWorkflowLoading(false)
    }
  }

  const appendMessageLineByRunId = (runId, nextText) => {
    if (!runId || !nextText) return
    const safeText = String(nextText)
    setMessages((prev) =>
      prev.map((message) => {
        if (message.role !== 'assistant' || message.runId !== runId) return message
        return { ...message, text: message.text ? `${message.text}\n${safeText}` : safeText }
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
    workflowExecutionRef.current = { runId, isRunning: true, activeNode: '' }
    clearWorkflowNodeHighlight()
  }

  const finishWorkflowRun = (runId) => {
    const state = workflowExecutionRef.current
    if (state.runId && state.runId !== runId) return
    clearWorkflowNodeHighlight()
    workflowExecutionRef.current = { ...state, runId, isRunning: false, activeNode: '' }
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
    if (shouldHighlight) applyWorkflowNodeHighlight(nodeId)
  }

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
      if (isPanelOpenRef.current) socket.send('get_graph')
      pingTimerId = window.setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) socket.send('ping')
      }, 20000)
    })

    socket.addEventListener('message', async (event) => {
      if (aborted) return
      try {
        const parsed = JSON.parse(event.data)
        if (parsed?.type === 'connected') {
          setWorkflowConnectionState('ready')
          setWorkflowError('')
          return
        }
        if (parsed?.type === 'graph' && parsed.payload) {
          setWorkflowFromPayload(parsed.payload)
          return
        }
        if (parsed?.type === 'workflow_started') {
          startWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 실행됨')
          return
        }
        if (parsed?.type === 'workflow_event' && parsed.run_id) {
          const nodeId = parsed.node || parsed.name
          const eventType = String(parsed.event || '').trim()
          const eventStage = String(parsed.stage || '').trim()
          if (isActiveRun(parsed.run_id) && nodeId && (eventType === 'node_started' || eventStage === 'start')) {
            updateActiveWorkflowNode(parsed.run_id, nodeId)
          }
          if (parsed.message) appendMessageLineByRunId(parsed.run_id, parsed.message)
          if (parsed.payload?.final_output) appendMessageLineByRunId(parsed.run_id, parsed.payload.final_output)
          return
        }
        if (parsed?.type === 'workflow_complete' && parsed.run_id) {
          finishWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 완료')
          updateRunMessageMetaByRunId(parsed.run_id, { status: 'done' })
          return
        }
        if (parsed?.type === 'workflow_error' && parsed.run_id) {
          finishWorkflowRun(parsed.run_id)
          appendMessageLineByRunId(parsed.run_id, parsed.message || 'workflow 오류')
          updateRunMessageMetaByRunId(parsed.run_id, { status: 'error' })
          return
        }
      } catch (_error) {
        if (typeof event.data === 'string' && event.data.startsWith('echo:')) return
      }
    })

    socket.addEventListener('error', () => {
      if (aborted) return
      const currentRunId = workflowExecutionRef.current.runId
      if (currentRunId && workflowExecutionRef.current.isRunning) {
        appendMessageLineByRunId(currentRunId, 'WebSocket 오류로 연결이 끊어졌습니다. 워크플로우 실행이 중단되었습니다.')
        updateRunMessageMetaByRunId(currentRunId, { status: 'disconnected' })
        finishWorkflowRun(currentRunId)
      }
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
        if (!aborted) {
          appendMessageLineByRunId(currentRunId, '연결이 끊어졌습니다. 워크플로우 실행이 중단되었습니다.')
          updateRunMessageMetaByRunId(currentRunId, { status: 'disconnected' })
        }
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

  return {
    workflowSocketRef,
    workflowExecutionRef,
    appendMessageLineByRunId,
    updateRunMessageMetaByRunId,
    applyWorkflowNodeHighlight,
    clearWorkflowNodeHighlight,
    loadWorkflowGraph,
  }
}
