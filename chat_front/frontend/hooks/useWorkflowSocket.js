import { useRef } from 'react'

const BACKEND_PORT = 10001
const _host = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const _isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:'
const GRAPH_API_URL = `${_isSecure ? 'https' : 'http'}://${_host}:${BACKEND_PORT}/graph`

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

const WORKFLOW_NODE_ALIASES = { '__start__': 'start', '__end__': 'end' }

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
  setWorkflowGraph,
  setWorkflowError,
  setIsWorkflowLoading,
  setWorkflowRenderError,
  setWorkflowConnectionState,
}) {
  const workflowExecutionRef = useRef({ runId: null, isRunning: false, activeNode: '' })

  const loadWorkflowGraph = async (opts = {}) => {
    const { signal } = opts
    if (!signal?.aborted) {
      setIsWorkflowLoading(true)
      setWorkflowError('')
      setWorkflowRenderError('')
      setWorkflowConnectionState('connecting')
    }
    try {
      const response = await fetch(GRAPH_API_URL, { signal })
      if (!response.ok) throw new Error(`워크플로우 API 응답 오류 (${response.status})`)
      const payload = await response.json()
      if (!signal?.aborted) {
        setWorkflowGraph(payload)
        setWorkflowError('')
        setIsWorkflowLoading(false)
        setWorkflowConnectionState('ready')
      }
    } catch (error) {
      if (error.name === 'AbortError') return
      if (!signal?.aborted) {
        setWorkflowError(error.message || '워크플로우 구성 조회 실패')
        setWorkflowConnectionState('error')
      }
    } finally {
      if (!signal?.aborted) setIsWorkflowLoading(false)
    }
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

  return {
    workflowExecutionRef,
    applyWorkflowNodeHighlight,
    clearWorkflowNodeHighlight,
    loadWorkflowGraph,
  }
}
