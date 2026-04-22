import { WORKFLOW_NODE_ALIASES, WORKFLOW_NODE_PALETTE } from '../constants'

export const normalizeNodeId = (value) => {
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

export const normalizeNodeLabel = (nodeId) => {
  const normalized = normalizeNodeId(nodeId)
  if (!normalized) return ''
  const cleaned = normalized.replace(/^__+|__+$/g, '')
  if (!cleaned) return ''
  if (cleaned.toUpperCase() === 'START') return 'START'
  if (cleaned.toUpperCase() === 'END') return 'END'
  return cleaned
}

export const normalizeNodeClass = (nodeId) => {
  const normalized = normalizeNodeId(nodeId).toUpperCase()
  if (normalized === '__START__' || normalized === 'START') return 'start'
  if (normalized === '__END__' || normalized === 'END') return 'end'
  return 'default'
}

export const resolveNodeVisual = (nodeId) => {
  const rawId = normalizeNodeId(nodeId).trim()
  const alias = WORKFLOW_NODE_ALIASES[rawId] || WORKFLOW_NODE_ALIASES[rawId.toUpperCase()]
  const candidate = (alias || rawId).toLowerCase()
  return WORKFLOW_NODE_PALETTE[candidate] || WORKFLOW_NODE_PALETTE.default
}
