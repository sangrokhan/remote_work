export const PANEL_WIDTH = '25%'

export const MODEL_LIST = ['gpt-4.1', 'gpt-4o-mini', 'gpt-4o']

export const INITIAL_LEFT_MESSAGES = [
  { id: 1, role: 'assistant', text: '안녕하세요. 무엇을 도와드릴까요?' },
]

export const INITIAL_RIGHT_MESSAGES = [
  { id: 1, role: 'assistant', text: '두 번째 시스템입니다. 무엇을 도와드릴까요?' },
]

export const WORKFLOW_NODE_PALETTE = {
  planner:     { kind: 'planner',     bg: '#d7ecff', border: '#68a8ee', text: '#12365f' },
  executor:    { kind: 'executor',    bg: '#d7f4dd', border: '#7dcf90', text: '#1a4f2f' },
  refiner:     { kind: 'refiner',     bg: '#fff1c7', border: '#e2be5e', text: '#5e4b17' },
  synthesizer: { kind: 'synthesizer', bg: '#f5d9fc', border: '#c78ce0', text: '#5d2e69' },
  start:       { kind: 'start',       bg: '#e7e7f8', border: '#9ca2df', text: '#32366c' },
  end:         { kind: 'end',         bg: '#dceeff', border: '#5f8bb0', text: '#263246' },
  default:     { kind: 'default',     bg: '#edf2ff', border: '#a9b6d7', text: '#2f3c56' },
}

export const WORKFLOW_NODE_ALIASES = {
  '__start__': 'start',
  '__end__': 'end',
}
