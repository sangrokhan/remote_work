const _host = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const _isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:'
const RUN_API_URL = `${_isSecure ? 'https' : 'http'}://${_host}:10001/api/run`

async function* parseSSE(body) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop()
    for (const block of blocks) {
      let eventType = 'message'
      let dataStr = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) eventType = line.slice(7).trim()
        if (line.startsWith('data: ')) dataStr = line.slice(6).trim()
      }
      if (dataStr) {
        try { yield { eventType, data: JSON.parse(dataStr) } } catch { /* skip malformed */ }
      }
    }
  }
}

export function useWorkflowSSE() {
  const streamWorkflow = async ({ params, replaceLine, updateMeta, onNodeEvent }) => {
    try {
      const res = await fetch(RUN_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        replaceLine(`HTTP 오류: ${res.status}`)
        updateMeta({ status: 'error' })
        return
      }
      for await (const { eventType, data } of parseSSE(res.body)) {
        if (eventType === 'run_started') {
          const ragLabel = data.agentic_rag ? ' [Agentic RAG]' : ''
          replaceLine(`모델: ${data.model}${ragLabel}`)
          continue
        }
        onNodeEvent?.(eventType, data)
        // 진행 상황은 이전 내용을 교체 (단계별 진행 표시)
        if (data.node) {
          const stage = data.stage === 'start' ? '실행 중...' : data.stage === 'end' ? '완료' : (data.message || '')
          const summary = data.payload ? JSON.stringify(data.payload).slice(0, 80) : ''
          replaceLine(`[${data.node}] ${stage}${summary ? `\n${summary}` : ''}`)
        } else if (data.message) {
          replaceLine(data.message)
        }
        if (data.payload?.final_output) replaceLine(data.payload.final_output)
        if (eventType === 'workflow_complete') {
          // 최종 결과만 표시 — 진행 내용 교체
          replaceLine(data.final_response || '')
          updateMeta({ status: 'done' })
        }
        if (eventType === 'workflow_error') {
          replaceLine(data.message || '오류가 발생했습니다.')
          updateMeta({ status: 'error' })
        }
      }
    } catch (err) {
      replaceLine(`연결 오류: ${err.message}`)
      updateMeta({ status: 'error' })
    }
  }

  return { streamWorkflow }
}
