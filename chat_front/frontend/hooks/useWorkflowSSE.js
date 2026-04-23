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
  const streamWorkflow = async ({ params, appendLine, updateMeta, onNodeEvent }) => {
    try {
      const res = await fetch(RUN_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        appendLine(`HTTP 오류: ${res.status}`)
        updateMeta({ status: 'error' })
        return
      }
      for await (const { eventType, data } of parseSSE(res.body)) {
        if (eventType === 'run_started') {
          const ragLabel = data.agentic_rag ? ' [Agentic RAG]' : ''
          appendLine(`모델: ${data.model}${ragLabel}`)
          continue
        }
        onNodeEvent?.(eventType, data)
        if (data.message) appendLine(data.message)
        if (data.payload?.final_output) appendLine(data.payload.final_output)
        if (eventType === 'workflow_complete') {
          if (data.final_response) appendLine(data.final_response)
          updateMeta({ status: 'done' })
        }
        if (eventType === 'workflow_error') updateMeta({ status: 'error' })
      }
    } catch (err) {
      appendLine(`연결 오류: ${err.message}`)
      updateMeta({ status: 'error' })
    }
  }

  return { streamWorkflow }
}
