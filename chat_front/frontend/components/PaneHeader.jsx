import { MODEL_LIST } from '../constants'

export function PaneHeader({ title, selectedModel, onModelChange, agenticRag, onAgenticRagToggle }) {
  return (
    <div className="pane-header">
      <span className="pane-title">{title}</span>
      <div className="pane-header-controls">
        <select
          className="pane-model-select"
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          aria-label={`${title} 모델 선택`}
        >
          {MODEL_LIST.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button
          className={`rag-toggle-btn${agenticRag ? ' active' : ''}`}
          aria-pressed={agenticRag}
          onClick={onAgenticRagToggle}
        >
          Agentic RAG
        </button>
      </div>
    </div>
  )
}
