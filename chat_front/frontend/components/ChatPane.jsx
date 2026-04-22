import { useScrollBehavior } from '../hooks/useScrollBehavior'
import { MessageBubble } from './MessageBubble'
import { PaneHeader } from './PaneHeader'

export function ChatPane({
  messages,
  showHeader,
  title,
  selectedModel,
  onModelChange,
  agenticRag,
  onAgenticRagToggle,
  isRight,
  models,
}) {
  const { containerRef, isScrollable, hasOverflow } = useScrollBehavior(messages)

  return (
    <div className={`chat-pane${isRight ? ' chat-pane-right' : ''}`}>
      {showHeader && (
        <PaneHeader
          title={title}
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          agenticRag={agenticRag}
          onAgenticRagToggle={onAgenticRagToggle}
          models={models}
        />
      )}
      <div
        ref={containerRef}
        className={`messages${hasOverflow && isScrollable ? ' messages-scrollbar-visible' : ''}`}
        role="log"
        aria-live="polite"
      >
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>
    </div>
  )
}
