export function MessageBubble({ message }) {
  return (
    <article className={`bubble ${message.role}${message.status ? ` bubble-${message.status}` : ''}`}>
      <p>{message.text}</p>
      {message.status === 'disconnected' && <p className="bubble-status-notice">연결 끊김</p>}
      {message.status === 'error' && <p className="bubble-status-notice">오류 발생</p>}
    </article>
  )
}
