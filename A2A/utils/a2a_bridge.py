import uuid
import asyncio
from typing import Optional, List, Any, Union
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, TextPart, Role, Part, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, MessageSendParams
from a2a.server.agent_execution.context import RequestContext
from a2a.server.agent_execution import SimpleRequestContextBuilder

class CaptureEventQueue(EventQueue):
    """
    An EventQueue implementation that captures the first generated message.
    Used for bridging to legacy request-response style calls.
    """
    def __init__(self):
        self.captured_message: Optional[Message] = None
        self._queue = []

    def enqueue_event(self, event: Union[Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent]) -> None:
        if isinstance(event, Message):
            if not self.captured_message:
                self.captured_message = event
        self._queue.append(event)
    
    def dequeue_event(self):
        if self._queue:
            return self._queue.pop(0)
        return None

    def clear_events(self):
        self._queue = []
    
    def close(self):
        pass
        
    def is_closed(self) -> bool:
        return False
        
    def tap(self):
        pass
        
    def task_done(self):
        pass


def create_text_message(text: str, role: Role = Role.user, metadata: Optional[dict] = None) -> Message:
    """Helper to create a simple text message conforming to a2a-sdk types."""
    return Message(
        messageId=str(uuid.uuid4()),
        role=role,
        parts=[TextPart(text=text)],
        metadata=metadata
    )

async def run_agent_sync(executor, message: Message) -> Message:
    """
    Runs an a2a-sdk AgentExecutor in a pseudo-synchronous way.
    
    Args:
        executor: The AgentExecutor instance.
        message: The input Message.
        
    Returns:
        The first Message emitted by the agent.
    """
    # 1. create dummy context
    # We need a context builder to get a context, or construct one manually if possible.
    # SimpleRequestContextBuilder usually takes a request and returns context.
    # Since we are bypassing the server layer, we can try to instantiate RequestContext directly 
    # if allowed, or mock it.
    
    # RequestContext signature (from inspection):
    # args might be complex. Let's try to mock or construct minimal one.
    # Actually, a2a.server.agent_execution.context.RequestContext seems to handle state.
    
    # Let's use a simple mock-like object if strictly typed, or try to construct.
    # For now, we will assume we can construct it or pass a dummy.
    # Ideally we use SimpleRequestContextBuilder.
    
    # Use builder to create context properly
    builder = SimpleRequestContextBuilder()
    params = MessageSendParams(message=message)
    context = await builder.build(params=params)
    
    queue = CaptureEventQueue()
    
    # Execute
    await executor.execute(context, queue)
    
    if queue.captured_message:
        return queue.captured_message
    
    # If no message returned, return an empty system message/error
    return create_text_message("No response from agent", role=Role.agent)
