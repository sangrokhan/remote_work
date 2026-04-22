export function WorkflowPanel({
  isWorkflowLoading,
  workflowError,
  workflowRenderError,
  workflowGraph,
  graphContainerRef,
}) {
  return (
    <div className="log-content" role="log" aria-live="polite">
      {isWorkflowLoading && (
        <p className="panel-state">워크플로우 구성을 조회하는 중입니다...</p>
      )}
      {workflowError && (
        <p className="panel-state panel-state-error">{workflowError}</p>
      )}
      {workflowRenderError && (
        <p className="panel-state panel-state-error">{workflowRenderError}</p>
      )}
      {!isWorkflowLoading && !workflowError && !workflowRenderError && !workflowGraph && (
        <p className="panel-state">패널을 열면 workflow 구성을 조회합니다.</p>
      )}
      {!isWorkflowLoading && !workflowError && !workflowRenderError && workflowGraph && (
        <div className="graph-view">
          <div ref={graphContainerRef} className="workflow-graph" aria-label="workflow graph" />
        </div>
      )}
    </div>
  )
}
