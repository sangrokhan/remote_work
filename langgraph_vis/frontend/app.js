(function () {
  try {
    const widget = window.createWorkflowGraphWidget({
      graphEndpoint: "/graph",
      runEndpoint: "/run",
      graphContainerId: "workflow-graph",
      runButtonId: "run-btn",
      outputId: "output",
      statusId: "graph-status",
    });

    const { init } = widget;
    window.workflowGraphWidget = widget;
    init();
  } catch (error) {
    const status = document.getElementById("graph-status");
    if (status) {
      status.textContent = `초기화 실패: ${error.message}`;
    } else {
      // eslint-disable-next-line no-console
      console.error(error);
    }
  }
})();
