import { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'
import { normalizeNodeId, normalizeNodeLabel, normalizeNodeClass, resolveNodeVisual } from '../utils/nodeUtils'

cytoscape.use(dagre)

export function useWorkflowGraph({
  isPanelOpen,
  workflowGraph,
  workflowExecutionRef,
  applyWorkflowNodeHighlight,
  clearWorkflowNodeHighlight,
  setWorkflowRenderError,
}) {
  const graphContainerRef = useRef(null)
  const cyRef = useRef(null)

  useEffect(() => {
    if (!isPanelOpen || !workflowGraph || !graphContainerRef.current) {
      if (!isPanelOpen && cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }
      return
    }

    try {
      const nodeSource = Array.isArray(workflowGraph.nodes) ? workflowGraph.nodes : []
      const edgeSource = Array.isArray(workflowGraph.edges) ? workflowGraph.edges : []
      const existingNodeIds = new Set()
      const nodeElements = []
      const edgeElements = []
      const seenEdges = new Set()

      for (const item of nodeSource) {
        const id = normalizeNodeId(item?.id ?? item)
        if (!id || existingNodeIds.has(id)) continue
        existingNodeIds.add(id)
        const visual = resolveNodeVisual(id)
        nodeElements.push({
          data: { id, label: normalizeNodeLabel(id) || id, bg: visual.bg, border: visual.border, text: visual.text },
          classes: `${normalizeNodeClass(id)} wf-${visual.kind}`,
        })
      }

      for (let i = 0; i < edgeSource.length; i++) {
        const edge = edgeSource[i] || {}
        const source = normalizeNodeId(edge.from ?? edge.source)
        const target = normalizeNodeId(edge.to ?? edge.target)
        if (!source || !target) continue

        for (const nodeId of [source, target]) {
          if (!existingNodeIds.has(nodeId)) {
            existingNodeIds.add(nodeId)
            const visual = resolveNodeVisual(nodeId)
            nodeElements.push({
              data: { id: nodeId, label: normalizeNodeLabel(nodeId), bg: visual.bg, border: visual.border, text: visual.text },
              classes: `${normalizeNodeClass(nodeId)} wf-${visual.kind}`,
            })
          }
        }

        const condition = edge.condition
        const conditionText = typeof condition === 'string' ? condition : condition == null ? '' : String(condition)
        const edgeId = `${i}-${source}->${target}${conditionText ? `:${conditionText}` : ''}`
        if (seenEdges.has(edgeId)) continue
        seenEdges.add(edgeId)
        edgeElements.push({
          data: { id: `e-${edgeId}`, source, target },
          classes: conditionText ? 'has-condition' : 'default-edge',
        })
      }

      if (nodeElements.length === 0 && edgeElements.length === 0) {
        setWorkflowRenderError('워크플로우 노드/엣지 정보를 읽을 수 없습니다.')
        if (cyRef.current) { cyRef.current.destroy(); cyRef.current = null }
        return
      }

      if (cyRef.current) { cyRef.current.destroy(); cyRef.current = null }

      cyRef.current = cytoscape({
        container: graphContainerRef.current,
        elements: [...nodeElements, ...edgeElements],
        boxSelectionEnabled: false,
        autounselectify: true,
        minZoom: 1,
        maxZoom: 2,
        zoom: 1,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': 'data(bg)',
              color: 'data(text)',
              content: (node) => node.data('label'),
              'text-wrap': 'wrap',
              'text-max-width': '130px',
              padding: 8,
              'font-size': 16,
              'min-zoomed-font-size': 16,
              'text-outline-width': 0,
              'font-weight': 600,
              'text-valign': 'center',
              'text-halign': 'center',
              'border-width': 1.2,
              'border-color': 'data(border)',
              'border-style': 'solid',
              width: 140,
              height: 40,
              'min-width': 140,
              'min-height': 40,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
              shape: 'round-rectangle',
            },
          },
          { selector: 'node.wf-active', style: { 'background-color': '#ffe8c7', 'border-color': '#ff9e2c', 'border-width': 3.8 } },
          { selector: 'node.start', style: { 'font-weight': 800 } },
          { selector: 'node.end', style: { 'font-style': 'italic' } },
          {
            selector: 'edge',
            style: {
              width: 2,
              'line-color': '#97a7bf',
              'target-arrow-color': '#97a7bf',
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'arrow-scale': 1.2,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
            },
          },
        ],
        layout: {
          name: 'dagre',
          nodeDimensionsIncludeLabels: true,
          rankDir: 'TB',
          ranker: 'network-simplex',
          nodeSep: 16,
          edgeSep: 10,
          rankSep: 30,
          spacingFactor: 0.95,
          fit: false,
          padding: 20,
        },
      })

      if (typeof cyRef.current.batch === 'function') {
        cyRef.current.batch(() => {
          cyRef.current.nodes().forEach((node) => {
            node.style({
              content: (node) => node.data('label'),
              'font-size': 16,
              'font-family': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
              'text-wrap': 'wrap',
              'text-max-width': '130px',
              'text-outline-width': 0,
            })
          })
        })
      }

      cyRef.current.center()

      const activeNode = workflowExecutionRef.current.activeNode
      if (activeNode) applyWorkflowNodeHighlight(activeNode)
      setWorkflowRenderError('')
    } catch (error) {
      setWorkflowRenderError(error instanceof Error ? error.message : '워크플로우 그래프 렌더링 실패')
      if (cyRef.current) { cyRef.current.destroy(); cyRef.current = null }
    }

    return () => {
      if (cyRef.current) {
        clearWorkflowNodeHighlight()
        cyRef.current.destroy()
        cyRef.current = null
      }
    }
  }, [isPanelOpen, workflowGraph])

  return { graphContainerRef, cyRef }
}
