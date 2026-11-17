import { useEffect } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from 'reactflow'
import { AgentNode, type AgentNodeData } from './AgentNode'
import { FlowingEdge } from './FlowingEdge'

const nodeTypes = { agent: AgentNode }
const edgeTypes = { flowing: FlowingEdge }

type Props = {
  nodes: Node<AgentNodeData>[]
  edges: Edge[]
  onNodeClick?: NodeMouseHandler
}

function Inner({ nodes, edges, onNodeClick }: Props) {
  const { fitView } = useReactFlow()
  const sig = nodes.map((n) => n.id).join('|')
  useEffect(() => {
    const t = setTimeout(() => fitView({ padding: 0.25, duration: 350, maxZoom: 1.05 }), 60)
    return () => clearTimeout(t)
  }, [sig, fitView])
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      defaultEdgeOptions={{ type: 'flowing' }}
      onNodeClick={onNodeClick}
      fitView
      fitViewOptions={{ padding: 0.25, maxZoom: 1.05 }}
      minZoom={0.2}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
    >
      <Background gap={24} size={1} color="#dfe5e1" />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable nodeColor="#cfd8d3" maskColor="rgba(238,241,239,.75)" />
    </ReactFlow>
  )
}

export default function FlowCanvas(props: Props) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  )
}
