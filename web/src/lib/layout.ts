import dagre from 'dagre'
import type { Edge, Node } from 'reactflow'
import type { AgentMeta, NodeStatus, PlanItem } from './types'
import type { AgentNodeData } from '../components/AgentNode'

const NODE_W = 190
const NODE_H = 66
const GATE_W = 110
const GATE_H = 42

type BuildArgs = {
  plan: PlanItem[]
  agents: AgentMeta[]
  statuses: Map<string, NodeStatus>
  edgeStatuses: Map<string, string>
}

function baseNode(
  id: string,
  label: string,
  kind: AgentNodeData['kind'],
  status: NodeStatus,
  meta?: string,
  detail?: string,
): Node<AgentNodeData> {
  return {
    id,
    type: 'agent',
    position: { x: 0, y: 0 },
    data: { label, agentKey: id, status, kind, meta, detail },
  }
}

export function buildGraph({ plan, agents, statuses, edgeStatuses }: BuildArgs): {
  nodes: Node<AgentNodeData>[]
  edges: Edge[]
} {
  const agentMap = new Map(agents.map((a) => [a.key, a]))
  const nodes: Node<AgentNodeData>[] = []
  const edges: Edge[] = []

  const orc = baseNode('orc', 'ORCHESTRATOR', 'orch', statuses.get('orc') || 'idle', 'ResearchTeam')
  nodes.push(orc)

  const addAgent = (key: string, label?: string) => {
    const meta = agentMap.get(key)
    nodes.push(
      baseNode(
        key,
        label || meta?.name || key,
        meta?.gate ? 'gate' : 'agent',
        statuses.get(key) || 'idle',
        meta ? `${meta.layer} · ${meta.tools.length} tools` : undefined,
        statuses.get(key) === 'active' ? 'running…' : undefined,
      ),
    )
  }

  if (plan.length === 0) {
    // 启动前不铺全量图: 14 个 Agent 全连边会炸开, 也不真实。
    // 等 pipeline 事件带来本次计划后再画。
    const end0 = baseNode('end', 'END', 'end', 'idle')
    nodes.push(end0)
    const g0 = new dagre.graphlib.Graph()
    g0.setDefaultEdgeLabel(() => ({}))
    g0.setGraph({ rankdir: 'TB', ranksep: 80, nodesep: 40 })
    g0.setNode('orc', { width: 190, height: 66 })
    g0.setNode('end', { width: 110, height: 42 })
    g0.setEdge('orc', 'end')
    dagre.layout(g0)
    orc.position = { x: g0.node('orc').x - 95, y: g0.node('orc').y - 33 }
    end0.position = { x: g0.node('end').x - 55, y: g0.node('end').y - 21 }
    edges.push({ id: 'e_orc_end', source: 'orc', target: 'end', type: 'smoothstep' })
    return { nodes, edges }
  } else {
    const batches = new Map<number, PlanItem[]>()
    for (const it of plan) {
      if (!batches.has(it.batch)) batches.set(it.batch, [])
      batches.get(it.batch)!.push(it)
    }
    for (const [, items] of [...batches.entries()].sort((a, b) => a[0] - b[0])) {
      for (const it of items) addAgent(it.key, it.name)
    }
  }

  const end = baseNode('end', 'END', 'end', statuses.get('end') || 'idle')
  nodes.push(end)

  const link = (from: string, to: string) => {
    const id = `e_${from}_${to}`
    const status = edgeStatuses.get(id)
    edges.push({
      id,
      source: from,
      target: to,
      type: 'flowing',
      animated: status === 'active',
      className: status || '',
      data: { status: status || '' },
      markerEnd: { type: 'arrowclosed', color: status === 'active' ? '#0b6b5e' : '#a7b6b0' } as any,
    })
  }

  if (plan.length === 0) {
    for (const a of agents) link('orc', a.key)
    for (const a of agents) link(a.key, 'end')
  } else {
    const batches = new Map<number, PlanItem[]>()
    for (const it of plan) {
      if (!batches.has(it.batch)) batches.set(it.batch, [])
      batches.get(it.batch)!.push(it)
    }
    const ordered = [...batches.entries()].sort((a, b) => a[0] - b[0])
    const first = ordered[0]?.[1] || []
    const last = ordered[ordered.length - 1]?.[1] || []
    for (const it of first) link('orc', it.key)
    for (let i = 1; i < ordered.length; i++) {
      const prev = ordered[i - 1][1]
      const cur = ordered[i][1]
      for (const p of prev) for (const c of cur) link(p.key, c.key)
    }
    for (const it of last) link(it.key, 'end')
  }

  // dagre 布局
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', ranksep: 74, nodesep: 44, marginx: 30, marginy: 30 })
  for (const n of nodes) {
    const isSmall = n.data.kind === 'gate' || n.data.kind === 'end' || n.data.kind === 'orch'
    g.setNode(n.id, { width: isSmall ? GATE_W : NODE_W, height: isSmall ? GATE_H : NODE_H })
  }
  for (const e of edges) g.setEdge(e.source, e.target)
  dagre.layout(g)
  for (const n of nodes) {
    const pos = g.node(n.id)
    const isSmall = n.data.kind === 'gate' || n.data.kind === 'end' || n.data.kind === 'orch'
    const w = isSmall ? GATE_W : NODE_W
    const h = isSmall ? GATE_H : NODE_H
    n.position = { x: pos.x - w / 2, y: pos.y - h / 2 }
  }

  return { nodes, edges }
}
