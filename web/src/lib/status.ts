import type { NodeStatus, PlanItem, Rec } from './types'

export function computeStatuses(rec: Rec[], upto: number, plan: PlanItem[]) {
  const node = new Map<string, NodeStatus>()
  const edge = new Map<string, string>()
  const planMap = new Map(plan.map((p) => [p.key, p]))
  let lastActive: string | null = null

  for (const e of rec) {
    if (e.t > upto) break
    const ev = e.ev
    if (ev.event === 'pipeline') {
      node.set('orc', 'active')
      lastActive = 'orc'
    } else if (ev.event === 'batch') {
      for (const k of ev.agents || []) node.set(k, 'active')
      lastActive = (ev.agents || [])[0] || lastActive
    } else if (ev.event === 'stage') {
      const k = ev.agent
      if (ev.status === 'running') {
        if (lastActive && lastActive !== k && lastActive !== 'orc') node.set(lastActive, 'done')
        node.set(k, 'active')
        lastActive = k
        const p = planMap.get(k)
        const inputs = (p?.inputs || []).filter((x) => x !== 'task')
        if (inputs.length === 0) {
          // 第一批: 从编排器进入
          edge.set(`e_orc_${k}`, 'active')
        } else {
          for (const from of inputs) edge.set(`e_${from}_${k}`, 'active')
        }
      } else if (ev.status === 'done') {
        node.set(k, 'done')
        for (const [id, s] of edge) {
          if (s === 'active' && id.endsWith(`_${k}`)) edge.set(id, 'traversed')
        }
      }
    } else if (ev.event === 'gate') {
      if (ev.agent) node.set(ev.agent, 'active')
    } else if (ev.event === 'team_done') {
      node.set('orc', 'done')
    } else if (ev.event === 'result') {
      for (const [id, s] of node) if (s === 'active') node.set(id, 'done')
      node.set('end', 'done')
      for (const [id, s] of edge) if (s === 'active') edge.set(id, 'traversed')
    } else if (ev.event === 'error') {
      node.set(ev.agent || 'orc', 'error')
    }
  }
  return { nodeStatuses: node, edgeStatuses: edge }
}
