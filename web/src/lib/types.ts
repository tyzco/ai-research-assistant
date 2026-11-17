export type AgentMeta = {
  key: string
  name: string
  layer: string
  role: string
  tools: string[]
  gate: string | null
  gate_name: string
}

export type PlanItem = {
  key: string
  name: string
  layer: string
  batch: number
  parallel: boolean
  gate: string
  inputs: string[]
}

export type GraphResp = {
  agents: AgentMeta[]
  layers: { id: string; name: string; desc: string; agents: string[] }[]
  gates: { id: string; name: string; detail: string }[]
  defaults: Record<string, number | string | boolean>
}

export type Rec = { t: number; ev: any }
export type RunListItem = {
  run_id: string
  task: string
  result: string
  status: string
  total_ms: number
  agent_count: number
  batch_count: number
  trace_id?: string
  created_at: string
}
export type Params = {
  rag_top_k: number
  rerank_min_score: number
  rag_rerank: boolean
  simple_max_len: number
  agent_max_rounds: number
  agent_timeout: number
  parallel_enabled: boolean
}

export type NodeStatus = 'idle' | 'active' | 'done' | 'error' | 'skipped'
