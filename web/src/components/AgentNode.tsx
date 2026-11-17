import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'

export type AgentNodeData = {
  label: string
  agentKey: string
  status: 'idle' | 'active' | 'done' | 'error' | 'skipped'
  kind: 'agent' | 'orch' | 'gate' | 'end' | 'start' | 'rag'
  meta?: string
  detail?: string
}

const STATUS_CLASS: Record<string, string> = {
  idle: 'idle',
  active: 'active',
  done: 'done',
  error: 'error',
  skipped: 'skipped',
}

export const AgentNode = memo(function AgentNode({ data }: NodeProps<AgentNodeData>) {
  return (
    <div className={`agent-node ${STATUS_CLASS[data.status] || 'idle'} ${data.kind}`}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div className="row">
        <span className="led" />
        <span className="name">{data.label}</span>
      </div>
      <span className="kindtag">{data.kind}</span>
      <div className="key">{data.agentKey}</div>
      {data.meta && <div className="meta"><span>{data.meta}</span></div>}
      {data.detail && (
        <div className="meta" style={{ display: 'block', color: 'var(--ink)', opacity: 0.75 }}>
          {data.detail}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  )
})
