import { useRef, useState } from 'react'
import { X } from 'lucide-react'

type Stage =
  | 'rewrite' | 'bm25' | 'vector' | 'rrf' | 'rerank' | 'gen' | 'parse' | 'chunk' | 'embed' | 'fts'

const NODES: { id: Stage; label: string; x: number; y: number }[] = [
  { id: 'parse', label: '解析', x: 30, y: 22 },
  { id: 'chunk', label: '分块', x: 118, y: 22 },
  { id: 'embed', label: '嵌入', x: 206, y: 22 },
  { id: 'fts', label: 'FTS', x: 294, y: 22 },
  { id: 'rewrite', label: '查询改写', x: 30, y: 92 },
  { id: 'bm25', label: 'BM25', x: 140, y: 70 },
  { id: 'vector', label: '向量', x: 140, y: 114 },
  { id: 'rrf', label: 'RRF', x: 246, y: 92 },
  { id: 'rerank', label: '两级重排', x: 340, y: 92 },
  { id: 'gen', label: '生成', x: 434, y: 92 },
]

const EDGES: [Stage, Stage, boolean][] = [
  ['parse', 'chunk', false], ['chunk', 'embed', false], ['embed', 'fts', false],
  ['rewrite', 'bm25', true], ['rewrite', 'vector', true],
  ['bm25', 'rrf', true], ['vector', 'rrf', true],
  ['rrf', 'rerank', false], ['rerank', 'gen', false],
]

const MAP: Record<string, Stage> = {
  coref: 'rewrite', rewrite: 'rewrite', vector: 'vector', bm25: 'bm25',
  rrf: 'rrf', rerank1: 'rerank', rerank2: 'rerank', retrieve: 'rrf', generate: 'gen',
}

export default function RagPip({ stage, onClose }: { stage: string; onClose: () => void }) {
  const [pos, setPos] = useState({ x: 16, y: 16 })
  const [collapsed, setCollapsed] = useState(false)
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const active = MAP[stage] || null
  const nodeMap = new Map(NODES.map((n) => [n.id, n]))

  return (
    <div
      style={{
        position: 'absolute', right: pos.x, bottom: pos.y, width: 540, zIndex: 20,
        background: 'rgba(248,249,248,.97)', border: '1px solid var(--line-strong)',
        borderRadius: 7, boxShadow: '0 12px 34px rgba(19,32,29,.14)', overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px',
          background: 'var(--ink)', color: 'var(--paper-2)', cursor: 'grab',
          fontFamily: '"IBM Plex Mono", monospace', fontSize: 8.5, letterSpacing: '.06em',
          textTransform: 'uppercase', userSelect: 'none',
        }}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, ox: pos.x, oy: pos.y }
          ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          if (!drag.current) return
          setPos({
            x: Math.max(8, drag.current.ox - (e.clientX - drag.current.x)),
            y: Math.max(8, drag.current.oy + (e.clientY - drag.current.y)),
          })
        }}
        onPointerUp={() => { drag.current = null }}
        onClick={() => setCollapsed((c) => !c)}
      >
        RAG 子流程 · 检索链
        <span style={{ marginLeft: 'auto', opacity: 0.7 }}>{collapsed ? '展开' : '折叠'}</span>
        <span onClick={(e) => { e.stopPropagation(); onClose() }} style={{ cursor: 'pointer' }}><X size={11} /></span>
      </div>
      {!collapsed && (
        <svg viewBox="0 0 540 150" style={{ width: '100%', display: 'block' }}>
          <defs>
            <marker id="rag-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill="#a7b6b0" />
            </marker>
          </defs>
          {EDGES.map(([a, b, dashed], i) => {
            const na = nodeMap.get(a)!
            const nb = nodeMap.get(b)!
            const x1 = na.x + 42, y1 = na.y + 14, x2 = nb.x, y2 = nb.y + 14
            const d = Math.abs(y1 - y2) < 2
              ? `M${x1} ${y1} L${x2} ${y2}`
              : `M${x1} ${y1} C ${x1 + 20} ${y1}, ${x2 - 20} ${y2}, ${x2} ${y2}`
            return (
              <path key={i} d={d} fill="none" stroke="#a7b6b0" strokeWidth="1.1"
                strokeDasharray={dashed ? '3 3' : undefined} markerEnd="url(#rag-arrow)" />
            )
          })}
          {NODES.map((n) => {
            const on = active === n.id
            return (
              <g key={n.id}>
                <rect x={n.x} y={n.y} width={84} height={28} rx={5}
                  fill={on ? '#e7f2ee' : '#f8f9f8'}
                  stroke={on ? '#0b6b5e' : '#cfd8d3'} strokeWidth={on ? 1.8 : 1} />
                <circle cx={n.x + 9} cy={n.y + 9} r={2.6} fill={on ? '#0b6b5e' : '#a7b6b0'} />
                <text x={n.x + 46} y={n.y + 18} textAnchor="middle" fontSize="9.5"
                  fill={on ? '#0b6b5e' : '#5c6b66'} fontWeight={on ? 600 : 400}
                  fontFamily='"IBM Plex Sans", sans-serif'>{n.label}</text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
