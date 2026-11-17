import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Pause, Play, PanelLeft, PanelRight, RotateCcw, SkipBack, SkipForward, X,
} from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import FlowCanvas from './components/FlowCanvas'
import RagPip from './components/RagPip'
import { buildGraph } from './lib/layout'
import { computeStatuses } from './lib/status'
import {
  compareRuns, compressEvents, fetchGraph, getRun, listRuns, saveRun, streamSSE,
} from './lib/api'
import type { GraphResp, Params, PlanItem, Rec, RunListItem } from './lib/types'

const EVENT_LABEL: Record<string, string> = {
  pipeline: '编排计划', batch: '并行批次', stage: 'Agent', gate: '闸门',
  team_done: '团队完成', result: '结果', error: '错误', rag: 'RAG',
  agent_turn: '轮次', agent_llm: '模型返回', agent_tool: '工具',
}

const DEFAULT_PARAMS: Params = {
  rag_top_k: 5, rerank_min_score: 0, rag_rerank: true,
  simple_max_len: 60, agent_max_rounds: 8, agent_timeout: 180,
  parallel_enabled: true,
}

export default function App() {
  const [graph, setGraph] = useState<GraphResp | null>(null)
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS)
  const [task, setTask] = useState('总结 RRF（倒数排名融合）的定义与用途')
  const [rec, setRec] = useState<Rec[]>([])
  const [plan, setPlan] = useState<PlanItem[]>([])
  const [live, setLive] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [playMs, setPlayMs] = useState(0)
  const [totalMs, setTotalMs] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [running, setRunning] = useState(false)
  const [output, setOutput] = useState('')
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [historyError, setHistoryError] = useState('')
  const [now, setNow] = useState({ badge: 'IDLE', title: '等待启动', detail: '', time: '--' })
  const [selected, setSelected] = useState<string | null>(null)
  const [compare, setCompare] = useState<any>(null)
  const [cmpA, setCmpA] = useState('')
  const [cmpB, setCmpB] = useState('')
  const [traceId, setTraceId] = useState('')
  const [showSide, setShowSide] = useState(false)
  const [showRight, setShowRight] = useState(false)
  const [ragStage, setRagStage] = useState('')
  const [ragOpen, setRagOpen] = useState(false)
  const [rightTab, setRightTab] = useState<'events' | 'output'>('events')
  const [eventFilter, setEventFilter] = useState<'key' | 'all'>('key')
  const ctl = useRef<AbortController | null>(null)
  const t0 = useRef(0)
  const recRef = useRef<Rec[]>([])
  const planRef = useRef<PlanItem[]>([])
  const outRef = useRef('')

  useEffect(() => {
    fetchGraph()
      .then((g) => { setGraph(g); setParams((p) => ({ ...p, ...(g.defaults || {}) })) })
      .catch(() => {})
    listRuns().then((d) => setRuns(d.runs || [])).catch(() => {})
  }, [])

  const upto = live ? totalMs : playMs

  const { nodeStatuses, edgeStatuses } = useMemo(
    () => computeStatuses(rec, upto, plan), [rec, upto, plan],
  )

  const base = useMemo(
    () => buildGraph({ plan, agents: graph?.agents || [], statuses: new Map(), edgeStatuses: new Map() }),
    [plan, graph],
  )

  const nodes = useMemo(
    () => base.nodes.map((n) => {
      const st = nodeStatuses.get(n.id) || 'idle'
      return {
        ...n,
        data: {
          ...n.data,
          status: st,
          detail: st === 'active' ? 'running…' : n.data.detail,
        },
      }
    }),
    [base.nodes, nodeStatuses],
  )

  const edges = useMemo(
    () => base.edges.map((e) => {
      const st = edgeStatuses.get(e.id) || ''
      return { ...e, className: st, animated: st === 'active', data: { ...(e.data || {}), status: st } }
    }),
    [base.edges, edgeStatuses],
  )

  const visibleEvents = useMemo(
    () => rec.filter((r) => r.t <= upto && r.ev?.event !== 'token').slice(-160).reverse(),
    [rec, upto],
  )

  const keyEvents = useMemo(
    () => visibleEvents.filter((e) =>
      ['stage', 'gate', 'result', 'error', 'pipeline', 'team_done'].includes(e.ev?.event)),
    [visibleEvents],
  )
  const shownEvents = eventFilter === 'key' ? keyEvents : visibleEvents

  const nowEvent = useMemo(() => {
    for (let i = rec.length - 1; i >= 0; i--) {
      if (rec[i].t <= upto && rec[i].ev?.event !== 'token') return rec[i]
    }
    return null
  }, [rec, upto])

  const batchCount = useMemo(() => new Set(plan.map((p) => p.batch)).size, [plan])
  const currentAgent = useMemo(() => {
    for (let i = rec.length - 1; i >= 0; i--) {
      const ev = rec[i].ev
      if (ev?.event === 'stage' && ev.turn !== undefined) return ev.agent
    }
    return plan.length ? plan[0].key : ''
  }, [rec, plan])
  const currentBatch = useMemo(() => {
    const p = plan.find((x) => x.key === currentAgent)
    return p ? p.batch : 0
  }, [plan, currentAgent])
  const agentDone = useMemo(() => {
    let n = 0
    nodeStatuses.forEach((st) => { if (st === 'done') n++ })
    return n
  }, [nodeStatuses])

  const setNowFromEvent = useCallback((ev: any, t: number) => {
    const awaitingApproval = ev.event === 'result' && ev.status === 'awaiting_approval'
    const badge = ev.event === 'stage' ? (ev.status === 'done' ? 'DONE' : 'AGENT')
      : ev.event === 'gate' || awaitingApproval ? 'GATE' : ev.event === 'result' ? 'END'
      : ev.event === 'rag' ? 'RAG' : ev.event === 'error' ? 'ERR'
      : (ev.event || '').toUpperCase()
    const title = ev.event === 'stage' ? (ev.name || ev.agent)
      : ev.event === 'agent_turn' ? `${ev.name || ev.agent} · 第 ${ev.round} 轮`
      : ev.event === 'agent_llm' ? `${ev.agent} · 模型返回`
      : ev.event === 'agent_tool' ? `${ev.tool || '工具'} · ${ev.agent || ''}`
      : ev.event === 'gate' || awaitingApproval ? '等待人工确认' : ev.event === 'result' ? '运行结束'
      : ev.event === 'rag' ? `RAG · ${ev.stage || ''}` : (ev.event || '')
    const detail = ev.detail || ev.preview || ev.title || ev.message || ev.request?.title
      || (ev.event === 'result' ? String(ev.result || '').slice(0, 120) : '')
    setNow({ badge, title, detail, time: `${(t / 1000).toFixed(1)}s` })
  }, [])

  useEffect(() => {
    if (nowEvent) setNowFromEvent(nowEvent.ev, nowEvent.t)
  }, [nowEvent, setNowFromEvent])

  const push = useCallback((ev: any) => {
    const t = Date.now() - t0.current
    recRef.current = [...recRef.current, { t, ev }]
    setRec(recRef.current)
    setTotalMs((prev) => Math.max(prev, t))
    if (ev.event === 'rag') { setRagStage(ev.stage || ''); setRagOpen(true) }
    if (ev.event === 'pipeline' && Array.isArray(ev.plan)) {
      setPlan(ev.plan); planRef.current = ev.plan
    }
    if (ev.event === 'token') {
      outRef.current += ev.token || ''
      setOutput(outRef.current)
    }
    if (ev.event === 'result') {
      setOutput(ev.result || ''); outRef.current = ev.result || ''
      setPlaying(false)
    }
    setNowFromEvent(ev, t)
  }, [setNowFromEvent])

  const start = useCallback(async () => {
    if (ctl.current) return
    setRec([]); setPlan([]); setOutput(''); setLive(true); setPlaying(false); setPlayMs(0); setTotalMs(0)
    setHistoryError('')
    recRef.current = []; planRef.current = []; outRef.current = ''
    setRagOpen(false); setRagStage(''); setTraceId('')
    setNow({ badge: 'RUN', title: '初始化', detail: '正在建立任务…', time: '0.0s' })
    setRunning(true)
    t0.current = Date.now()
    const controller = new AbortController()
    ctl.current = controller
    try {
      await streamSSE('/team/stream', { task, confirm: true, params }, push, controller.signal)
    } catch (e: any) {
      if (e.name !== 'AbortError') push({ event: 'error', message: e.message })
    } finally {
      setLive(false)
      const total = Date.now() - t0.current
      setPlayMs(total); setTotalMs((prev) => Math.max(prev, total))
      const snapshot = recRef.current
      const resultEv = snapshot.find((e) => e.ev?.event === 'result')?.ev
      const trId = resultEv?.trace_id || ''
      setTraceId(trId)
      try {
        await saveRun({
          task,
          params: { ...params, _mode: 'team' },
          events: compressEvents(snapshot),
          result: outRef.current,
          status: controller.signal.aborted ? 'cancelled' : resultEv?.status || 'error',
          total_ms: total,
          agent_count: planRef.current.length,
          batch_count: new Set(planRef.current.map((p) => p.batch)).size,
          trace_id: trId,
        })
        const history = await listRuns()
        setRuns(history.runs || [])
      } catch {
        setHistoryError('历史记录保存或刷新失败，仍可回放本次运行。')
      } finally {
        ctl.current = null
        setRunning(false)
      }
    }
  }, [task, params, push])

  const stop = useCallback(() => { ctl.current?.abort() }, [])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const tick = (t: number) => {
      const dt = t - last
      last = t
      setPlayMs((ms) => {
        const next = Math.min(totalMs, ms + dt * speed)
        if (next >= totalMs) setPlaying(false)
        return next
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, totalMs, speed])

  useEffect(() => {
    if (!compare) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setCompare(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [compare])

  const loadRun = useCallback(async (id: string) => {
    if (!id || ctl.current) return
    const d = await getRun(id).catch(() => {
      setHistoryError('无法加载这条历史记录。')
      return null
    })
    if (!d || ctl.current) return
    setHistoryError('')
    const events: Rec[] = (d.events || []).slice()
    setRec(events); recRef.current = events
    const total = events.reduce((m, e) => Math.max(m, e.t || 0), d.total_ms || 0)
    setTotalMs(total); setPlayMs(total); setLive(false); setPlaying(false)
    setOutput(d.result || ''); outRef.current = d.result || ''
    setParams((p) => ({ ...p, ...(d.params || {}) }))
    const pipe = events.find((e) => e.ev?.event === 'pipeline')
    setPlan(pipe?.ev?.plan || []); planRef.current = pipe?.ev?.plan || []
    setTask(d.task || '')
    setTraceId(d.trace_id || '')
    const last = events[events.length - 1]
    if (last) setNowFromEvent(last.ev, last.t || 0)
  }, [setNowFromEvent])

  const doCompare = useCallback(async () => {
    if (!cmpA || !cmpB || cmpA === cmpB) return
    setCompare(await compareRuns(cmpA, cmpB))
  }, [cmpA, cmpB])

  const advance = (dir: number) => {
    setLive(false); setPlaying(false)
    if (dir > 0) {
      const next = rec.find((e) => e.t > playMs + 1)
      setPlayMs(next ? next.t : totalMs)
    } else {
      const prev = [...rec].reverse().find((e) => e.t < playMs - 1)
      setPlayMs(prev ? prev.t : 0)
    }
  }

  const pct = totalMs ? Math.min(100, (upto / totalMs) * 100) : 0
  const selNode = selected ? graph?.agents.find((a) => a.key === selected) : null
  const cmpData = useMemo(() => {
    if (!compare) return []
    const a = compare.a, b = compare.b
    const items: any[] = [
      { name: '总耗时(s)', A: +(a.total_ms / 1000).toFixed(1), B: +(b.total_ms / 1000).toFixed(1) },
      { name: '结果字数', A: a.result_len, B: b.result_len },
      { name: '事件数', A: a.events, B: b.events },
    ]
    const keys = new Set([...Object.keys(a.agent_ms || {}), ...Object.keys(b.agent_ms || {})])
    keys.forEach((k) => items.push({ name: k, A: (a.agent_ms || {})[k] || 0, B: (b.agent_ms || {})[k] || 0 }))
    return items
  }, [compare])

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">P</span>
        <div>
          <div className="title">Agent Orchestration Instrument</div>
          <div className="subtitle">PaperLink · React Flow · live execution</div>
        </div>
        <button className="btn panel-toggle" aria-label="切换参数面板" onClick={() => setShowSide((v) => !v)}><PanelLeft size={12} /></button>
        <button className="btn panel-toggle" aria-label="切换事件面板" onClick={() => setShowRight((v) => !v)}><PanelRight size={12} /></button>
        <span className={`live-dot ${running ? 'on' : ''}`} />
        {traceId && (
          <a className="btn" href={`https://cloud.langfuse.com/trace/${traceId}`} target="_blank" rel="noreferrer">Langfuse</a>
        )}
        <span className="spacer" />
        <select className="select" value="" disabled={running} onChange={(e) => loadRun(e.target.value)} aria-label="历史运行">
          <option value="">历史运行</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.task.slice(0, 16)} · {Math.round(r.total_ms / 1000)}s</option>
          ))}
        </select>
      </header>

      <div className="body">
        <aside className={`sidebar ${showSide ? 'open' : ''}`}>
          <details open>
            <summary className="section-label">RAG 检索</summary>
            <ParamRow label="top_k" value={params.rag_top_k} min={1} max={20} step={1}
              onChange={(v) => setParams((p) => ({ ...p, rag_top_k: v }))} />
            <ParamRow label="重排阈值" value={params.rerank_min_score} min={0} max={1} step={0.05}
              onChange={(v) => setParams((p) => ({ ...p, rerank_min_score: v }))} />
            <label className="param-row">
              <input type="checkbox" checked={params.rag_rerank} aria-label="启用两级重排"
                onChange={(e) => setParams((p) => ({ ...p, rag_rerank: e.target.checked }))} />
              启用两级重排
            </label>
          </details>
          <details open>
            <summary className="section-label">编排</summary>
            <ParamRow label="简单任务阈值" value={params.simple_max_len} min={20} max={120} step={5}
              onChange={(v) => setParams((p) => ({ ...p, simple_max_len: v }))} />
            <ParamRow label="轮次上限" value={params.agent_max_rounds} min={2} max={8} step={1}
              onChange={(v) => setParams((p) => ({ ...p, agent_max_rounds: v }))} />
            <ParamRow label="超时(s)" value={params.agent_timeout} min={30} max={180} step={10}
              onChange={(v) => setParams((p) => ({ ...p, agent_timeout: v }))} />
            <label className="param-row">
              <input type="checkbox" checked={params.parallel_enabled} aria-label="同层并行"
                onChange={(e) => setParams((p) => ({ ...p, parallel_enabled: e.target.checked }))} />
              同层并行
            </label>
          </details>
          <div className="note">参数写入本次运行；历史回放显示当时的快照。</div>
          <details>
            <summary className="section-label">历史对比</summary>
            <select className="select" style={{ width: '100%', marginBottom: 6 }} value={cmpA} onChange={(e) => setCmpA(e.target.value)} aria-label="运行 A">
              <option value="">运行 A</option>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.task.slice(0, 14)}</option>)}
            </select>
            <select className="select" style={{ width: '100%', marginBottom: 6 }} value={cmpB} onChange={(e) => setCmpB(e.target.value)} aria-label="运行 B">
              <option value="">运行 B</option>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.task.slice(0, 14)}</option>)}
            </select>
            <button className="btn" style={{ width: '100%', justifyContent: 'center' }} onClick={doCompare}>对比并分析</button>
          </details>
        </aside>

        <div className="center">
          <div className="run-strip">
            <span className="mono">PHASE</span><b>{now.title || '—'}</b>
            <span className="sep" />
            <span className="mono">BATCH</span><b>{batchCount ? `${currentBatch + 1}/${batchCount}` : '—'}</b>
            <span className="sep" />
            <span className="mono">AGENTS</span><b>{plan.length ? `${agentDone}/${plan.length}` : '—'}</b>
            <span className="sep" />
            <span className="mono">ELAPSED</span><b>{(upto / 1000).toFixed(1)}s</b>
            <span className="spacer" />
            <span className={`live-dot ${running ? 'on' : ''}`} />
            <span className="mono">{running ? 'RUNNING' : live ? 'LIVE' : 'REPLAY'}</span>
          </div>
          <div className="canvas">
            <FlowCanvas nodes={nodes} edges={edges} onNodeClick={(_, n) => setSelected(n.id)} />
            {ragOpen && <RagPip stage={ragStage} onClose={() => setRagOpen(false)} />}
            {selNode && (
              <div className="node-sheet">
                <div className="panel-head">
                  {selNode.name}
                  <button className="btn" style={{ marginLeft: 'auto', padding: 2, minHeight: 20 }}
                    aria-label="关闭节点详情" onClick={() => setSelected(null)}><X size={11} /></button>
                </div>
                <div style={{ padding: '8px 10px', fontSize: 10.5, lineHeight: 1.6 }}>
                  <div className="mono" style={{ fontSize: 8.5, color: 'var(--muted)' }}>{selNode.key} · {selNode.layer}</div>
                  <div style={{ marginTop: 5 }}>{selNode.role}</div>
                  <div className="mono" style={{ fontSize: 8.5, marginTop: 6, color: 'var(--muted)' }}>
                    工具：{selNode.tools.join(', ') || '—'}
                    {selNode.gate && <> · 闸门：{selNode.gate_name}</>}
                  </div>
                </div>
              </div>
            )}
            <div className="legend">
              <span><i className="a" />当前</span><span><i className="d" />完成</span><span><i className="g" />闸门</span>
            </div>
          </div>

          <div className="transport">
            <button className="btn" onClick={() => advance(-1)} disabled={live} aria-label="上一步"><SkipBack size={12} /></button>
            <button className="btn" onClick={() => { if (live) setLive(false); setPlaying((p) => !p) }} aria-label={playing ? '暂停' : '播放'}>
              {playing ? <Pause size={12} /> : <Play size={12} />}
            </button>
            <button className="btn" onClick={() => advance(1)} disabled={live} aria-label="下一步"><SkipForward size={12} /></button>
            <div className="rail" role="slider" aria-label="回放进度" aria-valuemin={0} aria-valuemax={Math.round(totalMs)}
              aria-valuenow={Math.round(playMs)} tabIndex={0}
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect()
                const p = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
                setLive(false); setPlaying(false); setPlayMs(p * totalMs)
              }}>
              <div className="rail-fill" style={{ width: `${pct}%` }} />
            </div>
            <span className="mono" style={{ fontSize: 9, color: 'var(--muted)' }}>
              {live ? `LIVE ${(totalMs / 1000).toFixed(1)}s` : `${(playMs / 1000).toFixed(1)} / ${(totalMs / 1000).toFixed(1)}s`}
            </span>
            <select className="select" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="回放速度">
              {[0.1, 0.25, 0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
            <button className="btn" onClick={() => { setLive(true); setPlaying(false); setPlayMs(totalMs) }}>
              <RotateCcw size={12} /> 实时
            </button>
          </div>
        </div>

        <aside className={`rightbar ${showRight ? 'open' : ''}`}>
          <div style={{ padding: 10 }}>
            <div className="now">
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <span className="badge">{now.badge}</span>
                <span className="now-title">{now.title}</span>
                <span className="mono" style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--muted)' }}>{now.time}</span>
              </div>
              <div className="now-detail">{now.detail}</div>
            </div>
            <textarea className="input" style={{ marginTop: 8, height: 44, resize: 'none' }}
              value={task} onChange={(e) => setTask(e.target.value)} aria-label="任务描述" />
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <button className="btn primary" style={{ flex: 1, justifyContent: 'center' }} onClick={start} disabled={running} aria-busy={running}>启动</button>
              {running && <button className="btn danger" onClick={stop}>停止</button>}
            </div>
            {historyError && <p role="alert">{historyError}</p>}
          </div>

          <div className="panel" style={{ flex: 1, margin: '0 10px', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="panel-head" role="tablist">
              <button role="tab" aria-selected={rightTab === 'events'} className={`tab ${rightTab === 'events' ? 'on' : ''}`}
                onClick={() => setRightTab('events')}>事件</button>
              <button role="tab" aria-selected={rightTab === 'output'} className={`tab ${rightTab === 'output' ? 'on' : ''}`}
                onClick={() => setRightTab('output')}>输出</button>
              {rightTab === 'events' && (
                <button className="tab" style={{ marginLeft: 'auto' }}
                  onClick={() => setEventFilter((f) => (f === 'key' ? 'all' : 'key'))}>
                  {eventFilter === 'key' ? '关键' : '全部'}
                </button>
              )}
            </div>
            {rightTab === 'events' ? (
              <div className="panel-body" style={{ flex: 1 }} role="log" aria-live="polite" aria-relevant="additions">
                {shownEvents.map((e, i) => (
                  <div className="ev" key={i}>
                    <span className="t">{(e.t / 1000).toFixed(1)}s</span>
                    <span className={`k ${e.ev.event === 'error' ? 'err' : e.ev.event === 'result' ? 'ok' : ''}`}>
                      {EVENT_LABEL[e.ev.event] || e.ev.event}
                      {e.ev.event === 'stage' ? (e.ev.status === 'done' ? '·完成' : '·开始') : ''}
                    </span>
                    <span className="s">{e.ev.name || e.ev.agent || e.ev.title || e.ev.detail || ''}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="panel-body" style={{ flex: 1, padding: '8px 10px', fontSize: 10.5, lineHeight: 1.65, whiteSpace: 'pre-wrap' }}>
                {output || '完成后显示'}
              </div>
            )}
          </div>
        </aside>
      </div>

      {compare && (
        <div role="dialog" aria-modal="true" aria-label="运行对比"
          style={{ position: 'fixed', inset: 40, zIndex: 90, background: 'var(--paper-2)', border: '1px solid var(--line-strong)', borderRadius: 8, boxShadow: '0 24px 70px rgba(19,32,29,.18)', display: 'flex', flexDirection: 'column' }}>
          <div className="panel-head" style={{ fontSize: 12, padding: '10px 14px' }}>
            运行对比
            <button className="btn" style={{ marginLeft: 'auto', padding: 2, minHeight: 22 }} aria-label="关闭对比" onClick={() => setCompare(null)}><X size={13} /></button>
          </div>
          <div style={{ padding: 16, overflow: 'auto', fontSize: 11, lineHeight: 1.7 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {[['A', compare.a], ['B', compare.b]].map(([k, m]: any) => (
                <div key={k} className="panel" style={{ padding: 10 }}>
                  <div className="section-label">运行 {k}</div>
                  <div>{m.task}</div>
                  <div className="mono" style={{ fontSize: 9, color: 'var(--muted)', marginTop: 4 }}>{JSON.stringify(m.params)}</div>
                  <div className="mono" style={{ fontSize: 10, marginTop: 6 }}>
                    总耗时 {(m.total_ms / 1000).toFixed(1)}s · 结果 {m.result_len} 字 · 事件 {m.events}
                  </div>
                </div>
              ))}
            </div>
            <div className="panel" style={{ marginTop: 12, padding: 10 }}>
              <div className="section-label">指标对比（青 = A，铜 = B）</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={cmpData} margin={{ top: 8, right: 8, bottom: 48, left: 0 }}>
                  <CartesianGrid stroke="#e6e1d8" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-25} textAnchor="end" interval={0} height={60} />
                  <YAxis tick={{ fontSize: 9 }} width={40} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="A" fill="#0b6b5e" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="B" fill="#94662b" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ marginTop: 14, whiteSpace: 'pre-wrap' }}>{compare.report}</div>
          </div>
        </div>
      )}
    </div>
  )
}

function ParamRow(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="param-row">
      <span>{props.label}</span>
      <input type="range" aria-label={props.label} style={{ flex: 1, accentColor: 'var(--signal)' }}
        min={props.min} max={props.max} step={props.step} value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))} />
      <b>{props.step < 1 ? props.value.toFixed(2) : props.value}</b>
    </label>
  )
}
