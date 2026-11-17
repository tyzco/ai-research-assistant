import type { GraphResp, Rec, RunListItem } from './types'

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new Error(`${url} HTTP ${r.status}`)
  return r.json() as Promise<T>
}

export const fetchGraph = () => json<GraphResp>('/flow/graph')
export const listRuns = () => json<{ runs: RunListItem[] }>('/flow/runs?limit=10')
export const getRun = (id: string) => json<any>(`/flow/runs/${encodeURIComponent(id)}`)
export const listKbs = () => json<{ kbs: { table: string; rows: number }[] }>('/flow/kbs')
export const compareRuns = (a: string, b: string) =>
  json<any>('/flow/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ a, b }),
  })
export const saveRun = (payload: any) =>
  fetch('/flow/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

export async function streamSSE(
  url: string,
  body: any,
  onEvent: (ev: any) => void,
  signal?: AbortSignal,
) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok) throw new Error(`HTTP ${r.status} ${(await r.text()).slice(0, 160)}`)
  const reader = r.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const rd = await reader.read()
    if (rd.done) break
    buf += dec.decode(rd.value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() || ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        onEvent(JSON.parse(line.slice(6)))
      } catch {
        /* ignore malformed chunk */
      }
    }
  }
}

export function compressEvents(rec: Rec[]): Rec[] {
  const out: Rec[] = []
  for (const e of rec) {
    const last = out[out.length - 1]
    if (
      e.ev?.event === 'token' &&
      last?.ev?.event === 'token' &&
      e.t - last.t < 300
    ) {
      last.ev = { ...last.ev, token: (last.ev.token || '') + (e.ev.token || '') }
      last.t = e.t
    } else {
      out.push({ t: e.t, ev: e.ev })
    }
  }
  return out
}
