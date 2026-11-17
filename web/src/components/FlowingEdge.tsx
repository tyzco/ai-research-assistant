import { BaseEdge, getBezierPath, type EdgeProps } from 'reactflow'

/** 自定义边: 普通边细线; active 时沿线跑一个光点 (程序正在做这件事)。 */
export function FlowingEdge(props: EdgeProps) {
  const {
    id, sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition, markerEnd, style, data,
  } = props
  const [path] = getBezierPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY, targetPosition,
  })
  const status = ((data as { status?: string })?.status) || ''
  const active = status.includes('active')
  const color = status.includes('error') ? '#a13838'
    : active ? '#0b6b5e'
    : status.includes('traversed') ? '#4f8f7d'
    : '#c8c1b5'

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{ ...style, stroke: color, strokeWidth: active ? 1.8 : 1.1,
          strokeDasharray: active ? undefined : undefined }}
      />
      {active && (
        <>
          <circle r="3.2" fill="#0e8b78">
            <animateMotion dur="1.3s" repeatCount="indefinite" path={path} />
          </circle>
          <circle r="1.4" fill="#e6fffb">
            <animateMotion dur="1.3s" repeatCount="indefinite" path={path} />
          </circle>
        </>
      )}
    </>
  )
}
