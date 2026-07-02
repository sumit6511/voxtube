import { useRef } from 'react'
import { Download } from 'lucide-react'

interface Props {
  title: string
  filename: string
  children: React.ReactNode
  className?: string
}

async function exportToPng(container: HTMLDivElement, filename: string) {
  const svg = container.querySelector('svg')
  if (!svg) return

  const clone = svg.cloneNode(true) as SVGSVGElement
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  rect.setAttribute('width', '100%')
  rect.setAttribute('height', '100%')
  rect.setAttribute('fill', 'white')
  clone.insertBefore(rect, clone.firstChild)

  const { width, height } = svg.getBoundingClientRect()
  clone.setAttribute('width', String(Math.ceil(width)))
  clone.setAttribute('height', String(Math.ceil(height)))

  const svgStr  = new XMLSerializer().serializeToString(clone)
  const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' })
  const url     = URL.createObjectURL(svgBlob)

  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    const scale  = 2
    canvas.width  = Math.ceil(width)  * scale
    canvas.height = Math.ceil(height) * scale
    const ctx = canvas.getContext('2d')!
    ctx.scale(scale, scale)
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)

    const a = document.createElement('a')
    a.href = canvas.toDataURL('image/png')
    a.download = `${filename}.png`
    a.click()
    URL.revokeObjectURL(url)
  }
  img.onerror = () => URL.revokeObjectURL(url)
  img.src = url
}

export default function ChartCard({ title, filename, children, className = '' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <p className="label">{title}</p>
        <button
          onClick={() => containerRef.current && exportToPng(containerRef.current, filename)}
          title="Download as PNG"
          className="flex items-center gap-1 text-xs font-mono text-gray-600
                     hover:text-amber transition-colors px-2 py-1 rounded
                     border border-transparent hover:border-base-border"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <div ref={containerRef}>
        {children}
      </div>
    </div>
  )
}
