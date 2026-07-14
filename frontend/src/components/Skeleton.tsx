interface Props {
  className?: string
}

/** Single shimmering rectangle. */
export function Skeleton({ className = '' }: Props) {
  return (
    <div
      className={`rounded-lg bg-base-border animate-pulse ${className}`}
      style={{ opacity: 0.5 }}
    />
  )
}

/** Full dashboard loading layout — mirrors the actual Overview content. */
export function DashboardSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">

      {/* Header */}
      <div className="flex items-start gap-3 mb-5">
        <Skeleton className="w-24 h-14 flex-shrink-0" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="h-3 w-1/3" />
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card py-4 space-y-2">
            <Skeleton className="h-8 w-1/2 mx-auto" />
            <Skeleton className="h-3 w-2/3 mx-auto" />
          </div>
        ))}
      </div>

      {/* Tab bar */}
      <div className="flex gap-6 border-b border-base-border pb-0 mb-6">
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-4 w-16 mb-3" />
        ))}
      </div>

      {/* Row 1: two donut charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="card space-y-3">
            <Skeleton className="h-3 w-32" />
            <Skeleton className="h-44 w-44 rounded-full mx-auto" />
            <div className="grid grid-cols-3 gap-2">
              {[...Array(3)].map((_, j) => <Skeleton key={j} className="h-8" />)}
            </div>
          </div>
        ))}
      </div>

      {/* Row 2: timeline */}
      <div className="card space-y-3">
        <Skeleton className="h-3 w-40" />
        <Skeleton className="h-48" />
      </div>

      {/* Row 3: topics */}
      <div className="card space-y-3">
        <Skeleton className="h-3 w-36" />
        <Skeleton className="h-52" />
      </div>

    </div>
  )
}

/** History page loading layout — search/filter row + a handful of job rows. */
export function HistorySkeleton() {
  return (
    <div className="space-y-3 animate-pulse">

      {/* Search + status filter row */}
      <div className="flex flex-col sm:flex-row gap-2 mb-4">
        <Skeleton className="h-9 flex-1" />
        <div className="flex gap-1.5 flex-shrink-0">
          {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-9 w-20" />)}
        </div>
      </div>

      {/* Job rows */}
      {[...Array(5)].map((_, i) => (
        <div key={i} className="card flex items-start gap-4">
          <Skeleton className="w-5 h-5 rounded-full flex-shrink-0 mt-0.5" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-3 w-1/3" />
            <div className="flex gap-2 mt-2">
              <Skeleton className="h-5 w-16 rounded-full" />
              <Skeleton className="h-5 w-14" />
              <Skeleton className="h-5 w-16" />
            </div>
          </div>
          <Skeleton className="h-8 w-20 flex-shrink-0" />
        </div>
      ))}
    </div>
  )
}

/** Evaluate page loading layout — mirrors the stats/metrics/matrix sections. */
export function EvaluateSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">

      {/* Dataset overview stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card text-center py-3 space-y-2">
            <Skeleton className="h-7 w-12 mx-auto" />
            <Skeleton className="h-3 w-16 mx-auto" />
          </div>
        ))}
      </div>

      {/* XLM vs VADER metric cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="card space-y-3">
            <Skeleton className="h-3 w-28" />
            {[...Array(4)].map((_, j) => <Skeleton key={j} className="h-6" />)}
          </div>
        ))}
      </div>

      {/* Improvement badge */}
      <div className="card h-20" />

      {/* Confusion matrices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="card space-y-2">
            <Skeleton className="h-3 w-40" />
            <Skeleton className="h-40" />
          </div>
        ))}
      </div>
    </div>
  )
}
