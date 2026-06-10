import { create } from 'zustand'
import type { ResultsResponse, StatusResponse } from './api'

interface AppState {
  jobId:   string | null
  status:  StatusResponse | null
  results: ResultsResponse | null
  setJobId:   (id: string) => void
  setStatus:  (s: StatusResponse) => void
  setResults: (r: ResultsResponse) => void
  reset:      () => void
}

export const useStore = create<AppState>(set => ({
  jobId:   null,
  status:  null,
  results: null,
  setJobId:   jobId   => set({ jobId }),
  setStatus:  status  => set({ status }),
  setResults: results => set({ results }),
  reset:      ()      => set({ jobId: null, status: null, results: null }),
}))
