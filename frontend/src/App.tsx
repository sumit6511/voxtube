import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Home      from './pages/Home'
import Progress  from './pages/Progress'
import Dashboard from './pages/Dashboard'
import Evaluate  from './pages/Evaluate'
import History   from './pages/History'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                  element={<Home />} />
        <Route path="/progress/:jobId"   element={<Progress />} />
        <Route path="/dashboard/:jobId"  element={<Dashboard />} />
        <Route path="/evaluate"          element={<Evaluate />} />
        <Route path="/history"           element={<History />} />
        <Route path="*"                  element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
