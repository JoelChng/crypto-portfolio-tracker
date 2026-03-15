import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Portfolio from './pages/Portfolio'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/portfolio/:address" element={<Portfolio />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
