import { useState } from 'react'
import AudioRecorder from './components/AudioRecorder'
import Results from './components/Results'
import { predictAll } from './api'
import './App.css'

export default function App() {
  const [results, setResults]     = useState(null)
  const [transcript, setTranscript] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  const handleAudioReady = async (audioBlob, filename) => {
    setLoading(true)
    setError(null)
    setResults(null)
    setTranscript(null)

    try {
      const data = await predictAll(audioBlob, filename)

      // Izvuci transkript iz PULSE rezultata
      const pulse = data.results.find(r => r.model === 'PULSE_emotion')
      if (pulse?.transcribed_text) {
        setTranscript(pulse.transcribed_text)
      }

      setResults(data.results)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="app-header">
        <h1>Emotion Recognition</h1>
        <p className="subtitle">Snimi ili uploadaj audio za analizu emocija</p>
      </header>

      {/* ── Recorder ── */}
      <AudioRecorder onAudioReady={handleAudioReady} />

      {/* ── Loading ── */}
      {loading && (
        <div className="status-loading">
          <span className="spinner" />
          Analyzing audio...
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <p className="status-error">❌ {error}</p>
      )}


      {/* ── Rezultati ── */}
      {results && !loading && (
        <Results results={results} />
      )}

    </div>
  )
}
