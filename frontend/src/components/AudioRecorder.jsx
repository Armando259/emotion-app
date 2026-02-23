import { useState, useRef } from 'react'

// ── WAV encoder ──────────────────────────────────────────────────────────────
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(off + i, str.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)          // PCM
  view.setUint16(22, 1, true)          // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  let off = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    off += 2
  }
  return buffer
}

async function toWav16k(blob) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  const arrayBuf = await blob.arrayBuffer()
  const audioBuf = await audioCtx.decodeAudioData(arrayBuf)
  audioCtx.close()

  // Resample na 16 kHz mono (točno što model očekuje)
  const SR = 16000
  const offCtx = new OfflineAudioContext(
    1,
    Math.ceil(audioBuf.duration * SR),
    SR
  )
  const src = offCtx.createBufferSource()
  src.buffer = audioBuf
  src.connect(offCtx.destination)
  src.start()

  const resampled = await offCtx.startRendering()
  const samples = resampled.getChannelData(0)
  return new Blob([encodeWAV(samples, SR)], { type: 'audio/wav' })
}
// ─────────────────────────────────────────────────────────────────────────────

export default function AudioRecorder({ onAudioReady }) {
  const [recording, setRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState(null)   // originalni blob (za player)
  const [audioURL, setAudioURL] = useState(null)
  const [converting, setConverting] = useState(false)
  const mediaRecorder = useRef(null)
  const chunks = useRef([])

  // ── Mikrofon ──
  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorder.current = new MediaRecorder(stream)
      chunks.current = []
      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data)
      }
      mediaRecorder.current.onstop = () => {
        const raw = new Blob(chunks.current, { type: 'audio/webm' })
        setAudioBlob(raw)
        setAudioURL(URL.createObjectURL(raw))
      }
      mediaRecorder.current.start()
      setRecording(true)
      setAudioBlob(null)
      setAudioURL(null)
    } catch (err) {
      alert('Mikrofon nije dostupan: ' + err.message)
    }
  }

  const stop = () => {
    mediaRecorder.current?.stop()
    mediaRecorder.current?.stream.getTracks().forEach((t) => t.stop())
    setRecording(false)
  }

  // ── Upload ──
  const handleUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setAudioBlob(file)
    setAudioURL(URL.createObjectURL(file))
    setRecording(false)
  }

  // ── Reset ──
  const handleReset = () => {
    setAudioBlob(null)
    setAudioURL(null)
  }

  // ── Analyze — konvertira u WAV 16k pa šalje ──
  const handleAnalyze = async () => {
    if (!audioBlob) return
    setConverting(true)
    try {
      const wav = await toWav16k(audioBlob)
      onAudioReady(wav, 'recording.wav')
    } catch (err) {
      alert('Greška pri konverziji: ' + err.message)
    } finally {
      setConverting(false)
    }
  }

  return (
    <div className="recorder">
      {!audioBlob && (
        <div className="recorder-controls">
          {!recording ? (
            <button className="btn btn-record" onClick={start}>
              🎙️ Snimi
            </button>
          ) : (
            <button className="btn btn-stop" onClick={stop}>
              ⏹️ Stop
            </button>
          )}
          {!recording && (
            <>
              <span className="divider">ili</span>
              <label className="btn btn-upload">
                📂 Upload audio
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </>
          )}
        </div>
      )}

      {recording && (
        <div className="recording-indicator">
          <span className="dot" /> Recording...
        </div>
      )}

      {audioURL && !recording && (
        <div className="audio-preview">
          <audio controls src={audioURL} />
          <div className="audio-actions">
            <button
              className="btn btn-analyze"
              onClick={handleAnalyze}
              disabled={converting}
            >
              {converting ? '⏳ Converting...' : '🔍 Analyze'}
            </button>
            <button className="btn btn-reset" onClick={handleReset}>
              🔄 Re-record
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
