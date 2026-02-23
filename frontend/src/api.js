const BASE = '/api'

export async function predictAll(audioBlob, filename = 'recording.wav') {
  const form = new FormData()
  form.append('audio', audioBlob, filename)
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}

export async function predictLSTM(audioBlob, filename = 'recording.wav') {
  const form = new FormData()
  form.append('audio', audioBlob, filename)
  const res = await fetch(`${BASE}/predict/lstm`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}

export async function predictTransformer(audioBlob, filename = 'recording.wav') {
  const form = new FormData()
  form.append('audio', audioBlob, filename)
  const res = await fetch(`${BASE}/predict/transformer`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}

export async function predictPulse(audioBlob, filename = 'recording.wav') {
  const form = new FormData()
  form.append('audio', audioBlob, filename)
  const res = await fetch(`${BASE}/predict/pulse`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Server error: ${res.status}`)
  return res.json()
}
