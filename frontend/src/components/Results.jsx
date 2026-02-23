const emotionEmoji = {
  // LSTM / Transformer labels
  angry: '😡', calm: '😌', disgust: '🤢', fearful: '😨',
  happy: '😊', neutral: '😐', sad: '😢', surprised: '😲',
  // PULSE labels
  joy: '😄', sadness: '😢', anger: '😡', love: '🥰',
  surprise: '😲',
}

// ── Normalizira sve 3 strukture u jedan format ──
function normalise(r) {
  // LSTM: { model, emotion, confidence (0-1), probabilities: {label: 0-1} }
  if (r.probabilities) {
    return {
      model:       r.model,
      type:        'audio',
      emotion:     r.emotion,
      score:       r.confidence * 100,
      predictions: Object.entries(r.probabilities).map(([label, p]) => ({
        label,
        score: p * 100,
      })),
      transcription: null,
    }
  }

  // Transformer / PULSE: { model, top_emotion, top_score (0-100), all_predictions: [{label, score}] }
  return {
    model:         r.model,
    type:          r.type ?? 'audio',
    emotion:       r.top_emotion  ?? '—',
    score:         r.top_score    ?? 0,
    predictions:   r.all_predictions ?? [],
    transcription: r.transcribed_text ?? null,
    error:         r.error ?? null,
  }
}

export default function Results({ results }) {
  if (!results || results.length === 0) return null

  return (
    <div className="results">
      <h2>Results</h2>
      <div className="results-grid">
        {results.map((raw, i) => {
          const r = normalise(raw)

          // ── Error state ──
          if (r.error) {
            return (
              <div key={i} className="result-card result-card--error">
                <h3>{r.model}</h3>
                <p className="error-msg">❌ {r.error}</p>
              </div>
            )
          }

          const sorted = [...(r.predictions)].sort((a, b) => b.score - a.score)

          return (
            <div key={i} className="result-card">

              {/* Model name + tip */}
              <div className="card-header">
                <h3>{r.model}</h3>
                <span className="model-type">
                  {r.type === 'text' ? '📝 tekst' : '🎵 audio'}
                </span>
              </div>

              {/* Transkript (samo PULSE) */}
              {r.transcription && (
                <p className="transcription">
                  "{r.transcription}"
                </p>
              )}

              {/* Top emocija */}
              <div className="result-emotion">
                <span className="emotion-emoji">
                  {emotionEmoji[r.emotion] ?? '🎭'}
                </span>
                <span className="emotion-label">{r.emotion}</span>
                <span className="emotion-confidence">
                  {r.score.toFixed(1)}%
                </span>
              </div>

              {/* Probability bars */}
              {sorted.length > 0 && (
                <div className="probabilities">
                  {sorted.map(({ label, score }) => (
                    <div key={label} className="prob-row">
                      <span className="prob-label">
                        {emotionEmoji[label] ?? '🎭'} {label}
                      </span>
                      <div className="prob-bar-wrap">
                        <div
                          className="prob-bar"
                          style={{ width: `${score.toFixed(1)}%` }}
                        />
                      </div>
                      <span className="prob-value">
                        {score.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
