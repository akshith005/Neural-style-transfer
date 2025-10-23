import React, { useState } from 'react'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onFileChange = (e) => {
    const f = e.target.files[0]
    setFile(f)
    if (f) setPreview(URL.createObjectURL(f))
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    if (!file) return setError('Please choose an image')
    setLoading(true)
    setResultUrl(null)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch('/style', { method: 'POST', body: fd })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Server error')
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setResultUrl(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <div className="logo" />
          <div>
            <div className="title">Neural Style Transfer</div>
            <div className="subtitle">Apply artistic style to your photos</div>
          </div>
        </div>
        <div className="muted">Style image is fixed. Upload content and click Stylize.</div>
      </div>

      <div className="hero card">
        <h1 style={{margin:0}}>Turn your photos into artworks</h1>
        <p style={{marginTop:8}}>Upload a content image and apply the fixed style. High-quality results powered by deep learning.</p>
      </div>

      <div className="card">
        <form onSubmit={onSubmit} className="controls">
          <label className="file-input btn" aria-label="Choose image">
            {file ? file.name : 'Choose content image'}
            <input type="file" accept="image/*" onChange={onFileChange} />
          </label>
          <button className="btn" type="submit" disabled={loading}>Stylize</button>
          <div className="muted">Tip: use a portrait or landscape photo for best results</div>
        </form>
        {error && <p className="error">{error}</p>}

        <div className="images">
          <div className="image-card">
            <h3>Content</h3>
            {preview ? <img src={preview} alt="content" /> : <div className="muted">No content selected</div>}
          </div>

          <div className="image-card">
            <h3>Style (fixed)</h3>
            <img src="/style.png" alt="style" />
          </div>

          <div className="image-card">
            <h3>Output</h3>
            {loading ? <div className="muted">Processing...</div> : (resultUrl ? <img src={resultUrl} alt="output" /> : <div className="muted">No output yet</div>)}
          </div>
        </div>
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="loader" role="status" aria-live="polite">
            <div className="spinner" aria-hidden="true" />
            <div className="text">
              <h2>Server is currently serving 1000s of req — please wait</h2>
              <p className="muted">Processing image. This may take a few minutes depending on server load .</p>
              {/* <p className="caption-small">If it takes too long, try a smaller image or fewer steps.</p> */}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
