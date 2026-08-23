import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ImagePlus,
  LoaderCircle,
  RefreshCw,
  ScanSearch,
  Upload,
  X,
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8001'
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
const MAX_FILE_SIZE = 10 * 1024 * 1024
const PREDICTION_TIMEOUT_MS = 60_000

const CLASS_DETAILS = [
  { key: 'clear', label: 'Clear', color: '#21a179' },
  { key: 'partially_occluded', label: 'Partially Occluded', color: '#e77c39' },
  { key: 'heavily_occluded', label: 'Heavily Occluded', color: '#34495e' },
]

function displayLabel(className) {
  return CLASS_DETAILS.find(({ key }) => key === className)?.label ?? className
}

function formatPercent(probability = 0) {
  return `${(probability * 100).toFixed(1)}%`
}

function ProbabilityRows({ probabilities }) {
  return (
    <div className="probability-list" aria-label="Class probabilities">
      {CLASS_DETAILS.map(({ key, label, color }) => {
        const probability = probabilities[key] ?? 0
        return (
          <div className="probability-row" key={key}>
            <div className="probability-copy">
              <span>{label}</span>
              <strong>{formatPercent(probability)}</strong>
            </div>
            <div
              className="probability-track"
              role="progressbar"
              aria-label={`${label} probability`}
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(probability * 100)}
            >
              <span
                className="probability-fill"
                style={{ width: `${probability * 100}%`, backgroundColor: color }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), 4_000)

    fetch(`${API_URL}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('API health check failed')
        setApiStatus('online')
      })
      .catch(() => setApiStatus('offline'))
      .finally(() => window.clearTimeout(timeoutId))

    return () => {
      controller.abort()
      window.clearTimeout(timeoutId)
    }
  }, [])

  useEffect(() => {
    if (!file) {
      setPreviewUrl('')
      return undefined
    }

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  function chooseFile(selectedFile) {
    setError('')
    setResult(null)

    if (!selectedFile) return
    if (!ACCEPTED_TYPES.includes(selectedFile.type)) {
      setError('Choose a JPEG, PNG, WebP, or BMP image.')
      return
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('Choose an image that is 10 MB or smaller.')
      return
    }

    setFile(selectedFile)
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files[0])
  }

  function reset() {
    setFile(null)
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function predict() {
    if (!file || isLoading) return

    setIsLoading(true)
    setResult(null)
    setError('')

    const formData = new FormData()
    formData.append('file', file)
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), PREDICTION_TIMEOUT_MS)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      const contentType = response.headers.get('content-type') ?? ''
      if (!contentType.includes('application/json')) {
        throw new Error('The API returned an unexpected response. Check the backend URL.')
      }

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? 'Prediction failed. Please try again.')
      }
      setResult(payload)
    } catch (requestError) {
      const fallback = 'Could not reach the PickSense API at port 8001. Start the backend and try again.'
      if (requestError.name === 'AbortError') {
        setError('Prediction timed out after 60 seconds. Check the backend and try again.')
      } else {
        setError(requestError instanceof TypeError ? fallback : requestError.message)
      }
      setApiStatus('offline')
    } finally {
      window.clearTimeout(timeoutId)
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="PickSense home">
          <span className="brand-mark"><ScanSearch size={21} /></span>
          <span>PickSense</span>
        </a>
        <div className="model-status">
          <span className={`status-dot ${apiStatus}`} />
          ViT-B/16 · API {apiStatus}
        </div>
      </header>

      <section className="workspace" aria-labelledby="page-title">
        <div className="intro">
          <p className="eyebrow">Robotic perception experiment</p>
          <h1 id="page-title">How visible is the object?</h1>
          <p>
            Upload an object image and the trained PickSense model will estimate
            its visual occlusion level.
          </p>
        </div>

        <div className="inference-grid">
          <section className="upload-panel" aria-label="Image upload">
            <div className="panel-heading">
              <div>
                <span className="step-number">01</span>
                <h2>Input image</h2>
              </div>
              {file && (
                <button className="icon-button" type="button" onClick={reset} title="Remove image">
                  <X size={18} />
                </button>
              )}
            </div>

            {!file || !previewUrl ? (
              <button
                className={`dropzone ${isDragging ? 'is-dragging' : ''}`}
                type="button"
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault()
                  setIsDragging(true)
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <span className="upload-icon"><ImagePlus size={28} /></span>
                <strong>Drop an object image here</strong>
                <span>or click to browse your files</span>
                <small>JPEG, PNG, WebP or BMP · max 10 MB</small>
              </button>
            ) : (
              <div className="preview-frame">
                <img src={previewUrl} alt="Selected object preview" />
                <div className="preview-meta">
                  <div>
                    <strong>{file.name}</strong>
                    <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                  <CheckCircle2 size={20} />
                </div>
              </div>
            )}

            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept={ACCEPTED_TYPES.join(',')}
              onChange={(event) => chooseFile(event.target.files[0])}
            />

            <button className="primary-button" type="button" disabled={!file || isLoading} onClick={predict}>
              {isLoading ? (
                <><LoaderCircle className="spin" size={19} /> Analyzing image</>
              ) : (
                <><Upload size={19} /> Run prediction <ArrowRight size={18} /></>
              )}
            </button>
          </section>

          <section className="result-panel" aria-live="polite" aria-busy={isLoading}>
            <div className="panel-heading">
              <div>
                <span className="step-number">02</span>
                <h2>Model output</h2>
              </div>
              <span className="confidence-label">softmax confidence</span>
            </div>

            {error ? (
              <div className="state-message error-state">
                <AlertCircle size={28} />
                <strong>Prediction unavailable</strong>
                <p>{error}</p>
              </div>
            ) : isLoading ? (
              <div className="state-message loading-state">
                <div className="scanner"><ScanSearch size={38} /></div>
                <strong>Inspecting visual features</strong>
                <p>Resizing, normalizing, and running ViT inference.</p>
              </div>
            ) : result ? (
              <div className="result-content">
                <div className="prediction-summary">
                  <span>Prediction</span>
                  <h3>{displayLabel(result.prediction)}</h3>
                  <strong>{formatPercent(result.probability)}</strong>
                </div>
                <ProbabilityRows probabilities={result.probabilities} />
                <button className="secondary-button" type="button" onClick={reset}>
                  <RefreshCw size={18} /> Try another image
                </button>
              </div>
            ) : (
              <div className="state-message empty-state">
                <span className="result-icon"><ScanSearch size={32} /></span>
                <strong>Awaiting an image</strong>
                <p>Your prediction and all three class probabilities will appear here.</p>
              </div>
            )}
          </section>
        </div>

        <footer className="scope-note">
          <span>Model scope</span>
          PickSense estimates visual occlusion. It does not yet predict physical grasp success.
        </footer>
      </section>
    </main>
  )
}

export default App