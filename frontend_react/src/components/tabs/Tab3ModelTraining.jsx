import React, { useState } from 'react'
import './Tab3ModelTraining.css'
import { trainingApi } from '../../services/api'

export default function Tab3ModelTraining({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [modelType, setModelType] = useState('RF')
  const [trainType, setTrainType] = useState('Quick')
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('Ready')
  const [terminalLines, setTerminalLines] = useState([])

  const handleStartTraining = async () => {
    if (!sessionId) {
      setStatus('Error: Session not ready')
      return
    }

    if (!sessionState?.rfeCompleted) {
      setStatus('Error: Please complete Tab 2 (RFE) first')
      setTerminalLines(['> ✗ RFE must be completed before training'])
      return
    }

    setIsTraining(true)
    setProgress(0)
    setStatus('Training...')
    setTerminalLines([
      `> Starting ${modelType} model training...`,
      `> Training type: ${trainType}`,
      '> This may take 5-15 minutes...'
    ])

    try {
      const result = await trainingApi.trainModel(
        sessionId,
        modelType,
        trainType,
        (progressValue, statusMsg) => {
          setProgress(progressValue)
          setStatus(statusMsg)
          setTerminalLines(prev => [...prev, `> ${progressValue}% - ${statusMsg}`])
        }
      )

      setTerminalLines(prev => [
        ...prev,
        '> ✓ Training completed!',
        `> Model saved: ${result.model_name}`,
        `> Training samples: ${result.training_info.n_samples}`,
        `> Features used: ${result.training_info.n_features}`
      ])

      setSessionState(prev => ({
        ...prev,
        modelTrained: true,
        modelInfo: result
      }))

      setStatus('✅ Training Complete')
      setIsTraining(false)

      // Auto-advance to Tab 4
      setTimeout(() => setActiveTab(3), 2000)

    } catch (err) {
      setTerminalLines(prev => [...prev, `> ✗ Error: ${err.message}`])
      setStatus(`Error: ${err.message}`)
      setIsTraining(false)
    }
  }

  return (
    <div className="tab-container">
      <h3 className="tab-title">Model Training Configuration</h3>

      <p className="section-description">
        <strong>Train your downscaling model (Optional: Skip if you have a pre-trained model)</strong>
      </p>

      <div className="training-config">
        <div className="config-card">
          <label className="config-label">Model Type</label>
          <select 
            value={modelType} 
            onChange={(e) => setModelType(e.target.value)}
            className="config-select"
            disabled={isTraining || sessionState?.modelTrained}
          >
            <option value="RF">Random Forest</option>
            <option value="XGBoost">XGBoost</option>
          </select>
        </div>

        <div className="config-card">
          <label className="config-label">Training Type</label>
          <select 
            value={trainType} 
            onChange={(e) => setTrainType(e.target.value)}
            className="config-select"
            disabled={isTraining || sessionState?.modelTrained}
          >
            <option value="Quick">Quick (Faster, basic params)</option>
            <option value="Hyper">Hyper (Slower, optimized params)</option>
          </select>
        </div>
      </div>

      <div className="training-actions">
        <button 
          className="primary-btn"
          onClick={handleStartTraining}
          disabled={isTraining || sessionState?.modelTrained || !sessionState?.rfeCompleted}
        >
          {isTraining ? '⏳ Training...' : sessionState?.modelTrained ? '✅ Training Complete' : '▶ Start Training'}
        </button>
      </div>

      <div className="training-progress">
        <div className="progress-header">
          <h4>Training Progress</h4>
          <span className="status" style={{ color: status.includes('Error') ? '#ff0050' : '#00E5FF' }}>
            {status}
          </span>
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <p className="progress-text">{progress}% complete</p>
      </div>

      {terminalLines.length > 0 && (
        <div className="terminal-window" style={{ marginTop: '20px' }}>
          <div className="term-header">
            <div className="term-button close"></div>
            <div className="term-button minimize"></div>
            <div className="term-button maximize"></div>
          </div>
          <div className="term-body">
            {terminalLines.map((line, idx) => (
              <p key={idx} className="term-line" style={{ animationDelay: `${idx * 0.1}s` }}>
                {line}
              </p>
            ))}
            {isTraining && (
              <p className="term-line">
                &gt; Training in progress... <span className="cursor-blink">█</span>
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
