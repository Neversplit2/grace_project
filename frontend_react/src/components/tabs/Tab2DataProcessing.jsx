import React, { useState } from 'react'
import './Tab2DataProcessing.css'
import { dataProcessingApi } from '../../services/api'

export default function Tab2DataProcessing({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [showTerminal, setShowTerminal] = useState(false)
  const [modelRFE, setModelRFE] = useState('RF')
  const [numFeatures, setNumFeatures] = useState(5)
  const [terminalLines, setTerminalLines] = useState([])
  const [progress, setProgress] = useState(0)

  const handleDataPrep = async () => {
    if (!sessionId) {
      setTerminalLines(['> ✗ Session not ready'])
      setShowTerminal(true)
      return
    }

    setIsProcessing(true)
    setShowTerminal(true)
    setTerminalLines([
      '> system start data_prep pipeline',
      '> Checking session data...'
    ])

    try {
      const response = await dataProcessingApi.prepareData(sessionId)
      
      setTerminalLines(prev => [
        ...prev,
        '> ✓ Data validated!',
        `> Available features: ${response.data.available_features}`,
        `> Samples: ${response.data.n_samples}`,
        '> Ready for RFE...'
      ])
      setIsProcessing(false)
    } catch (err) {
      setTerminalLines(prev => [...prev, `> ✗ Error: ${err.message}`])
      setIsProcessing(false)
    }
  }

  const handleRFE = async () => {
    if (!sessionId) {
      setTerminalLines(['> ✗ Session not ready'])
      setShowTerminal(true)
      return
    }

    setIsProcessing(true)
    setShowTerminal(true)
    setProgress(0)
    setTerminalLines([
      `> Starting RFE with ${modelRFE} model...`,
      `> Selecting top ${numFeatures} features...`,
      '> This may take 2-5 minutes...'
    ])

    try {
      const result = await dataProcessingApi.runRFE(
        sessionId,
        modelRFE,
        numFeatures,
        (progressValue, status) => {
          setProgress(progressValue)
          setTerminalLines(prev => [...prev, `> ${progressValue}% - ${status}`])
        }
      )

      setTerminalLines(prev => [
        ...prev,
        '> ✓ RFE completed!',
        `> Selected features: ${result.selected_features.join(', ')}`,
        `> Total: ${result.n_features_selected}/${result.n_features_total}`
      ])

      setSessionState(prev => ({
        ...prev,
        rfeCompleted: true,
        selectedFeatures: result.selected_features,
      }))

      setIsProcessing(false)
      
      // Auto-advance to Tab 3
      setTimeout(() => setActiveTab(2), 2000)
      
    } catch (err) {
      setTerminalLines(prev => [...prev, `> ✗ Error: ${err.message}`])
      setIsProcessing(false)
    }
  }

  return (
    <div className="tab-container">
      <h3 className="tab-title">Data Preparation & RFE Analysis</h3>

      <p className="section-description">
        <strong>Run the data preparation pipeline to rank the best ERA5 features.</strong>
      </p>

      <div className="processing-layout">
        <div className="button-section">
          <button 
            className={`primary-btn ${isProcessing ? 'disabled' : ''}`}
            onClick={handleDataPrep}
            disabled={isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Data Prep'}
          </button>
          
          <button 
            className={`secondary-btn ${isProcessing ? 'disabled' : ''}`}
            onClick={handleRFE}
            disabled={isProcessing || sessionState?.rfeCompleted}
          >
            {isProcessing ? 'Running RFE...' : sessionState?.rfeCompleted ? '✅ RFE Complete' : 'RFE'}
          </button>
        </div>

        <div className="rfe-controls">
          <div className="control-group">
            <label htmlFor="model-select">Select Model for RFE</label>
            <select 
              id="model-select"
              value={modelRFE}
              onChange={(e) => setModelRFE(e.target.value)}
              disabled={isProcessing || sessionState?.rfeCompleted}
              className="model-dropdown"
            >
              <option value="RF">Random Forest</option>
              <option value="XGBoost">XGBoost</option>
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="features-input">Number of features to select</label>
            <input 
              id="features-input"
              type="number"
              min="1"
              max="15"
              value={numFeatures}
              onChange={(e) => setNumFeatures(parseInt(e.target.value))}
              disabled={isProcessing || sessionState?.rfeCompleted}
              className="features-input"
            />
          </div>
        </div>

        {progress > 0 && isProcessing && (
          <div style={{ marginTop: '15px' }}>
            <div style={{ 
              width: '100%', 
              height: '20px', 
              background: '#1a1a1a', 
              borderRadius: '10px',
              overflow: 'hidden',
              border: '1px solid #00E5FF'
            }}>
              <div style={{ 
                width: `${progress}%`, 
                height: '100%', 
                background: 'linear-gradient(90deg, #00E5FF, #FF00FF)',
                transition: 'width 0.3s'
              }} />
            </div>
            <p style={{ textAlign: 'center', color: '#00E5FF', marginTop: '5px' }}>{progress}%</p>
          </div>
        )}

        {showTerminal && (
          <div className="terminal-window">
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
              {isProcessing && (
                <p className="term-line">
                  &gt; Processing... <span className="cursor-blink">█</span>
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>Process Overview</h4>
        <ul>
          <li>{sessionState?.dataLoaded ? '✓' : '○'} Download ERA5 and GRACE data</li>
          <li>{sessionState?.dataLoaded ? '✓' : '○'} Merge spatial grids</li>
          <li>{sessionState?.rfeCompleted ? '✓' : '○'} Perform Recursive Feature Elimination (RFE)</li>
          <li>{sessionState?.rfeCompleted ? '✓' : '○'} Rank climate predictors by importance</li>
        </ul>
      </div>
    </div>
  )
}
