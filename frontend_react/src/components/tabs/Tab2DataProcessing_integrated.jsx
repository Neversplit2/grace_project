import React, { useState, useEffect } from 'react'
import './Tab2DataProcessing.css'
import { dataProcessingApi } from '../../services/api'

export default function Tab2DataProcessing({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [showTerminal, setShowTerminal] = useState(false)
  const [modelRFE, setModelRFE] = useState('RF')
  const [numFeatures, setNumFeatures] = useState(5)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [terminalMessages, setTerminalMessages] = useState([])
  const [progress, setProgress] = useState(0)
  const [availableFeatures, setAvailableFeatures] = useState(null)
  const [selectedFeatures, setSelectedFeatures] = useState(null)

  // Check if data is loaded from Tab 1
  useEffect(() => {
    if (!sessionState.dataLoaded) {
      setError('⚠️ Please complete Tab 1 first. Data must be loaded before processing.')
    }
  }, [sessionState.dataLoaded])

  // Handle data preparation
  const handleDataPrep = async () => {
    setError(null)
    setSuccess(null)
    setShowTerminal(true)
    setTerminalMessages([
      '> Checking session data...',
      '> Validating ERA5 and GRACE datasets...'
    ])

    try {
      const response = await dataProcessingApi.prepareData(sessionId)
      
      if (response.status === 'success') {
        const data = response.data
        setAvailableFeatures(data.available_features)
        setTerminalMessages(prev => [
          ...prev,
          `> ✓ Data validated successfully!`,
          `> Available features: ${data.available_features}`,
          `> Samples: ${data.n_samples}`,
          `> Ready for RFE...`
        ])
        setSuccess('✅ Data preparation complete! You can now run RFE.')
      }
    } catch (err) {
      setError(`❌ Data prep failed: ${err.message}`)
      setTerminalMessages(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    }
  }

  // Handle RFE execution
  const handleRFE = async () => {
    setError(null)
    setSuccess(null)
    setIsProcessing(true)
    setShowTerminal(true)
    setProgress(0)
    setTerminalMessages([
      `> Starting RFE with ${modelRFE} model...`,
      `> Selecting top ${numFeatures} features...`,
      '> This may take 2-5 minutes...'
    ])

    try {
      // Run RFE with progress callback
      const result = await dataProcessingApi.runRFE(
        sessionId,
        modelRFE,
        numFeatures,
        (progressValue, status) => {
          setProgress(progressValue)
          setTerminalMessages(prev => [
            ...prev,
            `> Progress: ${progressValue}% - ${status}`
          ])
        }
      )

      // Update terminal with results
      setTerminalMessages(prev => [
        ...prev,
        '> ✓ RFE completed successfully!',
        `> Selected features: ${result.selected_features.join(', ')}`,
        `> Total features: ${result.n_features_selected}/${result.n_features_total}`
      ])

      setSelectedFeatures(result.selected_features)
      
      // Update session state
      setSessionState(prev => ({
        ...prev,
        rfeCompleted: true,
        selectedFeatures: result.selected_features,
        modelType: modelRFE,
        nFeatures: numFeatures
      }))

      setSuccess(`✅ RFE complete! Selected ${result.n_features_selected} features. You can proceed to Tab 3.`)
      
      // Auto-advance to Tab 3 after 3 seconds
      setTimeout(() => {
        setActiveTab(2) // Tab 3 (index 2)
      }, 3000)

    } catch (err) {
      setError(`❌ RFE failed: ${err.message}`)
      setTerminalMessages(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="tab-container">
      <h3 className="tab-title">Data Preparation & RFE Analysis</h3>

      {/* Status Messages */}
      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}
      {success && (
        <div className="alert alert-success">
          {success}
        </div>
      )}
      {sessionState.rfeCompleted && (
        <div className="alert alert-info">
          ✅ RFE is complete. Selected features: {sessionState.selectedFeatures?.join(', ')}
        </div>
      )}

      <p className="section-description">
        <strong>Run the data preparation pipeline to rank the best ERA5 features.</strong>
      </p>

      <div className="processing-layout">
        <div className="button-section">
          <button 
            className={`primary-btn ${isProcessing || !sessionState.dataLoaded ? 'disabled' : ''}`}
            onClick={handleDataPrep}
            disabled={isProcessing || !sessionState.dataLoaded}
          >
            {isProcessing ? 'Processing...' : '📊 Data Prep'}
          </button>
          
          <button 
            className={`secondary-btn ${isProcessing || !availableFeatures || sessionState.rfeCompleted ? 'disabled' : ''}`}
            onClick={handleRFE}
            disabled={isProcessing || !availableFeatures || sessionState.rfeCompleted}
          >
            {isProcessing ? '⏳ Running RFE...' : '🧠 Run RFE'}
          </button>
        </div>

        <div className="rfe-controls">
          <div className="control-group">
            <label htmlFor="model-select">Select Model for RFE</label>
            <select 
              id="model-select"
              value={modelRFE}
              onChange={(e) => setModelRFE(e.target.value)}
              disabled={isProcessing || sessionState.rfeCompleted}
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
              disabled={isProcessing || sessionState.rfeCompleted}
              className="features-input"
            />
            <div className="hint">Recommended: 5-10 features</div>
          </div>

          {availableFeatures && (
            <div className="control-group">
              <label>Available Features</label>
              <div className="feature-count">{availableFeatures} ERA5 variables loaded</div>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {isProcessing && progress > 0 && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="progress-text">{progress}% complete</div>
          </div>
        )}

        {/* Terminal Output */}
        {showTerminal && (
          <div className="terminal-window">
            <div className="term-header">
              <div className="term-button close"></div>
              <div className="term-button minimize"></div>
              <div className="term-button maximize"></div>
              <span className="term-title">RFE Processing Terminal</span>
            </div>
            <div className="term-body">
              {terminalMessages.map((msg, idx) => (
                <p key={idx} className="term-line" style={{ animationDelay: `${idx * 0.1}s` }}>
                  {msg}
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

        {/* Selected Features Display */}
        {selectedFeatures && selectedFeatures.length > 0 && (
          <div className="results-box">
            <h4>✅ Selected Features ({selectedFeatures.length})</h4>
            <div className="feature-list">
              {selectedFeatures.map((feature, idx) => (
                <div key={idx} className="feature-item">
                  <span className="feature-rank">#{idx + 1}</span>
                  <span className="feature-name">{feature}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>Process Overview</h4>
        <ul>
          <li>{sessionState.dataLoaded ? '✅' : '⏳'} Download ERA5 and GRACE data</li>
          <li>{availableFeatures ? '✅' : '⏳'} Validate spatial grids</li>
          <li>{sessionState.rfeCompleted ? '✅' : '⏳'} Perform Recursive Feature Elimination (RFE)</li>
          <li>{selectedFeatures ? '✅' : '⏳'} Rank climate predictors by importance</li>
        </ul>
      </div>
    </div>
  )
}
