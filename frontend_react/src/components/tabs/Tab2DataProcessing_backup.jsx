import React, { useState } from 'react'
import './Tab2DataProcessing.css'

export default function Tab2DataProcessing({ sessionState, setSessionState }) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [showTerminal, setShowTerminal] = useState(false)
  const [modelRFE, setModelRFE] = useState('XGBoost')
  const [numFeatures, setNumFeatures] = useState(5)

  const handleDataPrep = () => {
    setIsProcessing(true)
    setShowTerminal(true)
    
    // Simulate processing
    setTimeout(() => {
      setIsProcessing(false)
    }, 3000)
  }

  const handleRFE = () => {
    setIsProcessing(true)
    setShowTerminal(true)
    
    // Simulate RFE processing
    setTimeout(() => {
      setIsProcessing(false)
    }, 3000)
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
            disabled={isProcessing}
          >
            RFE
          </button>
        </div>

        <div className="rfe-controls">
          <div className="control-group">
            <label htmlFor="model-select">Select Model for RFE</label>
            <select 
              id="model-select"
              value={modelRFE}
              onChange={(e) => setModelRFE(e.target.value)}
              disabled={isProcessing}
              className="model-dropdown"
            >
              <option value="XGBoost">XGBoost</option>
              <option value="Random Forest">Random Forest</option>
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="features-input">Number of features to select</label>
            <input 
              id="features-input"
              type="number"
              min="1"
              max="50"
              value={numFeatures}
              onChange={(e) => setNumFeatures(parseInt(e.target.value))}
              disabled={isProcessing}
              className="features-input"
            />
          </div>
        </div>

        {showTerminal && (
          <div className="terminal-window">
            <div className="term-header">
              <div className="term-button close"></div>
              <div className="term-button minimize"></div>
              <div className="term-button maximize"></div>
            </div>
            <div className="term-body">
              <p className="term-line delay-1">&gt; system start data_prep pipeline</p>
              <p className="term-line delay-2">&gt; Initializing ERA5 Data Pipeline...</p>
              <p className="term-line delay-3">&gt; Downloading and merging spatial grids...</p>
              <p className={`term-line ${isProcessing ? 'delay-4' : 'instant'}`}>
                &gt; Computing parameters... {isProcessing && <span className="cursor-blink">█</span>}
              </p>
              {!isProcessing && (
                <p className="term-line success">&gt; ✓ Data preparation complete!</p>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="info-box">
        <h4>Process Overview</h4>
        <ul>
          <li>✓ Download ERA5 and GRACE data</li>
          <li>✓ Merge spatial grids</li>
          <li>✓ Perform Recursive Feature Elimination (RFE)</li>
          <li>✓ Rank climate predictors by importance</li>
        </ul>
      </div>
    </div>
  )
}
