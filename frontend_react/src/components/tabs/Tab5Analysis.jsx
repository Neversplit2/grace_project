import React, { useState } from 'react'
import './Tab5Analysis.css'
import { analysisApi } from '../../services/api'

export default function Tab5Analysis({ sessionId, sessionState, setSessionState, setActiveTab }) {
  // Get bounds from Tab 1
  const rawLatMin = sessionState?.bounds?.lat_min || -17.0
  const rawLatMax = sessionState?.bounds?.lat_max || 5.0
  const rawLonMin = sessionState?.bounds?.lon_min || -80.0
  const rawLonMax = sessionState?.bounds?.lon_max || -50.0

  // Force to 0.1 resolution grid
  const limitLatMin = Math.round((Math.round(rawLatMin, 1) + 0.1) * 10) / 10
  const limitLatMax = Math.round((Math.round(rawLatMax, 1) - 0.1) * 10) / 10
  const limitLonMin = Math.round((Math.round(rawLonMin, 1) + 0.1) * 10) / 10
  const limitLonMax = Math.round((Math.round(rawLonMax, 1) - 0.1) * 10) / 10

  // State management
  const [latitude, setLatitude] = useState(limitLatMin)
  const [longitude, setLongitude] = useState(limitLonMin)
  const [startYear, setStartYear] = useState(2020)
  const [endYear, setEndYear] = useState(2021)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedVisualization, setSelectedVisualization] = useState(null) // 'evaluation' | 'importance'
  const [visualizationData, setVisualizationData] = useState(null)
  const [error, setError] = useState(null)
  const [terminalLines, setTerminalLines] = useState([])

  const handleEvaluation = async () => {
    if (!sessionId) {
      setError('Session not ready')
      return
    }

    if (!sessionState?.modelTrained) {
      setError('Please complete Tab 3 (model training) first')
      setTerminalLines(['> ✗ Model must be trained before evaluation'])
      return
    }

    setError(null)
    setIsLoading(true)
    setSelectedVisualization('evaluation')
    setTerminalLines([
      `> Running model evaluation...`,
      `> Location: (${latitude.toFixed(2)}, ${longitude.toFixed(2)})`,
      `> Period: ${startYear}-${endYear}`,
      '> Calculating metrics...'
    ])

    try {
      // Validation
      if (latitude < limitLatMin || latitude > limitLatMax || 
          longitude < limitLonMin || longitude > limitLonMax) {
        throw new Error('Invalid coordinates. Please check the bounds.')
      }

      if (endYear <= startYear) {
        throw new Error('End year must be after start year.')
      }

      const result = await analysisApi.evaluateModel(
        sessionId,
        latitude,
        longitude,
        startYear,
        endYear
      )

      setVisualizationData({
        type: 'evaluation',
        plotBase64: result.plot_base64,
        rScore: result.r_score?.toFixed(4) || 'N/A',
        rmse: result.rmse?.toFixed(4) || 'N/A',
        mae: result.mae?.toFixed(4) || 'N/A',
        latitude: latitude.toFixed(2),
        longitude: longitude.toFixed(2),
        startYear,
        endYear,
      })

      setTerminalLines(prev => [
        ...prev,
        '> ✓ Evaluation completed!',
        `> R-score: ${result.r_score?.toFixed(4)}`,
        `> RMSE: ${result.rmse?.toFixed(4)}`,
        `> MAE: ${result.mae?.toFixed(4)}`
      ])

      setSessionState(prev => ({
        ...prev,
        evaluationCompleted: true
      }))

    } catch (err) {
      setError(err.message)
      setTerminalLines(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeatureImportance = async () => {
    if (!sessionId) {
      setError('Session not ready')
      return
    }

    if (!sessionState?.modelTrained) {
      setError('Please complete Tab 3 (model training) first')
      setTerminalLines(['> ✗ Model must be trained before analyzing feature importance'])
      return
    }

    setError(null)
    setIsLoading(true)
    setSelectedVisualization('importance')
    setTerminalLines([
      '> Calculating feature importance...',
      `> Model: ${sessionState.modelInfo?.model_name || 'Unknown'}`,
      '> Analyzing feature contributions...'
    ])

    try {
      const result = await analysisApi.getFeatureImportance(sessionId)

      setVisualizationData({
        type: 'importance',
        plotBase64: result.plot_base64,
        features: result.feature_importance || []
      })

      setTerminalLines(prev => [
        ...prev,
        '> ✓ Feature importance calculated!',
        `> Top feature: ${result.feature_importance?.[0]?.feature || 'N/A'}`,
        `> Total features analyzed: ${result.feature_importance?.length || 0}`
      ])

    } catch (err) {
      setError(err.message)
      setTerminalLines(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!visualizationData?.plotBase64) return

    const filename = selectedVisualization === 'evaluation'
      ? `Model_Evaluation_${latitude}_${longitude}_${startYear}-${endYear}.png`
      : `Feature_Importance.png`
    
    // Create download link from base64 image
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${visualizationData.plotBase64}`
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    setTerminalLines(prev => [
      ...prev,
      `> ✓ Downloaded: ${filename}`
    ])
  }

  return (
    <div className="tab-container stats-container">
      <h3 className="tab-title">Perform Statistical Analysis</h3>

      <div className="stats-layout">
        {/* LEFT PANEL: CONTROLS */}
        <div className="stats-controls-panel">
          
          {/* Point Selection Section */}
          <div className="stats-section">
            <h4 className="section-heading">Choose points of interest</h4>
            
            <div className="stats-input-row">
              <div className="stats-input-group">
                <label htmlFor="latitude">Latitude</label>
                <input
                  id="latitude"
                  type="number"
                  min={limitLatMin}
                  max={limitLatMax}
                  step={0.1}
                  value={latitude}
                  onChange={(e) => setLatitude(parseFloat(e.target.value))}
                  disabled={isLoading}
                  className="stats-input"
                />
              </div>

              <div className="stats-input-group">
                <label htmlFor="longitude">Longitude</label>
                <input
                  id="longitude"
                  type="number"
                  min={limitLonMin}
                  max={limitLonMax}
                  step={0.1}
                  value={longitude}
                  onChange={(e) => setLongitude(parseFloat(e.target.value))}
                  disabled={isLoading}
                  className="stats-input"
                />
              </div>
            </div>

            <div className="stats-input-row">
              <div className="stats-input-group">
                <label htmlFor="startYear">Starting Year</label>
                <input
                  id="startYear"
                  type="number"
                  min={2002}
                  max={2024}
                  value={startYear}
                  onChange={(e) => setStartYear(parseInt(e.target.value))}
                  disabled={isLoading}
                  className="stats-input"
                />
              </div>

              <div className="stats-input-group">
                <label htmlFor="endYear">Ending Year</label>
                <input
                  id="endYear"
                  type="number"
                  min={startYear + 1}
                  max={2025}
                  value={endYear}
                  onChange={(e) => setEndYear(parseInt(e.target.value))}
                  disabled={isLoading}
                  className="stats-input"
                />
              </div>
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="stats-buttons-group">
            <button
              className={`stats-primary-btn ${isLoading && selectedVisualization === 'evaluation' ? 'loading' : ''}`}
              onClick={handleEvaluation}
              disabled={isLoading || !sessionState?.modelTrained}
            >
              {isLoading && selectedVisualization === 'evaluation' ? '⏳ Calculating...' : '📊 Evaluation'}
            </button>

            <button
              className={`stats-primary-btn ${isLoading && selectedVisualization === 'importance' ? 'loading' : ''}`}
              onClick={handleFeatureImportance}
              disabled={isLoading || !sessionState?.modelTrained}
            >
              {isLoading && selectedVisualization === 'importance' ? '⏳ Generating...' : '🥧 Feature Importance Pie'}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL: VISUALIZATION */}
        <div className="stats-visualization-panel">
          {isLoading ? (
            <div className="stats-skeleton-loader">
              <div className="skeleton-content">
                {selectedVisualization === 'evaluation' 
                  ? 'CALCULATING MODEL EVALUATION METRICS' 
                  : 'GENERATING FEATURE IMPORTANCE CHART'}
              </div>
            </div>
          ) : !visualizationData ? (
            <div className="stats-placeholder-empty">
              <div className="placeholder-content">
                <p className="placeholder-icon">📊</p>
                <p className="placeholder-text">No analysis generated yet</p>
                <p className="placeholder-subtext">
                  {!sessionState?.modelTrained 
                    ? 'Complete Tab 3 (model training) first'
                    : 'Select a location and run evaluation to visualize results'
                  }
                </p>
              </div>
            </div>
          ) : (
            <div className="stats-display-area">
              {/* Header with Download */}
              <div className="stats-header">
                <h4 className="stats-map-title">
                  {visualizationData.type === 'evaluation'
                    ? `Model Evaluation - Lat: ${visualizationData.latitude}, Lon: ${visualizationData.longitude}`
                    : 'Feature Importance Distribution'}
                </h4>
                <button
                  className="stats-download-btn"
                  onClick={handleDownload}
                  title="Download visualization as PNG"
                >
                  ↓
                </button>
              </div>

              {/* Visualization Content */}
              <div className="stats-canvas">
                {visualizationData.plotBase64 ? (
                  <img 
                    src={`data:image/png;base64,${visualizationData.plotBase64}`}
                    alt={visualizationData.type === 'evaluation' ? 'Model Evaluation Plot' : 'Feature Importance Plot'}
                    style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
                  />
                ) : (
                  <div className="stats-placeholder-empty">
                    <p>No visualization data available</p>
                  </div>
                )}
              </div>

              {/* Footer Info */}
              <div className="stats-info">
                <p className="stats-info-text">
                  {visualizationData.type === 'evaluation'
                    ? `R-score: ${visualizationData.rScore} | RMSE: ${visualizationData.rmse} | MAE: ${visualizationData.mae} | Period: ${visualizationData.startYear}-${visualizationData.endYear}`
                    : `Feature importance based on ${sessionState.modelInfo?.model_name || 'trained model'}`
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Terminal Output */}
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
            {isLoading && (
              <p className="term-line">
                &gt; {selectedVisualization === 'evaluation' ? 'Calculating...' : 'Generating...'} <span className="cursor-blink">█</span>
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
