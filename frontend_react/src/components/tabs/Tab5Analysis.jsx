import React, { useState } from 'react'
import './Tab5Analysis.css'
import { analysisApi, trainingApi } from '../../services/api'

export default function Tab5Analysis({ sessionId, sessionState, setSessionState, setActiveTab }) {
  // Get bounds from Tab 1
  const rawLatMin = sessionState?.bounds?.lat_min || -17.0
  const rawLatMax = sessionState?.bounds?.lat_max || 5.0
  const rawLonMin = sessionState?.bounds?.lon_min || -80.0
  const rawLonMax = sessionState?.bounds?.lon_max || -50.0

  // Force to 0.1 resolution grid without artificially shrinking the bounds
  const limitLatMin = Math.round(rawLatMin * 10) / 10
  const limitLatMax = Math.round(rawLatMax * 10) / 10
  const limitLonMin = Math.round(rawLonMin * 10) / 10
  const limitLonMax = Math.round(rawLonMax * 10) / 10

  // State management
  const [latitude, setLatitude] = useState(limitLatMin)
  const [longitude, setLongitude] = useState(limitLonMin)
  const [startYear, setStartYear] = useState(2020)
  const [endYear, setEndYear] = useState(2021)
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [selectedVisualization, setSelectedVisualization] = useState(null) // 'evaluation' | 'importance'
  const [visualizationData, setVisualizationData] = useState(null)
  const [error, setError] = useState(null)
  const [terminalLines, setTerminalLines] = useState([])

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    if (!sessionId) {
      setError('Session not ready')
      return
    }

    setIsUploading(true)
    setError(null)
    setTerminalLines(prev => [...prev, `> Uploading model file: ${file.name}...`])

    try {
      await trainingApi.uploadModel(sessionId, file)
      
      setSessionState(prev => ({
        ...prev,
        modelTrained: true,
        modelInfo: {
          model_name: file.name
        }
      }))
      
      setTerminalLines(prev => [...prev, `> ✓ Model ${file.name} uploaded successfully!`])
    } catch (err) {
      setError(err.message)
      setTerminalLines(prev => [...prev, `> ✗ Error uploading model: ${err.message}`])
    } finally {
      setIsUploading(false)
    }
  }

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

      // The image string from backend already includes the data:image/png;base64, prefix
      const plotData = result.data.plot.replace('data:image/png;base64,', '');

      setVisualizationData({
        type: 'evaluation',
        plotBase64: plotData,
        rScore: result.data.statistics.r_score?.toFixed(4) || 'N/A',
        rmse: 'N/A', // Not returned by backend
        mae: 'N/A',  // Not returned by backend
        latitude: latitude.toFixed(2),
        longitude: longitude.toFixed(2),
        startYear,
        endYear,
      })

      setTerminalLines(prev => [
        ...prev,
        '> ✓ Evaluation completed!',
        `> R-score: ${result.data.statistics.r_score?.toFixed(4)}`,
        `> P-value: ${result.data.statistics.p_value?.toExponential(2) || 'N/A'}`
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

      const plotData = result.data.plot.replace('data:image/png;base64,', '');

      setVisualizationData({
        type: 'importance',
        plotBase64: plotData,
        features: result.data.features || []
      })

      setTerminalLines(prev => [
        ...prev,
        '> ✓ Feature importance calculated!',
        `> Top feature: ${result.data.features?.[0]?.feature || 'N/A'}`,
        `> Total features analyzed: ${result.data.features?.length || 0}`
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

          <div className="stats-section" style={{ marginTop: '20px' }}>
            <h4 className="section-heading">Deploy model for Analysis</h4>
            <div className="upload-section">
              <p className="model-info" style={{ fontSize: '11px', color: '#8892B0', margin: '0 0 5px 0', fontFamily: 'monospace' }}>
                {sessionState?.modelTrained 
                  ? `✓ Using trained model: ${sessionState.modelInfo?.model_name || 'Unknown'}`
                  : '⚠ No model trained yet - complete Tab 3 or upload one below'
                }
              </p>
              
              <div className="file-upload-wrapper">
                <input 
                  type="file" 
                  accept=".pkl" 
                  onChange={handleFileUpload}
                  disabled={isUploading || isLoading}
                  id="model-upload-tab5"
                  style={{ display: 'none' }}
                />
                <label htmlFor="model-upload-tab5" style={{ 
                    cursor: 'pointer', 
                    display: 'inline-block',
                    padding: '4px 8px',
                    border: '1px solid #8892B0',
                    borderRadius: '4px',
                    color: '#8892B0',
                    fontFamily: 'monospace',
                    fontSize: '10px'
                  }}>
                  {isUploading ? '⏳ Uploading...' : '📁 Browse for .pkl Model'}
                </label>
              </div>
            </div>
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

        {/* RIGHT PANEL: TERMINAL */}
        <div className="terminal-panel-wrapper" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '400px' }}>
          <div className="terminal-window" style={{ flex: 1, margin: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="term-header">
              <div className="term-button close"></div>
              <div className="term-button minimize"></div>
              <div className="term-button maximize"></div>
            </div>
            <div className="term-body" style={{ flex: 1, overflowY: 'auto' }}>
              {terminalLines.length === 0 && (
                <p className="term-line" style={{ color: '#626A7F' }}>
                  &gt; System ready. Configure analysis parameters... <span className="cursor-blink">█</span>
                </p>
              )}
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
        </div>
      </div>

      {/* BOTTOM PANEL: VISUALIZATION */}
      <div className="stats-visualization-panel" style={{ marginTop: '30px', height: 'auto', minHeight: '500px' }}>
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
                  style={{ width: '100%', maxHeight: '800px', objectFit: 'contain', borderRadius: '8px' }}
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
                  ? `R-score: ${visualizationData.rScore} | Period: ${visualizationData.startYear}-${visualizationData.endYear}`
                  : `Feature importance based on ${sessionState.modelInfo?.model_name || 'trained model'}`
                }
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
