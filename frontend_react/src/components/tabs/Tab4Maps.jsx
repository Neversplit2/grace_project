import React, { useState } from 'react'
import './Tab4Maps.css'
import { mapsApi } from '../../services/api'

export default function Tab4Maps({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [year, setYear] = useState(2020)
  const [month, setMonth] = useState(5)
  const [eraVariable, setEraVariable] = useState('t2m')
  const [colormap, setColormap] = useState('viridis')
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentMapType, setCurrentMapType] = useState(null)
  const [mapImage, setMapImage] = useState(null)
  const [error, setError] = useState(null)
  const [terminalLines, setTerminalLines] = useState([])

  const handleGenerateERA5Map = async () => {
    if (!sessionId) {
      setError('Session not ready')
      return
    }

    if (!sessionState?.dataLoaded) {
      setError('Please complete Tab 1 (data loading) first')
      setTerminalLines(['> ✗ Data must be loaded before generating maps'])
      return
    }

    setIsGenerating(true)
    setCurrentMapType('era5')
    setError(null)
    setTerminalLines([
      `> Generating ERA5 map...`,
      `> Variable: ${eraVariable}`,
      `> Date: ${year}-${String(month).padStart(2, '0')}`,
      `> Colormap: ${colormap}`,
      '> Please wait...'
    ])

    try {
      const result = await mapsApi.generateERA5Map(
        sessionId,
        eraVariable,
        year,
        month,
        colormap
      )

      setMapImage(result.map_base64)
      setTerminalLines(prev => [
        ...prev,
        '> ✓ ERA5 map generated successfully!',
        `> Resolution: ${result.resolution || 'Native'}`,
        `> File ready for download`
      ])

      setSessionState(prev => ({
        ...prev,
        era5MapGenerated: true
      }))

    } catch (err) {
      setError(err.message)
      setTerminalLines(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    } finally {
      setIsGenerating(false)
    }
  }

  const handleGenerateGRACEMap = async () => {
    if (!sessionId) {
      setError('Session not ready')
      return
    }

    if (!sessionState?.modelTrained) {
      setError('Please complete Tab 3 (model training) first')
      setTerminalLines(['> ✗ Model must be trained before generating GRACE comparison maps'])
      return
    }

    setIsGenerating(true)
    setCurrentMapType('grace')
    setError(null)
    setTerminalLines([
      `> Generating GRACE comparison map...`,
      `> Date: ${year}-${String(month).padStart(2, '0')}`,
      `> Using trained model: ${sessionState.modelInfo?.model_name || 'Unknown'}`,
      `> Colormap: ${colormap}`,
      '> Please wait...'
    ])

    try {
      const result = await mapsApi.generateGRACEMap(
        sessionId,
        year,
        month,
        colormap
      )

      setMapImage(result.map_base64)
      setTerminalLines(prev => [
        ...prev,
        '> ✓ GRACE comparison map generated!',
        `> Model predictions vs actual GRACE data`,
        `> File ready for download`
      ])

      setSessionState(prev => ({
        ...prev,
        graceMapGenerated: true
      }))

    } catch (err) {
      setError(err.message)
      setTerminalLines(prev => [
        ...prev,
        `> ✗ Error: ${err.message}`
      ])
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownloadMap = () => {
    if (!mapImage) return

    const filename = currentMapType === 'era5' 
      ? `ERA5_${eraVariable}_${year}_${String(month).padStart(2, '0')}.png`
      : `GRACE_Comparison_${year}_${String(month).padStart(2, '0')}.png`
    
    // Create download link from base64 image
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${mapImage}`
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
    <div className="tab-container maps-container">
      <h3 className="tab-title">Geospatial Rendering Engine</h3>

      <div className="maps-layout">
        {/* LEFT SIDE: CONTROLS */}
        <div className="maps-controls-panel">
          
          {/* SECTION 1: ERA5 Controls */}
          <div className="maps-section">
            <h4 className="section-heading">Select an ERA5 variable to render</h4>
            
            <div className="era5-controls">
              <div className="control-group">
                <label htmlFor="year-input">Year for Map</label>
                <div className="input-spinner">
                  <button onClick={() => setYear(year - 1)} disabled={isGenerating}>−</button>
                  <input 
                    id="year-input"
                    type="number"
                    value={year}
                    onChange={(e) => setYear(parseInt(e.target.value))}
                    readOnly
                  />
                  <button onClick={() => setYear(year + 1)} disabled={isGenerating}>+</button>
                </div>
              </div>

              <div className="control-group">
                <label htmlFor="month-input">Month for Map</label>
                <div className="input-spinner">
                  <button 
                    onClick={() => setMonth(month === 1 ? 12 : month - 1)}
                    disabled={isGenerating}
                  >−</button>
                  <input 
                    id="month-input"
                    type="number"
                    value={month}
                    onChange={(e) => setMonth(parseInt(e.target.value))}
                    min="1"
                    max="12"
                    readOnly
                  />
                  <button 
                    onClick={() => setMonth(month === 12 ? 1 : month + 1)}
                    disabled={isGenerating}
                  >+</button>
                </div>
              </div>

              <div className="control-group">
                <label htmlFor="era-variable">ERA5 Variable</label>
                <select 
                  id="era-variable"
                  value={eraVariable}
                  onChange={(e) => setEraVariable(e.target.value)}
                  className="control-select"
                  disabled={isGenerating}
                >
                  <option value="t2m">t2m (Temperature 2m)</option>
                  <option value="tp">tp (Total Precipitation)</option>
                  <option value="u10">u10 (U-wind 10m)</option>
                  <option value="v10">v10 (V-wind 10m)</option>
                  <option value="sp">sp (Surface Pressure)</option>
                </select>
              </div>

              <div className="control-group">
                <label htmlFor="colormap">Colormap</label>
                <select 
                  id="colormap"
                  value={colormap}
                  onChange={(e) => setColormap(e.target.value)}
                  className="control-select"
                  disabled={isGenerating}
                >
                  <option value="viridis">Viridis</option>
                  <option value="plasma">Plasma</option>
                  <option value="coolwarm">Cool-Warm</option>
                  <option value="RdBu_r">Red-Blue</option>
                </select>
              </div>

              <div className="info-icon" title="Select ERA5 climate variable">ⓘ</div>
            </div>

            <button 
              className={`primary-btn era5-btn ${isGenerating && currentMapType === 'era5' ? 'loading' : ''}`}
              onClick={handleGenerateERA5Map}
              disabled={isGenerating || !sessionState?.dataLoaded}
            >
              {isGenerating && currentMapType === 'era5' ? '⏳ Generating...' : '🗺️ Generate ERA5 Map'}
            </button>
          </div>

          {/* SECTION 2: GRACE Model Controls */}
          <div className="maps-section">
            <h4 className="section-heading">Deploy model for GRACE comparison analysis</h4>
            
            <div className="upload-section">
              <p className="model-info">
                {sessionState?.modelTrained 
                  ? `✓ Using trained model: ${sessionState.modelInfo?.model_name || 'Unknown'}`
                  : '⚠ No model trained yet - complete Tab 3 first'
                }
              </p>
            </div>

            <button 
              className={`primary-btn grace-btn ${isGenerating && currentMapType === 'grace' ? 'loading' : ''}`}
              onClick={handleGenerateGRACEMap}
              disabled={isGenerating || !sessionState?.modelTrained}
            >
              {isGenerating && currentMapType === 'grace' ? '⏳ Generating...' : '🌍 Generate GRACE Comparison'}
            </button>
          </div>
        </div>

        {/* RIGHT SIDE: VISUALIZATION */}
        <div className="maps-visualization-panel">
          {mapImage ? (
            <div className="map-display-area">
              <div className="map-header">
                <h4 className="map-title">
                  {currentMapType === 'era5' 
                    ? `ERA5 Map - ${eraVariable.toUpperCase()} (${month}/${year})`
                    : `GRACE Comparison - ${month}/${year}`
                  }
                </h4>
                <button 
                  className="download-btn"
                  onClick={handleDownloadMap}
                  title="Download map as PNG"
                >
                  ↓
                </button>
              </div>
              <div className="map-canvas">
                <img 
                  src={`data:image/png;base64,${mapImage}`} 
                  alt={`${currentMapType} map`}
                  style={{ width: '100%', height: 'auto', borderRadius: '8px' }}
                />
              </div>
              <div className="map-info">
                <p className="info-text">Map ready for analysis and download</p>
              </div>
            </div>
          ) : (
            <div className="map-placeholder-empty">
              <div className="placeholder-content">
                <p className="placeholder-icon">🗺️</p>
                <p className="placeholder-text">No map generated yet</p>
                <p className="placeholder-subtext">
                  {!sessionState?.dataLoaded 
                    ? 'Complete Tab 1 (data loading) first'
                    : 'Select ERA5 variables or generate GRACE comparison to render maps'
                  }
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message" style={{ 
              marginTop: '20px', 
              padding: '15px', 
              background: 'rgba(255, 0, 80, 0.1)', 
              border: '1px solid #ff0050',
              borderRadius: '8px',
              color: '#ff0050'
            }}>
              ✗ {error}
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
            {isGenerating && (
              <p className="term-line">
                &gt; Generating map... <span className="cursor-blink">█</span>
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
