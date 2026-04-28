import React, { useState } from 'react'
import './Tab4Maps.css'
import { mapsApi, trainingApi } from '../../services/api'

export default function Tab4Maps({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [year, setYear] = useState(2020)
  const [month, setMonth] = useState(5)
  const [eraVariable, setEraVariable] = useState('t2m')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [currentMapType, setCurrentMapType] = useState(null)
  const [mapImage, setMapImage] = useState(null)
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
      '> Please wait...'
    ])

    try {
      const result = await mapsApi.generateERA5Map(
        sessionId,
        eraVariable,
        year,
        month,
        (progressValue, statusMsg) => {
          setTerminalLines(prev => [...prev, `> ${progressValue}% - ${statusMsg}`])
        }
      )

      setMapImage(result.image)
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
      '> Please wait...'
    ])

    try {
      const result = await mapsApi.generateGRACEMap(
        sessionId,
        year,
        month,
        (progressValue, statusMsg) => {
          setTerminalLines(prev => [...prev, `> ${progressValue}% - ${statusMsg}`])
        }
      )

      setMapImage(result.image)
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
    link.href = mapImage
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
                  <option value="e">e (Evaporation)</option>
                  <option value="pev">pev (Potential Evaporation)</option>
                  <option value="ssro">ssro (Sub-surface Runoff)</option>
                  <option value="sro">sro (Surface Runoff)</option>
                  <option value="evabs">evabs (Evaporation from Bare Soil)</option>
                  <option value="swvl1">swvl1 (Volumetric Soil Water Layer 1)</option>
                  <option value="swvl2">swvl2 (Volumetric Soil Water Layer 2)</option>
                  <option value="swvl3">swvl3 (Volumetric Soil Water Layer 3)</option>
                  <option value="swvl4">swvl4 (Volumetric Soil Water Layer 4)</option>
                  <option value="lai_hv">lai_hv (Leaf Area Index High Veg)</option>
                  <option value="lai_lv">lai_lv (Leaf Area Index Low Veg)</option>
                </select>
              </div>
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

            <div className="upload-section" style={{ marginBottom: '5px' }}>
              <p className="model-info" style={{ fontSize: '11px', color: '#8892B0', margin: '0 0 5px 0', fontFamily: 'monospace' }}>
                {sessionState?.modelTrained
                  ? `✓ Using trained model: ${sessionState.modelInfo?.model_name || 'Unknown'}`
                  : '⚠ No model trained yet - complete Tab 3 or upload one below'
                }
              </p>

              <div className="file-upload-wrapper" style={{ marginTop: '10px' }}>
                <input
                  type="file"
                  accept=".pkl"
                  onChange={handleFileUpload}
                  disabled={isUploading || isGenerating}
                  id="model-upload-tab4"
                  style={{ display: 'none' }}
                />
                <label htmlFor="model-upload-tab4" style={{
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

            <button
              className={`primary-btn grace-btn ${isGenerating && currentMapType === 'grace' ? 'loading' : ''}`}
              onClick={handleGenerateGRACEMap}
              disabled={isGenerating || !sessionState?.modelTrained}
            >
              {isGenerating && currentMapType === 'grace' ? '⏳ Generating...' : '🌍 Generate GRACE Comparison'}
            </button>
          </div>
        </div>

        {/* RIGHT SIDE: TERMINAL */}
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
                  &gt; System ready. Select parameters and generate map... <span className="cursor-blink">█</span>
                </p>
              )}
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
        </div>
      </div>

      {/* BOTTOM SIDE: VISUALIZATION */}
      <div className="maps-visualization-panel" style={{ marginTop: '30px', height: 'auto', minHeight: '500px' }}>
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
                src={mapImage}
                alt={`${currentMapType} map`}
                style={{ width: '100%', maxHeight: '800px', objectFit: 'contain', borderRadius: '8px' }}
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
            margin: '20px',
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
  )
}
