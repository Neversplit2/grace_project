import React, { useState, useEffect } from 'react'
import './Tab1Setup.css'
import PlotlyComponent from '../PlotlyComponent'
import { setupApi } from '../../services/api'

export default function Tab1Setup({ sessionId, sessionState, setSessionState, setActiveTab }) {
  const [latMin, setLatMin] = useState(-17.0)
  const [latMax, setLatMax] = useState(5.0)
  const [lonMin, setLonMin] = useState(-80.0)
  const [lonMax, setLonMax] = useState(-50.0)
  const [basinName, setBasinName] = useState('Amazon')
  const [graceDataset, setGraceDataset] = useState('CSR')
  const [coastlines, setCoastlines] = useState({ x: [], y: [], z: [] })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [validationStatus, setValidationStatus] = useState(null)

  // Fetch coastlines on mount
  useEffect(() => {
    const loadCoastlines = async () => {
      try {
        const response = await setupApi.getCoastlines()
        const data = response.data
        
        const xs = [], ys = [], zs = []
        
        const getXyz = (lon, lat, radius = 1.01) => {
          const latRad = (lat * Math.PI) / 180
          const lonRad = (lon * Math.PI) / 180
          return [
            radius * Math.cos(latRad) * Math.cos(lonRad),
            radius * Math.cos(latRad) * Math.sin(lonRad),
            radius * Math.sin(latRad)
          ]
        }
        
        const addLine = (coords) => {
          for (const [lon, lat] of coords) {
            const [x, y, z] = getXyz(lon, lat)
            xs.push(x)
            ys.push(y)
            zs.push(z)
          }
          xs.push(null)
          ys.push(null)
          zs.push(null)
        }
        
        for (const feature of data.features || []) {
          const geom = feature.geometry
          if (!geom) continue
          if (geom.type === 'Polygon') {
            for (const poly of geom.coordinates) addLine(poly)
          } else if (geom.type === 'MultiPolygon') {
            for (const multipoly of geom.coordinates) {
              for (const poly of multipoly) addLine(poly)
            }
          }
        }
        
        setCoastlines({ x: xs, y: ys, z: zs })
      } catch (err) {
        console.error('Failed to load coastlines:', err)
        setError('Failed to load map data')
      }
    }

    loadCoastlines()
  }, [])

  // Validate bounds
  const handleValidateBounds = async () => {
    setError(null)
    setValidationStatus(null)
    
    try {
      const bounds = {
        lat_min: latMin,
        lat_max: latMax,
        lon_min: lonMin,
        lon_max: lonMax,
      }

      const response = await setupApi.validateBounds(sessionId, bounds)
      
      if (response.status === 'success') {
        setValidationStatus('✅ Bounds are valid!')
        setSuccess('Bounds validated successfully. You can now load data.')
      }
    } catch (err) {
      setError(`❌ Validation failed: ${err.message}`)
      setValidationStatus('❌ Invalid bounds')
    }
  }

  // Load data
  const handleLoadData = async () => {
    setError(null)
    setSuccess(null)
    setIsLoading(true)

    try {
      const bounds = {
        lat_min: latMin,
        lat_max: latMax,
        lon_min: lonMin,
        lon_max: lonMax,
      }

      const response = await setupApi.loadData(sessionId, bounds, graceDataset)

      if (response.status === 'success') {
        const data = response.data
        
        // Update session state
        setSessionState(prev => ({
          ...prev,
          dataLoaded: true,
          bounds: data.bounds,
          graceDataset: data.grace_dataset,
          dateRange: data.date_range,
          eraVariables: data.era5_variables,
        }))

        setSuccess(`✅ Data loaded successfully!
          - ERA5 shape: ${data.era5_shape.join('x')}
          - GRACE shape: ${data.csr_shape.join('x')}
          - Date range: ${data.date_range.start} to ${data.date_range.end}
          - Variables: ${data.era5_variables.length} features`)
        
        // Auto-advance to next tab after 2 seconds
        setTimeout(() => {
          setActiveTab(1)
        }, 2000)
      }
    } catch (err) {
      setError(`❌ Failed to load data: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="tab-container">
      <h3 className="tab-title">Define Area Of Interest</h3>

      {/* Status Messages */}
      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}
      {success && (
        <div className="alert alert-success">
          <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{success}</pre>
        </div>
      )}
      {sessionState.dataLoaded && (
        <div className="alert alert-info">
          ✅ Data is loaded. You can proceed to Tab 2 (Data Processing).
        </div>
      )}

      <div className="layout-grid">
        <div className="input-section">
          <div className="input-group">
            <label className="input-label">Latitude Bounds</label>
            <div className="input-row">
              <div className="input-field">
                <span className="field-label">Min</span>
                <input
                  type="number"
                  value={latMin}
                  onChange={(e) => setLatMin(parseFloat(e.target.value))}
                  step={0.25}
                  className="number-input"
                  disabled={isLoading}
                />
              </div>
              <div className="input-field">
                <span className="field-label">Max</span>
                <input
                  type="number"
                  value={latMax}
                  onChange={(e) => setLatMax(parseFloat(e.target.value))}
                  step={0.25}
                  className="number-input"
                  disabled={isLoading}
                />
              </div>
            </div>
            <div className="hint">Valid range: -90 to 90 degrees</div>
          </div>

          <div className="input-group">
            <label className="input-label">Longitude Bounds</label>
            <div className="input-row">
              <div className="input-field">
                <span className="field-label">Min</span>
                <input
                  type="number"
                  value={lonMin}
                  onChange={(e) => setLonMin(parseFloat(e.target.value))}
                  step={0.25}
                  className="number-input"
                  disabled={isLoading}
                />
              </div>
              <div className="input-field">
                <span className="field-label">Max</span>
                <input
                  type="number"
                  value={lonMax}
                  onChange={(e) => setLonMax(parseFloat(e.target.value))}
                  step={0.25}
                  className="number-input"
                  disabled={isLoading}
                />
              </div>
            </div>
            <div className="hint">Valid range: -180 to 180 degrees</div>
          </div>

          <div className="input-group">
            <label className="input-label">Target Region & Data Source Selection</label>
            
            <div className="form-field">
              <label htmlFor="basin-name" className="form-label">Basin/Region Name</label>
              <input
                id="basin-name"
                type="text"
                value={basinName}
                onChange={(e) => setBasinName(e.target.value)}
                placeholder="Region/Basin name will appear on your map titles."
                className="text-input"
                disabled={isLoading}
              />
            </div>

            <div className="form-field">
              <label htmlFor="grace-dataset" className="form-label">GRACE Dataset</label>
              <select
                id="grace-dataset"
                value={graceDataset}
                onChange={(e) => setGraceDataset(e.target.value)}
                className="select-input"
                disabled={isLoading}
              >
                <option value="CSR">CSR (Center for Space Research)</option>
                <option value="JPL">JPL (Jet Propulsion Laboratory)</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="action-buttons">
            <button
              onClick={handleValidateBounds}
              disabled={isLoading}
              className="btn btn-secondary"
            >
              🔍 Validate Bounds
            </button>
            {validationStatus && (
              <span className="validation-status">{validationStatus}</span>
            )}
          </div>

          <div className="action-buttons">
            <button
              onClick={handleLoadData}
              disabled={isLoading || sessionState.dataLoaded}
              className="btn btn-primary"
            >
              {isLoading ? '⏳ Loading Data...' : '📥 Load Data'}
            </button>
          </div>

          {isLoading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Loading GRACE and ERA5 data... This may take 1-3 minutes.</p>
            </div>
          )}
        </div>

        <div className="globe-section">
          <PlotlyComponent 
            latMin={latMin}
            latMax={latMax}
            lonMin={lonMin}
            lonMax={lonMax}
            basinName={basinName}
            coastlines={coastlines}
          />
        </div>
      </div>
    </div>
  )
}
