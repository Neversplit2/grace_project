import React, { useState, useEffect, useRef } from 'react'
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

  // Fetch coastlines on mount - using a simpler, faster source
  useEffect(() => {
    // Use Natural Earth data which is more optimized
    fetch('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.geojson')
      .then(res => res.json())
      .then(data => {
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
        
        // Process only features (skip properties to reduce data)
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
      })
      .catch(err => {
        // Fallback to simpler source if Natural Earth fails
        console.log('Using fallback coastline source')
        fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')
          .then(res => res.json())
          .then(data => {
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
          })
          .catch(err => console.log('Coastline load failed:', err))
      })
  }, [])

  // Load data from backend
  const handleLoadData = async () => {
    if (!sessionId) {
      setError('Session not ready. Please wait...')
      return
    }

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
        
        setSessionState(prev => ({
          ...prev,
          dataLoaded: true,
          bounds: data.bounds,
          graceDataset: data.grace_dataset,
        }))

        setSuccess(`✅ Data loaded! ERA5: ${data.era5_shape.join('x')}, GRACE: ${data.csr_shape.join('x')}`)
        
        setTimeout(() => {
          setActiveTab(1) // Go to Tab 2
        }, 2000)
      }
    } catch (err) {
      setError(`❌ Failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="tab-container">
      <h3 className="tab-title">Define Area Of Interest</h3>

      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}
      {sessionState?.dataLoaded && (
        <div className="alert alert-info">✅ Data loaded. Proceed to Tab 2.</div>
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
                />
              </div>
            </div>
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
                />
              </div>
            </div>
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
              />
            </div>

            <div className="form-field">
              <label htmlFor="grace-dataset" className="form-label">GRACE Dataset</label>
              <select
                id="grace-dataset"
                value={graceDataset}
                onChange={(e) => setGraceDataset(e.target.value)}
                className="select-input"
              >
                <option value="CSR">CSR</option>
                <option value="JPL">JPL</option>
              </select>
            </div>
          </div>

          <div className="input-group" style={{ marginTop: '10px' }}>
            <button
              onClick={handleLoadData}
              disabled={isLoading || sessionState?.dataLoaded}
              className="process-button"
            >
              {isLoading ? '⏳ Processing...' : sessionState?.dataLoaded ? '✅ Data Processed' : '🚀 Process Data for Selected Region'}
            </button>
            {isLoading && (
              <div style={{ marginTop: '15px', textAlign: 'center', color: '#00E5FF', fontSize: '14px' }}>
                This may take 1-3 minutes...
              </div>
            )}
          </div>
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
