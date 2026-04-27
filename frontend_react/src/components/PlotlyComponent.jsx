import React, { useEffect, useRef } from 'react'

export default function PlotlyComponent({ latMin, latMax, lonMin, lonMax, basinName, coastlines }) {
  const plotRef = useRef(null)

  useEffect(() => {
    // Load Plotly from CDN
    if (!window.Plotly) {
      const script = document.createElement('script')
      script.src = 'https://cdn.plot.ly/plotly-latest.min.js'
      script.async = true
      script.onload = () => renderPlot()
      document.head.appendChild(script)
    } else {
      renderPlot()
    }

    function renderPlot() {
      if (!plotRef.current) return

      const getXyz = (lon, lat, radius = 1.0) => {
        const latRad = (lat * Math.PI) / 180
        const lonRad = (lon * Math.PI) / 180
        return [
          radius * Math.cos(latRad) * Math.cos(lonRad),
          radius * Math.cos(latRad) * Math.sin(lonRad),
          radius * Math.sin(latRad)
        ]
      }

      // Calculate coordinates
      const centerLon = (lonMin + lonMax) / 2
      const centerLat = (latMin + latMax) / 2

      const [x1, y1, z1] = getXyz(lonMin, latMin)
      const [x2, y2, z2] = getXyz(lonMax, latMin)
      const [x3, y3, z3] = getXyz(lonMax, latMax)
      const [x4, y4, z4] = getXyz(lonMin, latMax)
      const [sx, sy, sz] = getXyz(centerLon, centerLat, 1.6)

      // Create Earth sphere with better resolution
      const u = []
      const v = []
      for (let i = 0; i <= 30; i++) u.push((2 * Math.PI * i) / 30)  // Close the loop by adding wrap-around
      for (let i = 0; i < 15; i++) v.push((Math.PI * i) / 15)

      const xSph = u.map(u_val => v.map(v_val => Math.cos(u_val) * Math.sin(v_val)))
      const ySph = u.map(u_val => v.map(v_val => Math.sin(u_val) * Math.sin(v_val)))
      const zSph = u.map(u_val => v.map(v_val => Math.cos(v_val)))

      // Build plot data
      const data = [
        // Earth sphere
        {
          type: 'surface',
          x: xSph,
          y: ySph,
          z: zSph,
          colorscale: [[0, '#0E1117'], [1, '#1A1C23']],
          opacity: 1,
          showscale: false,
          hoverinfo: 'skip',
          name: 'Earth'
        }
      ]

      // Add coastlines if available
      if (coastlines.x && coastlines.x.length > 0) {
        data.push({
          type: 'scatter3d',
          x: coastlines.x,
          y: coastlines.y,
          z: coastlines.z,
          mode: 'lines',
          line: { color: '#4A5568', width: 1.5 },
          hoverinfo: 'skip',
          name: 'Coastlines'
        })
      }

      // Target region boundary
      data.push({
        type: 'scatter3d',
        x: [x1, x2, x3, x4, x1],
        y: [y1, y2, y3, y4, y1],
        z: [z1, z2, z3, z4, z1],
        mode: 'lines',
        line: { color: '#00E5FF', width: 5 },
        hoverinfo: 'skip',
        name: 'Target Region'
      })

      // Pyramid lines from satellite to corners
      const corners = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
      corners.forEach(([cx, cy, cz]) => {
        data.push({
          type: 'scatter3d',
          x: [sx, cx],
          y: [sy, cy],
          z: [sz, cz],
          mode: 'lines',
          line: { color: 'rgba(0, 229, 255, 0.4)', width: 3, dash: 'dash' },
          hoverinfo: 'skip',
          name: 'Hologram'
        })
      })

      // Satellite marker
      data.push({
        type: 'scatter3d',
        x: [sx],
        y: [sy],
        z: [sz],
        mode: 'markers+text',
        marker: { size: 12, color: '#00E5FF', symbol: 'diamond' },
        text: ['🛰️'],
        textposition: 'top center',
        textfont: { color: '#00E5FF', size: 32 },
        hoverinfo: 'text',
        hovertext: `GRACE Satellite<br>Region: ${basinName}`,
        name: 'Satellite'
      })

      // Layout
      const layout = {
        title: {
          text: `🛰️ 3D Globe - ${basinName}<br><sub>Lat: ${latMin.toFixed(1)}° to ${latMax.toFixed(1)}° | Lon: ${lonMin.toFixed(1)}° to ${lonMax.toFixed(1)}°</sub>`,
          x: 0.5,
          xanchor: 'center',
          font: { color: '#00E5FF', size: 16 }
        },
        autosize: true,
        height: 600,
        margin: { r: 0, t: 80, l: 0, b: 0 },
        paper_bgcolor: '#0E1117',
        plot_bgcolor: '#0E1117',
        font: { color: '#A0AEC0', family: 'monospace' },
        showlegend: false,
        scene: {
          xaxis: { visible: false },
          yaxis: { visible: false },
          zaxis: { visible: false },
          camera: {
            eye: { x: sx * 0.8, y: sy * 0.8, z: sz * 0.8 }
          },
          bgcolor: '#0E1117'
        }
      }

      const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['hoverclosest', 'hovercompare'],
        toImageButtonOptions: { format: 'png', filename: `grace_globe_${basinName}` }
      }

      window.Plotly.newPlot(plotRef.current, data, layout, config)
    }

    return () => {
      if (plotRef.current && window.Plotly) {
        window.Plotly.purge(plotRef.current)
      }
    }
  }, [latMin, latMax, lonMin, lonMax, basinName, coastlines])

  return (
    <div 
      ref={plotRef} 
      style={{ 
        width: '100%', 
        height: '600px',
        borderRadius: '8px',
        overflow: 'hidden'
      }}
    />
  )
}
