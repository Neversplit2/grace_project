import React from 'react'
import './SystemMetrics.css'

export default function SystemMetrics() {
  const links = [
    { label: '📡JPL GRACE Satellite Download-link', href: 'https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/' },
    { label: '📡CSR GRACE Satellite Download-link', href: 'https://www2.csr.utexas.edu/grace/RL06_mascons.html' },
    { label: '☁️ERA5-Land Download-link', href: 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download' },
  ]

  return (
    <section className="system-metrics">
      <div className="metric-card">
        <div className="metric-label">Primary Target</div>
        <div className="metric-value">LWE Thickness</div>
        <a href={links[0].href} target="_blank" rel="noopener noreferrer" className="metric-link">
          {links[0].label}
        </a>
        <a href={links[1].href} target="_blank" rel="noopener noreferrer" className="metric-link">
          {links[1].label}
        </a>
      </div>

      <div className="metric-card">
        <div className="metric-label">Climate Predictors</div>
        <div className="metric-value">ERA5 Reanalysis</div>
        <a href={links[2].href} target="_blank" rel="noopener noreferrer" className="metric-link">
          {links[2].label}
        </a>
      </div>

      <div className="metric-card">
        <div className="metric-label">Engine resolution</div>
        <div className="metric-value">0.1° (~10km)</div>
        <div className="metric-info">
          📅 Data Range: 2002 - 2024
        </div>
      </div>
    </section>
  )
}
