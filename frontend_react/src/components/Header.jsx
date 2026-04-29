import React from 'react'
import './Header.css'

export default function Header() {
  return (
    <header className="header">
      <h1 className="main-title">
        DOWNSCALING ENGINE FOR GRACE & GRACE-FO LWE DATA
      </h1>
      <div className="header-credits">
        ENGINEERED & DESIGNED BY:
        <div className="credit-item">
          <span className="cyber-tooltip">
            NEVERSPLIT
            <span className="tooltip-text">
              <span className="credential">[NEVERSPLIT]</span><br />
              Alexandros Karachles<br />
              <i>akarachle@topo.auth.gr</i>
            </span>
          </span>
        </div>
        &nbsp;|&nbsp;
        <div className="credit-item">
          <span className="cyber-tooltip">
            ANASTRIA-LAB
            <span className="tooltip-text">
              <span className="credential">[ANASTRIA-LAB]</span><br />
              Anastasia I. Triantafyllou, PhDc.<br />
              <i>anastria@topo.auth.gr</i>
            </span>
          </span>
        </div>
      </div>
    </header>
  )
}
