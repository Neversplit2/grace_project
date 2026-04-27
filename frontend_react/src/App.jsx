import React, { useState, useEffect } from 'react'
import './App.css'
import Header from './components/Header'
import Ticker from './components/Ticker'
import SystemMetrics from './components/SystemMetrics'
import TabNavigation from './components/TabNavigation'
import Tab1Setup from './components/tabs/Tab1Setup'
import Tab2DataProcessing from './components/tabs/Tab2DataProcessing'
import Tab3ModelTraining from './components/tabs/Tab3ModelTraining'
import Tab4Maps from './components/tabs/Tab4Maps'
import Tab5Analysis from './components/tabs/Tab5Analysis'
import { sessionApi } from './services/api'

function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [sessionId, setSessionId] = useState(null)
  const [sessionState, setSessionState] = useState({
    dataLoaded: false,
    rfeCompleted: false,
    modelTrained: false,
    bounds: null,
    selectedFeatures: null,
    modelInfo: null,
  })

  // Create session on mount
  useEffect(() => {
    let mounted = true
    let createdSessionId = null

    const initSession = async () => {
      try {
        const response = await sessionApi.create()
        createdSessionId = response.session_id
        if (mounted) {
          setSessionId(createdSessionId)
          console.log('✅ Session created:', createdSessionId)
        }
      } catch (error) {
        console.error('❌ Failed to create session:', error)
      }
    }

    initSession()

    // Cleanup session on unmount
    return () => {
      mounted = false
      if (createdSessionId) {
        sessionApi.delete(createdSessionId).catch(console.error)
      }
    }
  }, [])

  const tabs = [
    { id: 0, label: '⚙️ 1. Setup & Area of interest', component: Tab1Setup },
    { id: 1, label: '🧠 2. Data Processing', component: Tab2DataProcessing },
    { id: 2, label: '🦾 3. Model Training', component: Tab3ModelTraining },
    { id: 3, label: '🗺️ 4. Maps', component: Tab4Maps },
    { id: 4, label: '📊 5. Statistical Analysis', component: Tab5Analysis },
  ]

  const ActiveComponent = tabs[activeTab].component

  return (
    <div className="app-container">
      <Header />
      <Ticker />
      <SystemMetrics />
      
      <main className="content-wrapper">
        <TabNavigation tabs={tabs} activeTab={activeTab} setActiveTab={setActiveTab} />
        
        <div className="tab-content">
          <ActiveComponent 
            sessionId={sessionId}
            sessionState={sessionState} 
            setSessionState={setSessionState}
            setActiveTab={setActiveTab}
          />
        </div>
      </main>
    </div>
  )
}

export default App
