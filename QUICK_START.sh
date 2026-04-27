#!/bin/bash
# Quick start scripts for GRACE Downscaling Engine

echo "═══════════════════════════════════════════════════════════════"
echo "🛰️  GRACE Downscaling Engine - Quick Start"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if commands exist
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed"
        return 1
    else
        echo "✅ $1 found"
        return 0
    fi
}

echo "Checking dependencies..."
check_command python3
check_command npm
check_command node

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Available Commands:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 STREAMLIT UI:"
echo "   ./start-streamlit.sh      Start Python Streamlit UI"
echo "   URL: http://localhost:8501"
echo ""
echo "⚛️  REACT UI:"
echo "   ./start-react.sh          Start React development server"
echo "   URL: http://localhost:3000"
echo ""
echo "🚀 BOTH UIs:"
echo "   ./start-all.sh            Start both UIs simultaneously"
echo ""
echo "🔨 BUILD REACT:"
echo "   ./build-react.sh          Build React for production"
echo ""
echo "═══════════════════════════════════════════════════════════════"
