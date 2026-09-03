#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  DocMind Agent · Non-Human Identity Edition — Start Script
# ─────────────────────────────────────────────────────────────
set -e

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   DocMind Agent · NHI Edition             ║"
echo "  ║   Scoped agent identity + Ollama AI        ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found."
  exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt -q --break-system-packages 2>/dev/null || pip install -r requirements.txt -q

if command -v ollama &>/dev/null; then
  echo "✅ Ollama found"
  if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "🚀 Starting Ollama server in background..."
    ollama serve &>/dev/null &
    sleep 2
  fi
  echo ""
  echo "📋 Available Ollama models:"
  ollama list 2>/dev/null || echo "   (none — run: ollama pull llama3)"
else
  echo "⚠️  Ollama not found — will use Claude API fallback if ANTHROPIC_API_KEY is set"
  echo "   Install Ollama: https://ollama.com then run: ollama pull llama3"
fi

mkdir -p uploads logs static

echo ""
echo "🌐 Starting server at http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
