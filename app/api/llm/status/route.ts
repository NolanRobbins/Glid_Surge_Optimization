import { NextResponse } from 'next/server'

// FastAPI backend (default 8002 in demo mode; may be overridden via env)
const API_BASE = process.env.API_BASE_URL || 'http://localhost:8002'
// Direct NIM/vLLM endpoint for status checks (default 8001 in demo mode; may be overridden via env)
const LLM_DIRECT_URL = process.env.LLM_DIRECT_URL || 'http://localhost:8001'

export async function GET() {
  try {
    // Try FastAPI backend first
    const response = await fetch(`${API_BASE}/llm/status`, {
      cache: 'no-store',
    })
    
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
    
    // Fallback: check NIM/vLLM directly
    const directResponse = await fetch(`${LLM_DIRECT_URL}/v1/models`, {
      cache: 'no-store',
    })
    
    if (directResponse.ok) {
      return NextResponse.json({
        available: true,
        model: 'nemotron-49b',
        modes: ['assistant', 'route_optimizer', 'surge_analyst'],
        features: ['reasoning', 'tool_calling', '128k_context'],
        source: 'direct'
      })
    }
    
    return NextResponse.json({ available: false }, { status: 200 })
  } catch (error) {
    console.error('LLM status check failed:', error)
    return NextResponse.json({ available: false }, { status: 200 })
  }
}

