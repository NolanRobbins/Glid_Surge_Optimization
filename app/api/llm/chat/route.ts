import { NextRequest, NextResponse } from 'next/server'

// FastAPI backend (default 8002 in demo mode; may be overridden via env)
const API_BASE = process.env.API_BASE_URL || 'http://localhost:8002'
// Direct NIM/vLLM endpoint (default 8001 in demo mode; may be overridden via env)
const LLM_DIRECT_URL = process.env.LLM_DIRECT_URL || 'http://localhost:8001'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Try FastAPI backend first
    try {
      const response = await fetch(`${API_BASE}/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      
      if (response.ok) {
        const data = await response.json()
        return NextResponse.json(data)
      }
    } catch {
      // Fall through to direct LLM call
    }
    
    // Fallback: Call NIM/vLLM directly
    const messages = [
      { role: 'system', content: getSystemPrompt(body.mode) },
      { role: 'user', content: body.message }
    ]
    
    if (body.context) {
      messages[1].content = `Context:\n${JSON.stringify(body.context, null, 2)}\n\n${body.message}`
    }
    
    const llmResponse = await fetch(`${LLM_DIRECT_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nemotron-49b',
        messages,
        temperature: 0.6,
        top_p: 0.95,
        max_tokens: 4096,
      }),
    })
    
    if (!llmResponse.ok) {
      const error = await llmResponse.text()
      console.error('LLM direct call error:', error)
      return NextResponse.json(
        { error: 'LLM request failed', details: error },
        { status: llmResponse.status }
      )
    }
    
    const llmData = await llmResponse.json()
    const content = llmData.choices?.[0]?.message?.content || ''
    
    // Extract reasoning from <think> tags
    let reasoning: string | undefined
    let cleanContent = content
    if (content.includes('<think>') && content.includes('</think>')) {
      const start = content.indexOf('<think>') + 7
      const end = content.indexOf('</think>')
      reasoning = content.substring(start, end).trim()
      cleanContent = content.substring(end + 8).trim()
    }
    
    return NextResponse.json({
      response: cleanContent,
      reasoning,
      usage: llmData.usage,
    })
  } catch (error) {
    console.error('LLM chat request failed:', error)
    return NextResponse.json(
      { error: 'Failed to connect to LLM service' },
      { status: 503 }
    )
  }
}

function getSystemPrompt(mode: string): string {
  const prompts: Record<string, string> = {
    route_optimizer: `You are an expert logistics route optimizer for the Glid Surge platform.
Analyze multi-modal routes (road + rail) and provide actionable insights.
Be concise and data-driven.`,
    surge_analyst: `You are a surge prediction analyst for the Glid Surge platform.
Interpret GNN model predictions and explain surge patterns at ports.
Recommend optimal dispatch windows to avoid congestion.`,
    assistant: `You are an AI assistant for the Glid Surge logistics optimization platform.
Help users with route planning, surge predictions, and logistics questions.`
  }
  return prompts[mode] || prompts.assistant
}

