'use client'

import { useState, useRef, useEffect, useCallback } from 'react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  timestamp: Date
  isLoading?: boolean
}

interface AIAssistantProps {
  routeContext?: unknown
  surgeContext?: unknown
  onRouteOptimize?: (suggestion: unknown) => void
}

const MODES = {
  assistant: { label: 'Assistant', icon: '💬', color: 'from-blue-500 to-cyan-400' },
  route_optimizer: { label: 'Route Optimizer', icon: '🛤️', color: 'from-emerald-500 to-teal-400' },
  surge_analyst: { label: 'Surge Analyst', icon: '📊', color: 'from-amber-500 to-orange-400' },
}

type LLMChatResponse = {
  response: string
  reasoning?: string
  suggestion?: unknown
}

export default function AIAssistant({ routeContext, surgeContext, onRouteOptimize }: AIAssistantProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<keyof typeof MODES>('assistant')
  const [isLoading, setIsLoading] = useState(false)
  const [llmStatus, setLlmStatus] = useState<{ available: boolean } | null>(null)
  const [showReasoning, setShowReasoning] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Check LLM status on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/llm/status')
        if (res.ok) {
          setLlmStatus(await res.json())
        }
      } catch {
        setLlmStatus({ available: false })
      }
    }
    checkStatus()
  }, [])

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus()
    }
  }, [isOpen])

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Add loading message
    const loadingId = crypto.randomUUID()
    setMessages(prev => [...prev, {
      id: loadingId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    }])

    try {
      const context = mode === 'route_optimizer' && routeContext ? routeContext :
                      mode === 'surge_analyst' && surgeContext ? surgeContext : undefined

      const res = await fetch('/api/llm/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          mode,
          context,
          enable_tools: mode !== 'assistant',
        }),
      })

      if (!res.ok) {
        throw new Error('Failed to get response')
      }

      const data: LLMChatResponse = await res.json()

      setMessages(prev => prev.map(msg =>
        msg.id === loadingId ? {
          ...msg,
          content: data.response,
          reasoning: data.reasoning,
          isLoading: false,
        } : msg
      ))

      if (mode === 'route_optimizer' && onRouteOptimize && data.suggestion !== undefined) {
        onRouteOptimize(data.suggestion)
      }
    } catch {
      setMessages(prev => prev.map(msg =>
        msg.id === loadingId ? {
          ...msg,
          content: 'Sorry, I encountered an error. Please ensure the Nemotron 49B model is running.',
          isLoading: false,
        } : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }, [input, isLoading, mode, routeContext, surgeContext])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-[2000] group"
        title="Open AI Assistant"
      >
        <div className="relative">
          {/* Animated gradient ring */}
          <div className="absolute inset-0 bg-gradient-to-r from-violet-600 via-fuchsia-500 to-amber-500 rounded-full blur-md opacity-75 group-hover:opacity-100 animate-pulse" />
          
          {/* Button */}
          <div className="relative w-14 h-14 bg-gray-900 rounded-full flex items-center justify-center shadow-2xl border border-white/10 group-hover:scale-110 transition-transform">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-white">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            
            {/* Status indicator */}
            <div className={`absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-900 ${
              llmStatus?.available ? 'bg-emerald-400' : 'bg-gray-500'
            }`} />
          </div>
        </div>
      </button>
    )
  }

  return (
    <div 
      className={`fixed z-[2000] transition-all duration-300 ease-out ${
        isExpanded 
          ? 'inset-4 md:inset-8' 
          : 'bottom-6 right-6 w-[420px] h-[600px]'
      }`}
    >
      <div className="w-full h-full bg-gray-900/95 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/10 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex-shrink-0 px-4 py-3 bg-gradient-to-r from-gray-800 to-gray-900 border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Animated logo */}
              <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${MODES[mode].color} flex items-center justify-center text-xl shadow-lg`}>
                {MODES[mode].icon}
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Nemotron 49B</h3>
                <p className="text-gray-400 text-xs">{MODES[mode].label}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-1">
              <button
                onClick={() => setShowReasoning(!showReasoning)}
                className={`p-2 rounded-lg transition-colors ${
                  showReasoning ? 'bg-violet-500/20 text-violet-400' : 'text-gray-400 hover:text-white hover:bg-white/10'
                }`}
                title="Toggle reasoning visibility"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                  <path d="M12 17h.01"/>
                </svg>
              </button>
              <button
                onClick={clearChat}
                className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                title="Clear chat"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                </svg>
              </button>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                title={isExpanded ? 'Minimize' : 'Expand'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  {isExpanded ? (
                    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/>
                  ) : (
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
                  )}
                </svg>
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                title="Close"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>
            </div>
          </div>
          
          {/* Mode selector */}
          <div className="flex gap-2 mt-3">
            {Object.entries(MODES).map(([key, { label, icon, color }]) => (
              <button
                key={key}
                onClick={() => setMode(key as keyof typeof MODES)}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-medium transition-all ${
                  mode === key
                    ? `bg-gradient-to-r ${color} text-white shadow-lg`
                    : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                }`}
              >
                <span className="mr-1">{icon}</span>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-6">
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${MODES[mode].color} flex items-center justify-center text-3xl mb-4 shadow-lg`}>
                {MODES[mode].icon}
              </div>
              <h4 className="text-white font-medium mb-2">Glid Surge AI Assistant</h4>
              <p className="text-gray-400 text-sm mb-6">
                {mode === 'assistant' && 'Ask me anything about logistics, routes, or surge predictions.'}
                {mode === 'route_optimizer' && 'I\'ll analyze routes and suggest optimizations.'}
                {mode === 'surge_analyst' && 'I\'ll interpret surge data and recommend dispatch windows.'}
              </p>
              <div className="grid grid-cols-1 gap-2 w-full max-w-xs">
                {mode === 'assistant' && (
                  <>
                    <SuggestionButton onClick={() => setInput('What are the current surge levels at major ports?')} text="📊 Current surge levels" />
                    <SuggestionButton onClick={() => setInput('Optimize my route from Long Beach to Kansas')} text="🛤️ Optimize a route" />
                  </>
                )}
                {mode === 'route_optimizer' && (
                  <>
                    <SuggestionButton onClick={() => setInput('Analyze the current route and suggest improvements')} text="🔍 Analyze current route" />
                    <SuggestionButton onClick={() => setInput('What\'s the most cost-effective route option?')} text="💰 Most cost-effective route" />
                  </>
                )}
                {mode === 'surge_analyst' && (
                  <>
                    <SuggestionButton onClick={() => setInput('When is the best time to dispatch to avoid surge?')} text="⏰ Best dispatch time" />
                    <SuggestionButton onClick={() => setInput('Predict surge levels for the next 72 hours')} text="📈 72-hour forecast" />
                  </>
                )}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] ${msg.role === 'user' ? 'order-2' : ''}`}>
                    {msg.isLoading ? (
                      <div className="bg-white/5 rounded-2xl px-4 py-3 flex items-center gap-2">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-2 h-2 bg-fuchsia-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-gray-400 text-sm">Thinking...</span>
                      </div>
                    ) : (
                      <>
                        {/* Reasoning (collapsible) */}
                        {msg.reasoning && showReasoning && (
                          <div className="mb-2 bg-violet-500/10 border border-violet-500/20 rounded-xl px-3 py-2">
                            <div className="text-violet-300 text-xs font-medium mb-1 flex items-center gap-1">
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <path d="M12 17h.01"/>
                              </svg>
                              Reasoning
                            </div>
                            <p className="text-violet-200/70 text-xs leading-relaxed">{msg.reasoning}</p>
                          </div>
                        )}
                        
                        {/* Message content */}
                        <div className={`rounded-2xl px-4 py-3 ${
                          msg.role === 'user'
                            ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white'
                            : 'bg-white/5 text-gray-100'
                        }`}>
                          <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                        </div>
                        
                        <div className={`text-xs text-gray-500 mt-1 ${msg.role === 'user' ? 'text-right' : ''}`}>
                          {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        <div className="flex-shrink-0 p-4 bg-gray-800/50 border-t border-white/10">
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask ${MODES[mode].label}...`}
              rows={1}
              className="flex-1 bg-white/5 text-white placeholder-gray-500 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-violet-500/50 border border-white/10"
              disabled={isLoading || !llmStatus?.available}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !input.trim() || !llmStatus?.available}
              className={`px-4 rounded-xl font-medium text-sm transition-all ${
                isLoading || !input.trim() || !llmStatus?.available
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : `bg-gradient-to-r ${MODES[mode].color} text-white shadow-lg hover:shadow-xl hover:scale-105`
              }`}
            >
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25"/>
                  <path d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" fill="currentColor" className="opacity-75"/>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
              )}
            </button>
          </div>
          
          {!llmStatus?.available && (
            <p className="text-amber-400/80 text-xs mt-2 flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
              </svg>
              Nemotron 49B not connected. Start the model server to enable AI features.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function SuggestionButton({ onClick, text }: { onClick: () => void; text: string }) {
  return (
    <button
      onClick={onClick}
      className="w-full py-2 px-3 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 hover:text-white text-sm text-left transition-colors border border-white/5"
    >
      {text}
    </button>
  )
}

