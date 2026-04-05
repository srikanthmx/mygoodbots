import { useState } from 'react'
import BotSelector from './components/BotSelector'
import ChatWindow from './components/ChatWindow'
import ApprovalPanel from './components/ApprovalPanel'

function generateSessionId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return Math.random().toString(36).slice(2)
}

export default function App() {
  const [sessionId] = useState(() => generateSessionId())
  const [selectedBotId, setSelectedBotId] = useState('genai')

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-gray-700 px-4 py-3">
        <h1 className="text-lg font-bold text-blue-400">🤖 Multi-Bot AI</h1>
        <BotSelector selectedBotId={selectedBotId} onSelect={setSelectedBotId} />
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-hidden">
        <ChatWindow sessionId={sessionId} botId={selectedBotId} />
      </main>

      {/* Approval panel — shown only when there are pending proposals */}
      <ApprovalPanel />
    </div>
  )
}
