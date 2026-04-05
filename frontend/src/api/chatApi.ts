export interface BotInfo {
  bot_id: string
  description: string
  model: string
  available: boolean
}

export interface ChatRequest {
  session_id: string
  bot_id: string
  message: string
  stream?: boolean
}

export interface ChatResponse {
  session_id: string
  bot_id: string
  reply: string
  tokens_used: number
  latency_ms: number
  error?: string
}

export interface PendingApproval {
  approval_id: string
  diff: string
  summary: string
  status: string
  created_at: string
  expires_at: string | null
}

const API_KEY = import.meta.env.VITE_API_KEY ?? ''

function authHeaders(): HeadersInit {
  return API_KEY ? { 'X-API-Key': API_KEY } : {}
}

export async function fetchBots(): Promise<BotInfo[]> {
  const res = await fetch('/api/v1/bots', { headers: authHeaders() })
  if (!res.ok) throw new Error(`fetchBots failed: ${res.status}`)
  return res.json()
}

export async function sendMessage(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ ...req, stream: false }),
  })
  if (!res.ok) throw new Error(`sendMessage failed: ${res.status}`)
  return res.json()
}

export function streamMessage(
  req: ChatRequest,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void
): () => void {
  const controller = new AbortController()

  fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ ...req, stream: true }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok || !res.body) throw new Error(`streamMessage failed: ${res.status}`)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') { onDone(); return }
            try {
              const parsed = JSON.parse(data)
              if (parsed.reply) onToken(parsed.reply)
            } catch { /* ignore parse errors */ }
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err)
    })

  return () => controller.abort()
}

export async function fetchPendingApprovals(): Promise<PendingApproval[]> {
  const res = await fetch('/api/v1/approvals/pending', { headers: authHeaders() })
  if (!res.ok) throw new Error(`fetchPendingApprovals failed: ${res.status}`)
  return res.json()
}

export async function approvePending(id: string): Promise<void> {
  const res = await fetch(`/api/v1/approvals/${id}/approve`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error(`approvePending failed: ${res.status}`)
}

export async function rejectPending(id: string, reason: string): Promise<void> {
  const res = await fetch(`/api/v1/approvals/${id}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ reason }),
  })
  if (!res.ok) throw new Error(`rejectPending failed: ${res.status}`)
}
