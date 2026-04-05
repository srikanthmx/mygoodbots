import { useQuery } from '@tanstack/react-query'
import { fetchBots, type BotInfo } from '../api/chatApi'

interface Props {
  selectedBotId: string
  onSelect: (botId: string) => void
}

export default function BotSelector({ selectedBotId, onSelect }: Props) {
  const { data: bots, isLoading, error } = useQuery<BotInfo[]>({
    queryKey: ['bots'],
    queryFn: fetchBots,
  })

  if (isLoading) return <div className="text-sm text-gray-400">Loading bots…</div>
  if (error) return <div className="text-sm text-red-400">Failed to load bots</div>

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="bot-select" className="text-sm font-medium text-gray-300">
        Bot:
      </label>
      <select
        id="bot-select"
        value={selectedBotId}
        onChange={(e) => onSelect(e.target.value)}
        className="rounded bg-gray-700 px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {bots?.map((bot) => (
          <option key={bot.bot_id} value={bot.bot_id} disabled={!bot.available}>
            {bot.bot_id} — {bot.description}
          </option>
        ))}
      </select>
    </div>
  )
}
