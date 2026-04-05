import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchPendingApprovals,
  approvePending,
  rejectPending,
  type PendingApproval,
} from '../api/chatApi'

export default function ApprovalPanel() {
  const qc = useQueryClient()
  const [rejectReason, setRejectReason] = useState<Record<string, string>>({})

  const { data: pending, isLoading } = useQuery<PendingApproval[]>({
    queryKey: ['approvals'],
    queryFn: fetchPendingApprovals,
    refetchInterval: 10_000,
  })

  const approveMutation = useMutation({
    mutationFn: (id: string) => approvePending(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['approvals'] }),
  })

  const rejectMutation = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      rejectPending(id, reason),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['approvals'] }),
  })

  if (isLoading) return null
  if (!pending?.length) return null

  return (
    <div className="border-t border-yellow-600 bg-gray-900 p-4 space-y-4">
      <h2 className="text-sm font-semibold text-yellow-400">
        ⏳ Pending Approvals ({pending.length})
      </h2>
      {pending.map((req) => (
        <div key={req.approval_id} className="rounded border border-gray-700 bg-gray-800 p-3 space-y-2">
          <p className="text-xs text-gray-400">ID: {req.approval_id}</p>
          <p className="text-sm text-gray-200">{req.summary}</p>
          <pre className="overflow-x-auto rounded bg-gray-900 p-2 text-xs text-green-400">
            {req.diff}
          </pre>
          <div className="flex gap-2 items-center">
            <button
              onClick={() => approveMutation.mutate(req.approval_id)}
              disabled={approveMutation.isPending}
              className="rounded bg-green-700 px-3 py-1 text-xs text-white hover:bg-green-600 disabled:opacity-50"
            >
              ✅ Approve
            </button>
            <input
              type="text"
              placeholder="Rejection reason…"
              value={rejectReason[req.approval_id] ?? ''}
              onChange={(e) =>
                setRejectReason((prev) => ({ ...prev, [req.approval_id]: e.target.value }))
              }
              className="flex-1 rounded bg-gray-700 px-2 py-1 text-xs text-white placeholder-gray-400 focus:outline-none"
            />
            <button
              onClick={() =>
                rejectMutation.mutate({
                  id: req.approval_id,
                  reason: rejectReason[req.approval_id] ?? 'Rejected by user',
                })
              }
              disabled={rejectMutation.isPending}
              className="rounded bg-red-700 px-3 py-1 text-xs text-white hover:bg-red-600 disabled:opacity-50"
            >
              ❌ Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
