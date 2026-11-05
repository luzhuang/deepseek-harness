/** Ordered ACP transcript replay without live turn bookkeeping. */

import type { Context } from '@deepseek-ai/cordis'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { isAppendSurfaceEvent } from '@deepseek-ai/dsh-session/surface'
import type { SessionUpdate } from '@agentclientprotocol/sdk'
import { assistantBlockToAcp } from './content.ts'
import { assistantContentUpdates, toolCallUpdate, toolResultUpdate } from './updates.ts'

/**
 * Convert the visible transcript in log order, excluding injected context and
 * model-only replacements. Conversion failures reject the load operation.
 * @param ctx - context carrying the attachment store.
 * @param events - committed events in sequence order.
 * @param signal - optional cancellation checked between events and blocks.
 * @returns standard user, assistant and tool updates in transcript order.
 */
export async function* replayUpdates(
  ctx: Context,
  events: readonly SessionEvent[],
  signal?: AbortSignal,
): AsyncGenerator<SessionUpdate> {
  for (const event of events) {
    signal?.throwIfAborted()
    if (event.type === 'tool/call') {
      yield toolCallUpdate(event)
      continue
    }
    if (!isAppendSurfaceEvent(event)) continue
    if (event.type === 'user/message' && event.data.source.kind === 'user') {
      for (const block of event.data.content) {
        signal?.throwIfAborted()
        const content = await assistantBlockToAcp(ctx, block)
        if (content !== undefined) {
          yield { sessionUpdate: 'user_message_chunk', messageId: event.data.id, content }
        }
      }
    } else if (event.type === 'assistant/message') {
      yield* await assistantContentUpdates(ctx, event)
    } else if (event.type === 'tool/result') {
      yield await toolResultUpdate(ctx, event)
    }
  }
}
