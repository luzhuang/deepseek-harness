/** Standard ACP updates derived from committed DSH session events. */

import type { Context } from '@deepseek-ai/cordis'
import type { SessionUpdate, ToolCallContent, ToolKind } from '@agentclientprotocol/sdk'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-token-meter'
import { assistantBlockToAcp } from './content.ts'

/**
 * Convert one committed assistant message and its context usage in block order.
 * @param ctx - bridge context carrying attachment and token-meter services.
 * @param session - durable session used for context pressure.
 * @param event - committed assistant message event.
 * @returns ordered standard thought, message, and optional usage updates.
 */
export async function assistantUpdates(
  ctx: Context,
  session: Session,
  event: SessionEvent<'assistant/message'>,
): Promise<SessionUpdate[]> {
  const updates = await assistantContentUpdates(ctx, event)
  const usage = usageUpdate(ctx, session, event)
  if (usage !== undefined) updates.push(usage)
  return updates
}

/**
 * Translate assistant content without measuring current usage for past turns.
 * @param ctx - context carrying the attachment store.
 * @param event - committed assistant message.
 * @returns ordered thought and message updates.
 */
export async function assistantContentUpdates(
  ctx: Context,
  event: SessionEvent<'assistant/message'>,
): Promise<SessionUpdate[]> {
  const updates: SessionUpdate[] = []
  for (const block of event.data.message.content) {
    if (block.type === 'reasoning') {
      if (block.text.length > 0) {
        updates.push({
          sessionUpdate: 'agent_thought_chunk',
          messageId: event.data.message.id,
          content: { type: 'text', text: block.text },
        })
      }
      continue
    }
    const content = await assistantBlockToAcp(ctx, block)
    if (content !== undefined) {
      updates.push({
        sessionUpdate: 'agent_message_chunk',
        messageId: event.data.message.id,
        content,
      })
    }
  }
  return updates
}

/**
 * Map a harness tool name onto ACP's tool-kind vocabulary so clients can pick
 * the right icon, grouping, and affordance instead of rendering every call as
 * a generic "other" entry. Unknown tools stay `other` by design.
 * @param name - the model-facing tool name from the committed call fact.
 * @returns the closest standard ACP tool kind.
 */
export function toolKind(name: string): ToolKind {
  switch (name) {
    case 'bash':
    case 'shell':
    case 'execute':
      return 'execute'
    case 'read':
      return 'read'
    case 'edit':
    case 'write':
    case 'multiedit':
      return 'edit'
    case 'grep':
    case 'glob':
    case 'web_search':
      return 'search'
    case 'web_fetch':
      return 'fetch'
    case 'todo_write':
    case 'todo_read':
      return 'think'
    default:
      return 'other'
  }
}

/**
 * Start one generic ACP tool lifecycle from the durable call fact.
 * @param event - committed DSH tool-call event.
 * @returns the standard generic tool-call update.
 */
export function toolCallUpdate(event: SessionEvent<'tool/call'>): SessionUpdate {
  return {
    sessionUpdate: 'tool_call',
    toolCallId: event.data.callId,
    title: event.data.name,
    kind: toolKind(event.data.name),
    status: 'in_progress',
    rawInput: parseToolArguments(event.data.arguments),
  }
}

/**
 * Finish one generic ACP tool lifecycle from its committed model-facing result.
 * @param ctx - bridge context carrying the attachment store.
 * @param event - committed DSH tool-result event.
 * @returns the standard completed or failed tool-call update.
 */
export async function toolResultUpdate(
  ctx: Context,
  event: SessionEvent<'tool/result'>,
): Promise<SessionUpdate> {
  const message = event.data.message
  const content: ToolCallContent[] = []
  for (const block of message.content) {
    const converted = await assistantBlockToAcp(ctx, block)
    if (converted !== undefined) content.push({ type: 'content' as const, content: converted })
  }
  return {
    sessionUpdate: 'tool_call_update',
    toolCallId: message.toolCallId,
    status: message.isError === true ? 'failed' : 'completed',
    content,
  }
}

/** Report current context occupancy only when DSH has both usage and capacity facts. */
function usageUpdate(
  ctx: Context,
  session: Session,
  event: SessionEvent<'assistant/message'>,
): SessionUpdate | undefined {
  if (event.data.usage === undefined) return undefined
  const size = session.requestContext()?.contextWindow
  const meter = ctx.get('tokenMeter')
  if (size === undefined || meter === undefined) return undefined
  return {
    sessionUpdate: 'usage_update',
    used: meter.measure(session).totalTokens,
    size,
  }
}

/** Preserve malformed model output as opaque input instead of dropping the call update. */
function parseToolArguments(value: string): unknown {
  try {
    return JSON.parse(value) as unknown
  } catch (_invalidModelJson) {
    return value
  }
}
