import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PROTOCOL_VERSION } from '@agentclientprotocol/sdk'
import { makeBridgeHarness, textResponse, type BridgeHarness, type CapturedUpdate } from './harness.ts'

/** Concatenated text chunks of one update kind for a session. */
function textFor(
  updates: { sessionId: string; update: CapturedUpdate }[],
  sessionId: string,
  kind: 'user_message_chunk' | 'agent_message_chunk',
): string {
  return updates.flatMap(({ sessionId: owner, update }) => (
    owner === sessionId && update.sessionUpdate === kind && update.content.type === 'text'
      ? [update.content.text]
      : []
  )).join('')
}

describe('ACP session/load history replay', () => {
  let harnesses: BridgeHarness[] = []
  let root: string | undefined

  afterEach(async () => {
    for (const harness of harnesses) await harness.dispose()
    harnesses = []
    if (root !== undefined) await rm(root, { recursive: true, force: true })
    root = undefined
  })

  it('advertises the loadSession capability', async () => {
    const harness = await makeBridgeHarness()
    harnesses.push(harness)
    const result = await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    expect(result.agentCapabilities.loadSession).toBe(true)
  })

  it('replays stored user and assistant turns to a reconnecting client', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-acp-load-'))

    const first = await makeBridgeHarness({ persistenceRoot: root, script: [textResponse('stored-answer')] })
    harnesses.push(first)
    await first.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await first.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await first.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'remember-this' }] })
    await vi.waitFor(() => {
      expect(textFor(first.sessionUpdates, sessionId, 'agent_message_chunk')).toBe('stored-answer')
    })
    await first.dispose()
    harnesses = []

    const second = await makeBridgeHarness({ persistenceRoot: root })
    harnesses.push(second)
    await second.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const loaded = await second.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    expect(loaded.configOptions.length).toBeGreaterThan(0)

    const replayedUser = textFor(second.sessionUpdates, sessionId, 'user_message_chunk')
    const replayedAgent = textFor(second.sessionUpdates, sessionId, 'agent_message_chunk')
    expect(replayedUser).toContain('remember-this')
    expect(replayedAgent).toContain('stored-answer')
  })

  it('rejects loading a session whose workspace moved', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-acp-load-'))
    const first = await makeBridgeHarness({ persistenceRoot: root, script: [textResponse('x')] })
    harnesses.push(first)
    await first.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await first.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await first.dispose()
    harnesses = []

    const second = await makeBridgeHarness({ persistenceRoot: root })
    harnesses.push(second)
    await second.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    await expect(
      second.client.loadSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
    ).rejects.toThrow(/cwd does not match/)
  })
})
