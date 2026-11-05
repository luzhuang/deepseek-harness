import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PROTOCOL_VERSION } from '@agentclientprotocol/sdk'
import { createUserMessage, type StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import { makeBridgeHarness, textResponse, type BridgeHarness, type CapturedUpdate } from './harness.ts'
import { AcpSession } from '../src/session.ts'

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
    expect(result.agentCapabilities?.loadSession).toBe(true)
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
    expect(loaded.configOptions?.length).toBeGreaterThan(0)

    const replayedUser = textFor(second.sessionUpdates, sessionId, 'user_message_chunk')
    const replayedAgent = textFor(second.sessionUpdates, sessionId, 'agent_message_chunk')
    expect(replayedUser).toContain('remember-this')
    expect(replayedAgent).toContain('stored-answer')
  })

  it('preserves the exact order of multiple user and assistant turns', async () => {
    const harness = await makeBridgeHarness({ script: [textResponse('answer-1'), textResponse('answer-2')] })
    harnesses.push(harness)
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    for (const text of ['question-1', 'question-2']) {
      await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text }] })
    }
    await harness.client.closeSession({ sessionId })
    harness.updates.length = 0
    await harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    expect(harness.updates.flatMap(update => (
      (update.sessionUpdate === 'user_message_chunk' || update.sessionUpdate === 'agent_message_chunk')
        && update.content.type === 'text' ? [update.content.text] : []
    ))).toEqual(['question-1', 'answer-1', 'question-2', 'answer-2'])
  })

  it('does not attribute injected context or replacement messages to the user', async () => {
    const harness = await makeBridgeHarness({ script: [textResponse('answer')] })
    harnesses.push(harness)
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    const agent = harness.ctx.agents.get(SessionId(sessionId))!
    agent.inject(createUserMessage({ content: [{ type: 'text', text: 'private injected context' }], source: { kind: 'system-prompt' } }))
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'human question' }] })
    const original = agent.session.snapshotEvents().find(event => event.type === 'user/message' && event.data.source.kind === 'user')!
    agent.session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'model-only replacement' }], source: { kind: 'user' },
    }), { surfaceOp: { op: 'replace', startSeq: original.seq, endSeq: original.seq }, sourceEventSeqs: [original.seq] })
    await harness.client.closeSession({ sessionId })
    harness.sessionUpdates.length = 0
    await harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    expect(textFor(harness.sessionUpdates, sessionId, 'user_message_chunk')).toBe('human question')
  })

  it('restores user images between text blocks', async () => {
    const harness = await makeBridgeHarness({ imageCapable: true, script: [textResponse('answer')] })
    harnesses.push(harness)
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    const prompt = [
      { type: 'text', text: 'before' },
      { type: 'image', data: 'AQ==', mimeType: 'image/png' },
      { type: 'text', text: 'after' },
    ] as const
    await harness.client.prompt({ sessionId, prompt: [...prompt] })
    await harness.client.closeSession({ sessionId })
    harness.updates.length = 0
    await harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    expect(harness.updates.flatMap(update => update.sessionUpdate === 'user_message_chunk' ? [update.content] : [])).toEqual(prompt)
  })

  it('rejects an incomplete replay and releases the session for retry', async () => {
    const script: StreamChunk[][] = []
    const harness = await makeBridgeHarness({ script })
    harnesses.push(harness)
    const ref = await harness.attachments!.saveImage({ data: Uint8Array.of(1), mediaType: 'image/png' })
    script.push([
      { type: 'block-start', index: 0, blockType: 'image' },
      { type: 'block-end', index: 0, block: { type: 'image', attachment: ref } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'show image' }] })
    await harness.client.closeSession({ sessionId })
    harness.attachments!.beforeRead = () => Promise.reject(new Error('attachment read failed'))
    await expect(harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })).rejects.toThrow()
    expect(harness.ctx.agents.get(SessionId(sessionId))).toBeUndefined()
    harness.attachments!.beforeRead = undefined
    await expect(harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })).resolves.toHaveProperty('configOptions')
  })

  it('blocks new prompts during replay and releases ownership on request cancellation', async () => {
    const script: StreamChunk[][] = []
    const harness = await makeBridgeHarness({ script })
    harnesses.push(harness)
    const ref = await harness.attachments!.saveImage({ data: Uint8Array.of(2), mediaType: 'image/png' })
    script.push([
      { type: 'block-start', index: 0, blockType: 'image' },
      { type: 'block-end', index: 0, block: { type: 'image', attachment: ref } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'show image' }] })
    await harness.client.closeSession({ sessionId })
    const reading = Promise.withResolvers<undefined>()
    const release = Promise.withResolvers<undefined>()
    harness.attachments!.beforeRead = () => { reading.resolve(undefined); return release.promise }
    const cancellation = new AbortController()
    const open = harness.ctx.sessionPersistence.open.bind(harness.ctx.sessionPersistence)
    let serverSignal: AbortSignal | undefined
    const spy = vi.spyOn(harness.ctx.sessionPersistence, 'open').mockImplementation((id, access, options) => {
      if (access === 'read') serverSignal = options?.signal
      return open(id, access, options)
    })
    const load = harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] }, {
      cancellationSignal: cancellation.signal,
    })
    try {
      await reading.promise
      await expect(harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'too early' }] }))
        .rejects.toThrow(/activating/)
      await expect(harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] }))
        .rejects.toThrow(/already active/)
      cancellation.abort()
      await vi.waitFor(() => { expect(serverSignal?.aborted).toBe(true) })
      release.resolve(undefined)
      await expect(load).rejects.toThrow()
      expect(harness.ctx.agents.get(SessionId(sessionId))).toBeUndefined()
      harness.attachments!.beforeRead = undefined
      await expect(harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] }))
        .resolves.toHaveProperty('configOptions')
    } finally {
      release.resolve(undefined)
      spy.mockRestore()
      await load.catch((_loadFailure: unknown) => {})
    }
  })

  it('aborts and joins a pending cold read before bridge disposal returns', async () => {
    const harness = await makeBridgeHarness({ script: [textResponse('answer')] })
    harnesses.push(harness)
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'question' }] })
    await harness.client.closeSession({ sessionId })
    harness.updates.length = 0
    const reading = Promise.withResolvers<undefined>()
    const aborted = Promise.withResolvers<undefined>()
    const release = Promise.withResolvers<undefined>()
    let readClosed = false
    const open = harness.ctx.sessionPersistence.open.bind(harness.ctx.sessionPersistence)
    const spy = vi.spyOn(harness.ctx.sessionPersistence, 'open').mockImplementation(async (id, access, options) => {
      const handle = await open(id, access, options)
      if (access === 'read') {
        const read = handle.read.bind(handle)
        vi.spyOn(handle, 'read').mockImplementation(async (...args) => {
          options?.signal?.addEventListener('abort', () => { aborted.resolve(undefined) }, { once: true })
          reading.resolve(undefined)
          await release.promise
          return read(...args)
        })
        const close = handle.close.bind(handle)
        vi.spyOn(handle, 'close').mockImplementation(async () => { await close(); readClosed = true })
      }
      return handle
    })
    const load = harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    const settledLoad = load.catch((_loadFailure: unknown) => {})
    let disposed = false
    try {
      await reading.promise
      const disposal = harness.acpFiber.dispose().then(() => { disposed = true })
      await aborted.promise
      expect(disposed).toBe(false)
      release.resolve(undefined)
      await expect(load).rejects.toThrow()
      await disposal
      expect(readClosed).toBe(true)
      expect(harness.ctx.agents.get(SessionId(sessionId))).toBeUndefined()
      expect(harness.updates).toEqual([])
    } finally {
      release.resolve(undefined)
      await settledLoad
      spy.mockRestore()
    }
  })

  it('joins an in-progress replay notification before disposing the bridge', async () => {
    const harness = await makeBridgeHarness({ script: [textResponse('answer')] })
    harnesses.push(harness)
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    const { sessionId } = await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'question' }] })
    await harness.client.closeSession({ sessionId })
    harness.updates.length = 0
    const notifying = Promise.withResolvers<undefined>()
    const release = Promise.withResolvers<undefined>()
    const closing = Promise.withResolvers<undefined>()
    const resume = AcpSession.resume.bind(AcpSession)
    const spy = vi.spyOn(AcpSession, 'resume').mockImplementation((ctx, options) => resume(ctx, {
      ...options,
      notify: async (notification) => {
        if (notification.update.sessionUpdate === 'user_message_chunk') {
          options.signal.addEventListener('abort', () => { closing.resolve(undefined) }, { once: true })
          notifying.resolve(undefined)
          await release.promise
        }
        await options.notify(notification)
      },
    }))
    const load = harness.client.loadSession({ sessionId, cwd: process.cwd(), mcpServers: [] })
    const settledLoad = load.catch((_loadFailure: unknown) => {})
    let disposed = false
    try {
      await notifying.promise
      const disposal = harness.acpFiber.dispose().then(() => { disposed = true })
      await closing.promise
      expect(disposed).toBe(false)
      release.resolve(undefined)
      await expect(load).rejects.toThrow()
      await disposal
      expect(harness.updates.some(update => update.sessionUpdate === 'agent_message_chunk')).toBe(false)
      expect(harness.ctx.agents.get(SessionId(sessionId))).toBeUndefined()
    } finally {
      release.resolve(undefined)
      await settledLoad
      spy.mockRestore()
    }
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
