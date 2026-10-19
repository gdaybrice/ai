import { buildSessionContext } from '@earendil-works/pi-coding-agent';
import { describe, expect, it } from 'vitest';
import { piLifecycleDataFromUIMessages } from './pi-history-from-parts';
import { PI_HISTORY_VERSION } from './pi-history-metadata';

const usage = {
  input: 10,
  output: 5,
  cacheRead: 100,
  cacheWrite: 0,
  totalTokens: 115,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const assistantMetadata = (stopReason: string, signatures: string[] = []) => ({
  pi: {
    assistant: {
      version: PI_HISTORY_VERSION,
      message: {
        api: 'anthropic-messages',
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
        usage,
        stopReason,
        timestamp: 1,
      },
      thinking: signatures.map(signature => ({ signature, redacted: false })),
    },
  },
});

const contextOf = (entries: ReadonlyArray<unknown>) =>
  buildSessionContext(entries.slice(1) as never).messages.map(message => {
    const { timestamp: _timestamp, ...rest } = message as Record<
      string,
      unknown
    >;
    return rest;
  });

describe('piLifecycleDataFromUIMessages', () => {
  it('rebuilds turns with thinking signatures and raw tool output', () => {
    const data = piLifecycleDataFromUIMessages({
      messages: [
        { role: 'user', parts: [{ type: 'text', text: 'Read notes.txt' }] },
        {
          role: 'assistant',
          parts: [
            { type: 'step-start' },
            {
              type: 'reasoning',
              text: 'I will read it.',
              providerMetadata: assistantMetadata('toolUse', ['sig-1']),
            },
            {
              type: 'tool-read',
              toolCallId: 'call-1',
              state: 'output-error',
              input: { path: 'missing.txt' },
              errorText: JSON.stringify({
                content: [{ type: 'text', text: 'ENOENT: missing.txt' }],
              }),
              callProviderMetadata: {
                pi: { rawArguments: { path: '/work/missing.txt' } },
              },
              resultProviderMetadata: {
                pi: { rawText: ['ENOENT: /work/missing.txt'] },
              },
            },
            { type: 'step-start' },
            {
              type: 'text',
              text: 'It is missing.',
              providerMetadata: assistantMetadata('stop'),
            },
          ],
        },
      ],
    });

    expect(data.pendingToolBatch).toBeUndefined();
    expect(contextOf(data.entries!)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'Read notes.txt' }] },
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: 'I will read it.',
            thinkingSignature: 'sig-1',
          },
          {
            type: 'toolCall',
            id: 'call-1',
            name: 'read',
            arguments: { path: '/work/missing.txt' },
          },
        ],
        api: 'anthropic-messages',
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
        usage,
        stopReason: 'toolUse',
      },
      {
        role: 'toolResult',
        toolCallId: 'call-1',
        toolName: 'read',
        content: [{ type: 'text', text: 'ENOENT: /work/missing.txt' }],
        isError: true,
      },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'It is missing.' }],
        api: 'anthropic-messages',
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
        usage,
        stopReason: 'stop',
      },
    ]);
  });

  it('places a compaction where Pi made it', () => {
    const turn = (text: string) => [
      { role: 'user', parts: [{ type: 'text', text }] },
      {
        role: 'assistant',
        parts: [
          { type: 'step-start' },
          {
            type: 'text',
            text: `answer ${text}`,
            providerMetadata: assistantMetadata('stop'),
          },
        ],
      },
    ];
    const data = piLifecycleDataFromUIMessages({
      messages: [
        ...turn('one'),
        ...turn('two'),
        {
          role: 'assistant',
          parts: [
            { type: 'step-start' },
            {
              type: 'dynamic-tool',
              toolName: 'compaction',
              toolCallId: 'harness-compaction-1',
              state: 'output-available',
              input: {},
              output: { trigger: 'auto', summary: 'Earlier: one.' },
              callProviderMetadata: {
                pi: {
                  compaction: {
                    messageCountBefore: 4,
                    firstKeptMessageIndex: 2,
                    tokensBefore: 900,
                    details: null,
                    usage: null,
                  },
                },
              },
            },
          ],
        },
        ...turn('three'),
      ],
    });

    const context = contextOf(data.entries!);
    expect(context.map(message => message.role)).toEqual([
      'compactionSummary',
      'user',
      'assistant',
      'user',
      'assistant',
    ]);
    expect(context[0]).toMatchObject({ summary: 'Earlier: one.' });
    expect(context[1]).toMatchObject({
      content: [{ type: 'text', text: 'two' }],
    });
  });

  it('returns a tool batch that waits on host input', () => {
    const data = piLifecycleDataFromUIMessages({
      messages: [
        { role: 'user', parts: [{ type: 'text', text: 'Ask my name' }] },
        {
          role: 'assistant',
          parts: [
            { type: 'step-start' },
            {
              type: 'dynamic-tool',
              toolName: 'getUserName',
              toolCallId: 'call-host',
              state: 'input-available',
              input: {},
              callProviderMetadata: assistantMetadata('toolUse'),
            },
            {
              type: 'tool-read',
              toolCallId: 'call-read',
              state: 'output-available',
              input: { path: 'a.txt' },
              output: { content: [{ type: 'text', text: 'a' }] },
            },
          ],
        },
      ],
    });

    expect(data.pendingToolBatch).toEqual({
      awaitingToolCallIds: ['call-host'],
      completedToolResults: [],
      nativeApprovalDecisions: [],
    });
    expect(contextOf(data.entries!).map(message => message.role)).toEqual([
      'user',
      'assistant',
      'toolResult',
    ]);
  });

  it('drops a waiting batch once a later prompt follows it', () => {
    const data = piLifecycleDataFromUIMessages({
      messages: [
        {
          role: 'assistant',
          parts: [
            { type: 'step-start' },
            {
              type: 'tool-getUserName',
              toolCallId: 'call-host',
              state: 'input-available',
              input: {},
              callProviderMetadata: assistantMetadata('toolUse'),
            },
          ],
        },
        { role: 'user', parts: [{ type: 'text', text: 'Never mind' }] },
      ],
    });

    expect(data.pendingToolBatch).toBeUndefined();
  });

  it('leaves out a last step that was cut off before Pi finished it', () => {
    const data = piLifecycleDataFromUIMessages({
      messages: [
        { role: 'user', parts: [{ type: 'text', text: 'Read a.txt' }] },
        {
          role: 'assistant',
          parts: [
            { type: 'step-start' },
            {
              type: 'text',
              text: 'Reading it.',
              providerMetadata: assistantMetadata('toolUse'),
            },
            { type: 'step-start' },
            { type: 'text', text: 'It says' },
            {
              type: 'tool-read',
              toolCallId: 'call-cut',
              state: 'input-available',
              input: { path: 'a.txt' },
            },
          ],
        },
      ],
    });

    expect(data.pendingToolBatch).toBeUndefined();
    expect(contextOf(data.entries!)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'Read a.txt' }] },
      expect.objectContaining({
        role: 'assistant',
        content: [{ type: 'text', text: 'Reading it.' }],
      }),
    ]);
  });

  it('throws for an earlier step without Pi history metadata', () => {
    expect(() =>
      piLifecycleDataFromUIMessages({
        messages: [
          { role: 'user', parts: [{ type: 'text', text: 'Hi' }] },
          {
            role: 'assistant',
            parts: [{ type: 'step-start' }, { type: 'text', text: 'Hello' }],
          },
          { role: 'user', parts: [{ type: 'text', text: 'Again' }] },
        ],
      }),
    ).toThrow('without Pi history metadata');
  });

  it('throws for Pi history metadata of another version', () => {
    expect(() =>
      piLifecycleDataFromUIMessages({
        messages: [
          { role: 'user', parts: [{ type: 'text', text: 'Hi' }] },
          {
            role: 'assistant',
            parts: [
              { type: 'step-start' },
              {
                type: 'text',
                text: 'Hello',
                providerMetadata: {
                  pi: {
                    assistant: {
                      ...assistantMetadata('stop').pi.assistant,
                      version: 2,
                    },
                  },
                },
              },
            ],
          },
        ],
      }),
    ).toThrow('Pi history version 2');
  });
});
