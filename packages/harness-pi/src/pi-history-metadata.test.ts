import { describe, expect, it } from 'vitest';
import {
  createPiHistoryAnnotator,
  PI_HISTORY_VERSION,
} from './pi-history-metadata';

const message = (id: string, role: string) => ({
  type: 'message',
  id,
  parentId: null,
  timestamp: '',
  message: { role, content: [], timestamp: 0 },
});

describe('createPiHistoryAnnotator', () => {
  it('records where a compaction cut the active branch as message counts', () => {
    const branch = [
      message('s', 'system'),
      message('a', 'user'),
      message('b', 'assistant'),
      { type: 'model_change', id: 'm', parentId: null, timestamp: '' },
      message('c', 'user'),
      message('d', 'assistant'),
      {
        type: 'compaction',
        id: 'x',
        parentId: null,
        timestamp: '',
        summary: 'Earlier.',
        firstKeptEntryId: 'c',
        tokensBefore: 900,
      },
    ];
    const annotator = createPiHistoryAnnotator({
      sessionWorkDir: '/work',
      getBranch: () => branch as never,
    });

    const [part] = annotator.annotate({ type: 'compaction_end' }, [
      { type: 'compaction', trigger: 'auto', summary: 'Earlier.' },
    ]);

    expect(part).toEqual({
      type: 'compaction',
      trigger: 'auto',
      summary: 'Earlier.',
      harnessMetadata: {
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
    });
  });

  it('keeps the raw text of a failed tool result', () => {
    const annotator = createPiHistoryAnnotator({
      sessionWorkDir: '/work',
      getBranch: () => [],
    });

    const [part] = annotator.annotate(
      {
        type: 'tool_execution_end',
        isError: true,
        result: { content: [{ type: 'text', text: 'Denied by policy' }] },
      },
      [
        {
          type: 'tool-result',
          toolCallId: 'call-1',
          toolName: 'bash',
          result: 'Denied by policy',
          isError: true,
        } as never,
      ],
    );

    expect(part).toMatchObject({
      providerMetadata: { pi: { rawText: ['Denied by policy'] } },
    });
  });

  it('copies the call metadata onto a failed result of a tool-call-only answer', () => {
    const annotator = createPiHistoryAnnotator({
      sessionWorkDir: '/work',
      getBranch: () => [],
    });
    annotator.annotate(
      {
        type: 'message_end',
        message: {
          role: 'assistant',
          content: [
            { type: 'toolCall', id: 'call-1', name: 'read', arguments: {} },
          ],
          stopReason: 'toolUse',
        },
      },
      [],
    );

    const [call] = annotator.annotate({ type: 'tool_execution_start' }, [
      {
        type: 'tool-call',
        toolCallId: 'call-1',
        toolName: 'read',
        input: '{}',
      } as never,
    ]);
    const [result] = annotator.annotate(
      {
        type: 'tool_execution_end',
        isError: true,
        result: { content: [{ type: 'text', text: 'Invalid input' }] },
      },
      [
        {
          type: 'tool-result',
          toolCallId: 'call-1',
          toolName: 'read',
          result: 'Invalid input',
          isError: true,
        } as never,
      ],
    );

    const assistant = {
      version: PI_HISTORY_VERSION,
      message: { stopReason: 'toolUse' },
      thinking: [],
    };
    expect(call).toMatchObject({ providerMetadata: { pi: { assistant } } });
    expect(result).toMatchObject({
      providerMetadata: { pi: { assistant, rawText: ['Invalid input'] } },
    });
  });
});
