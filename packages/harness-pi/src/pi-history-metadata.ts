import type { HarnessV1Metadata, HarnessV1StreamPart } from '@ai-sdk/harness';
import { secureJsonParse } from '@ai-sdk/provider-utils';
import type { SessionEntry } from '@earendil-works/pi-coding-agent';
import type { PiSessionEvent } from './pi-events';

type PiMetadata = HarnessV1Metadata[string];
type JSONValue = PiMetadata[string];

export type PiAssistantMetadata = {
  message: Record<string, JSONValue>;
  thinking: { signature: string | null; redacted: boolean }[];
};

export type PiCompactionMetadata = {
  messageCountBefore: number;
  firstKeptMessageIndex: number;
  tokensBefore: number;
  details: JSONValue;
  usage: JSONValue;
};

export interface PiHistoryAnnotator {
  annotate(
    event: PiSessionEvent,
    parts: HarnessV1StreamPart[],
  ): HarnessV1StreamPart[];
}

const ANNOTATED_EVENT_TYPES = new Set([
  'message_end',
  'tool_execution_start',
  'tool_execution_end',
  'tool_result',
  'compaction_end',
]);

const withPi = (
  part: HarnessV1StreamPart,
  pi: PiMetadata,
): HarnessV1StreamPart => {
  const key =
    part.type === 'tool-call' || part.type === 'tool-result'
      ? 'providerMetadata'
      : 'harnessMetadata';
  const current = (part as Record<string, unknown>)[key] as
    | HarnessV1Metadata
    | undefined;
  return {
    ...part,
    [key]: { ...current, pi: { ...current?.pi, ...pi } },
  } as HarnessV1StreamPart;
};

const toJSON = (value: unknown): JSONValue =>
  secureJsonParse(JSON.stringify(value ?? null)) as JSONValue;

function compactionMetadata(
  branch: ReadonlyArray<SessionEntry>,
): PiCompactionMetadata | undefined {
  const entry = branch.at(-1);
  if (entry?.type !== 'compaction') return undefined;
  let messageCountBefore = 0;
  let firstKeptMessageIndex: number | undefined;
  for (const candidate of branch.slice(0, -1)) {
    if (candidate.id === entry.firstKeptEntryId) {
      firstKeptMessageIndex = messageCountBefore;
    }
    if (candidate.type === 'message' && candidate.message.role !== 'system') {
      messageCountBefore += 1;
    }
  }
  return {
    messageCountBefore,
    firstKeptMessageIndex: firstKeptMessageIndex ?? messageCountBefore,
    tokensBefore: entry.tokensBefore,
    details: toJSON(entry.details),
    usage: toJSON(entry.usage),
  };
}

export function createPiHistoryAnnotator({
  sessionWorkDir,
  getBranch,
}: {
  sessionWorkDir: string;
  getBranch: () => ReadonlyArray<SessionEntry>;
}): PiHistoryAnnotator {
  let pendingAssistant: PiAssistantMetadata | undefined;
  const toolCalls = new Map<string, PiMetadata>();

  return {
    annotate(event, parts) {
      if (!ANNOTATED_EVENT_TYPES.has(event.type)) return parts;
      const message = event.message;
      if (event.type === 'message_end' && message?.role === 'assistant') {
        const content = Array.isArray(message.content)
          ? (message.content as Record<string, unknown>[])
          : [];
        toolCalls.clear();
        for (const block of content) {
          if (block.type !== 'toolCall') continue;
          const serialized = JSON.stringify(block.arguments ?? null);
          toolCalls.set(block.id as string, {
            nativeName: block.name as string,
            ...(serialized.includes(sessionWorkDir)
              ? { rawArguments: secureJsonParse(serialized) as JSONValue }
              : {}),
          });
        }
        const { content: _content, role: _role, ...fields } = message;
        const assistant: PiAssistantMetadata = {
          message: toJSON(fields) as Record<string, JSONValue>,
          thinking: content
            .filter(block => block.type === 'thinking')
            .map(block => ({
              signature: (block.thinkingSignature as string) ?? null,
              redacted: block.redacted === true,
            })),
        };
        let lastEnd = -1;
        parts.forEach((part, index) => {
          if (part.type === 'text-end' || part.type === 'reasoning-end')
            lastEnd = index;
        });
        if (lastEnd === -1) {
          pendingAssistant = assistant;
          return parts;
        }
        pendingAssistant = undefined;
        return parts.map((part, index) =>
          index === lastEnd ? withPi(part, { assistant: assistant }) : part,
        );
      }
      return parts.map(part => {
        if (part.type === 'compaction') {
          const compaction = compactionMetadata(getBranch());
          return compaction ? withPi(part, { compaction: compaction }) : part;
        }
        if (part.type === 'tool-call') {
          const { nativeName, ...call } = toolCalls.get(part.toolCallId) ?? {};
          toolCalls.delete(part.toolCallId);
          const pi: PiMetadata = {
            ...call,
            ...(nativeName != null && nativeName !== part.toolName
              ? { nativeName }
              : {}),
            ...(pendingAssistant ? { assistant: pendingAssistant } : {}),
          };
          pendingAssistant = undefined;
          return Object.keys(pi).length === 0 ? part : withPi(part, pi);
        }
        if (part.type === 'tool-result') {
          const result = (event as { result?: { content?: unknown } }).result;
          const content = Array.isArray(result?.content)
            ? (result.content as Record<string, unknown>[])
            : [];
          const texts = content
            .filter(block => block.type === 'text')
            .map(block => block.text as string);
          return event.isError === true ||
            texts.some(text => text.includes(sessionWorkDir))
            ? withPi(part, { rawText: texts })
            : part;
        }
        return part;
      });
    },
  };
}
