import { HarnessError } from '@ai-sdk/harness';
import { secureJsonParse } from '@ai-sdk/provider-utils';
import { CURRENT_SESSION_VERSION } from '@earendil-works/pi-coding-agent';
import {
  PI_HISTORY_VERSION,
  type PiAssistantMetadata,
  type PiCompactionMetadata,
} from './pi-history-metadata';
import type { PiLifecycleData, PiSessionEntries } from './pi-lifecycle-state';
import { serializeToolOutput } from './pi-utils';

type UIPart = Record<string, any>;
type PiBlock = Record<string, unknown>;
type PiMessage = Record<string, unknown>;

/**
 * The parts of an AI SDK `UIMessage` that the rebuild reads. Messages
 * persisted with their provider metadata satisfy it.
 */
export interface PiHistoryUIMessage {
  readonly role: string;
  readonly parts: ReadonlyArray<UIPart>;
}

const OUTPUT_STATES = new Set([
  'output-available',
  'output-error',
  'output-denied',
]);

const piMetadataOf = (part: UIPart): Record<string, any> | undefined =>
  (
    part.providerMetadata ??
    part.callProviderMetadata ??
    part.resultProviderMetadata
  )?.pi;

const isHarnessSyntheticTool = (part: UIPart) =>
  part.toolCallId.startsWith('harness-');

function toolNameOf(part: UIPart): string | undefined {
  if (part.type === 'dynamic-tool') return part.toolName;
  return part.type.startsWith('tool-')
    ? part.type.slice('tool-'.length)
    : undefined;
}

function parseErrorText(text: string): unknown {
  try {
    return secureJsonParse(text);
  } catch {
    return text;
  }
}

function toolResultMessage(part: UIPart, toolName: string): PiMessage {
  const isError = part.state !== 'output-available';
  const envelope = isError
    ? typeof part.errorText === 'string'
      ? parseErrorText(part.errorText)
      : part.errorText
    : part.output;
  const record = envelope as { content?: PiBlock[]; details?: unknown };
  const rawText: string[] | undefined =
    part.resultProviderMetadata?.pi?.rawText;
  let textIndex = 0;
  const content = (
    Array.isArray(record?.content)
      ? record.content
      : [{ type: 'text', text: serializeToolOutput(envelope) }]
  ).map(block =>
    rawText && block.type === 'text'
      ? { ...block, text: rawText[textIndex++] ?? block.text }
      : block,
  );
  return {
    role: 'toolResult',
    toolCallId: part.toolCallId,
    toolName,
    content,
    ...(record?.details !== undefined ? { details: record.details } : {}),
    isError,
    timestamp: 0,
  };
}

function splitSteps(parts: ReadonlyArray<UIPart>): UIPart[][] {
  const steps: UIPart[][] = [[]];
  for (const part of parts) {
    if (part.type === 'step-start') steps.push([]);
    else steps.at(-1)!.push(part);
  }
  return steps.filter(step => step.length > 0);
}

/**
 * Rebuild the Pi lifecycle data of a session from its persisted UI messages,
 * so the UI messages can be the only stored copy of the conversation. The
 * messages must have been streamed by the Pi harness and keep their provider
 * metadata. A last step whose tool calls have no output yet is a tool batch
 * that waits on host input, and comes back as `pendingToolBatch`. A last step
 * without metadata was cut off before Pi finished it and is left out. Any
 * other step without metadata, or metadata of another version, throws a
 * `HarnessError`.
 */
export function piLifecycleDataFromUIMessages({
  messages,
}: {
  messages: ReadonlyArray<PiHistoryUIMessage>;
}): PiLifecycleData {
  const history: PiMessage[] = [];
  const compactions: (PiCompactionMetadata & { summary: string })[] = [];
  let awaitingToolCallIds: string[] = [];

  for (const message of messages) {
    if (message.role === 'user') {
      awaitingToolCallIds = [];
      history.push({
        role: 'user',
        content: message.parts
          .filter(part => part.type === 'text')
          .map(part => ({ type: 'text', text: part.text })),
        timestamp: 0,
      });
      continue;
    }
    if (message.role !== 'assistant') continue;
    const steps = splitSteps(message.parts);
    for (const step of steps) {
      let assistant: PiAssistantMetadata | undefined;
      for (const part of step) {
        assistant ??= piMetadataOf(part)?.assistant;
      }
      let thinkingIndex = 0;
      const content: PiBlock[] = [];
      const toolResults: PiMessage[] = [];
      const awaiting: string[] = [];
      for (const part of step) {
        if (part.type === 'reasoning') {
          const meta = assistant?.thinking[thinkingIndex++];
          content.push({
            type: 'thinking',
            thinking: part.text,
            ...(meta?.signature != null
              ? { thinkingSignature: meta.signature }
              : {}),
            ...(meta?.redacted ? { redacted: true } : {}),
          });
          continue;
        }
        if (part.type === 'text') {
          content.push({ type: 'text', text: part.text });
          continue;
        }
        const uiName = toolNameOf(part);
        if (uiName == null) continue;
        const pi = piMetadataOf(part);
        if (isHarnessSyntheticTool(part)) {
          if (pi?.compaction && typeof part.output?.summary === 'string') {
            compactions.push({
              ...pi.compaction,
              summary: part.output.summary,
            });
          }
          continue;
        }
        const name = pi?.nativeName ?? uiName;
        content.push({
          type: 'toolCall',
          id: part.toolCallId,
          name,
          arguments: pi?.rawArguments ?? part.input,
        });
        if (OUTPUT_STATES.has(part.state)) {
          toolResults.push(toolResultMessage(part, name));
        } else {
          awaiting.push(part.toolCallId);
        }
      }
      if (content.length === 0) continue;
      if (assistant == null) {
        if (message === messages.at(-1) && step === steps.at(-1)) continue;
        throw new HarnessError({
          message:
            'The UI messages contain an assistant step without Pi history metadata, so the Pi session cannot be rebuilt from them.',
        });
      }
      if (assistant.version !== PI_HISTORY_VERSION) {
        throw new HarnessError({
          message: `The UI messages use Pi history version ${assistant.version}, but this version of the Pi harness rebuilds version ${PI_HISTORY_VERSION}.`,
        });
      }
      awaitingToolCallIds = awaiting;
      history.push({ role: 'assistant', content, ...assistant.message });
      history.push(...toolResults);
    }
  }

  const compactionsAt = new Map<number, typeof compactions>();
  for (const compaction of compactions) {
    const at = Math.min(compaction.messageCountBefore, history.length);
    compactionsAt.set(at, [...(compactionsAt.get(at) ?? []), compaction]);
  }
  const timestamp = new Date(0).toISOString();
  const entries: Record<string, unknown>[] = [
    {
      type: 'session',
      version: CURRENT_SESSION_VERSION,
      id: 'rebuilt',
      timestamp,
      cwd: '',
    },
  ];
  const messageIds: string[] = [];
  let parentId: string | null = null;
  const nextId = () => entries.length.toString(16).padStart(8, '0');
  const append = (entry: Record<string, unknown>) => {
    const id = nextId();
    entries.push({ ...entry, id, parentId, timestamp });
    parentId = id;
    return id;
  };
  const appendCompactions = (messageCountBefore: number) => {
    for (const compaction of compactionsAt.get(messageCountBefore) ?? []) {
      append({
        type: 'compaction',
        summary: compaction.summary,
        firstKeptEntryId:
          messageIds[compaction.firstKeptMessageIndex] ?? nextId(),
        tokensBefore: compaction.tokensBefore,
        ...(compaction.details != null ? { details: compaction.details } : {}),
        ...(compaction.usage != null ? { usage: compaction.usage } : {}),
      });
    }
  };
  history.forEach((message, index) => {
    appendCompactions(index);
    messageIds.push(append({ type: 'message', message }));
  });
  appendCompactions(history.length);

  return {
    entries: entries as unknown as PiSessionEntries,
    ...(awaitingToolCallIds.length > 0
      ? {
          pendingToolBatch: {
            awaitingToolCallIds,
            completedToolResults: [],
            nativeApprovalDecisions: [],
          },
        }
      : {}),
  };
}
