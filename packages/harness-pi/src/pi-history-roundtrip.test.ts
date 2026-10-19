import type { HarnessV1NetworkSandboxSession } from '@ai-sdk/harness';
import {
  HarnessAgent,
  type HarnessAgentContinueTurnState,
  type HarnessAgentResumeSessionState,
} from '@ai-sdk/harness/agent';
import { createJustBashNetworkSandboxSession } from '@ai-sdk/sandbox-just-bash';
import {
  readUIMessageStream,
  tool,
  type ToolApprovalResponse,
  type ToolResultPart,
  type UIMessage,
  type UIMessageChunk,
} from 'ai';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { z } from 'zod/v4';
import { piLifecycleDataFromUIMessages } from './pi-history-from-parts';
import type { PiLifecycleData } from './pi-lifecycle-state';
import {
  close,
  createFakePi,
  createScriptedModelServerThenDone,
  listen,
  textChunks,
  toolCallChunks,
} from './test-helpers';

const model = createScriptedModelServerThenDone();
const agentDir = mkdtempSync(path.join(tmpdir(), 'pi-history-roundtrip-'));
const workDir = '/sandbox/workspace';
let modelUrl = '';
let sandboxSession: HarnessV1NetworkSandboxSession;

const executions: string[] = [];
let releaseSlow = () => {};

beforeAll(async () => {
  process.env.PI_CODING_AGENT_DIR = agentDir;
  modelUrl = await listen(model.server);
});
afterAll(async () => {
  await close(model.server);
  rmSync(agentDir, { recursive: true, force: true });
});
beforeEach(async () => {
  model.reset();
  executions.length = 0;
  writeFileSync(path.join(agentDir, 'settings.json'), '{}');
  sandboxSession = await createJustBashNetworkSandboxSession({ cwd: workDir });
});

const tools = {
  getUserName: tool({
    description: 'Ask the user for their name.',
    inputSchema: z.object({}),
  }),
  weather: tool({
    description: 'Get the weather for a city.',
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => {
      executions.push(`weather:${city}`);
      return { city, celsius: 12 };
    },
  }),
  lookup: tool({
    description: 'Look up a fact.',
    inputSchema: z.object({ topic: z.string() }),
    execute: async ({ topic }) => {
      executions.push(`lookup:${topic}`);
      return `${topic} is fine`;
    },
  }),
  slow: tool({
    description: 'A slow lookup.',
    inputSchema: z.object({}),
    execute: () =>
      new Promise<string>(resolve => {
        releaseSlow = () => resolve('late');
      }),
  }),
};

type LifecycleJSON = HarnessAgentResumeSessionState['data'];

type Lifecycle = {
  resumeFrom?: HarnessAgentResumeSessionState;
  continueFrom?: HarnessAgentContinueTurnState;
};

type Continuations = {
  toolApprovalContinuations?: ToolApprovalResponse[];
  toolResultContinuations?: ToolResultPart[];
};

async function startSession(lifecycle: Lifecycle = {}) {
  const agent = new HarnessAgent({
    harness: createFakePi(modelUrl, { reattachInProcess: false, agentDir }),
    model: 'fake/fake-model',
    tools,
    toolApproval: { weather: 'user-approval' },
  });
  const session = await agent.createSession({
    sandboxSession,
    sessionId: 'roundtrip',
    ...lifecycle,
  });
  return { agent, session };
}

async function collect(
  result: { toUIMessageStream: () => ReadableStream<UIMessageChunk> },
  onMessage: (message: UIMessage) => void = () => {},
): Promise<UIMessage> {
  let latest: UIMessage | undefined;
  for await (const message of readUIMessageStream({
    stream: result.toUIMessageStream(),
  })) {
    latest = message;
    onMessage(message);
  }
  return latest!;
}

async function waitFor(condition: () => boolean) {
  while (!condition()) await new Promise(resolve => setTimeout(resolve, 10));
}

const rebuild = (prompt: string, answer: UIMessage) =>
  piLifecycleDataFromUIMessages({
    messages: [
      { role: 'user', parts: [{ type: 'text', text: prompt }] },
      answer,
    ],
  });

const resumeState = (
  data: PiLifecycleData | LifecycleJSON,
): HarnessAgentResumeSessionState => ({
  type: 'resume-session',
  harnessId: 'pi',
  specificationVersion: 'harness-v1',
  data: data as LifecycleJSON,
});

async function firstRequest(
  lifecycle: Lifecycle,
  next: { prompt: string } | Continuations,
) {
  const { agent, session } = await startSession(lifecycle);
  const before = model.requests.length;
  await collect(
    'prompt' in next
      ? await agent.stream({ session, prompt: next.prompt })
      : await agent.continueStream({ session, ...next }),
  );
  await session.stop();
  return model.requests[before];
}

describe('Pi history rebuilt from UI messages', () => {
  it('sends the same request for the next prompt after a tool turn', async () => {
    await sandboxSession.writeTextFile({
      path: `${workDir}/pi-roundtrip/notes.txt`,
      content: 'The code word is PELICAN.\n',
    });
    model.enqueue(
      toolCallChunks({
        id: 'call-bad',
        name: 'read',
        args: { path: 'notes.txt' },
      }),
      toolCallChunks(
        { id: 'call-read', name: 'read', args: { file_path: 'notes.txt' } },
        {
          id: 'call-missing',
          name: 'read',
          args: { file_path: 'missing.txt' },
        },
      ),
      textChunks('The code word is PELICAN.'),
    );
    const { agent, session } = await startSession();
    const prompt = 'What is the code word in notes.txt?';
    const answer = await collect(await agent.stream({ session, prompt }));
    const native = (await session.detach()).data;
    const next = { prompt: 'Thanks' };

    expect(
      await firstRequest(
        { resumeFrom: resumeState(rebuild(prompt, answer)) },
        next,
      ),
    ).toEqual(await firstRequest({ resumeFrom: resumeState(native) }, next));
  });

  it('continues an answer paused mid-stream the same way', async () => {
    model.enqueue({
      chunks: textChunks(
        'Once ',
        'upon ',
        'a ',
        'time ',
        'there ',
        'was ',
        'a fox.',
      ),
      delayMs: 50,
    });
    const { agent, session } = await startSession();
    const prompt = 'Tell me a story.';
    let partial: UIMessage | undefined;
    const reading = agent
      .stream({ session, prompt })
      .then(result => collect(result, message => (partial = message)));
    await waitFor(
      () => partial?.parts.some(part => part.type === 'text') === true,
    );
    const continueFrom = await session.suspendTurn();
    const rebuilt = rebuild(prompt, await reading);

    expect(
      await firstRequest(
        { continueFrom: { ...continueFrom, data: rebuilt as LifecycleJSON } },
        {},
      ),
    ).toEqual(await firstRequest({ continueFrom }, {}));
  });

  it('continues a paused tool batch without running a finished tool again', async () => {
    model.enqueue(
      toolCallChunks(
        { id: 'call-lookup', name: 'lookup', args: { topic: 'foxes' } },
        { id: 'call-weather', name: 'weather', args: { city: 'Paris' } },
        { id: 'call-name', name: 'getUserName', args: {} },
      ),
    );
    const { agent, session } = await startSession();
    const prompt = 'Look up foxes, get the weather in Paris, and ask my name.';
    const answer = await collect(await agent.stream({ session, prompt }));
    const approval = answer.parts.find(
      part => 'approval' in part && part.state === 'approval-requested',
    ) as { approval: { id: string } };
    const continueFrom = await session.suspendTurn();
    expect(executions.splice(0)).toEqual(['lookup:foxes']);
    const rebuilt = rebuild(prompt, answer);
    const continuations: Continuations = {
      toolApprovalContinuations: [
        {
          type: 'tool-approval-response',
          approvalId: approval.approval.id,
          approved: true,
        },
      ],
      toolResultContinuations: [
        {
          type: 'tool-result',
          toolCallId: 'call-name',
          toolName: 'getUserName',
          output: { type: 'json', value: { name: 'Felix' } },
        },
      ],
    };

    expect(new Set(rebuilt.pendingToolBatch?.awaitingToolCallIds)).toEqual(
      new Set(
        (continueFrom.data as PiLifecycleData).pendingToolBatch
          ?.awaitingToolCallIds,
      ),
    );
    const native = await firstRequest({ continueFrom }, continuations);
    const nativeRuns = executions.splice(0);
    const fromRebuild = await firstRequest(
      { continueFrom: { ...continueFrom, data: rebuilt as LifecycleJSON } },
      continuations,
    );
    expect(fromRebuild).toEqual(native);
    expect(executions).toEqual(nativeRuns);
    expect(executions).toEqual(['weather:Paris']);
  });

  it('sends the same request after an automatic compaction', async () => {
    writeFileSync(
      path.join(agentDir, 'settings.json'),
      JSON.stringify({
        compaction: {
          enabled: true,
          reserveTokens: 127_950,
          keepRecentTokens: 1,
        },
      }),
    );
    model.enqueue(
      textChunks('A long answer about foxes and their habits in the wild.'),
      textChunks('Summary: we talked about foxes.'),
    );
    const { agent, session } = await startSession();
    const prompt = 'Tell me about foxes.';
    const answer = await collect(await agent.stream({ session, prompt }));
    const native = (await session.detach()).data as PiLifecycleData;
    expect(native.entries?.some(entry => entry.type === 'compaction')).toBe(
      true,
    );
    const next = { prompt: 'And wolves?' };

    expect(
      await firstRequest(
        { resumeFrom: resumeState(rebuild(prompt, answer)) },
        next,
      ),
    ).toEqual(await firstRequest({ resumeFrom: resumeState(native) }, next));
  });

  it('starts a valid next prompt after a cancelled tool call', async () => {
    model.enqueue(toolCallChunks({ id: 'call-slow', name: 'slow', args: {} }));
    const { agent, session } = await startSession();
    const prompt = 'Look it up.';
    const cancel = new AbortController();
    let partial: UIMessage | undefined;
    const reading = agent
      .stream({ session, prompt, abortSignal: cancel.signal })
      .then(result => collect(result, message => (partial = message)));
    await waitFor(
      () => partial?.parts.some(part => part.type === 'tool-slow') === true,
    );
    cancel.abort();
    releaseSlow();
    await reading.catch(() => {});
    await session.stop().catch(() => {});
    const rebuilt = rebuild(prompt, {
      ...partial!,
      parts: partial!.parts.filter(
        part =>
          !part.type.startsWith('tool-') ||
          ('state' in part && String(part.state).startsWith('output-')),
      ),
    });

    expect(rebuilt.pendingToolBatch).toBeUndefined();
    const request = await firstRequest(
      { resumeFrom: resumeState(rebuilt) },
      { prompt: 'Try again later.' },
    );
    expect(
      request.messages
        .flatMap(message => message.tool_calls ?? [])
        .map(call => call.id),
    ).toEqual(
      request.messages.flatMap(message =>
        message.role === 'tool' ? [message.tool_call_id] : [],
      ),
    );
  });
});
