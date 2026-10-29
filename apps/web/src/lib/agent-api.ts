import { apiOrigin } from './api';
import type { components } from './api.generated';

export type AgentProvider = components['schemas']['ProviderStatus'];
export type AgentSession = components['schemas']['Session'];
export type AgentEvent = components['schemas']['Event'];
export type AgentSessionRequest = components['schemas']['SessionCreate'];
export type AgentProviderRequest = components['schemas']['ProviderInput'];

export class AgentRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'AgentRequestError';
  }
}

/** Mutations are never retried: reconcile the saved session after an uncertain response. */
export async function agentRequest<T>(
  path: string,
  method = 'GET',
  body?: object,
  options: { timeoutMs?: number; keepalive?: boolean } = {},
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? (path.endsWith('/check') ? 130000 : 15000));
  try {
    const response = await fetch(`${apiOrigin}/api/v1${path}`, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: 'no-store',
      keepalive: options.keepalive,
      signal: controller.signal,
    });
    const value = await response.json();
    if (!response.ok)
      throw new AgentRequestError(
        typeof value.detail === 'string' ? value.detail : 'Agent Control request could not be verified.',
        response.status,
      );
    return value as T;
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error('No response. Refresh the saved session before trying another action.', { cause: error });
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

const sessionPath = (id: string) => `/agent-sessions/${encodeURIComponent(id)}`;
export const agentApi = {
  providers: () => agentRequest<AgentProvider[]>('/agent/providers'),
  saveProvider: (id: string, body: AgentProviderRequest) =>
    agentRequest<AgentProvider>(`/agent/providers/${encodeURIComponent(id)}`, 'PUT', body),
  deleteProvider: (id: string) => agentRequest<unknown>(`/agent/providers/${encodeURIComponent(id)}`, 'DELETE'),
  checkProvider: (id: string) =>
    agentRequest<AgentProvider>(`/agent/providers/${encodeURIComponent(id)}/check`, 'POST', {}),
  sessions: (project: string) =>
    agentRequest<AgentSession[]>(`/projects/${encodeURIComponent(project)}/agent-sessions`),
  create: (project: string, body: AgentSessionRequest) =>
    agentRequest<AgentSession>(`/projects/${encodeURIComponent(project)}/agent-sessions`, 'POST', body),
  session: (id: string) => agentRequest<AgentSession>(sessionPath(id)),
  events: (id: string, after = 0) => agentRequest<AgentEvent[]>(`${sessionPath(id)}/events?after=${after}`),
  propose: (id: string, requestId: string) =>
    agentRequest<AgentSession>(`${sessionPath(id)}/proposals`, 'POST', { request_id: requestId }),
  approve: (
    id: string,
    proposalId: string,
    authorityEpoch: number,
    admission?: {
      execution_observation_id: string;
      execution_observation_sha256: string;
      scene_unchanged_confirmed: boolean;
    },
  ) =>
    agentRequest<AgentSession>(`${sessionPath(id)}/approve`, 'POST', {
      proposal_id: proposalId,
      authority_epoch: authorityEpoch,
      ...admission,
    }),
  review: (
    id: string,
    body: { authority_epoch: number; observation_id: string; outcome: 'success' | 'failure' | 'unknown'; note: string },
  ) => agentRequest<AgentSession>(`${sessionPath(id)}/review`, 'POST', body),
  run: (id: string) => agentRequest<AgentSession>(`${sessionPath(id)}/run`, 'POST', {}),
  executionObservation: (id: string) =>
    agentRequest<AgentSession>(`${sessionPath(id)}/execution-observation`, 'POST', {}, { timeoutMs: 2000 }),
  heartbeat: (id: string, authorityEpoch: number) =>
    agentRequest<AgentSession>(
      `${sessionPath(id)}/heartbeat`,
      'POST',
      {
        authority_epoch: authorityEpoch,
        operator_present: true,
      },
      { timeoutMs: 1000 },
    ),
  imageUrl: (id: string, observationId: string) =>
    `${apiOrigin}/api/v1${sessionPath(id)}/observations/${encodeURIComponent(observationId)}/image`,
  stop: (id: string, keepalive = false) =>
    agentRequest<AgentSession>(`${sessionPath(id)}/stop`, 'POST', {}, { keepalive }),
  action: (id: string, action: 'observe' | 'cancel-inference' | 'pause' | 'stop') =>
    agentRequest<AgentSession>(`${sessionPath(id)}/${action}`, 'POST', {}),
};
