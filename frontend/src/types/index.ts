// --- Enums ---

export type TokenStatus = 'pending' | 'accepted' | 'rejected' | 'resampled' | 'bonus';
export type EventType = 'draft_token' | 'verify_result' | 'metrics' | 'done' | 'error';

// --- Incoming request ---

export interface StartGenerationRequest {
  prompt: string;
  maxTokens: number;
  temperature: number;
  k: number;
}

// --- Shared ---

export interface TopToken {
  token: string;
  logprob: number;
}

// --- Events from server ---

export interface DraftTokenEvent {
  type: 'draft_token';
  round: number;
  position: number;
  token: string;
  tokenId: number;
  logprob: number;
  entropy: number;
  topTokens: TopToken[];
  draftTimeMs: number;
}

export interface VerifyResultEvent {
  type: 'verify_result';
  round: number;
  position: number;
  token: string;
  tokenId: number;
  status: TokenStatus;
  draftLogprob: number;
  targetLogprob: number | null;
  acceptanceProb: number | null;
  targetEntropy: number | null;
  targetTopTokens: TopToken[];
  verifyTimeMs: number;
}

export interface MetricsEvent {
  type: 'metrics';
  round: number;
  acceptanceRate: number;
  roundAccepted: number;
  roundTotal: number;
  effectiveTps: number;
  baselineTps: number;
  speedup: number;
  draftLatencyMs: number;
  verifyLatencyMs: number;
  totalTokensGenerated: number;
}

export interface GenerationDoneEvent {
  type: 'done';
  totalTokens: number;
  totalRounds: number;
  finalAcceptanceRate: number;
  averageSpeedup: number;
  generatedText: string;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
  round: number | null;
}

export type ServerEvent =
  | DraftTokenEvent
  | VerifyResultEvent
  | MetricsEvent
  | GenerationDoneEvent
  | ErrorEvent;

// --- Tree nodes for D3 visualization ---

export interface TreeNode {
  id: string;
  token: string;
  status: TokenStatus;
  round: number;
  position: number;
  entropy: number;
  logprob: number;
  acceptanceProb: number | null;
  children: TreeNode[];
}

// --- App state ---

export interface TokenInfo {
  token: string;
  status: TokenStatus;
  round: number;
  position: number;
  logprob: number;
  entropy: number;
  acceptanceProb: number | null;
}

export interface MetricsSnapshot {
  round: number;
  acceptanceRate: number;
  effectiveTps: number;
  baselineTps: number;
  speedup: number;
  draftLatencyMs: number;
  verifyLatencyMs: number;
}
