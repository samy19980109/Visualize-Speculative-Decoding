import { useReducer, useCallback } from 'react';
import type {
  ServerEvent,
  TokenInfo,
  TreeNode,
  MetricsSnapshot,
  TokenStatus,
} from '../types';

export interface SpecDecState {
  isGenerating: boolean;
  generatedText: string;
  tokens: TokenInfo[];
  treeRoots: TreeNode[];
  metricsHistory: MetricsSnapshot[];
  currentRound: number;
  error: string | null;
  finalStats: {
    totalTokens: number;
    totalRounds: number;
    acceptanceRate: number;
    speedup: number;
  } | null;
}

const initialState: SpecDecState = {
  isGenerating: false,
  generatedText: '',
  tokens: [],
  treeRoots: [],
  metricsHistory: [],
  currentRound: 0,
  error: null,
  finalStats: null,
};

type Action =
  | { type: 'START_GENERATION' }
  | { type: 'STOP_GENERATION' }
  | { type: 'DRAFT_TOKEN'; event: ServerEvent & { type: 'draft_token' } }
  | { type: 'VERIFY_RESULT'; event: ServerEvent & { type: 'verify_result' } }
  | { type: 'METRICS_UPDATE'; event: ServerEvent & { type: 'metrics' } }
  | { type: 'GENERATION_DONE'; event: ServerEvent & { type: 'done' } }
  | { type: 'ERROR'; event: ServerEvent & { type: 'error' } };

/** Recursively find a node by round+position. */
function findNode(node: TreeNode, round: number, position: number): TreeNode | null {
  if (node.round === round && node.position === position) return node;
  for (const child of node.children) {
    const found = findNode(child, round, position);
    if (found) return found;
  }
  return null;
}

/** Find the deepest node in the tree (rightmost leaf). */
function findDeepest(node: TreeNode): TreeNode {
  if (node.children.length === 0) return node;
  return findDeepest(node.children[node.children.length - 1]);
}

function reducer(state: SpecDecState, action: Action): SpecDecState {
  switch (action.type) {
    case 'START_GENERATION':
      return { ...initialState, isGenerating: true };

    case 'STOP_GENERATION':
      return { ...state, isGenerating: false };

    case 'DRAFT_TOKEN': {
      const e = action.event;
      // Deep clone to avoid StrictMode double-invocation issues
      const newRoots: TreeNode[] = structuredClone(state.treeRoots);

      // Find or create round node
      let roundNode = newRoots.find((r) => r.round === e.round);
      if (!roundNode) {
        roundNode = {
          id: `round-${e.round}`,
          token: `R${e.round}`,
          status: 'pending',
          round: e.round,
          position: -1,
          entropy: 0,
          logprob: 0,
          acceptanceProb: null,
          children: [],
        };
        newRoots.push(roundNode);
      }

      // Check if node already exists (StrictMode guard)
      if (findNode(roundNode, e.round, e.position)) {
        return state;
      }

      const draftNode: TreeNode = {
        id: `r${e.round}-d${e.position}`,
        token: e.token,
        status: 'pending',
        round: e.round,
        position: e.position,
        entropy: e.entropy,
        logprob: e.logprob,
        acceptanceProb: null,
        children: [],
      };

      // Build linear chain: position 0 → roundNode, else find parent recursively
      if (e.position === 0) {
        roundNode.children.push(draftNode);
      } else {
        const parent = findNode(roundNode, e.round, e.position - 1);
        if (parent) {
          parent.children.push(draftNode);
        } else {
          // Fallback: append to deepest node
          const deepest = findDeepest(roundNode);
          deepest.children.push(draftNode);
        }
      }

      const newToken: TokenInfo = {
        token: e.token,
        status: 'pending',
        round: e.round,
        position: e.position,
        logprob: e.logprob,
        entropy: e.entropy,
        acceptanceProb: null,
      };

      return {
        ...state,
        treeRoots: newRoots,
        tokens: [...state.tokens, newToken],
        currentRound: e.round,
      };
    }

    case 'VERIFY_RESULT': {
      const e = action.event;
      const newRoots = structuredClone(state.treeRoots);

      // For bonus tokens, add a new node
      if (e.status === 'bonus') {
        const roundNode = newRoots.find((r) => r.round === e.round);
        if (roundNode) {
          const deepest = findDeepest(roundNode);
          deepest.children.push({
            id: `r${e.round}-bonus`,
            token: e.token,
            status: 'bonus',
            round: e.round,
            position: e.position,
            entropy: 0,
            logprob: 0,
            acceptanceProb: 1.0,
            children: [],
          });
        }
      } else {
        // Update existing node status
        for (const root of newRoots) {
          const node = findNode(root, e.round, e.position);
          if (node) {
            // Only update if this is a status progression (don't re-update with rejected after resampled)
            if (node.status === 'pending' || e.status === 'resampled') {
              node.status = e.status as TokenStatus;
              node.acceptanceProb = e.acceptanceProb;
              if (e.status === 'resampled') {
                node.token = e.token;
              }
            }
            break;
          }
        }
      }

      // Update tokens list
      const newTokens = [...state.tokens];
      const tokenIdx = newTokens.findIndex(
        (t) => t.round === e.round && t.position === e.position
      );
      if (tokenIdx >= 0) {
        newTokens[tokenIdx] = {
          ...newTokens[tokenIdx],
          status: e.status as TokenStatus,
          acceptanceProb: e.acceptanceProb,
        };
        if (e.status === 'resampled') {
          newTokens[tokenIdx].token = e.token;
        }
      } else if (e.status === 'bonus') {
        newTokens.push({
          token: e.token,
          status: 'bonus',
          round: e.round,
          position: e.position,
          logprob: 0,
          entropy: 0,
          acceptanceProb: 1.0,
        });
      }

      // Build generated text from accepted/resampled/bonus tokens
      const accepted = newTokens.filter((t) =>
        ['accepted', 'resampled', 'bonus'].includes(t.status)
      );
      const generatedText = accepted.map((t) => t.token).join('');

      return {
        ...state,
        treeRoots: newRoots,
        tokens: newTokens,
        generatedText,
      };
    }

    case 'METRICS_UPDATE': {
      const e = action.event;
      const snapshot: MetricsSnapshot = {
        round: e.round,
        acceptanceRate: e.acceptanceRate,
        effectiveTps: e.effectiveTps,
        baselineTps: e.baselineTps,
        speedup: e.speedup,
        draftLatencyMs: e.draftLatencyMs,
        verifyLatencyMs: e.verifyLatencyMs,
      };
      return {
        ...state,
        metricsHistory: [...state.metricsHistory, snapshot],
      };
    }

    case 'GENERATION_DONE': {
      const e = action.event;
      return {
        ...state,
        isGenerating: false,
        generatedText: e.generatedText,
        finalStats: {
          totalTokens: e.totalTokens,
          totalRounds: e.totalRounds,
          acceptanceRate: e.finalAcceptanceRate,
          speedup: e.averageSpeedup,
        },
      };
    }

    case 'ERROR':
      return {
        ...state,
        isGenerating: false,
        error: action.event.message,
      };

    default:
      return state;
  }
}

export function useSpecDecState() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'draft_token':
        dispatch({ type: 'DRAFT_TOKEN', event });
        break;
      case 'verify_result':
        dispatch({ type: 'VERIFY_RESULT', event });
        break;
      case 'metrics':
        dispatch({ type: 'METRICS_UPDATE', event });
        break;
      case 'done':
        dispatch({ type: 'GENERATION_DONE', event });
        break;
      case 'error':
        dispatch({ type: 'ERROR', event });
        break;
    }
  }, []);

  const startGeneration = useCallback(() => {
    dispatch({ type: 'START_GENERATION' });
  }, []);

  const stopGeneration = useCallback(() => {
    dispatch({ type: 'STOP_GENERATION' });
  }, []);

  return { state, handleEvent, startGeneration, stopGeneration };
}
