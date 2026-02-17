import { useReducer, useCallback, useMemo } from 'react';
import { findNode, findDeepest } from '../lib/treeUtils';
import type {
  ServerEvent,
  TokenInfo,
  TreeNode,
  MetricsSnapshot,
  TokenStatus,
} from '../types';

export interface SpecDecState {
  isGenerating: boolean;
  tokens: TokenInfo[];
  treeRoots: TreeNode[];
  metricsHistory: MetricsSnapshot[];
  // Incremental list of accepted token strings for text generation
  acceptedTokens: { token: string; round: number; position: number }[];
  // Final generated text from done event (authoritative)
  finalGeneratedText: string | null;
}

const initialState: SpecDecState = {
  isGenerating: false,
  tokens: [],
  treeRoots: [],
  metricsHistory: [],
  acceptedTokens: [],
  finalGeneratedText: null,
};

type Action =
  | { type: 'START_GENERATION' }
  | { type: 'STOP_GENERATION' }
  | { type: 'DRAFT_TOKEN'; event: ServerEvent & { type: 'draft_token' } }
  | { type: 'VERIFY_RESULT'; event: ServerEvent & { type: 'verify_result' } }
  | { type: 'METRICS_UPDATE'; event: ServerEvent & { type: 'metrics' } }
  | { type: 'GENERATION_DONE'; event: ServerEvent & { type: 'done' } }
  | { type: 'ERROR'; event: ServerEvent & { type: 'error' } };

/**
 * Surgical spine-copy: clone only the nodes on the path from root to the
 * target node, leaving all other subtrees shared with the previous state.
 */
function cloneRootWithUpdate(
  root: TreeNode,
  round: number,
  position: number,
  updater: (node: TreeNode) => TreeNode
): TreeNode {
  if (root.round === round && root.position === position) {
    return updater(root);
  }
  const newChildren = root.children.map((child) => {
    const found = findNode(child, round, position);
    if (found) {
      return cloneRootWithUpdate(child, round, position, updater);
    }
    return child; // unchanged subtree — no clone
  });
  return { ...root, children: newChildren };
}

function reducer(state: SpecDecState, action: Action): SpecDecState {
  switch (action.type) {
    case 'START_GENERATION':
      return { ...initialState, isGenerating: true };

    case 'STOP_GENERATION':
      return { ...state, isGenerating: false };

    case 'DRAFT_TOKEN': {
      const e = action.event;

      // Find existing round root or prepare to add new one
      const existingRootIdx = state.treeRoots.findIndex((r) => r.round === e.round);

      let newRoots: TreeNode[];

      if (existingRootIdx === -1) {
        // New round — create round node with draft as child
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
        const roundNode: TreeNode = {
          id: `round-${e.round}`,
          token: `R${e.round}`,
          status: 'pending',
          round: e.round,
          position: -1,
          entropy: 0,
          logprob: 0,
          acceptanceProb: null,
          children: e.position === 0 ? [draftNode] : [],
        };
        newRoots = [...state.treeRoots, roundNode];
      } else {
        const roundNode = state.treeRoots[existingRootIdx];

        // StrictMode guard: check if node already exists
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

        // Surgical update: clone spine to insertion point
        let updatedRound: TreeNode;
        if (e.position === 0) {
          updatedRound = { ...roundNode, children: [...roundNode.children, draftNode] };
        } else {
          const parent = findNode(roundNode, e.round, e.position - 1);
          if (parent) {
            updatedRound = cloneRootWithUpdate(roundNode, e.round, e.position - 1, (node) => ({
              ...node,
              children: [...node.children, draftNode],
            }));
          } else {
            // Fallback: append to deepest
            const deepest = findDeepest(roundNode);
            updatedRound = cloneRootWithUpdate(roundNode, deepest.round, deepest.position, (node) => ({
              ...node,
              children: [...node.children, draftNode],
            }));
          }
        }

        newRoots = [...state.treeRoots];
        newRoots[existingRootIdx] = updatedRound;
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
      };
    }

    case 'VERIFY_RESULT': {
      const e = action.event;

      // Update tree
      let newRoots: TreeNode[];
      if (e.status === 'bonus') {
        const rootIdx = state.treeRoots.findIndex((r) => r.round === e.round);
        if (rootIdx >= 0) {
          const roundNode = state.treeRoots[rootIdx];
          const deepest = findDeepest(roundNode);
          const bonusNode: TreeNode = {
            id: `r${e.round}-bonus`,
            token: e.token,
            status: 'bonus',
            round: e.round,
            position: e.position,
            entropy: 0,
            logprob: 0,
            acceptanceProb: 1.0,
            children: [],
          };
          const updated = cloneRootWithUpdate(roundNode, deepest.round, deepest.position, (node) => ({
            ...node,
            children: [...node.children, bonusNode],
          }));
          newRoots = [...state.treeRoots];
          newRoots[rootIdx] = updated;
        } else {
          newRoots = state.treeRoots;
        }
      } else {
        // Update existing node status via surgical spine copy
        newRoots = state.treeRoots.map((root) => {
          const node = findNode(root, e.round, e.position);
          if (node && (node.status === 'pending' || e.status === 'resampled')) {
            return cloneRootWithUpdate(root, e.round, e.position, (n) => ({
              ...n,
              status: e.status as TokenStatus,
              acceptanceProb: e.acceptanceProb,
              ...(e.status === 'resampled' ? { token: e.token } : {}),
            }));
          }
          return root;
        });
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
          ...(e.status === 'resampled' ? { token: e.token } : {}),
        };
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

      // Incremental: append to accepted tokens list if this is a new acceptance
      let newAccepted = state.acceptedTokens;
      if (['accepted', 'resampled', 'bonus'].includes(e.status)) {
        newAccepted = [...state.acceptedTokens, { token: e.token, round: e.round, position: e.position }];
      }

      return {
        ...state,
        treeRoots: newRoots,
        tokens: newTokens,
        acceptedTokens: newAccepted,
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
        finalGeneratedText: e.generatedText,
      };
    }

    case 'ERROR':
      return {
        ...state,
        isGenerating: false,
      };

    default:
      return state;
  }
}

export function useSpecDecState() {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Derive generated text from incremental accepted tokens list
  const generatedText = useMemo(() => {
    if (state.finalGeneratedText !== null) return state.finalGeneratedText;
    return state.acceptedTokens.map((t) => t.token).join('');
  }, [state.acceptedTokens, state.finalGeneratedText]);

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

  return { state, generatedText, handleEvent, startGeneration, stopGeneration };
}
