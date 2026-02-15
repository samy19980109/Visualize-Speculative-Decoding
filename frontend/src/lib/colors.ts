import type { TokenStatus } from '../types';

export const STATUS_COLORS: Record<TokenStatus, string> = {
  accepted: '#22c55e',
  rejected: '#ef4444',
  resampled: '#f59e0b',
  bonus: '#3b82f6',
  pending: '#94a3b8',
};

/** Map entropy (0–4 nats typical) to circle radius (8–24px). */
export function entropyToRadius(entropy: number): number {
  const clamped = Math.max(0, Math.min(entropy, 4));
  return 8 + (clamped / 4) * 16;
}

/** Map acceptance probability (0–1) to opacity (0.4–1.0). */
export function acceptanceProbToOpacity(prob: number | null): number {
  if (prob === null) return 0.6;
  return 0.4 + Math.min(prob, 1) * 0.6;
}

/** Acceptance rate to color: red (<50%) → yellow (50-75%) → green (>75%). */
export function acceptanceRateColor(rate: number): string {
  if (rate < 0.5) return '#ef4444';
  if (rate < 0.75) return '#f59e0b';
  return '#22c55e';
}
