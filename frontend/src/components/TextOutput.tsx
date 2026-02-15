import { STATUS_COLORS } from '../lib/colors';
import type { TokenInfo } from '../types';

interface TextOutputProps {
  tokens: TokenInfo[];
  generatedText?: string;
  isGenerating: boolean;
}

export function TextOutput({ tokens, isGenerating }: TextOutputProps) {
  // Show color-coded tokens that have been accepted/resampled/bonus
  const visibleTokens = tokens.filter((t) =>
    ['accepted', 'resampled', 'bonus'].includes(t.status)
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      <h2 style={{ fontSize: 14, fontWeight: 600, color: '#94a3b8', margin: 0, textTransform: 'uppercase', letterSpacing: 1 }}>
        Generated Text
      </h2>

      <div style={{
        flex: 1,
        backgroundColor: '#1e293b',
        borderRadius: 8,
        padding: 12,
        fontSize: 14,
        lineHeight: 1.8,
        overflow: 'auto',
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      }}>
        {visibleTokens.length === 0 && !isGenerating && (
          <span style={{ color: '#475569' }}>Output will appear here...</span>
        )}
        {visibleTokens.map((token, i) => (
          <span
            key={`${token.round}-${token.position}-${i}`}
            title={`Round ${token.round}, Pos ${token.position}\nStatus: ${token.status}\nLogprob: ${token.logprob.toFixed(3)}\nEntropy: ${token.entropy.toFixed(3)}${token.acceptanceProb !== null ? `\nAcceptance: ${(token.acceptanceProb * 100).toFixed(1)}%` : ''}`}
            style={{
              color: STATUS_COLORS[token.status],
              borderBottom: `2px solid ${STATUS_COLORS[token.status]}40`,
              cursor: 'help',
              transition: 'opacity 300ms',
            }}
          >
            {token.token}
          </span>
        ))}
        {isGenerating && (
          <span style={{
            display: 'inline-block',
            width: 8,
            height: 16,
            backgroundColor: '#6366f1',
            marginLeft: 2,
            animation: 'blink 1s infinite',
          }} />
        )}
      </div>

      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}
