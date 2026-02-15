interface SpeedupIndicatorProps {
  speedup: number;
}

export function SpeedupIndicator({ speedup }: SpeedupIndicatorProps) {
  const displayValue = speedup > 0 ? speedup.toFixed(1) : '—';
  const color = speedup >= 2 ? '#22c55e' : speedup >= 1.5 ? '#f59e0b' : '#94a3b8';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 4,
    }}>
      <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Speedup</span>
      <span style={{
        fontSize: 36,
        fontWeight: 800,
        color,
        lineHeight: 1,
        fontVariantNumeric: 'tabular-nums',
      }}>
        {displayValue}x
      </span>
      <span style={{ fontSize: 10, color: '#64748b' }}>vs autoregressive</span>
    </div>
  );
}
