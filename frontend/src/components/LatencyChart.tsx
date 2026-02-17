import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { tooltipStyle, chartEmptyStyle } from '../lib/styles';
import type { MetricsSnapshot } from '../types';

interface LatencyChartProps {
  history: MetricsSnapshot[];
}

export function LatencyChart({ history }: LatencyChartProps) {
  // Show last 10 rounds
  const recent = history.slice(-10);
  const data = recent.map((m) => ({
    round: `R${m.round}`,
    draft: Math.round(m.draftLatencyMs),
    verify: Math.round(m.verifyLatencyMs),
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Latency (ms)</span>
      {data.length === 0 ? (
        <div style={chartEmptyStyle}>Waiting for data...</div>
      ) : (
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <XAxis dataKey="round" fontSize={9} tick={{ fill: '#64748b' }} />
            <YAxis hide />
            <Tooltip
              contentStyle={tooltipStyle.contentStyle}
              labelStyle={tooltipStyle.labelStyle}
            />
            <Bar dataKey="draft" fill="#22c55e" name="Draft (local)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="verify" fill="#f59e0b" name="Verify (API)" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
