import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';
import { tooltipStyle, chartEmptyStyle } from '../lib/styles';
import type { MetricsSnapshot } from '../types';

interface TPSChartProps {
  history: MetricsSnapshot[];
}

export function TPSChart({ history }: TPSChartProps) {
  const data = history.map((m) => ({
    round: m.round,
    effective: Math.round(m.effectiveTps * 10) / 10,
    baseline: Math.round(m.baselineTps * 10) / 10,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Tokens/sec</span>
      {data.length === 0 ? (
        <div style={chartEmptyStyle}>Waiting for data...</div>
      ) : (
        <ResponsiveContainer width="100%" height={100}>
          <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <XAxis dataKey="round" hide />
            <YAxis hide domain={[0, 'auto']} />
            <Tooltip
              contentStyle={tooltipStyle.contentStyle}
              labelStyle={tooltipStyle.labelStyle}
            />
            <Area
              type="monotone"
              dataKey="effective"
              fill="#6366f120"
              stroke="none"
            />
            <Line type="monotone" dataKey="effective" stroke="#6366f1" strokeWidth={2} dot={false} name="Speculative" />
            <Line type="monotone" dataKey="baseline" stroke="#ef4444" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Baseline" />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
