import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { acceptanceRateColor } from '../lib/colors';

interface AcceptanceGaugeProps {
  rate: number; // 0–1
}

export function AcceptanceGauge({ rate }: AcceptanceGaugeProps) {
  const pct = Math.round(rate * 100);
  const color = acceptanceRateColor(rate);

  const data = [
    { name: 'accepted', value: rate },
    { name: 'rejected', value: 1 - rate },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, width: '100%' }}>
      <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Acceptance Rate</span>
      <div style={{ position: 'relative', width: '100%', maxWidth: 120, aspectRatio: '1' }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius="60%"
              outerRadius="85%"
              startAngle={90}
              endAngle={-270}
              dataKey="value"
              stroke="none"
              animationDuration={500}
            >
              <Cell fill={color} />
              <Cell fill="#1e293b" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          fontSize: 18,
          fontWeight: 700,
          color,
        }}>
          {pct}%
        </div>
      </div>
    </div>
  );
}
