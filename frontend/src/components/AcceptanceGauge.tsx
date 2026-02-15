import { PieChart, Pie, Cell } from 'recharts';
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Acceptance Rate</span>
      <div style={{ position: 'relative', width: 100, height: 100 }}>
        <PieChart width={100} height={100}>
          <Pie
            data={data}
            cx={45}
            cy={45}
            innerRadius={30}
            outerRadius={42}
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
