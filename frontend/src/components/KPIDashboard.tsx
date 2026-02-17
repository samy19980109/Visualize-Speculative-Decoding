import { AcceptanceGauge } from './AcceptanceGauge';
import { TPSChart } from './TPSChart';
import { LatencyChart } from './LatencyChart';
import { SpeedupIndicator } from './SpeedupIndicator';
import { sectionHeaderStyle, cardStyle } from '../lib/styles';
import type { MetricsSnapshot } from '../types';

interface KPIDashboardProps {
  history: MetricsSnapshot[];
}

export function KPIDashboard({ history }: KPIDashboardProps) {
  const latest = history[history.length - 1];
  const acceptanceRate = latest?.acceptanceRate ?? 0;
  const speedup = latest?.speedup ?? 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      <h2 style={sectionHeaderStyle}>Metrics</h2>

      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        gap: 12,
      }}>
        <div style={cardStyle}>
          <AcceptanceGauge rate={acceptanceRate} />
        </div>
        <div style={cardStyle}>
          <SpeedupIndicator speedup={speedup} />
        </div>
        <div style={{ ...cardStyle, alignItems: 'stretch', justifyContent: 'stretch' }}>
          <TPSChart history={history} />
        </div>
        <div style={{ ...cardStyle, alignItems: 'stretch', justifyContent: 'stretch' }}>
          <LatencyChart history={history} />
        </div>
      </div>
    </div>
  );
}
