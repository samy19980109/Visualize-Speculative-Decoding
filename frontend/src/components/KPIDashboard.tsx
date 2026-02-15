import { AcceptanceGauge } from './AcceptanceGauge';
import { TPSChart } from './TPSChart';
import { LatencyChart } from './LatencyChart';
import { SpeedupIndicator } from './SpeedupIndicator';
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
      <h2 style={{ fontSize: 14, fontWeight: 600, color: '#94a3b8', margin: 0, textTransform: 'uppercase', letterSpacing: 1 }}>
        Metrics
      </h2>

      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        gap: 12,
      }}>
        <div style={{ backgroundColor: '#1e293b', borderRadius: 8, padding: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <AcceptanceGauge rate={acceptanceRate} />
        </div>
        <div style={{ backgroundColor: '#1e293b', borderRadius: 8, padding: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <SpeedupIndicator speedup={speedup} />
        </div>
        <div style={{ backgroundColor: '#1e293b', borderRadius: 8, padding: 8 }}>
          <TPSChart history={history} />
        </div>
        <div style={{ backgroundColor: '#1e293b', borderRadius: 8, padding: 8 }}>
          <LatencyChart history={history} />
        </div>
      </div>
    </div>
  );
}
