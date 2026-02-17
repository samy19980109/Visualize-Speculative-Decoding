import type { ReactNode } from 'react';

interface LayoutProps {
  promptPanel: ReactNode;
  treePanel: ReactNode;
  textPanel: ReactNode;
  kpiPanel: ReactNode;
}

export function Layout({ promptPanel, treePanel, textPanel, kpiPanel }: LayoutProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: '#0f172a',
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    }}>
      {/* Header */}
      <header style={{
        padding: '16px 24px',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexShrink: 0,
      }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>
          SpeculatoViz
        </h1>
        <span style={{ fontSize: 12, color: '#64748b' }}>
          Real-time Speculative Decoding Visualization
        </span>
      </header>

      {/* Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '380px 1fr',
        gridTemplateRows: '1fr 320px',
        gap: 1,
        flex: 1,
        minHeight: 0,
        backgroundColor: '#1e293b',
      }}>
        {/* Top-left: Prompt input */}
        <div style={{ backgroundColor: '#0f172a', padding: 16, overflow: 'auto' }}>
          {promptPanel}
        </div>

        {/* Top-right: Token tree */}
        <div style={{ backgroundColor: '#0f172a', padding: 16, overflow: 'hidden' }}>
          {treePanel}
        </div>

        {/* Bottom-left: Text output */}
        <div style={{ backgroundColor: '#0f172a', padding: 16, overflow: 'auto' }}>
          {textPanel}
        </div>

        {/* Bottom-right: KPI dashboard */}
        <div style={{ backgroundColor: '#0f172a', padding: 16, overflow: 'auto' }}>
          {kpiPanel}
        </div>
      </div>
    </div>
  );
}
