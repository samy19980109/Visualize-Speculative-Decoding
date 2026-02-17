import type { CSSProperties } from 'react';

export const sectionHeaderStyle: CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: '#94a3b8',
  margin: 0,
  textTransform: 'uppercase',
  letterSpacing: 1,
};

export const cardStyle: CSSProperties = {
  backgroundColor: '#1e293b',
  borderRadius: 8,
  padding: 8,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};

export const tooltipStyle = {
  contentStyle: {
    backgroundColor: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 8,
    fontSize: 11,
  } as CSSProperties,
  labelStyle: { color: '#94a3b8' } as CSSProperties,
};

export const chartEmptyStyle: CSSProperties = {
  height: 100,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#475569',
  fontSize: 12,
};
