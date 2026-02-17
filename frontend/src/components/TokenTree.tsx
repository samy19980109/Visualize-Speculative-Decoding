import { useRef, useMemo } from 'react';
import { computeTreeLayout } from '../lib/treeLayout';
import { STATUS_COLORS, entropyToRadius, acceptanceProbToOpacity } from '../lib/colors';
import { sectionHeaderStyle } from '../lib/styles';
import type { TreeNode } from '../types';

interface TokenTreeProps {
  roots: TreeNode[];
}

function Legend() {
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      {Object.entries(STATUS_COLORS).map(([status, color]) => (
        <div key={status} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: color }} />
          <span style={{ fontSize: 11, color: '#94a3b8', textTransform: 'capitalize' }}>{status}</span>
        </div>
      ))}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 8 }}>
        <span style={{ fontSize: 11, color: '#64748b' }}>Size = entropy | Opacity = acceptance prob</span>
      </div>
    </div>
  );
}

export function TokenTree({ roots }: TokenTreeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const dimensions = useMemo(() => {
    const maxDepth = roots.reduce((max, root) => {
      function depth(node: TreeNode): number {
        if (node.children.length === 0) return 1;
        return 1 + Math.max(...node.children.map(depth));
      }
      return Math.max(max, depth(root));
    }, 0);

    return {
      width: Math.max(600, roots.length * 120),
      height: Math.max(400, maxDepth * 80 + 100),
    };
  }, [roots]);

  const layoutRoot = useMemo(
    () => computeTreeLayout(roots, dimensions.width, dimensions.height),
    [roots, dimensions]
  );

  if (roots.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
        <h2 style={sectionHeaderStyle}>Token Tree</h2>
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#475569',
          fontSize: 14,
        }}>
          Tree visualization will appear during generation...
        </div>
        <Legend />
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      <h2 style={sectionHeaderStyle}>Token Tree</h2>

      <div style={{ flex: 1, overflow: 'auto' }}>
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
        >
          <g transform="translate(40, 40)">
            {/* Links */}
            {layoutRoot?.links().filter(l => l.source.data.id !== 'root').map((link, i) => {
              const isRejectedBranch = link.target.data.status === 'rejected';
              return (
                <line
                  key={`link-${i}`}
                  x1={link.source.x}
                  y1={link.source.y}
                  x2={link.target.x}
                  y2={link.target.y}
                  stroke={isRejectedBranch ? '#ef444480' : '#334155'}
                  strokeWidth={isRejectedBranch ? 1 : 2}
                  strokeDasharray={isRejectedBranch ? '4,4' : undefined}
                  style={{ transition: 'all 500ms ease' }}
                />
              );
            })}
            {/* Lines from root to round nodes */}
            {layoutRoot?.children?.map((child, i) => (
              <line
                key={`root-link-${i}`}
                x1={layoutRoot.x}
                y1={layoutRoot.y}
                x2={child.x}
                y2={child.y}
                stroke="#334155"
                strokeWidth={2}
                strokeDasharray="2,4"
              />
            ))}
            {/* Nodes */}
            {layoutRoot?.descendants().filter(d => d.data.id !== 'root').map((node, idx) => {
              const d = node.data;
              const r = d.position === -1 ? 12 : entropyToRadius(d.entropy);
              const opacity = d.position === -1 ? 1 : acceptanceProbToOpacity(d.acceptanceProb);
              const color = d.position === -1 ? '#6366f1' : STATUS_COLORS[d.status];

              return (
                <g
                  key={`${d.id}-${idx}`}
                  transform={`translate(${node.x}, ${node.y})`}
                  style={{ transition: 'transform 500ms ease, opacity 500ms ease' }}
                  opacity={opacity}
                >
                  <circle
                    r={r}
                    fill={`${color}30`}
                    stroke={color}
                    strokeWidth={2}
                    style={{ transition: 'fill 500ms, stroke 500ms' }}
                  />
                  <text
                    textAnchor="middle"
                    dy={4}
                    fontSize={Math.min(10, r)}
                    fill="#e2e8f0"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >
                    {d.token.length > 6 ? d.token.slice(0, 5) + '...' : d.token}
                  </text>
                  {/* Label below */}
                  {d.position >= 0 && (
                    <text
                      textAnchor="middle"
                      dy={r + 14}
                      fontSize={9}
                      fill="#64748b"
                      style={{ pointerEvents: 'none' }}
                    >
                      {d.status}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      <Legend />
    </div>
  );
}
