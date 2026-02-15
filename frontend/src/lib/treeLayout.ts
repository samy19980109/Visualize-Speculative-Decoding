import * as d3 from 'd3';
import type { TreeNode } from '../types';

export interface D3TreeNode extends d3.HierarchyPointNode<TreeNode> {}

/**
 * Create a virtual root that parents all round nodes, then compute
 * D3 tree layout positions.
 */
export function computeTreeLayout(
  roots: TreeNode[],
  width: number,
  height: number
): D3TreeNode | null {
  if (roots.length === 0) return null;

  // Virtual root
  const virtualRoot: TreeNode = {
    id: 'root',
    token: '',
    status: 'pending',
    round: 0,
    position: -1,
    entropy: 0,
    logprob: 0,
    acceptanceProb: null,
    children: roots,
  };

  const hierarchy = d3.hierarchy(virtualRoot);
  const treeLayout = d3.tree<TreeNode>().size([width - 80, height - 80]);
  return treeLayout(hierarchy);
}
