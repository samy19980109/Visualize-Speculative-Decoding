import type { TreeNode } from '../types';

/** Recursively find a node by round+position. */
export function findNode(node: TreeNode, round: number, position: number): TreeNode | null {
  if (node.round === round && node.position === position) return node;
  for (const child of node.children) {
    const found = findNode(child, round, position);
    if (found) return found;
  }
  return null;
}

/** Find the deepest node in the tree (rightmost leaf). */
export function findDeepest(node: TreeNode): TreeNode {
  if (node.children.length === 0) return node;
  return findDeepest(node.children[node.children.length - 1]);
}
