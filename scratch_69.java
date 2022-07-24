import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.sequenceReconstruction(new int[]{4, 1, 5, 2, 6, 3},
                new int[][]{{5, 2, 6, 3}, {4, 1, 5, 2}}));

    }

    // LC444 JZOF II 115
    public boolean sequenceReconstruction(int[] nums, int[][] sequences) {
        int n = nums.length;
        List<Integer>[] outEdge = new List[n + 1];
        int[] indegree = new int[n + 1];
        BitSet bs = new BitSet(n + 1);
        for (int[] s : sequences) {
            Integer prev = null;
            for (int cur : s) {
                bs.set(cur);
                if (prev != null) {
                    indegree[cur]++;
                    if (outEdge[prev] == null) {
                        outEdge[prev] = new ArrayList<>();
                    }
                    outEdge[prev].add(cur);
                }
                prev = cur;
            }
        }
        if (bs.cardinality() != n) return false;
        Deque<Integer> q = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
                break;
            }
        }
        if (q.size() != 1) return false;
        List<Integer> topo = new ArrayList<>(n + 1);
        while (!q.isEmpty()) {
            int qs = q.size();
            if (qs > 1) return false;
            int p = q.poll();
            topo.add(p);
            if (outEdge[p] == null) continue;
            for (int next : outEdge[p]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        if (topo.size() != n) return false;
        for (int i = 0; i < n; i++) {
            if (topo.get(i) != nums[i]) return false;
        }
        return true;
    }

    // LC1184
    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start == destination) return 0;
        int forward = 0, backward = 0, startPoint = destination > start ? start : destination,
                endPoint = destination > startPoint ? destination : start, total = Arrays.stream(distance).sum();

        for (int i = startPoint; i < endPoint; i++) {
            forward += distance[i];
        }
        backward = total - forward;
        return Math.min(backward, forward);
    }

    // LC558 ** Quad Tree
    public Node intersect(Node quadTree1, Node quadTree2) {
        if (quadTree1.isLeaf) {
            if (quadTree1.val) {
                return new Node() {{
                    val = true;
                    isLeaf = true;
                }};
            }
            return new Node(quadTree2.val, quadTree2.isLeaf, quadTree2.topLeft, quadTree2.topRight, quadTree2.bottomLeft, quadTree2.bottomRight);
        }
        if (quadTree2.isLeaf) {
            return intersect(quadTree2, quadTree1);
        }
        Node o1 = intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node o2 = intersect(quadTree1.topRight, quadTree2.topRight);
        Node o3 = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node o4 = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node() {{
                val = o1.val;
                isLeaf = true;
            }};
        }
        return new Node(false, false, o1, o2, o3, o4);
    }

}

// Definition for a QuadTree node.
class Node {
    public boolean val;
    public boolean isLeaf; // true means all value of the 4 corner is the same
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;

    public Node() {
    }

    public Node(boolean _val, boolean _isLeaf, Node _topLeft, Node _topRight, Node _bottomLeft, Node _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
}
