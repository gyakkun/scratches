import java.util.*;
import java.util.stream.Collectors;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

//        System.out.println(s.domino(3, 3, new int[][]{}));
        System.out.println(s.minBuildTime(new int[]{94961, 39414, 41263, 7809, 41473},
                90));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1602
    public TreeNode findNearestRightNode(TreeNode root, TreeNode u) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                TreeNode p = q.poll();
                if (p == u) {
                    if (i == qs - 1) return null;
                    return q.poll();
                }
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
        }
        return null;
    }

    // LC1199 Hard 可以二分, 也可以直接优先队列解决, 非常优雅
    public int minBuildTime(int[] blocks, int split) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i : blocks) {
            pq.offer(i);
        }
        while (pq.size() > 1) {
            // 队列里最小的两个任务耗时中的大者+分裂时间, 可视为大者分裂出另一个工人完成小两者中较小的任务后, 再完成自己任务所耗的总时间
            // 这样较小的任务的时间就被"掩盖"了
            pq.offer(Math.max(pq.poll(), pq.poll()) + split);
        }
        return pq.poll();
    }


    // LCP 04 匈牙利算法 二分图的最大匹配 Hard **
    public int domino(int n, int m, int[][] broken) {
        // 统计
        Set<Integer> brokenSet = new HashSet<>();
        int[][] direction = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] b : broken) {
            brokenSet.add(b[0] * m + b[1]);
        }
        // 建图
        List<List<Integer>> mtx = new ArrayList<>(m * n); // 邻接矩阵
        for (int i = 0; i < m * n; i++) {
            mtx.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int idx = i * m + j;
                if (!brokenSet.contains(idx)) {
                    for (int[] d : direction) {
                        if (lcp04Check(i + d[0], j + d[1], n, m, brokenSet)) {
                            int nextIdx = (i + d[0]) * m + j + d[1];
                            mtx.get(idx).add(nextIdx);
                        }
                    }
                }
            }
        }
        boolean[] visited;
        int[] p = new int[m * n];
        Arrays.fill(p, -1);
        int result = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if ((i + j) % 2 == 0 && !brokenSet.contains(i * m + j)) {
                    visited = new boolean[m * n];
                    if (lcp04(i * m + j, visited, mtx, p, brokenSet)) {
                        result++;
                    }
                }
            }
        }
        return result;
    }

    private boolean lcp04(int i, boolean[] visited, List<List<Integer>> mtx, int[] p, Set<Integer> brokenSet) {
        if (brokenSet.contains(i)) return false;
        for (int next : mtx.get(i)) {
            if (!visited[next]) {
                visited[next] = true;
                if (p[next] == -1 || lcp04(p[next], visited, mtx, p, brokenSet)) {
                    p[next] = i;
                    return true;
                }
            }
        }
        return false;
    }

    private boolean lcp04Check(int row, int col, int n, int m, Set<Integer> brokenSet) {
        return row >= 0 && row < n && col >= 0 && col < m && !brokenSet.contains(row * m + col);
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}