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

    // LC1199
    public int minBuildTime(int[] blocks, int split) {
        // int lo = Arrays.stream(blocks).max().getAsInt();
        // int hi = Arrays.stream(blocks).sum() + split * blocks.length;
        int lo = 0;
        int hi = (int) 1e9;
        Arrays.sort(blocks);
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (check(mid, blocks, split)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private boolean check(int totalTime, int[] blocks, int split) {
        Deque<int[]> q = new LinkedList<>();
        int ptr = blocks.length - 2;
        // [诞生时间, 任务耗时]
        q.offer(new int[]{0, blocks[blocks.length - 1]});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            // 在保证完成自己的工作的前提下尽可能分配多的工人
            if (totalTime - p[0] < p[1]) return false;
            int maxWorker = (totalTime - p[0] - p[1]) / split;
            int workerCtr = 0;
            while (workerCtr != maxWorker && ptr != -1) {
                q.offer(new int[]{p[0] + (workerCtr + 1) * split, blocks[ptr--]});
                workerCtr++;
            }
        }
        return ptr == -1;
    }


    // LCP 04 匈牙利算法 二分图的最大匹配
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