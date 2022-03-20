import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.networkBecomesIdle(
                new int[][]{{5, 7}, {15, 18}, {12, 6}, {5, 1}, {11, 17}, {3, 9}, {6, 11}, {14, 7}, {19, 13}, {13, 3}, {4, 12}, {9, 15}, {2, 10}, {18, 4}, {5, 14}, {17, 5}, {16, 2}, {7, 1}, {0, 16}, {10, 19}, {1, 8}},
                new int[]{0, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1}
        ));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC2039
    public int networkBecomesIdle(int[][] edges, int[] patience) {
        final int n = patience.length, INF = Integer.MAX_VALUE / 2;
        // Dijkstra first
        Map<Integer, List<Integer>> m = new HashMap<>(n);
        for (int i = 0; i < n; i++) m.put(i, new ArrayList<>());
        for (int[] e : edges) {
            m.get(e[0]).add(e[1]);
            m.get(e[1]).add(e[0]);
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i[1]));
        pq.offer(new int[]{0, 0});
        Set<Integer> visited = new HashSet<>(n);
        int[] shortestDistanceToZero = new int[n];
        Arrays.fill(shortestDistanceToZero, INF);
        shortestDistanceToZero[0] = 0;
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int cur = p[0], distanceFromCurToZero = p[1];
            if (visited.contains(cur)) continue;
            visited.add(cur);
            for (int next : m.get(cur)) {
                if (cur != next && !visited.contains(next)) {
                    if (shortestDistanceToZero[next] > distanceFromCurToZero + 1) {// 1 为邻接点间的距离
                        shortestDistanceToZero[next] = distanceFromCurToZero + 1;
                        pq.offer(new int[]{next, distanceFromCurToZero + 1});
                    }
                }
            }
        }

        int result = 0;
        for (int i = 0; i < n; i++) {
            if (patience[i] == 0) continue;
            // rtt: 2 x shortest distance
            result = Math.max(result, ((2 * shortestDistanceToZero[i] - 1) / patience[i]) * patience[i] + 2 * shortestDistanceToZero[i]);
            //                                                        ^ ^ -1 是因为第一秒才开始检查, 下面的错误做法认为从第0秒就开始检查

            // Wrong:
            // if ((2 * shortestDistanceToZero[i]) % patience[i] == 0) {
            //     result = Math.max(result, ((2 * shortestDistanceToZero[i] / patience[i]) - 1) + 2 * shortestDistanceToZero[i]);
            // } else {
            //    result = Math.max(result, (2 * shortestDistanceToZero[i] / patience[i]) * patience[i] + 2 * shortestDistanceToZero[i]);
            //}
        }
        return result + 1;
    }

    // LC2044
    public int countMaxOrSubsets(int[] nums) {
        int n = nums.length;
        int max = Integer.MIN_VALUE, maxCount = 0;
        for (int mask = 1; mask < (1 << n); mask++) {
            int tmp = 0;
            for (int j = 0; j < n; j++) {
                if (((mask >> j) & 1) == 1) {
                    tmp |= nums[j];
                }
            }
            if (tmp > max) {
                max = tmp;
                maxCount = 1;
            } else if (tmp == max) {
                maxCount++;
            }
        }
        return maxCount;
    }
}