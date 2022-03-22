import javax.sound.midi.Soundbank;
import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        int a = (int) (Math.random() * 1024);
        StringBuilder sb = new StringBuilder(a);
        Random r = new Random();
        for (int i = 0; i < a; i++) {
            if (r.nextBoolean()) {
                sb.append('A');
            } else {
                sb.append('B');
            }
        }
        System.out.println(sb);
        System.out.println(s.winnerOfGame(sb.toString()));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC2038
    public boolean winnerOfGame(String colors) {
        int a = 0, b = 0, n = colors.length();
        char[] ca = colors.toCharArray();
        for (int i = 0; i < n; i++) {
            if (i - 1 >= 0 && i + 1 < n && ca[i - 1] == ca[i] && ca[i + 1] == ca[i]) {
                switch (ca[i]) {
                    case 'A':
                        a++;
                        continue;
                    case 'B':
                        b++;
                        continue;
                }
            }
        }
        return a > b;
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