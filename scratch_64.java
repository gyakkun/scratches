import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.missingRolls(new int[]{1, 5, 6}, 3, 4));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC2028
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int m = rolls.length;
        int curSum = Arrays.stream(rolls).sum();
        int totalCount = m + n;
        int remain = mean * totalCount - curSum;
        if (remain > 6 * n || remain < n) return new int[]{};
        int avg = remain / n;
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = avg;
            remain -= avg;
            if (i != n - 1) avg = remain / (n - i - 1);
        }
        return result;
    }

    // 220327 LYJJ
    int backwardOne;
    int[] op;
    Integer[][][] memo;

    public int minOnes(int[] arr, int m) { // op: 0 - &, 1 - | ,2 - ^
        int n = arr.length + 1;
        backwardOne = m;
        op = arr;
        memo = new Integer[2][n][1 << m];
        return helper(1, n - 1, 0);
    }

    private int helper(int target, int numIdx, int mask) {
        boolean isBackwardMOne = ((mask >> (backwardOne - 1)) & 1) == 1;
        boolean isBackwardMMinusOneOne = ((mask >> (backwardOne - 2)) & 1) == 1;
        if (numIdx == 1) { // 边界条件, 轮到正数第二个数(numIdx==1), 此时只剩下一个运算符
            switch (op[0]) {
                case 0:
                    if (target == 1) { // 此时只能两侧各填1, 所以前m个数和前m-1个数都不能是1, 否则返回极大值
                        if (isBackwardMOne || isBackwardMMinusOneOne)
                            return Integer.MAX_VALUE / 2;
                        return 2;
                    } else if (target == 0) {
                        return 0;
                    }
                case 1: // 这里或运算和异或运算所要判断的情形是一致的
                case 2:
                    if (target == 1) { // 前m和前m-1不能同时是1
                        if (isBackwardMOne && isBackwardMMinusOneOne)
                            return Integer.MAX_VALUE / 2;
                        return 1;
                    } else if (target == 0) {
                        return 0;
                    }
            }
        }
        if (memo[target][numIdx][mask] != null) return memo[target][numIdx][mask];
        int result = -1;
        int newMaskWithOne = ((mask << 1) | 1) & ((1 << backwardOne) - 1);
        int newMaskWithZero = ((mask << 1) | 0) & ((1 << backwardOne) - 1);
        switch (op[numIdx - 1]) {
            case 0:
                if (target == 1) { // 此时两侧都要填1, 只要第前m个数是1, 就返回极大值
                    if (isBackwardMOne) return Integer.MAX_VALUE / 2;
                    result = 1 + helper(1, numIdx - 1, newMaskWithOne);
                } else if (target == 0) { // (0,0),(0,1),(1,0) 中最小的
                    result = Math.min(
                            Math.min(
                                    0 + helper(0, numIdx - 1, newMaskWithZero),
                                    0 + helper(1, numIdx - 1, newMaskWithOne)
                            ),
                            1 + helper(0, numIdx - 1, newMaskWithZero)
                    );
                }
                break;
            case 1:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0),(1,1) 中最小的
                        result = Math.min(
                                Math.min(
                                        0 + helper(1, numIdx - 1, newMaskWithZero),
                                        1 + helper(0, numIdx - 1, newMaskWithOne)),
                                1 + helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                } else if (target == 0) { // 意味着两侧都要填0
                    result = 0 + helper(0, numIdx - 1, newMaskWithZero);
                }
                break;
            case 2:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0)中最小的
                        result = Math.min(
                                0 + helper(1, numIdx - 1, newMaskWithZero),
                                1 + helper(0, numIdx - 1, newMaskWithOne)
                        );

                    }
                } else if (target == 0) { // (0,0), (1,1)
                    if (isBackwardMOne) {
                        result = 0 + helper(0, numIdx - 1, newMaskWithZero);
                    } else {
                        result = Math.min(
                                0 + helper(0, numIdx - 1, newMaskWithZero),
                                1 + helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                }
        }
        return memo[target][numIdx][mask] = result;
    }


    // LC682
    public int calPoints(String[] ops) {
        List<Integer> points = new ArrayList<>(ops.length);
        for (String o : ops) {
            char op = o.charAt(0);
            if (Character.isDigit(op) || op == '-') {
                points.add(Integer.parseInt(o));
            } else {
                switch (op) {
                    case '+':
                        points.add(points.get(points.size() - 1) + points.get(points.size() - 2));
                        break;
                    case 'D':
                        points.add(points.get(points.size() - 1) * 2);
                        break;
                    case 'C':
                        points.remove(points.size() - 1);
                        break;
                }
            }
        }
        return points.stream().reduce(0, Integer::sum);
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