import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.cutOffTree(List.of(List.of(54581641, 64080174, 24346381, 69107959), List.of(86374198, 61363882, 68783324, 79706116), List.of(668150, 92178815, 89819108, 94701471), List.of(83920491, 22724204, 46281641, 47531096), List.of(89078499, 18904913, 25462145, 60813308))));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC675
    public int cutOffTree(List<List<Integer>> forest) {
        int m = forest.size(), n = forest.get(0).size();
        int[] mtx = new int[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                mtx[i * n + j] = forest.get(i).get(j);
            }
        }
        List<Pair<Integer, Integer>> seq = new ArrayList<>();
        for (int i = 0; i < m * n; i++) {
            seq.add(new Pair<>(i, mtx[i]));
        }
        seq = seq.stream().filter(i -> i.getValue() > 1).collect(Collectors.toList());
        Collections.sort(seq, Comparator.comparingInt(Pair::getValue));
        int cur = 0, result = 0;
        for (Pair<Integer, Integer> p : seq) {
            // < IDX, VAL>
            int next = p.getKey();
            int step = helper(cur, next, m, n, mtx);
            if (step == -1) return -1;
            mtx[next] = 1;
            result += step;
            cur = next;
        }
        return result;
    }

    final int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    private int helper(int cur, int target, int m, int n, int[] mtx) {
        if (cur == target) return 0;
        Deque<Integer> q = new LinkedList<>();
        q.offer(cur);
        Set<Integer> visited = new HashSet<>();
        int layer = -1;
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int p = q.poll();
                if (visited.contains(p)) continue;
                visited.add(p);
                int r = p / n, c = p % n;
                for (int[] d : directions) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n) continue;
                    int nidx = nr * n + nc;
                    if (nidx == target) return layer + 1;
                    if (mtx[nidx] == 0 || visited.contains(nidx)) continue;
                    q.offer(nidx);
                }
            }
        }
        return -1;
    }


    // LC479 **
    public long largestPalindrome(int n) {
        if (n == 1) {
            return 9;
        }
        int upper = (int) Math.pow(10, n) - 1;
        long ans = 0;
        for (int left = upper; ans == 0; --left) { // 枚举回文数的左半部分
            long p = left;
            for (int x = left; x > 0; x /= 10) {
                p = p * 10 + x % 10; // 翻转左半部分到其自身末尾，构造回文数 p
            }
            for (long x = upper; x * x >= p; --x) {
                if (p % x == 0) { // x 是 p 的因子
                    ans = p;
                    break;
                }
            }
        }
        return ans % 1337L;
    }

    // 220327 LYJJ
    int backwardOne;
    int[] op;
    byte[][][] memo;

    public int maxOnes(int[] arr, int m) { // op: 0 - &, 1 - | ,2 - ^
        int n = arr.length + 1;
        backwardOne = m;
        op = arr;
        memo = new byte[2][n][1 << m];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < 1 << m; k++) {
                    memo[i][j][k] = -1;
                }
            }
        }
        return lyjjHelper(1, n - 1, 0);
    }

    private int lyjjHelper(int target, int numIdx, int mask) {
        boolean isBackwardMOne = ((mask >> (backwardOne - 1)) & 1) == 1;
        boolean isBackwardMMinusOneOne = ((mask >> (backwardOne - 2)) & 1) == 1;
        if (numIdx == 1) { // 边界条件, 轮到正数第二个数(numIdx==1), 此时只剩下一个运算符
            switch (op[0]) {
                case 0:
                    if (target == 1) { // 此时只能两侧各填1, 所以前m个数和前m-1个数都不能是1, 否则返回极大值
                        if (isBackwardMOne || isBackwardMMinusOneOne) {
                            return Integer.MIN_VALUE / 2;
                        }
                        return 2;
                    } else if (target == 0) {
                        return 0;
                    }
                case 1: // 这里或运算和异或运算所要判断的情形是一致的
                case 2:
                    if (target == 1) { // 前m和前m-1不能同时是1
                        if (isBackwardMOne && isBackwardMMinusOneOne) {
                            return Integer.MIN_VALUE / 2;
                        }
                        return 1;
                    } else if (target == 0) {
                        return 0;
                    }
            }
        }
        if (memo[target][numIdx][mask] != -1) {
            return memo[target][numIdx][mask];
        }
        int result = -1;
        int newMaskWithOne = ((mask << 1) | 1) & ((1 << backwardOne) - 1);
        int newMaskWithZero = ((mask << 1) | 0) & ((1 << backwardOne) - 1);
        switch (op[numIdx - 1]) {
            case 0:
                if (target == 1) { // 此时两侧都要填1, 只要第前m个数是1, 就返回极大值
                    if (isBackwardMOne) {
                        return Integer.MIN_VALUE / 2;
                    }
                    result = 1 + lyjjHelper(1, numIdx - 1, newMaskWithOne);
                } else if (target == 0) { // (0,0),(0,1),(1,0) 中最大的
                    result = Math.max(
                            Math.max(
                                    0 + lyjjHelper(0, numIdx - 1, newMaskWithZero),
                                    0 + lyjjHelper(1, numIdx - 1, newMaskWithZero)
                            ),
                            1 + lyjjHelper(0, numIdx - 1, newMaskWithOne)
                    );
                }
                break;
            case 1:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + lyjjHelper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0),(1,1) 中最大的
                        result = Math.max(
                                Math.max(
                                        0 + lyjjHelper(1, numIdx - 1, newMaskWithZero),
                                        1 + lyjjHelper(0, numIdx - 1, newMaskWithOne)),
                                1 + lyjjHelper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                } else if (target == 0) { // 意味着两侧都要填0
                    result = 0 + lyjjHelper(0, numIdx - 1, newMaskWithZero);
                }
                break;
            case 2:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + lyjjHelper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0)中最小的
                        result = Math.max(
                                0 + lyjjHelper(1, numIdx - 1, newMaskWithZero),
                                1 + lyjjHelper(0, numIdx - 1, newMaskWithOne)
                        );

                    }
                } else if (target == 0) { // (0,0), (1,1)
                    if (isBackwardMOne) {
                        result = 0 + lyjjHelper(0, numIdx - 1, newMaskWithZero);
                    } else {
                        result = Math.max(
                                0 + lyjjHelper(0, numIdx - 1, newMaskWithZero),
                                1 + lyjjHelper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                }
        }
        return memo[target][numIdx][mask] = (byte) result;
    }

}