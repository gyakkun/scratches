import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.isInterleave(
                "aabcc",
                "dbbca",
                "aadbbcbcac"
        ));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // JZ Offer II 096
    // LC097
    char[] ca1, ca2, ca3;
    Boolean[][][] memo;

    public boolean isInterleave(String s1, String s2, String s3) {
        ca1 = s1.toCharArray();
        ca2 = s2.toCharArray();
        ca3 = s3.toCharArray();
        if (ca1.length + ca2.length != ca3.length) return false;
        memo = new Boolean[ca1.length + 1][ca2.length + 1][ca3.length + 1];
        return helper(0, 0, 0);
    }

    private boolean helper(int p1, int p2, int p3) {
        if (p3 == ca3.length && p2 == ca2.length && p1 == ca1.length) return true;
        if (memo[p1][p2][p3] != null) return memo[p1][p2][p3];
        char cur = ca3[p3];
        boolean b1 = false, b2 = false;
        if (p1 < ca1.length && ca1[p1] == cur) {
            b1 = helper(p1 + 1, p2, p3 + 1);
            if (b1) return memo[p1][p2][p3] = true;
        }
        if (p2 < ca2.length && ca2[p2] == cur) {
            b2 = helper(p1, p2 + 1, p3 + 1);
            if (b2) return memo[p1][p2][p3] = true;
        }
        return memo[p1][p2][p3] = false;
    }

    // Interview 17.11
    public int findClosest(String[] words, String word1, String word2) {
        int p1 = -1, p2 = -1, result = Integer.MAX_VALUE / 2;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                if (p2 != -1) {
                    result = Math.min(result, i - p2);
                }
                p1 = i;
            }
            if (words[i].equals(word2)) {
                if (p1 != -1) {
                    result = Math.min(result, i - p1);
                }
                p2 = i;
            }
        }
        return result;
    }

    // LC2196
    public TreeNode createBinaryTree(int[][] descriptions) {
        Map<Integer, TreeNode> m = new HashMap<>();
        Set<Integer> parentSet = new HashSet<>();
        Set<Integer> childrenSet = new HashSet<>();
        for (int[] nd : descriptions) {
            int par = nd[0], cur = nd[1], isLeft = nd[2];
            TreeNode curNode = m.getOrDefault(cur, new TreeNode(cur));
            TreeNode parNode = m.getOrDefault(par, new TreeNode(par));
            m.putIfAbsent(par, parNode);
            m.putIfAbsent(cur, curNode);
            if (isLeft == 1) {
                parNode.left = curNode;
            } else {
                parNode.right = curNode;
            }
            parentSet.add(par);
            childrenSet.add(cur);
        }
        parentSet.removeAll(childrenSet);
        return m.get(parentSet.stream().findFirst().get());
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
            int step = lc675Helper(cur, next, m, n, mtx);
            if (step == -1) return -1;
            mtx[next] = 1;
            result += step;
            cur = next;
        }
        return result;
    }

    final int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    private int lc675Helper(int cur, int target, int m, int n, int[] mtx) {
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
    byte[][][] lyjjMemo;

    public int maxOnes(int[] arr, int m) { // op: 0 - &, 1 - | ,2 - ^
        int n = arr.length + 1;
        backwardOne = m;
        op = arr;
        lyjjMemo = new byte[2][n][1 << m];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < 1 << m; k++) {
                    lyjjMemo[i][j][k] = -1;
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
        if (lyjjMemo[target][numIdx][mask] != -1) {
            return lyjjMemo[target][numIdx][mask];
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
        return lyjjMemo[target][numIdx][mask] = (byte) result;
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