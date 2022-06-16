import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.findDiagonalOrder(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC532
    public int findPairs(int[] nums, int k) {
        Map<Integer, Integer> collect = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        int result = 0;
        if (k == 0) {
            for (Integer v : collect.values()) {
                if (v > 1) result++;
            }
            return result;
        } else {
            TreeMap<Integer, Integer> tm = new TreeMap<>(collect);
            for (Integer i : tm.keySet()) {
                if (tm.containsKey(i + k)) {
                    result++;
                }
            }
            return result;
        }
    }

    // LC498 **
    public int[] findDiagonalOrder(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int[] res = new int[m * n];
        int pos = 0;
        for (int i = 0; i < m + n - 1; i++) {
            if (i % 2 == 1) {
                int x = i < n ? 0 : i - n + 1;
                int y = i < n ? i : n - 1;
                while (x < m && y >= 0) {
                    res[pos] = mat[x][y];
                    pos++;
                    x++;
                    y--;
                }
            } else {
                int x = i < m ? i : m - 1;
                int y = i < m ? 0 : i - m + 1;
                while (x >= 0 && y < n) {
                    res[pos] = mat[x][y];
                    pos++;
                    x--;
                    y++;
                }
            }
        }
        return res;
    }

    // LC890
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        int len = pattern.length();
        Character[] map, rMap;
        char[] pa = pattern.toCharArray();
        List<String> result = new ArrayList<>();
        outer:
        for (String w : words) {
            if (w.length() != len) continue outer;
            char[] ca = w.toCharArray();
            map = new Character[128];
            rMap = new Character[128];
            inner:
            for (int i = 0; i < len; i++) {
                if (map[pa[i]] == null && rMap[ca[i]] == null) {
                    map[pa[i]] = ca[i];
                    rMap[ca[i]] = pa[i];
                } else {
                    if (map[pa[i]] != null && ca[i] == map[pa[i]] && rMap[ca[i]] != null && pa[i] == rMap[ca[i]])
                        continue inner;
                    else continue outer;
                }
            }
            result.add(w);
        }
        return result;
    }

    // LC730 ** Hard
    Long[][] lc730Memo;
    long lc730Mod = 1000000007l;
    int[] lc730Pre, lc730Next;
    char[] lc730Ca;

    public int countPalindromicSubsequences(String s) {
        int n = s.length();
        lc730Ca = s.toCharArray();
        lc730Memo = new Long[n + 1][n + 1];
        lc730Pre = new int[n];
        lc730Next = new int[n];
        int[] lookBack = new int[128], lookForward = new int[128];
        Arrays.fill(lc730Pre, -1);
        Arrays.fill(lc730Next, n);
        Arrays.fill(lookBack, -1);
        Arrays.fill(lookForward, n);
        for (int i = 0; i < n; i++) {
            int r = n - i - 1;
            if (lookBack[lc730Ca[i]] != -1) {
                lc730Pre[i] = lookBack[lc730Ca[i]];
            }
            lookBack[lc730Ca[i]] = i;

            if (lookForward[lc730Ca[r]] != n) {
                lc730Next[r] = lookForward[lc730Ca[r]];
            }
            lookForward[lc730Ca[r]] = r;
        }
        return (int) (lc730Helper(0, n - 1) % lc730Mod);
    }

    private long lc730Helper(int i, int j) {
        if (i > j) return 0l;
        if (i == j) {
            if (lc730Ca[i] == lc730Ca[j]) return 1l;
            return 0l;
        }
        if (lc730Memo[i][j] != null) return lc730Memo[i][j];
        if (lc730Ca[i] != lc730Ca[j]) {
            return lc730Memo[i][j] = (lc730Helper(i + 1, j) + lc730Helper(i, j - 1) - lc730Helper(i + 1, j - 1) + lc730Mod) % lc730Mod;
        }
        // if ca[i] == ca[j]
        int lo = lc730Next[i], hi = lc730Pre[j];
        long result = 2 * lc730Helper(i + 1, j - 1);
        if (lo > hi) { // no middle x between i,j
            result += 2;
        } else if (lo == hi) { // one middle x between i,j
            result += 1;
        } else {
            result -= lc730Helper(lo + 1, hi - 1);
        }
        return lc730Memo[i][j] = (result + lc730Mod) % lc730Mod;
    }


    // LC875
    public int minEatingSpeed(int[] piles, int h) {
        if (h == piles.length) return Arrays.stream(piles).max().getAsInt();
        int lo = 1, hi = Arrays.stream(piles).max().getAsInt();
        while (lo < hi) {
            // 满足条件的最小值
            int mid = lo + (hi - lo) / 2;
            if (lc875Judge(mid, piles, h)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private boolean lc875Judge(int targetRate, int[] piles, int givenHours) {
        // Ceil divide: (x+divider-1) / divider
        return Arrays.stream(piles).map(i -> (i + targetRate - 1) / targetRate).sum() <= givenHours;
    }

    // LC829 Hard Math ** 推导不难
    public int consecutiveNumbersSum(int n) {
        int upperBound = 2 * n, result = 0;
        for (int i = 1; i * (i + 1) <= upperBound; i++) {
            if ((i & 1) == 1 && n % i == 0) { // odd
                result++;
            } else if (n % i != 0 && (2 * n) % i == 0) { // even
                result++;
            }
        }
        return result;
    }

    Map<Integer, Map<Integer, Boolean>> lc473Memo;

    // LC473
    public boolean makesquare(int[] matchsticks) {
        if (matchsticks.length < 4) return false;
        int sum = 0;
        for (int i : matchsticks) {
            sum += i;
        }
        if (sum % 4 != 0) return false;
        int target = sum / 4;
        for (int i : matchsticks) {
            if (i > target) return false;
        }
        Arrays.sort(matchsticks);
        lc473Memo = new HashMap<>();
        return lc473Helper(0, 4, target, matchsticks, target);
    }

    private boolean lc473Helper(int origMask, int remain, int target, int[] matchsticks, int quarter) {
        if (origMask == (1 << matchsticks.length) - 1 && remain == 0) return true;
        if (lc473Memo.containsKey(origMask) && lc473Memo.get(origMask).containsKey(target)) {
            return lc473Memo.get(origMask).get(target);
        }
        lc473Memo.putIfAbsent(origMask, new HashMap<>());
        int mask = origMask;
        for (int i = 0; i < matchsticks.length; i++) {
            if (((mask >> i) & 1) == 1) continue;
            if (matchsticks[i] > target) continue;
            mask |= 1 << i;
            boolean result;
            if (target == matchsticks[i]) {
                result = lc473Helper(mask, remain - 1, quarter, matchsticks, quarter);
            } else {
                result = lc473Helper(mask, remain, target - matchsticks[i], matchsticks, quarter);
            }
            if (result) {
                return true;
            }
            mask ^= 1 << i;
        }
        lc473Memo.get(origMask).put(target, false);
        return false;
    }

    // JZ Offer II 096
    // LC097
    char[] ca1, ca2, ca3;
    Boolean[][][] lc097Memo;

    public boolean isInterleave(String s1, String s2, String s3) {
        ca1 = s1.toCharArray();
        ca2 = s2.toCharArray();
        ca3 = s3.toCharArray();
        if (ca1.length + ca2.length != ca3.length) return false;
        lc097Memo = new Boolean[ca1.length + 1][ca2.length + 1][ca3.length + 1];
        return lc097Helper(0, 0, 0);
    }

    private boolean lc097Helper(int p1, int p2, int p3) {
        if (p3 == ca3.length && p2 == ca2.length && p1 == ca1.length) return true;
        if (lc097Memo[p1][p2][p3] != null) return lc097Memo[p1][p2][p3];
        char cur = ca3[p3];
        boolean b1 = false, b2 = false;
        if (p1 < ca1.length && ca1[p1] == cur) {
            b1 = lc097Helper(p1 + 1, p2, p3 + 1);
            if (b1) return lc097Memo[p1][p2][p3] = true;
        }
        if (p2 < ca2.length && ca2[p2] == cur) {
            b2 = lc097Helper(p1, p2 + 1, p3 + 1);
            if (b2) return lc097Memo[p1][p2][p3] = true;
        }
        return lc097Memo[p1][p2][p3] = false;
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