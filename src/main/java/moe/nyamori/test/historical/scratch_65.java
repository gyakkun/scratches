package moe.nyamori.test.historical;


import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class scratch_65 {
    public static void main(String[] args) {
        scratch_65 s = new scratch_65();
        long timing = System.currentTimeMillis();

//        System.out.println(s.minRefuelStops(
//                100,
//                10,
//                new int[][]{{10, 60}, {20, 30}, {30, 30}, {60, 40}}
//        ));
//        System.out.println(s.minRefuelStops(
//                1000,
//                10,
//                new int[][]{{10, 201}, {201, 100}, {303, 330}, {600, 401}}
//        ));
//        System.out.println(s.minRefuelStops(
//                1000,
//                1,
//                new int[][]{{125, 480}, {162, 46}, {175, 490}, {194, 207}, {355, 252}, {369, 75}, {433, 360}, {553, 95}, {562, 171}, {566, 12}}
//        ));
        System.out.println(s.minRefuelStops(
                1000,
                83,
                new int[][]{{25, 27}, {36, 187}, {140, 186}, {378, 6}, {492, 202}, {517, 89}, {579, 234}, {673, 86}, {808, 53}, {954, 49}}
        ));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC921
    public int minAddToMakeValid(String s) {
        int ctr = 0, extra = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') ctr++;
            else if (ctr > 0 && c == ')') ctr--;
            else if (ctr <= 0 && c == ')') extra++;
        }
        return ctr + extra;
    }

    // LC1252 **
    public int oddCells(int m, int n, int[][] indices) {
        int[] row = new int[m], col = new int[n];
        for (int[] i : indices) {
            row[i[0]]++;
            col[i[1]]++;
        }
        int oddx = 0, oddy = 0;
        for (int i = 0; i < m; i++) {
            if ((row[i] & 1) != 0) {
                oddx++;
            }
        }
        for (int i = 0; i < n; i++) {
            if ((col[i] & 1) != 0) {
                oddy++;
            }
        }
        return oddx * (n - oddy) + (m - oddx) * oddy;
    }

    // LC741
    Integer[][][] lc741Memo;
    int[][] lc741Grid;
    int lc741MtxLen;

    public int cherryPickup(int[][] grid) {
        this.lc741Grid = grid;
        int n = grid.length;
        this.lc741MtxLen = n;
        int total = 2 * n - 2;
        lc741Memo = new Integer[total + 1][n + 1][n + 1];
        return Math.max(helper(0, 0, 0), 0);
    }

    private int helper(int accu, int x1, int x2) {
        int y1 = accu - x1, y2 = accu - x2;
        if (lc741Grid[x1][y1] == -1 || lc741Grid[x2][y2] == -1) {
            return Integer.MIN_VALUE / 2;
        }
        if (lc741Memo[accu][x1][x2] != null) return lc741Memo[accu][x1][x2];
        int thisStepGain = 0;
        if (lc741Grid[x1][y1] == 1) thisStepGain++;
        if (lc741Grid[x2][y2] == 1) thisStepGain++;
        if (x1 == x2 && lc741Grid[x1][y1] == 1) thisStepGain--;
        if (accu == 2 * lc741MtxLen - 2) return thisStepGain;
        // Make choice
        // X1 RIGHT, X2 RIGHT 0,0
        // X1 RIGHT, X2 DOWN 0,1
        // X1 DOWN, X2 RIGHT 1,0
        // X1 DOWN, X2 DOWN 1,1
        int mask = (1 << 2) - 1;
        int result = Integer.MIN_VALUE / 2;
        for (int i = 0; i < 4; i++) {
            int nextX1 = -1, nextX2 = -1;
            int x1Bit = ((i & mask) >> 1) & 1, x2Bit = (i & mask) & 1;
            if (x1Bit == 0) { // right
                if (x1 + 1 >= lc741MtxLen) continue;
                nextX1 = x1 + 1;
            } else if (x1Bit == 1) { // down
                if (y1 + 1 >= lc741MtxLen) continue;
                nextX1 = x1;
            }

            if (x2Bit == 0) {
                if (x2 + 1 >= lc741MtxLen) continue;
                nextX2 = x2 + 1;
            } else if (x2Bit == 1) {
                if (y2 + 1 >= lc741MtxLen) continue;
                nextX2 = x2;
            }
            result = Math.max(result, helper(accu + 1, nextX1, nextX2));
        }
        return lc741Memo[accu][x1][x2] = result + thisStepGain;
    }

    // LC257
    List<String> lc257Result = new ArrayList<>();

    public List<String> binaryTreePaths(TreeNode65 root) {
        lc257Helper(root, new StringBuilder());
        return lc257Result;
    }

    public void lc257Helper(TreeNode65 cur, StringBuilder sb) {
        sb.append(cur.val);
        if (cur.left == null && cur.right == null) {
            lc257Result.add(sb.toString());
            return;
        }
        sb.append("->");
        if (cur.left != null) {
            lc257Helper(cur.left, new StringBuilder(sb));
        }
        if (cur.right != null) {
            lc257Helper(cur.right, new StringBuilder(sb));
        }
    }

    public int nextGreaterElement(int n) {
        char[] ca = ("" + n).toCharArray();
        long next = Long.parseLong(new String(getNextPermutation(ca)));
        if (next > Integer.MAX_VALUE) return -1;
        if (next <= n) return -1;
        return (int) next;
    }

    private char[] getNextPermutation(char[] ca) {
        int n = ca.length;
        int right = ca.length - 2;
        while (right >= 0 && ca[right] >= ca[right + 1]) {
            right--;
        }
        if (right >= 0) {
            int left = n - 1;
            while (left >= 0 && ca[right] >= ca[left]) {
                left--;
            }
            arraySwap(ca, left, right);
        }
        arrayReverse(ca, right + 1, n - 1);
        return ca;
    }

    private void arraySwap(char[] ca, int i, int j) {
        if (i != j) {
            char orig = ca[j];
            ca[j] = ca[i];
            ca[i] = orig;
        }
    }

    private void arrayReverse(char[] ca, int from, int to) {
        if (from < 0 || from >= ca.length || to < 0 || to >= ca.length) return;
        int origFrom = from;
        from = from > to ? to : from;
        to = from == origFrom ? to : origFrom;
        int mid = (from + to + 1) / 2;
        for (int i = from; i < mid; i++) {
            arraySwap(ca, i, to - (i - from));
        }
    }

    // LC871
    int lc871StartFuel, lc871Target;
    int[][] lc871Stations;
    Integer[][] lc871Memo;

    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        if (startFuel >= target) return 0;
        this.lc871Target = target;
        this.lc871StartFuel = startFuel;
        this.lc871Stations = stations;
        int n = stations.length;
        if (n == 0) return -1;
        lc871Memo = new Integer[n + 1][n + 1];
        // dp[i][j] 表示经过前 i 个油站, 在其中的j个油站中加了油, 最多还可以剩下多少升油?
        int result = Integer.MAX_VALUE / 2;
        for (int i = n; i >= 0; i--) {
            if (lc871Helper(n - 1, i) >= target - stations[n - 1][0]) {
                result = Math.min(result, i);
            }
        }
        if (result == Integer.MAX_VALUE / 2) return -1;
        return result;
    }

    private int lc871Helper(int current, int stopCount) {
        if (stopCount > current + 1) {
            return -1;
        }
        if (current < 0) {
            return lc871StartFuel;
        }
        if (current == 0 && stopCount == 1) {
            if (lc871StartFuel < lc871Stations[0][0]) return -1;
            return lc871StartFuel - lc871Stations[0][0] + lc871Stations[0][1];
        }
        if (stopCount == 0) {
            int remain = lc871StartFuel - lc871Stations[current][0];
            return lc871Memo[current][stopCount] = remain;
        }
        if (stopCount < 0) {
            return -1;
        }
        if (lc871Memo[current][stopCount] != null) {
            return lc871Memo[current][stopCount];
        }
        int result = -1;

        for (int i = 0; i < 2; i++) {
            int remain = lc871Helper(current - 1, stopCount - i);
            if (remain < 0) continue;
            int consumption;
            if (current == 0) {
                consumption = lc871Stations[0][0];
            } else {
                consumption = lc871Stations[current][0] - lc871Stations[current - 1][0];
            }
            if (remain - consumption < 0) continue;
            if (i == 0) { // 本站不加油
                result = Math.max(result, remain - consumption);
            } else if (i == 1) { // 本站 加油
                result = Math.max(result, remain - consumption + lc871Stations[current][1]);
            }
        }
        return lc871Memo[current][stopCount] = result;
    }

    // LC241 **
    List<Integer>[][] lc241Memo;
    List<Integer> lc241TokenList = new ArrayList<>();
    List<Integer> lc241OpPosition = new ArrayList<>();

    public List<Integer> diffWaysToCompute(String expression) {
        int[] ops = new int[256];
        ops['+'] = -1;
        ops['-'] = -2;
        ops['*'] = -3;
        char[] ca = expression.toCharArray();
        int prev = 0, cur = 0;
        while (cur < ca.length) {
            while (cur < ca.length && Character.isDigit(ca[cur])) cur++;
            lc241TokenList.add(Integer.parseInt(expression.substring(prev, cur)));
            if (cur >= ca.length) break;
            lc241TokenList.add(ops[ca[cur]]);
            lc241OpPosition.add(cur);
            cur++;
            prev = cur;
        }
        lc241Memo = new List[lc241TokenList.size()][lc241TokenList.size()];
        IntStream.range(0, lc241TokenList.size()).forEachOrdered(i -> {
            IntStream.range(0, lc241TokenList.size()).forEachOrdered(j -> {
                lc241Memo[i][j] = new ArrayList<>();
            });
        });
        return lc241Helper(0, lc241TokenList.size() - 1).stream().toList();
    }

    private List<Integer> lc241Helper(int left, int right) { // 左闭右闭
        if (lc241Memo[left][right].isEmpty()) {
            if (left == right) {
                lc241Memo[left][right].add(lc241TokenList.get(left));
            } else {
                for (int i = left; i < right; i += 2) {
                    List<Integer> l = lc241Helper(left, i);
                    List<Integer> r = lc241Helper(i + 2, right);
                    for (int j : l) {
                        for (int k : r) {
                            switch (lc241TokenList.get(i + 1)) {
                                case -1:
                                    lc241Memo[left][right].add(j + k);
                                    break;
                                case -2:
                                    lc241Memo[left][right].add(j - k);
                                    break;
                                case -3:
                                    lc241Memo[left][right].add(j * k);
                                    break;
                            }
                        }
                    }

                }
            }
        }
        return lc241Memo[left][right];
    }


    // LC522
    public int findLUSlength(String[] strs) {
        int result = -1;
        outer:
        for (int i = 0; i < strs.length; i++) {
            inner:
            for (int j = 0; j < strs.length; j++) {
                if (i == j) continue;
                int pi = 0, pj = 0;
                char[] ci = strs[i].toCharArray(), cj = strs[j].toCharArray();
                while (pi < ci.length && pj < cj.length) {
                    if (ci[pi] == cj[pj]) pi++;
                    pj++;
                }
                if (pi == ci.length) continue outer;
            }
            result = Math.max(result, strs[i].length());
        }
        return result;
    }


    // LC513
    public int findBottomLeftValue(TreeNode65 root) {
        Deque<TreeNode65> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qs = q.size(), nextLayerCount = 0;
            TreeNode65 leftMostNode = null;
            for (int i = 0; i < qs; i++) {
                TreeNode65 p = q.poll();
                if (i == 0) {
                    leftMostNode = p;
                }
                if (p.left != null) {
                    q.offer(p.left);
                    nextLayerCount++;
                }
                if (p.right != null) {
                    q.offer(p.right);
                    nextLayerCount++;
                }
            }
            if (nextLayerCount == 0) {
                return leftMostNode.val;
            }
        }
        return -1;
    }

    // LC1089
    public void duplicateZeros(int[] arr) {
        int ctr = 0, pivot = 0, n = arr.length;
        for (int i : arr) {
            if (i == 0) {
                ctr += 2;
            } else {
                ctr++;
            }
            if (ctr >= n) break;
            pivot++;
        }
        for (int i = n - 1; i >= 0; i--) {
            if (arr[pivot] == 0) {
                if (i == n - 1 && ctr > n) {
                    arr[i] = 0;
                } else {
                    arr[i] = arr[pivot];
                    arr[--i] = 0;
                }
            } else {
                arr[i] = arr[pivot];
            }
            pivot--;
        }
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
    public TreeNode65 createBinaryTree(int[][] descriptions) {
        Map<Integer, TreeNode65> m = new HashMap<>();
        Set<Integer> parentSet = new HashSet<>();
        Set<Integer> childrenSet = new HashSet<>();
        for (int[] nd : descriptions) {
            int par = nd[0], cur = nd[1], isLeft = nd[2];
            TreeNode65 curNode = m.getOrDefault(cur, new TreeNode65(cur));
            TreeNode65 parNode = m.getOrDefault(par, new TreeNode65(par));
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
                    result = Math.max(Math.max(0 + lyjjHelper(0, numIdx - 1, newMaskWithZero), 0 + lyjjHelper(1, numIdx - 1, newMaskWithZero)), 1 + lyjjHelper(0, numIdx - 1, newMaskWithOne));
                }
                break;
            case 1:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + lyjjHelper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0),(1,1) 中最大的
                        result = Math.max(Math.max(0 + lyjjHelper(1, numIdx - 1, newMaskWithZero), 1 + lyjjHelper(0, numIdx - 1, newMaskWithOne)), 1 + lyjjHelper(1, numIdx - 1, newMaskWithOne));
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
                        result = Math.max(0 + lyjjHelper(1, numIdx - 1, newMaskWithZero), 1 + lyjjHelper(0, numIdx - 1, newMaskWithOne));

                    }
                } else if (target == 0) { // (0,0), (1,1)
                    if (isBackwardMOne) {
                        result = 0 + lyjjHelper(0, numIdx - 1, newMaskWithZero);
                    } else {
                        result = Math.max(0 + lyjjHelper(0, numIdx - 1, newMaskWithZero), 1 + lyjjHelper(1, numIdx - 1, newMaskWithOne));
                    }
                }
        }
        return lyjjMemo[target][numIdx][mask] = (byte) result;
    }

}

class TreeNode65 {
    int val;
    TreeNode65 left;
    TreeNode65 right;

    TreeNode65() {
    }

    TreeNode65(int val) {
        this.val = val;
    }

    TreeNode65(int val, TreeNode65 left, TreeNode65 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}