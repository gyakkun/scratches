package moe.nyamori.test.historical;


import javafx.util.Pair;

import java.math.BigInteger;
import java.util.*;
import java.util.function.Function;


class scratch_54 {
    public static void main(String[] args) {
        scratch_54 s = new scratch_54();
        long timing = System.currentTimeMillis();

        TreeNode54 one = new TreeNode54(1);
        TreeNode54 two = new TreeNode54(2);
        one.right = two;

        System.out.println(s.flipMatchVoyage(one, new int[]{1, 2}));

        System.out.println(s.computeArea(-3, 0, 3, 4, 0, -1, 9, 2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC971
    List<Integer> result;
    int idx = -1;
    int[] voy;
    boolean flag = true;

    public List<Integer> flipMatchVoyage(TreeNode54 root, int[] voyage) {
        result = new ArrayList<>();
        voy = voyage;
        helper(root);
        if (!flag) return Arrays.asList(-1);
        return result;
    }

    private void helper(TreeNode54 root) {
        if (root == null) return;
        idx++;
        if (voy[idx] != root.val) {
            flag = false;
            return;
        }
        if (root.left != null && root.right != null) {
            if (root.left.val != voy[idx + 1]) {
                if (root.right.val != voy[idx + 1]) {
                    flag = false;
                    return;
                }
                result.add(root.val);
                TreeNode54 tmp = root.left;
                root.left = root.right;
                root.right = tmp;
            }
        } else if (root.left != null && root.right == null) {
            if (root.left.val != voy[idx + 1]) {
                flag = false;
                return;
            }
        } else if (root.left == null && root.right != null) {
            if (root.right.val != voy[idx + 1]) {
                flag = false;
                return;
            }
        }
        helper(root.left);
        helper(root.right);
    }

    // LC223
    public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        // 扫描线求投影面积, 用总面积间去投影面积等于重叠部分面积
        final int OPEN = 0, CLOSE = 1;
        int[][] events = new int[][]{
                {ay1, OPEN, ax1, ax2},
                {ay2, CLOSE, ax1, ax2},
                {by1, OPEN, bx1, bx2},
                {by2, CLOSE, bx1, bx2}
        };
        Arrays.sort(events, Comparator.comparingInt(o -> o[0]));
        // 活动中的与X轴平行的线: activeXs : [ [起始x, 结束x] , 个数 ]
        TreeMap<Pair<Integer, Integer>, Integer> activeXs = new TreeMap<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                return !o1.getKey().equals(o2.getKey()) ? o1.getKey() - o2.getKey() : o1.getValue() - o2.getValue();
            }
        });
        activeXs.put(new Pair<>(events[0][2], events[0][3]), 1);
        int project = 0, curY = events[0][0];

        for (int i = 1; i < 4; i++) {
            int height = events[i][0] - curY; // 必然是正数
            int length = 0;
            int cur = Integer.MIN_VALUE; // 遍历当前所有活动中的X线, 取全长
            for (Pair<Integer, Integer> xPair : activeXs.keySet()) { // 从左到右的X坐标们
                cur = Math.max(cur, xPair.getKey());
                length += Math.max(0, xPair.getValue() - cur);
                cur = Math.max(cur, xPair.getValue());
            }
            project += height * length;
            Pair<Integer, Integer> thisX = new Pair<>(events[i][2], events[i][3]);
            if (events[i][1] == OPEN) {
                int val = 0;
                if (activeXs.get(thisX) != null) val = activeXs.get(thisX);
                activeXs.put(thisX, val + 1);
            } else {
                activeXs.put(thisX, activeXs.get(thisX) - 1);
                if (activeXs.get(thisX) == 0) {
                    activeXs.remove(thisX);
                }
            }
            curY = events[i][0]; // 更新Y坐标
        }
        return project;
    }

    // LCP13 TSP
    int start = -1, end = -1;

    public int minimalSteps(String[] maze) {
        // S - start , T - target, O - stones, M - mechanism, # - wall, . - road
        //
        // 图中的必经点: 起点、机关, 点的个数不超过17个(1+16), 可以DP
        // 两点之间的代价为两点之间到最近一个石堆的路程之和, 石堆有40个, 考虑逐一双向BFS比较以获得最短路径
        char[][] grid = new char[maze.length][];
        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int i = 0; i < grid.length; i++) {
            grid[i] = maze[i].toCharArray();
        }
        int m = grid.length, n = grid[0].length;
        Set<Integer> mSet = new HashSet<>(), oSet = new HashSet<>(), wSet = new HashSet<>();
        extractElements(grid, m, n, mSet, oSet, wSet);

        if (mSet.size() == 0) return minLenToTarget(start, end, m, n, directions, grid);

        Set<Integer> pointSet = new HashSet<>(mSet);
        pointSet.add(start);

        int pointLen = pointSet.size();
        int[] pointsArr = new int[pointLen];
        Iterator<Integer> it = pointSet.iterator();
        int startIdxInPointsArr = -1;
        for (int i = 0; i < pointLen; i++) {
            pointsArr[i] = it.next();
            if (pointsArr[i] == start) {
                startIdxInPointsArr = i;
            }
        }
        int[][] minLenToStoneCache = new int[pointLen][pointLen];
        for (int i = 0; i < pointLen; i++) Arrays.fill(minLenToStoneCache[i], -2); // -2表示未初始化, -1表示不可达

        // 双向BFS 处理点之间的最近距离
        long timing = System.currentTimeMillis();
        getPointToPointMinLen(directions, m, n, oSet, grid, pointsArr, minLenToStoneCache);
        timing = System.currentTimeMillis() - timing;
        System.err.println("getPointToPointMinLen: " + timing + "ms.");

        int fullMask = (1 << pointLen) - 1;
        int[][] dp = new int[fullMask + 1][pointLen];
        for (int i = 0; i <= fullMask; i++) {
            Arrays.fill(dp[i], Integer.MAX_VALUE / 2);
        }
        // 最开始的时候, 只有到起点的距离是0
        timing = System.currentTimeMillis();
        dp[1 << startIdxInPointsArr][startIdxInPointsArr] = 0;
        for (int mask = 0; mask <= fullMask; mask++) {
            // 看看已经选了哪几个点
            for (int selected = 0; selected < pointLen; selected++) {
                if (((mask >> selected) & 1) == 1) {
                    int parentMask = mask ^ (1 << selected);
                    if (parentMask == 0) continue;
                    for (int prevEnd = 0; prevEnd < pointLen; prevEnd++) {
                        if (((parentMask >> prevEnd) & 1) == 1) {
                            if (minLenToStoneCache[selected][prevEnd] == -1) continue;
                            int len = dp[parentMask][prevEnd] + minLenToStoneCache[selected][prevEnd]; // 找到当前点与之前点到石头的最短距离
                            dp[mask][selected] = Math.min(dp[mask][selected], len);
                        }
                    }
                }
            }
        }
        timing = System.currentTimeMillis() - timing;
        System.err.println("dp: " + timing + "ms.");
        int minLen = Integer.MAX_VALUE / 2, minEndButOneIdx = -1;
        // 直接取dp[fullMask][end 在 pointsArr的下标] ?
        for (int i = 0; i < pointSet.size(); i++) {
            int finalStep = minLenToTarget(pointsArr[i], end, m, n, directions, grid);
            if (finalStep != -1) minLen = Math.min(minLen, dp[fullMask][i] + finalStep);
        }

        if (minLen == Integer.MAX_VALUE / 2) {
            return -1;
        }

        return minLen;
    }

    private void getPointToPointMinLen(int[][] directions, int m, int n, Set<Integer> oSet, char[][] grid, int[] pointsArr, int[][] minLenToStoneCache) {
        for (int i = 0; i < pointsArr.length; i++) {
            for (int j = i + 1; j < pointsArr.length; j++) {
                if (minLenToStoneCache[i][j] != -2) continue;
                int minToStoneLen = Integer.MAX_VALUE;
                // 选一个石堆
                for (int stone : oSet) {
                    int fromI = minLenToTarget(pointsArr[i], stone, m, n, directions, grid);
                    int fromJ = minLenToTarget(pointsArr[j], stone, m, n, directions, grid);
                    if (fromI == -1 || fromJ == -1) continue;// 选另一个石头
                    int lenToThisStone = fromI + fromJ;
                    if (lenToThisStone < minToStoneLen) {
                        minToStoneLen = lenToThisStone;
                    }
                }
                if (minToStoneLen == Integer.MAX_VALUE) {
                    minToStoneLen = -1;
                }
                minLenToStoneCache[i][j] = minToStoneLen;
                minLenToStoneCache[j][i] = minToStoneLen;
            }
        }
    }

    Map<Integer, Map<Integer, Integer>> memo = new HashMap<>();

    private int minLenToTarget(int from, int target, int m, int n, int[][] directions, char[][] grid) {
        if (memo.containsKey(from) && memo.get(from).containsKey(target)) return memo.get(from).get(target);
        memo.putIfAbsent(from, new HashMap<>());
        Deque<Integer> q = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        q.add(from);
        int layer = 0;

        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int k = 0; k < qs; k++) {
                int p = q.poll();
                if (visited.contains(p)) continue;
                visited.add(p);
                int[] idxArr = idxConvert(p, m, n);
                int r = idxArr[0], c = idxArr[1];
                for (int[] d : directions) {
                    int nr = r + d[0], nc = c + d[1];
                    int nIdx = nr * n + nc;
                    if (!visited.contains(nIdx) && checkLegal(nr, nc, m, n, grid)) {
                        if (nIdx == target) {
                            memo.get(from).put(target, layer);
                            return layer;
                        }
                        q.offer(nIdx);
                    }
                }
            }
        }
        memo.get(from).put(target, -1);
        return -1;
    }

    private int[] idxConvert(int idx, int m, int n) {
        return new int[]{idx / n, idx % n};
    }

    private int idxConvert(int[] idx, int m, int n) {
        return idx[0] * n + idx[1];
    }

    private boolean checkLegal(int row, int col, int m, int n, char[][] grid) {
        int idx = row * n + col;
        return row >= 0 && row < m && col >= 0 && col < n && grid[row][col] != '#';
    }

    private void extractElements(char[][] grid, int m, int n, Set<Integer> mSet, Set<Integer> oSet, Set<Integer> wSet) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                switch (grid[i][j]) {
                    case '#':
                        wSet.add(idx);
                        break;
                    case 'S':
                        start = idx;
                        break;
                    case 'T':
                        end = idx;
                        break;
                    case 'O':
                        oSet.add(idx);
                        break;
                    case 'M':
                        mSet.add(idx);
                        break;
                }
            }
        }
    }

    // LC1886
    public boolean findRotation(int[][] mat, int[][] target) {
        int n = mat.length;
        // 原地旋转90°算法, 做4次(最后一次起始和起始相同, 不管了)
        for (int dummy = 0; dummy < 4; dummy++) {
            // 先沿着竖着的轴线对折
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n / 2; col++) {
                    if (mat[row][col] != mat[row][n - 1 - col]) {
                        mat[row][col] ^= mat[row][n - 1 - col];
                        mat[row][n - 1 - col] ^= mat[row][col];
                        mat[row][col] ^= mat[row][n - 1 - col];
                    }
                }
            }
            // 然后沿着左上-右下的对角线对折 (逆时针, 不影响)
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < row; col++) {
                    if (mat[row][col] != mat[col][row]) {
                        mat[row][col] ^= mat[col][row];
                        mat[col][row] ^= mat[row][col];
                        mat[row][col] ^= mat[col][row];
                    }
                }
            }
            // 然后校验
            boolean legal = true;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (mat[i][j] != target[i][j]) {
                        legal = false;
                        break;
                    }
                }
            }
            if (legal) return true;
        }
        return false;
    }

    // LC943 Hard Revenge! 状压DP TSP
    public String shortestSuperstring(String[] words) {
        int n = words.length;
        int fullMask = (1 << n) - 1;
        int[][] dp = new int[fullMask + 1][n];
        int[][] parent = new int[fullMask + 1][n]; // 记录转移到dp[mask][i] 的上一个词的下标
        int[][] overlapCache = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    int maxLen = Math.min(words[i].length(), words[j].length());
                    for (int k = maxLen; k >= 0; k--) {
                        // j 接在 i 跟后, 最多能重叠多少个
                        if (words[i].endsWith(words[j].substring(0, k))) {
                            overlapCache[i][j] = k;
                            break;
                        }
                    }
                }
            }
        }

        // dp[mask][j] 表示 在选了 mask 里面的单词, 以j为最后一个单词时候, 重叠部分的最大长度

        for (int mask = 1; mask <= fullMask; mask++) {
            Arrays.fill(parent[mask], -1);
            for (int selected = 0; selected < n; selected++) {
                if (((mask >> selected) & 1) == 1) {
                    // 他从哪个选项而来?
                    int parentMask = mask ^ (1 << selected);
                    if (parentMask == 0) continue; // 不可能由空的而来 直接跳过

                    // 在上一个选项中, 可以以哪个为结尾? 遍历上一个选项中mask的所有可能
                    for (int prevEnd = 0; prevEnd < n; prevEnd++) {
                        if (((parentMask >> prevEnd) & 1) == 1) {
                            // 如果是以prevEnd 结尾转移过来
                            int overlap = overlapCache[prevEnd][selected] + dp[parentMask][prevEnd];
                            if (overlap > dp[mask][selected]) {
                                parent[mask][selected] = prevEnd;
                                dp[mask][selected] = overlap;
                            }
                        }
                    }
                }
            }
        }

        int maxOverlap = Integer.MIN_VALUE, lastEleIdx = -1;
        for (int i = 0; i < n; i++) {
            if (dp[fullMask][i] > maxOverlap) {
                maxOverlap = dp[fullMask][i];
                lastEleIdx = i;
            }
        }

        List<Integer> perm = new ArrayList<>(); // 开始遍历parent
        int curEleIdx = lastEleIdx, curMask = fullMask;
        boolean[] visited = new boolean[n];
        while (curEleIdx != -1) {
            perm.add(curEleIdx);
            visited[curEleIdx] = true;
            int nextEleIdx = parent[curMask][curEleIdx];
            curMask ^= (1 << curEleIdx); // 去除掉mask当前的词
            curEleIdx = nextEleIdx;
        }
        Collections.reverse(perm);
        StringBuilder sb = new StringBuilder();
        sb.append(words[perm.get(0)]);
        for (int i = 1; i < perm.size(); i++) {
            sb.append(words[perm.get(i)].substring(overlapCache[perm.get(i - 1)][perm.get(i)]));
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                sb.append(words[i]);
            }
        }
        return sb.toString();
    }


    // LC1289 Hard
    public int minFallingPathSumDp(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[2][n];
        for (int i = 0; i < 2; i++) Arrays.fill(dp[i], Integer.MAX_VALUE / 2);
        for (int i = 0; i < n; i++) {
            dp[(m - 1) % 2][i] = grid[m - 1][i]; // 递归终止条件
        }

        for (int row = m - 2; row >= 0; row--) {
            for (int col = 0; col < n; col++) {
                for (int down = 0; down < n; down++) {
                    if (down != col) {
                        dp[row % 2][col] = Math.min(dp[row % 2][col], grid[row][col] + dp[(row + 1) % 2][down]);
                    }
                }
            }
            if (row >= 1) { // 保留最顶上两行的结果, 其余时候都要将下一行用最大值填充
                Arrays.fill(dp[(row - 1) % 2], Integer.MAX_VALUE / 2);
            }
        }
        int result = Integer.MAX_VALUE / 2;
        for (int i = 0; i < n; i++) {
            result = Math.min(result, dp[0][i]);
        }
        return result;
    }

    Integer[][] lc1289Memo;

    public int minFallingPathSum(int[][] grid) {
        int col = grid[0].length, row = grid.length;
        int result = Integer.MAX_VALUE;
        lc1289Memo = new Integer[row + 1][col + 1];
        for (int i = 0; i < col; i++) {
            result = Math.min(result, lc1289Helper(i, 0, grid));
        }
        return result;
    }

    private int lc1289Helper(int curCol, int curRow, int[][] grid) {
        int curVal = grid[curRow][curCol];
        if (curRow == grid.length - 1) return curVal;
        if (lc1289Memo[curRow][curCol] != null) return lc1289Memo[curRow][curCol];
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < grid[0].length; i++) {
            if (i != curCol) {
                result = Math.min(result, curVal + lc1289Helper(i, curRow + 1, grid));
            }
        }
        return lc1289Memo[curRow][curCol] = result;
    }

    // LC1022
    int lc1022Result = 0;

    public int sumRootToLeaf(TreeNode54 root) {
        lc1022Helper(root, 0);
        return lc1022Result;
    }

    private void lc1022Helper(TreeNode54 root, int val) {
        val = (val << 1) + root.val;
        if (root.left == null && root.right == null) {
            lc1022Result += val;
        }
        if (root.left != null) {
            lc1022Helper(root.left, val);
        }
        if (root.right != null) {
            lc1022Helper(root.right, val);
        }
    }

    // LC517 Hard **
    public int findMinMoves(int[] machines) {
        int n = machines.length, sum = 0, result = 0;
        for (int i : machines) sum += i;
        if (sum % n != 0) return -1;
        // 选定一个范围, 这个范围内的数组的数相邻之间可以互相+1 -1
        int avg = sum / n;
        sum = 0;
        for (int i : machines) {
            i -= avg;
            sum += i;
            result = Math.max(result, Math.max(Math.abs(sum), i));
        }
        return result;
    }

    // Interview 17.24 ** 最大子序和 二维版 O(n^3)
    public int[] getMaxMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] prefix = new int[m + 1][n + 1];
        int globalMax = Integer.MIN_VALUE / 2;
        int[] result = new int[]{-1, -1, -1, -1};

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }

        for (int top = 0; top < m; top++) { // 枚举上下边界
            for (int bottom = top; bottom < m; bottom++) {
                int localMax = Integer.MIN_VALUE / 2, left = -1, right = -1;
                int[] dp = new int[n + 1];
                int[] startColumn = new int[n + 1];
                Arrays.fill(startColumn, -1);
                Arrays.fill(dp, Integer.MIN_VALUE / 2);
                // 最大子序和
                for (int col = 0; col < n; col++) {
                    // 这一列的和
                    int thisColumn = prefix[bottom + 1][col + 1] - prefix[top][col + 1] - prefix[bottom + 1][col] + prefix[top][col];
                    dp[col + 1] = Math.max(dp[col] + thisColumn, thisColumn); // 暂不考虑溢出
                    if (dp[col + 1] == thisColumn) {
                        startColumn[col + 1] = col;
                    } else {
                        startColumn[col + 1] = startColumn[col];
                    }
                    if (dp[col + 1] > localMax) {
                        localMax = dp[col + 1];
                        if (localMax > globalMax) {
                            globalMax = localMax;
                            result = new int[]{top, startColumn[col + 1], bottom, col};
                        }
                    }

                }
            }
        }
        return result;
    }

    // LC1139
    public int largest1BorderedSquare(int[][] grid) {
        int maxSideLen = 0, m = grid.length, n = grid[0].length;
        int[][] upOnes = new int[m + 1][n + 1];
        int[][] leftOnes = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (grid[i - 1][j - 1] == 1) {
                    upOnes[i][j] = 1 + upOnes[i - 1][j];
                    leftOnes[i][j] = 1 + leftOnes[i][j - 1];
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                // 以[i,j] 为右下角
                int limit = Math.min(leftOnes[i + 1][j + 1], upOnes[i + 1][j + 1]);
                if (limit <= maxSideLen) continue;

                for (int sideLen = limit; sideLen >= 0; sideLen--) {
                    if (sideLen <= maxSideLen) break;
                    int upMost = i - sideLen + 1, leftMost = j - sideLen + 1;
                    if (leftOnes[upMost + 1][j + 1] >= sideLen && upOnes[i + 1][leftMost + 1] >= sideLen) {
                        maxSideLen = Math.max(maxSideLen, sideLen);
                        break;
                    }
                }
            }
        }
        return maxSideLen * maxSideLen;
    }

    // LC1903
    public String largestOddNumber(String num) {
        int n = num.length();
        for (int i = n - 1; i >= 0; i--) {
            if ((num.charAt(i) - '0') % 2 == 1) return num.substring(0, i + 1);
        }
        return "";
    }

    // LC774 ** 非常好
    public double minmaxGasDist(int[] stations, int k) {
        int n = stations.length;
        int maxInt = Integer.MIN_VALUE;
        for (int i = 1; i < n; i++) {
            int d = stations[i] - stations[i - 1];
            maxInt = Math.max(maxInt, d);
        }
        double lo = 0, hi = maxInt;
        while (Math.abs(hi - lo) >= 1e-6) {
            double mid = (hi + lo) / 2;
            if (lc774Check(mid, k, stations)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return lo;
    }

    private boolean lc774Check(double maxDist, int k, int[] stations) {
        int require = 0;
        for (int i = 1; i < stations.length; i++) {
            require += (int) ((stations[i] - stations[i - 1] + 0d) / maxDist);
        }
        return require <= k;
    }

    // LC1995
    public int countQuadruplets(int[] nums) {
        int n = nums.length, result = 0;
        for (int i = 0; i < n - 3; i++) {
            for (int j = i + 1; j < n - 2; j++) {
                for (int o = j + 1; o < n - 1; o++) {
                    for (int p = o + 1; p < n; p++) {
                        if (nums[i] + nums[j] + nums[o] == nums[p]) {
                            result++;
                        }
                    }
                }
            }
        }
        return result;
    }

    // JZOF II 021 LC19
    public ListNode54 removeNthFromEnd(ListNode54 head, int n) {
        if (n <= 0) return head;
        ListNode54 fast = head, dummy = new ListNode54();
        dummy.next = head;
        int ctr = 0;
        while (ctr != n) {
            fast = fast.next;
            ctr++;
        }
        ListNode54 slow = head, pre = dummy;
        while (fast != null) {
            fast = fast.next;
            pre = slow;
            slow = slow.next;
        }
        pre.next = slow.next;

        return dummy.next;
    }


    // Interview 04.06 **
    public TreeNode54 inorderSuccessorTraverse(TreeNode54 root, TreeNode54 p) {
        Deque<TreeNode54> q = new LinkedList<>();
        boolean find = false;
        while (root != null || !q.isEmpty()) {
            while (root != null) {
                q.push(root);
                root = root.left;
            }
            root = q.pop();
            if (find) return root;
            if (root == p) find = true;
            root = root.right;
        }
        return null;
    }

    public TreeNode54 inorderSuccessor(TreeNode54 root, TreeNode54 p) { // BST!!
        TreeNode54 pre = null;
        while (root.val != p.val) { // 先找到p, 并记录前驱
            if (root.val < p.val) {
                root = root.right;
            } else {
                pre = root; // 只有当左拐的时候, 前驱才会出现在中序遍历的下一个位置, 所以这时候更新前驱
                root = root.left;
            }
        }
        if (root.right == null) return pre;
        root = root.right;
        while (root.left != null) root = root.left;
        return root;
    }


    // LC363 二维前缀和
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;
        int result = Integer.MIN_VALUE;
        int[][] prefix = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                for (int o = 0; o < i; o++) {
                    for (int p = 0; p < j; p++) {
                        int area = prefix[i][j] - prefix[o][j] - prefix[i][p] + prefix[o][p];
                        if (area <= k) {
                            result = Math.max(result, area);
                        }
                    }
                }
            }
        }
        return result;
    }

    // LC568
    Integer[][] lc568Memo;

    public int maxVacationDays(int[][] flights, int[][] days) { // N*N, N*K
        lc568Memo = new Integer[days[0].length + 1][flights.length + 1];
        return lc568Helper(0, 0, flights, days);
    }

    private int lc568Helper(int kth, int curCity, int[][] flights, int[][] days) {
        // 假设每次进入函数都是周一
        if (kth == days[0].length) return 0; // 如果到了最后一周 不能休息 直接返回0
        if (lc568Memo[kth][curCity] != null) return lc568Memo[kth][curCity];
        // 如果不飞
        int result = days[curCity][kth] + lc568Helper(kth + 1, curCity, flights, days);
        // 看能飞到哪里去
        for (int targetCity = 0; targetCity < flights.length; targetCity++) {
            if (flights[curCity][targetCity] == 1) {
                // 当天飞走, 则必须要在目标机场度过这一周, 将这一周的休息时间计入贡献
                result = Math.max(result, days[targetCity][kth] + lc568Helper(kth + 1, targetCity, flights, days));
            }
        }
        return lc568Memo[kth][curCity] = result;
    }

    // LC661
    public int[][] imageSmoother(int[][] img) {
        int[][] result = new int[img.length][img[0].length];
        for (int i = 0; i < img.length; i++) {
            for (int j = 0; j < img[0].length; j++) {
                int sum = 0;
                int ctr = 0;
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <= 1; y++) {
                        if (i + x >= 0 && i + x < img.length && j + y >= 0 && j + y < img[0].length) {
                            sum += img[i + x][j + y];
                            ctr++;
                        }
                    }
                }
                result[i][j] = sum / ctr;
            }
        }
        return result;
    }

    // LC1872 STONE GAME VIII **
    Integer[] lc1872Memo;

    public int stoneGameVIII(int[] stones) {
        int n = stones.length;
        int[] prefix = new int[n + 1];
        lc1872Memo = new Integer[n + 3];
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + stones[i];
        }
        return lc1872Helper(prefix, 1);
    }

    private int lc1872Helper(int[] prefix, int idx) {
        if (idx >= prefix.length - 2) { // 如果一个选手直接取了所有石子, 则结果(两者之差)显然是所有石子的价值之和
            return prefix[prefix.length - 1];
        }
        if (lc1872Memo[idx] != null) return lc1872Memo[idx];
        return lc1872Memo[idx] = Math.max(lc1872Helper(prefix, idx + 1), prefix[idx + 1] - lc1872Helper(prefix, idx + 1));
    }


    // LC437
    public int pathSum(TreeNode54 root, int targetSum) {
        Map<Integer, Integer> prefix = new HashMap<>();
        prefix.put(0, 1);
        return lc437Helper(root, 0, targetSum, prefix);
    }

    private int lc437Helper(TreeNode54 root, int cur, int target, Map<Integer, Integer> prefix) {
        if (root == null) return 0;
        cur += root.val;
        int result = prefix.getOrDefault(cur - target, 0);
        prefix.put(cur, prefix.getOrDefault(cur, 0) + 1);
        result += lc437Helper(root.left, cur, target, prefix);
        result += lc437Helper(root.right, cur, target, prefix);
        prefix.put(cur, prefix.get(cur) - 1);
        return result;
    }

    // LC1933
    public boolean isDecomposable(String s) {
        int twoCount = 0, threeCount = 0;
        char prev = '\0';
        int ctr = 0;
        for (char c : s.toCharArray()) {
            if (c == prev) {
                ctr++;
            } else {
                if (ctr != 0) {
                    if (ctr % 3 != 0 && ctr % 3 != 2) {
                        return false;
                    }
                    if (ctr % 3 == 0) threeCount += ctr / 3;
                    if (ctr % 3 == 2) {
                        twoCount++;
                        threeCount += ctr / 3;
                        if (twoCount > 1) return false;
                    }
                }
                ctr = 1;
                prev = c;
            }
        }
        if (ctr != 0) {
            if (ctr % 3 != 0 && ctr % 3 != 2) {
                return false;
            }
            if (ctr % 3 == 0) threeCount += ctr / 3;
            if (ctr % 3 == 2) {
                twoCount++;
                threeCount += ctr / 3;
                if (twoCount > 1) return false;
            }
        }
        return twoCount == 1;
    }

    // LCP24 **
    public int[] numsGame(int[] nums) {
        // 对前i个数, 满足nums[0] ~ nums[i-1] 是一个公差为1的等差数列, 对每个数至少要操作多少次? (i>0)
        // 即 nums[a] +1 = nums[a+1]  <=> nums[a] - a = nums[a+1] - (a+1)
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            nums[i] -= i;
        }
        int[] result = new int[n];
        result[0] = 0;
        MedianFinderMod mf = new MedianFinderMod();
        mf.addNum(nums[0]);
        for (int i = 1; i < n; i++) {
            mf.addNum(nums[i]);

            int median = mf.findMedian();
            long r = (mf.minSum - median * mf.minPq.size() + median * mf.maxPq.size() - mf.maxSum);
            result[i] = (int) (r % 1000000007);

        }
        return result;
    }

    // LC1791 有O(1)的方法
    public int findCenter(int[][] edges) {
        int n = edges.length + 1;
        int[] indegree = new int[n];
        for (int[] e : edges) {
            indegree[e[0] - 1]++;
            indegree[e[1] - 1]++;
        }
        for (int i = 0; i < n; i++) {
            if (indegree[i] == n - 1) return i + 1;
        }
        return -1;
    }

    // LC1849
    public boolean splitString(String s) {
        return lc1849Helper(Long.MAX_VALUE, 0, s, 0);
    }

    private boolean lc1849Helper(long prevVal, int curIdx, String s, int parts) {
        if (curIdx == s.length()) {
            return parts > 1; // 只有分出超过1part才正确
        }
        for (int len = 1; len <= s.length() - curIdx; len++) {
            String next = s.substring(curIdx, curIdx + len);
            next = next.replaceAll("^0+(.*)$", "$1"); // 删除前缀零
            if (next.length() > (s.length() + 1) / 2) break; // 实际长度超过一半 后面就不可能分出比他小的值
            if (next.length() == 0) next = "0";
            long val = Long.parseLong(next, 10);
            if (val >= prevVal) return false;  // 只有比前一个值小才有可能符合, 否则立即剪枝
            if (parts == 0 || val == prevVal - 1) { // 如果前面没有值 (第一个值), 或者前面有值且当前值使前一个值减一
                if (lc1849Helper(val, curIdx + len, s, parts + 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    // LC273
    class Lc273 {
        public String numberToWords(int num) {
            if (num == 0) return "Zero";
            int bil = num / 1000000000;
            int mil = (num / 1000000) % 1000;
            int tho = (num / 1000) % 1000;
            int rest = num % 1000;
            List<String> result = new ArrayList<>();
            if (bil != 0) result.add(threeDigit(bil) + " Billion");
            if (mil != 0) result.add(threeDigit(mil) + " Million");
            if (tho != 0) result.add(threeDigit(tho) + " Thousand");
            if (rest != 0) result.add(threeDigit(rest));
            return String.join(" ", result);
        }

        private String one(int num) {
            switch (num) {
                case 1:
                    return "One";
                case 2:
                    return "Two";
                case 3:
                    return "Three";
                case 4:
                    return "Four";
                case 5:
                    return "Five";
                case 6:
                    return "Six";
                case 7:
                    return "Seven";
                case 8:
                    return "Eight";
                case 9:
                    return "Nine";
            }
            return "";
        }

        private String belowTwenty(int num) {
            switch (num) {
                case 10:
                    return "Ten";
                case 11:
                    return "Eleven";
                case 12:
                    return "Twelve";
                case 13:
                    return "Thirteen";
                case 14:
                    return "Fourteen";
                case 15:
                    return "Fifteen";
                case 16:
                    return "Sixteen";
                case 17:
                    return "Seventeen";
                case 18:
                    return "Eighteen";
                case 19:
                    return "Nineteen";
            }
            return "";
        }

        private String ten(int num) {
            switch (num) {
                case 2:
                    return "Twenty";
                case 3:
                    return "Thirty";
                case 4:
                    return "Forty";
                case 5:
                    return "Fifty";
                case 6:
                    return "Sixty";
                case 7:
                    return "Seventy";
                case 8:
                    return "Eighty";
                case 9:
                    return "Ninety";
            }
            return "";
        }

        private String twoDigit(int num) {
            if (num == 0) return "";
            if (num < 10) return one(num);
            if (num < 20) return belowTwenty(num);
            int ten = num / 10, one = num % 10;
            if (one == 0) return ten(ten);
            return ten(ten) + " " + one(one);
        }

        private String threeDigit(int num) {
            if (num == 0) return "";
            int hundred = num / 100, rest = num % 100;
            if (hundred != 0 && rest != 0) {
                return one(hundred) + " Hundred " + twoDigit(rest);
            }
            if (hundred == 0 && rest != 0) {
                return twoDigit(rest);
            }
            if (hundred != 0 && rest == 0) {
                return one(hundred) + " Hundred";
            }
            return "";
        }
    }

    // LC1335 Hard 可以抽象成: 将一个数组分成k份使这k个子数组的最大值的和最小
    Integer[][] lc1335Memo;

    public int minDifficulty(int[] jobDifficulty, int d) {
        int n = jobDifficulty.length;
        if (d > n) return -1;
        lc1335Memo = new Integer[d + 1][n + 1];
        return lc1335Helper(jobDifficulty, 0, 0, d);
    }

    private int lc1335Helper(int[] jobDifficulty, int curDay, int curJob, int daysLeft) {
        if (curJob == jobDifficulty.length) return 0;
        if (lc1335Memo[curDay][curJob] != null) return lc1335Memo[curDay][curJob];
        if (daysLeft == 0) { // 如果只剩0天了, 那肯定要在当天赶DDL把剩下的做完
            int max = 0;
            for (int i = curJob; i < jobDifficulty.length; i++) {
                max = Math.max(max, jobDifficulty[i]);
            }
            return lc1335Memo[curDay][curJob] = max;
        } else {
            // 保证剩下的每天都有工作可以分配, 找一个当天可以做的工作的数量上界
            int remainJobs = jobDifficulty.length - curJob;
            int maxJobs = remainJobs - daysLeft + 1;
            // 做前几个工作?
            int max = 0;
            int result = Integer.MAX_VALUE;
            for (int i = 1; i <= maxJobs; i++) {
                max = Math.max(max, jobDifficulty[curJob + i - 1]);
                result = Math.min(result, max + lc1335Helper(jobDifficulty, curDay + 1, curJob + i, daysLeft - 1));
            }
            return lc1335Memo[curDay][curJob] = result;
        }
    }

    // LC276 ** 打家劫舍?
    public int numWays(int n, int k) {
        int[][] dp = new int[2][2]; //dp[i][0] 表示末尾没有重复颜色, dp[i][1] 表示刷到i最后两个颜色重复的刷法
        dp[1][0] = k;
        dp[1][1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i % 2][0] = (k - 1) * (dp[(i - 1) % 2][0] + dp[(i - 1) % 2][1]); // 乘以k-1是因为不能刷和上一个相同的颜色(0表示末尾没有重复颜色)
            dp[i % 2][1] = dp[(i - 1) % 2][0];
        }
        return dp[n % 2][0] + dp[n % 2][1];
    }


    // LC639
    Long[] lc639Memo;
    final long lc639Mod = 1000000007;

    public int numDecodingsDp(String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        long[] dp = new long[n + 1];
        dp[n] = 1;
        for (int i = n - 1; i >= 0; i--) {
            if (ca[i] == '0') continue;
            long one = 0;
            if (Character.isDigit(ca[i])) {
                one += dp[i + 1];
            } else {
                one += (9 * dp[i + 1]) % lc639Mod;
            }

            long two = 0;
            if (i + 2 <= ca.length) {
                if (ca[i] == '*' && ca[i + 1] == '*') {
                    two += 15 * dp[i + 2];
                } else if (ca[i] == '*' && ca[i + 1] != '*') {
                    two += dp[i + 2];
                    if (ca[i + 1] >= '0' && ca[i + 1] <= '6') {
                        two += dp[i + 2];
                    }
                } else if (ca[i] != '*' && ca[i + 1] == '*') {
                    if (ca[i] == '1') {
                        two += 9 * dp[i + 2];
                    } else if (ca[i] == '2') {
                        two += 6 * dp[i + 2];
                    }
                } else {
                    if (s.substring(i, i + 2).compareTo("26") <= 0) {
                        two += dp[i + 2];
                    }
                }
            }
            dp[i] = (one + two) % lc639Mod;
        }
        return (int) (dp[0] % lc639Mod);
    }

    public int numDecodings(String s) {
        int n = s.length();
        lc639Memo = new Long[n + 1];
        return (int) (lc639Helper(s, 0) % lc639Mod);
    }

    private long lc639Helper(String s, int idx) {
        if (idx >= s.length()) return 1;
        if (s.charAt(idx) == '0') return 0;
        if (lc639Memo[idx] != null) return lc639Memo[idx];

        long one = 0;
        if (Character.isDigit(s.charAt(idx))) {
            one += lc639Helper(s, idx + 1);
        } else {
            one += (9 * (long) lc639Helper(s, idx + 1)) % lc639Mod;
        }

        long two = 0;
        // 如果两位都是星号 有15种可能(11~19, 21~26)
        if (idx + 2 <= s.length()) {
            if (s.charAt(idx) == '*' && s.charAt(idx + 1) == '*') {
                two += 15 * lc639Helper(s, idx + 2);
            } else if (s.charAt(idx) == '*' && s.charAt(idx + 1) != '*') {
                // 如果有这一位是*, 又要考虑两位产生的贡献, 则只能取1或者2 (3~9不产生贡献)
                // 如果*=1, 则第二位可以取任何值, 产生了一种可能
                two += lc639Helper(s, idx + 2);

                /// 如果*=2, 则第二位只能取0~6之间的值, 否则不会产生任何新的可能
                if (s.charAt(idx + 1) >= '0' && s.charAt(idx + 1) <= '6') {
                    two += lc639Helper(s, idx + 2);
                }
            } else if (s.charAt(idx) != '*' && s.charAt(idx + 1) == '*') {
                // 同样讨论第一位是1还是2
                if (s.charAt(idx) == '1') { // 如果第一位是1, 则*可以取1~9, 供9种
                    two += 9 * lc639Helper(s, idx + 2);
                } else if (s.charAt(idx) == '2') { // 如果第一位是2, *可以取1~6, 共6种
                    two += 6 * lc639Helper(s, idx + 2);
                }
            } else { // 如果两位都是数字的情况
                if (s.substring(idx, idx + 2).compareTo("26") <= 0) {
                    two += lc639Helper(s, idx + 2);
                }
            }
        }
        return lc639Memo[idx] = ((one + two) % lc639Mod);
    }


    // LC634 ** 错位排列个数
    public int findDerangement(int n) {
        if (n == 0) return 1; //??
        if (n == 1) return 0;
        final int mod = 1000000007;
        // dp[n] = (n-1) * (dp[n-2] + dp[n-1])
        long[] dp = new long[3]; // 滚数组
        dp[0] = 1;
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i % 3] = ((i - 1) * ((dp[(i - 2) % 3] + dp[(i - 1) % 3]))) % mod;
        }
        return (int) dp[n % 3];
    }

    // LC1138
    public String alphabetBoardPath(String target) {
        StringBuilder result = new StringBuilder();
        String[] boardArr = {"abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"};
        char[][] boardMtx = new char[6][];
        for (int i = 0; i < 6; i++) {
            boardMtx[i] = boardArr[i].toCharArray();
        }
        Map<Character, int[]> letterIdxMap = new HashMap<>(26);
        for (int i = 0; i < boardMtx.length; i++) {
            for (int j = 0; j < boardMtx[i].length; j++) {
                letterIdxMap.put(boardMtx[i][j], new int[]{i, j});
            }
        }
        int[] cur = {0, 0};
        for (char c : target.toCharArray()) {
            int[] targetIdx = letterIdxMap.get(c);
            int y = targetIdx[0] - cur[0], x = targetIdx[1] - cur[1];
            // Z对策
            if (c == 'z') {
                // 先横向移动
                if (x < 0) {
                    while (x++ != 0) result.append('L');
                } else {
                    while (x-- != 0) result.append('R');
                }
                // 再纵向移动
                if (y < 0) {
                    while (y++ != 0) result.append('U');
                } else {
                    while (y-- != 0) result.append('D');
                }
            } else {
                if (y < 0) {
                    while (y++ != 0) result.append('U');
                } else {
                    while (y-- != 0) result.append('D');
                }

                if (x < 0) {
                    while (x++ != 0) result.append('L');
                } else {
                    while (x-- != 0) result.append('R');
                }
            }
            result.append('!');
            cur = targetIdx;
        }
        return result.toString();
    }

    // JZOF II 004 LC137
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int i = 0; i < Integer.SIZE; i++) {
            int total = 0;
            for (int j : nums) {
                total += (j >> i) & 1;
            }
            if (total % 3 != 0) { // 出现了3次的话 这一位上为1的数目mod3肯定为0
                result |= (1 << i);
            }
        }
        return result;
    }

    // LC1290
    public int getDecimalValue(ListNode54 head) {
        int result = 0;
        ListNode54 cur = head;
        while (cur != null) {
            result <<= 1;
            result += cur.val;
            cur = cur.next;
        }
        return result;
    }

    // LC648 JZOF II 063 **
    public String replaceWords(String[] dictionary, String sentence) {
        Trie54 trie = new Trie54();
        for (String w : dictionary) trie.addWord(w);
        List<String> result = new ArrayList<>();
        String[] split = sentence.split("\\s+");
        for (String w : split) {
            TrieNode54 cur = trie.root;
            StringBuilder appending = new StringBuilder();
            for (char c : w.toCharArray()) {
                if (!cur.children.containsKey(c) || cur.end != 0) {
                    break;
                }
                cur = cur.children.get(c);
                appending.append(c);
            }
            if (cur.end != 0) {
                result.add(appending.toString());
            } else {
                result.add(w);
            }
        }
        return String.join(" ", result);
    }

    // LC1880
    public boolean isSumEqual(String firstWord, String secondWord, String targetWord) {
        return getStringVal(targetWord) == getStringVal(firstWord) + getStringVal(secondWord);
    }

    private long getStringVal(String w) {
        long result = 0;
        for (char c : w.toCharArray()) {
            result = result * 10 + (c - 'a');
        }
        return result;
    }

    // LC1509 思想就是每次修改最大或最小的数 使之变成非最大非最小
    public int minDifference(int[] nums) {
        if (nums.length <= 4) return 0;
        int n = nums.length;
        Arrays.sort(nums);
        int pos1 = Math.min(nums[n - 1] - nums[3], nums[n - 4] - nums[0]);
        int pos2 = Math.min(nums[n - 2] - nums[2], nums[n - 3] - nums[1]);
        return Math.min(pos1, pos2);
    }

    // LC968
    class Lc968 {
        final int NEED = 0, HAS = 1, NO_NEED = 2;
        int result = 0;
        Map<TreeNode54, Integer> memo = new HashMap<>();

        public int minCameraCover(TreeNode54 root) {
            if (dfs(root) == NEED) result++;
            return result;
        }

        private int dfs(TreeNode54 root) {
            if (root == null) return NO_NEED;
            int left = dfs(root.left), right = dfs(root.right);
            if (left == NEED || right == NEED) {
                result++;
                return HAS;
            }
            return (left == HAS || right == HAS) ? NO_NEED : NEED;
        }
    }

    // LC979 **
    int lc979Result = 0;

    public int distributeCoins(TreeNode54 root) {
        lc979Dfs(root);
        return lc979Result;
    }

    private int lc979Dfs(TreeNode54 root) {
        if (root == null) return 0;
        int left = lc979Dfs(root.left), right = lc979Dfs(root.right);
        lc979Result += Math.abs(left) + Math.abs(right);
        return root.val + left + right - 1;
    }

    // LC834 ** O(n) from solution
    int[] lc834Result;
    Map<Integer, Set<Integer>> lc834Mtx;
    int[] lc834Size;
    int[] lc834Dp;

    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        lc834Result = new int[n];
        lc834Dp = new int[n];
        lc834Size = new int[n];
        lc834Mtx = new HashMap<>(n);
        for (int i = 0; i < n; i++) lc834Mtx.put(i, new HashSet<>());
        for (int[] e : edges) {
            lc834Mtx.get(e[0]).add(e[1]);
            lc834Mtx.get(e[1]).add(e[0]);
        }
        lc834Helper(0, -1);
        lc834Helper2(0, -1);
        return lc834Result;
    }

    private void lc834Helper(int root, int parent) { // 返回的是这个节点下面共有多少个子节点
        lc834Size[root] = 1;
        lc834Dp[root] = 0;
        for (int child : lc834Mtx.get(root)) {
            if (child != parent) {
                lc834Helper(child, root);
                lc834Dp[root] += lc834Dp[child] + lc834Size[child];
                lc834Size[root] += lc834Size[child];
            }
        }
    }

    private void lc834Helper2(int root, int parent) {
        lc834Result[root] = lc834Dp[root];
        for (int child : lc834Mtx.get(root)) {
            if (child != parent) {
                int origDpRoot = lc834Dp[root], origDpChild = lc834Dp[child], origSizeRoot = lc834Size[root], origSizeChild = lc834Size[child];

                // 换根, 将root 变成 child的孩子, 此时dp[root] 要减去 child的贡献, 即 dp[root] -= (origDpChild + origSizeChild)
                // size[root] -= origSizeChild
                lc834Dp[root] -= origDpChild + origSizeChild;
                lc834Size[root] -= origSizeChild;
                // 而child则多了 root作为孩子的贡献
                lc834Dp[child] += lc834Dp[root] + lc834Size[root];
                lc834Size[child] += lc834Size[root];
                lc834Helper2(child, root);

                lc834Dp[child] = origDpChild;
                lc834Size[child] = origSizeChild;
                lc834Dp[root] = origDpRoot;
                lc834Size[root] = origSizeRoot;
            }
        }
    }


    // Interview 01.06
    public String compressString(String s) {
        char[] ca = s.toCharArray();
        StringBuilder sb = new StringBuilder();
        char prev = '\0';
        int ctr = 0;
        for (char c : ca) {
            if (c == prev) ctr++;
            else {
                if (ctr != 0) {
                    sb.append(prev);
                    sb.append(ctr);
                }
                ctr = 1;
                prev = c;
            }
        }
        if (ctr != 0) {
            sb.append(prev);
            sb.append(ctr);
        }
        return s.length() <= sb.length() ? s : sb.toString();
    }

    // LC1140 ** 注意状态定义
    int[] lc1140Prefix;
    int[][] lc1140Memo;
    BitSet lc1140Visit;

    public int stoneGameII(int[] piles) {
        int n = piles.length;
        lc1140Prefix = new int[n + 1];
        for (int i = 0; i < n; i++) lc1140Prefix[i + 1] = lc1140Prefix[i] + piles[i];
        lc1140Memo = new int[n + 1][n + 1];
        lc1140Visit = new BitSet((n + 1) * (n + 1));
        return lc1140Helper(n, 1, 0);
    }

    private int lc1140Helper(int n, int m, int ptr) { // 返回的是先手能从剩下的石堆中能取到的最多的石子数量
        if (ptr == n) return 0;
        if (lc1140Visit.get(m * (n + 1) + ptr)) return lc1140Memo[m][ptr];
        lc1140Visit.set(m * (n + 1) + ptr);
        // 如果取值范围覆盖到了n, 则全部拿走
        if (ptr + 2 * m >= n) return lc1140Memo[m][ptr] = lc1140Prefix[n] - lc1140Prefix[ptr];
        int maxGain = Integer.MIN_VALUE;
        for (int len = 1; len <= 2 * m; len++) {
            if (ptr + len > n) break;
            int remain = lc1140Prefix[n] - lc1140Prefix[ptr];
            int myGainThisTime = lc1140Prefix[len + ptr] - lc1140Prefix[ptr];
            int advGain = lc1140Helper(n, Math.max(len, m), ptr + len);
            // 现在剩余的所有石子- 己方本轮的所有石子 -  对方将来得到得到的所有石子 = 己方将来得到的所有石子
            int myGainFuture = remain - myGainThisTime - advGain;
            maxGain = Math.max(maxGain, myGainFuture + myGainThisTime);
        }
        return lc1140Memo[m][ptr] = maxGain;
    }

    // LC1025
    Boolean[] lc1025Memo = new Boolean[1001];

    public boolean divisorGame(int n) {
        if (n <= 1) return false;
        if (lc1025Memo[n] != null) return lc1025Memo[n];
        int sqrt = (int) Math.sqrt(n);
        for (int i = 2; i <= sqrt; i++) {
            if (n % i == 0) {
                if (!divisorGame(n - i) || !divisorGame(n - n / i)) {
                    return lc1025Memo[n] = true;
                }
            }
        }
        if (!divisorGame(n - 1)) return lc1025Memo[n] = true;
        return lc1025Memo[n] = false;
    }

    // LC1686 **
    public int stoneGameVI(int[] aliceValues, int[] bobValues) {
        int n = aliceValues.length;
        List<int[]> totalValuesAndIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int totalValues = aliceValues[i] + bobValues[i];
            totalValuesAndIdx.add(new int[]{totalValues, i});
        }
        Collections.sort(totalValuesAndIdx, Comparator.comparingInt(o -> -o[0]));
        int alice = 0, bob = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) alice += aliceValues[totalValuesAndIdx.get(i)[1]];
            else bob += bobValues[totalValuesAndIdx.get(i)[1]];
        }
        if (alice == bob) return 0;
        if (alice > bob) return 1;
        return -1;
    }

    // LC877
    Boolean[] lc877Memo;

    public boolean stoneGame(int[] piles) {
        LinkedList<Integer> pileList = new LinkedList<>();
        for (int i : piles) pileList.add(i);
        lc877Memo = new Boolean[piles.length];
        return lc877Helper(pileList, 0, 0, 0);
    }

    private boolean lc877Helper(LinkedList<Integer> pileList, int myGain, int advGain, int status) {
        if (pileList.size() == 0) {
            return myGain > advGain;
        }
        if (lc877Memo[status] != null) return lc877Memo[status];
        // 左侧
        int left = pileList.getFirst(), right = pileList.getLast();
        pileList.removeFirst();
        boolean result = lc877Helper(pileList, advGain, myGain + left, status + 1);
        pileList.addFirst(left);
        if (!result) return lc877Memo[status] = true;

        pileList.removeLast();
        result = lc877Helper(pileList, advGain, myGain + right, status);
        pileList.addLast(right);
        if (!result) return lc877Memo[status] = true;

        return lc877Memo[status] = false;
    }


    // LC1908
    BitSet lc1908Visited;
    BitSet lc1908Memo;
    int lc1908Base;

    public boolean nimGame(int[] piles) {
        lc1908Base = Arrays.stream(piles).max().getAsInt() + 1;
        int status = calStatus(piles);
        lc1908Memo = new BitSet(status + 1);
        lc1908Visited = new BitSet(status + 1);
        return lc1908Helper(piles);
    }

    private boolean lc1908Helper(int[] piles) {
        int status = calStatus(piles);
        if (status == 0) return false;
        if (lc1908Visited.get(status)) return lc1908Memo.get(status);
        lc1908Visited.set(status);
        for (int i = 0; i < piles.length; i++) {
            if (piles[i] > 0) {
                for (int j = 1; j <= piles[i]; j++) {
                    piles[i] -= j;
                    boolean lose = lc1908Helper(piles);
                    piles[i] += j;
                    if (!lose) {
                        lc1908Memo.set(status);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private int calStatus(int[] piles) {
        int status = 0;
        for (int i : piles) {
            status *= lc1908Base;
            status += i;
        }
        return status;
    }


    // LC1973
    int lc1973Result = 0;

    public int equalToDescendants(TreeNode54 root) {
        if (root == null) return 0;
        lc1973Dfs(root);
        return lc1973Result;
    }

    private int lc1973Dfs(TreeNode54 root) {
        if (root == null) return 0;
        int sum = lc1973Dfs(root.left) + lc1973Dfs(root.right);
        if (root.val == sum) lc1973Result++;
        return sum + root.val;
    }

    // LC430
    class Lc430 {
        public Node flatten(Node head) {
            if (head == null) return null;
            Deque<Node> dfs = new LinkedList<>(); // for dfs
            dfs.push(head);
            Deque<Node> stack = new LinkedList<>(); // for linking
            while (!dfs.isEmpty()) {
                Node p = dfs.pop();
                stack.push(p);
                if (p.next != null) dfs.push(p.next);
                if (p.child != null)
                    dfs.push(p.child); // rule: if child is not null, make child next, so last push child (for first pop)
            }
            Node next = null;
            while (!stack.isEmpty()) {
                Node p = stack.pop();
                p.child = null; // make the child null or error occur
                p.next = next;
                next = p;
                if (!stack.isEmpty()) {
                    p.prev = stack.peek();
                } else {
                    p.prev = null;
                }
            }
            return head;
        }

        class Node {
            public int val;
            public Node prev;
            public Node next;
            public Node child;
        }
    }

    // Interview 04.01
    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        Map<Integer, Set<Integer>> edge = new HashMap<>();
        for (int[] e : graph) {
            edge.putIfAbsent(e[0], new HashSet<>());
            edge.get(e[0]).add(e[1]);
        }
        Deque<Integer> q = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int p = q.poll();
            if (visited.contains(p)) continue;
            if (p == target) return true;
            visited.add(p);
            for (int next : edge.getOrDefault(p, new HashSet<>())) {
                if (!visited.contains(next)) {
                    q.offer(next);
                }
            }
        }
        return false;
    }

    // LC440 ** 第k大字典序数
    public int findKthNumber(int upperBound, int k) {
        int cur = 1;
        k--; // 从1开始。 如果从0开始不需要减
        while (k != 0) {
            int num = lc440Helper(cur, upperBound);
            if (num <= k) {
                cur++;
                k -= num;
            } else {
                cur *= 10;
                k--;
            }
        }
        return cur;
    }

    private int lc440Helper(int prefix, int upperBound) {// inclusive
        int count = 0;
        for (long cur = prefix, next = prefix + 1; cur <= upperBound; cur *= 10, next *= 10) {
            count += Math.min(upperBound + 1, next) - cur;
        }
        // 如: 数字位数1位的时候 有多少个以1开头的个数?
        // a=1, b=2, count += 2-1
        // 上界为12
        // a=10 b=20, 10 11 12, count += 12+1-10
        return count;
    }

    // LC708 ** JZOF II 029
    class Lc708 {
        public Node insert(Node head, int insertVal) {
            if (head == null) {
                Node n = new Node(insertVal);
                n.next = n;
                return n;
            }
            Function<Node, Node> deal = cur -> {
                Node n = new Node(insertVal);
                Node origNext = cur.next;
                cur.next = n;
                n.next = origNext;
                return head;
            };
            Node cur = head;
            if (cur.next == head) {
                return deal.apply(cur);
            }
            while (true) {
                if (cur.val > cur.next.val) { // cur就是分界点, 就是末尾
                    if (insertVal >= cur.val || insertVal <= cur.next.val) {
                        return deal.apply(cur);
                    }
                } else if (insertVal >= cur.val && insertVal <= cur.next.val) {
                    return deal.apply(cur);
                }
                cur = cur.next;
                // 说明已经转了一圈, 还没有收获, 即圈上所有数都是同一个值
                if (cur.next == head) {
                    return deal.apply(cur);
                }
            }
        }

        class Node {
            public int val;
            public Node next;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, Node _next) {
                val = _val;
                next = _next;
            }
        }

    }

    // LC548 Hard **
    public boolean splitArray(int[] nums) {
        // 要找到三个点, 去除这三个点的数分成四段, 使得四段的和相等
        // 0 1 2 3 4 5 6 i=1 j=3 k = 5, 至少要有7个数
        // 1 <= 1 <= n-6
        // 3 <= i+2 <= j <= n-4
        // 5 <= j+2 <= k <= n-2
        if (nums.length < 7) return false;
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + nums[i];
        // 要点: 枚举中间的点, i,j,k 的j
        for (int j = 3; j <= n - 4; j++) {
            Set<Integer> set = new HashSet<>();
            for (int i = 1; i <= j - 2; i++) {
                if (prefix[i] == prefix[j] - prefix[i + 1]) {
                    set.add(prefix[i]);
                }
            }
            for (int k = j + 2; k <= n - 2; k++) {
                int sum = prefix[n] - prefix[k + 1];
                if (sum == prefix[k] - prefix[j + 1] && set.contains(sum))
                    return true;
            }
        }
        return false;
    }


    // LC325
    public int maxSubArrayLen(int[] nums, int k) {
        int result = 0, n = nums.length, prefix = 0;
        Map<Integer, Integer> prefixIdxMap = new HashMap<>();
        prefixIdxMap.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix += nums[i];
            if (prefixIdxMap.containsKey(prefix - k)) {
                result = Math.max(result, i - prefixIdxMap.get(prefix - k));
            }
            prefixIdxMap.putIfAbsent(prefix, i);
        }
        return result;
    }

    // LC768 Hard **  LC769 Medium 单调栈 非常精妙
    // https://leetcode-cn.com/problems/max-chunks-to-make-sorted-ii/solution/zui-duo-neng-wan-cheng-pai-xu-de-kuai-ii-deng-jie-/
    public int maxChunksToSorted(int[] arr) {
        Deque<Integer> stack = new LinkedList<>(); // 单调递增栈
        for (int i : arr) {
            if (!stack.isEmpty() && i < stack.peek()) {
                int head = stack.pop();
                while (!stack.isEmpty() && i < stack.peek()) stack.pop();
                stack.push(head);
            } else {
                stack.push(i);
            }
        }
        return stack.size();
    }

    // LC1788 Hard
    public int maximumBeauty(int[] flowers) {
        final int OFFSET = 10000;
        int[] firstAppear = new int[20001];
        Arrays.fill(firstAppear, -1);
        int prefix = 0, result = Integer.MIN_VALUE;
        for (int i : flowers) {
            if (firstAppear[i + OFFSET] == -1) {
                firstAppear[i + OFFSET] = prefix;
                prefix += i > 0 ? i : 0;
            } else {
                result = Math.max(result, (prefix += (i > 0 ? i : 0)) - firstAppear[i + OFFSET] + (i < 0 ? i << 1 : 0));
            }
        }
        return result;
    }

    // JZOF II 070 二分 LC540
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid % 2 == 1) {
                mid--;
            }
            if (nums[mid] == nums[mid + 1]) {
                lo = mid + 2;
            } else {
                hi = mid;
            }
        }
        return nums[lo];
    }

    // LC1546 **
    public int maxNonOverlapping(int[] nums, int target) {
        int n = nums.length;
        int prefix = 0;
        int idx = 0;
        int result = 0;
        while (idx < n) {
            Set<Integer> prefixSet = new HashSet<>();
            prefixSet.add(0);
            prefix = 0;
            while (idx < n) {
                prefix = prefix + nums[idx];
                if (prefixSet.contains(prefix - target)) {
                    result++;
                    break;
                } else {
                    prefixSet.add(prefix);
                    idx++;
                }
            }
            idx++;
        }
        return result;
    }

    // LC889 ** 前序 后续 重建二叉树 不唯一
    public TreeNode54 constructFromPrePost(int[] preorder, int[] postorder) {
        if (preorder.length == 0) return null;
        TreeNode54 root = new TreeNode54(preorder[0]);
        if (preorder.length == 1) return root;
        int len = 0;
        for (int i = 0; i < postorder.length; i++) {
            if (postorder[i] == preorder[1]) {
                len = i + 1; // + 1 方便算len
                break;
            }
        }

        root.left = constructFromPrePost(Arrays.copyOfRange(preorder, 1, 1 + len), Arrays.copyOfRange(postorder, 0, len));
        root.right = constructFromPrePost(Arrays.copyOfRange(preorder, 1 + len, preorder.length), Arrays.copyOfRange(postorder, len, postorder.length - 1));
        return root;
    }

    // LC640
    public String solveEquation(String equation) {
        final String INF_RES = "Infinite solutions";
        final String NO_SOL = "No solution";
        final String SOL_IS_ZERO = "x=0";

        String[] parts = equation.split("=");
        int[] left = lc640Helper(parts[0]), right = lc640Helper(parts[1]);

        int leftFactor = left[0], leftConst = left[1];
        int rightFactor = right[0], rightConst = right[1];

        // 无解 / 无限解
        if (leftFactor == rightFactor) {
            if (leftConst == rightConst) return INF_RES;
            else return NO_SOL;
        }
        // 解只有一个且为0
        if (leftConst == rightConst) {
            if (leftFactor != rightFactor) return SOL_IS_ZERO;
            else return INF_RES;
        }
        return "x=" + (rightConst - leftConst) / (leftFactor - rightFactor);
    }

    private int[] lc640Helper(String halfEq) {
        // 左侧
        int factor = 0, constVal = 0;
        boolean comesDigit = false;
        int cur = 0, sign = 1;
        for (char c : halfEq.toCharArray()) {
            if (c == 'x') {
                if (cur == 0 && !comesDigit) factor += 1 * sign;
                else factor += cur * sign;
                sign = 1;
                cur = 0;
                comesDigit = false;
            } else if (c == '+') {
                constVal += cur * sign;
                cur = 0;
                sign = 1;
                comesDigit = false;
            } else if (c == '-') {
                constVal += cur * sign;
                cur = 0;
                sign = -1;
                comesDigit = false;
            } else if (Character.isDigit(c)) {
                cur = cur * 10 + (c - '0');
                comesDigit = true;
            }
        }
        constVal += cur * sign;
        return new int[]{factor, constVal};
    }

    // LC1110
    public List<TreeNode54> delNodes(TreeNode54 root, int[] to_delete) {
        List<TreeNode54> result = new ArrayList<>();
        Set<Integer> toDelete = new HashSet<>();
        for (int i : to_delete) toDelete.add(i);
        Function<TreeNode54, TreeNode54> dfs = new Function<TreeNode54, TreeNode54>() {
            @Override
            public TreeNode54 apply(TreeNode54 root) {
                if (root == null) return null;
                root.left = this.apply(root.left);
                root.right = this.apply(root.right);
                if (toDelete.contains(root.val)) {
                    if (root.left != null) {
                        result.add(root.left);
                    }
                    if (root.right != null) {
                        result.add(root.right);
                    }
                    root = null;
                }
                return root;
            }
        };
        root = dfs.apply(root);
        if (root != null) result.add(root);
        return result;
    }

    // LC1410
    public String entityParser(String text) {
        StringBuilder sb = new StringBuilder();
        int ptr = 0;
        while (ptr < text.length()) {
            if (text.charAt(ptr) != '&') {
                sb.append(text.charAt(ptr));
                ptr++;
                continue;
            }
            if (ptr + 6 <= text.length() && text.startsWith("&quot;", ptr)) {
                sb.append("\"");
                ptr += 6;
            } else if (ptr + 6 <= text.length() && text.startsWith("&apos;", ptr)) {
                sb.append("\'");
                ptr += 6;
            } else if (ptr + 5 <= text.length() && text.startsWith("&amp;", ptr)) {
                sb.append("&");
                ptr += 5;
            } else if (ptr + 4 <= text.length() && text.startsWith("&gt;", ptr)) {
                sb.append(">");
                ptr += 4;
            } else if (ptr + 4 <= text.length() && text.startsWith("&lt;", ptr)) {
                sb.append("<");
                ptr += 4;
            } else if (ptr + 7 <= text.length() && text.startsWith("&frasl;", ptr)) {
                sb.append("/");
                ptr += 7;
            } else {
                sb.append(text.charAt(ptr));
                ptr++;
            }
        }
        return sb.toString();
    }

    // LC1105 ** DP DFS
    public int minHeightShelves(int[][] books, int shelfWidth) {
        int n = books.length;
        Integer[][] memo = new Integer[n + 1][n + 1];
        return lc1105Helper(0, 0, books, shelfWidth, memo);
    }

    private int lc1105Helper(int cur, int floor, int[][] books, int shelfWidth, Integer[][] memo) {
        if (cur == books.length) {
            return 0;
        }
        if (memo[cur][floor] != null) return memo[cur][floor];
        int width = 0, height = 0;
        int result = Integer.MAX_VALUE / 2;
        int i;
        for (i = cur; i < books.length; i++) {
            if (width + books[i][0] > shelfWidth) break;
            width += books[i][0];
            height = Math.max(height, books[i][1]);
            int next = height + lc1105Helper(i + 1, floor + 1, books, shelfWidth, memo);
            result = Math.min(result, next);
        }
        return memo[cur][floor] = result;
    }

    // LC2007
    public int[] findOriginalArray(int[] changed) {
        List<Integer> result = new ArrayList<>();
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i : changed) freq.put(i, freq.getOrDefault(i, 0) + 1);
        // 如果有0, 则一定有2的倍数个0
        if (freq.containsKey(0)) {
            if (freq.get(0) % 2 == 1) return new int[0];
            for (int i = 0; i < freq.get(0) / 2; i++) result.add(0);
            freq.remove(0);
        }
        Arrays.sort(changed);  // 从小往大找, 规避如[16,32,8,64]的测例
        for (int i : changed) {
            if (!freq.containsKey(i) || freq.get(i) == 0) continue;
            if (i % 2 == 0) {
                int lack = 0;
                if (!freq.containsKey(i * 2)) lack++;
                if (!freq.containsKey(i / 2)) lack++;
                if (lack == 2) return new int[0];

                if (freq.containsKey(i * 2)) {
                    result.add(i);
                    freq.put(i, freq.get(i) - 1);
                    if (freq.get(i) == 0) freq.remove(i);
                    freq.put(i * 2, freq.get(i * 2) - 1);
                    if (freq.get(i * 2) == 0) freq.remove(i * 2);
                }

                if (freq.containsKey(i / 2)) {
                    result.add(i / 2);
                    freq.put(i / 2, freq.get(i / 2) - 1);
                    if (freq.get(i / 2) == 0) freq.remove(i / 2);
                    if (!freq.containsKey(i)) return new int[0];
                    freq.put(i, freq.get(i) - 1);
                    if (freq.get(i) == 0) freq.remove(i);
                }
            } else {
                if (!freq.containsKey(i * 2)) return new int[0];
                result.add(i);
                freq.put(i, freq.get(i) - 1);
                if (freq.get(i) == 0) freq.remove(i);
                freq.put(i * 2, freq.get(i * 2) - 1);
                if (freq.get(i * 2) == 0) freq.remove(i * 2);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC725
    public ListNode54[] splitListToParts(ListNode54 head, int k) {
        ListNode54[] result = new ListNode54[k];
        int len = 0;
        ListNode54 cur = head;
        while (cur != null) {
            len++;
            cur = cur.next;
        }
        int partLen = len / k;
        int remain = 0;
        if (partLen == 0) {
            partLen = 1;
        } else {
            remain = len - k * partLen;
        }
        cur = head;
        int ptr = 0;
        int resultCtr = 0;
        ListNode54 partHead = head;
        while (ptr < len) {
            ptr++;
            if (ptr % partLen == 0) {
                if ((remain - 1) >= 0) {
                    cur = cur.next;
                    remain--;
                }
                if (cur == null) break;
                ListNode54 origNext = cur.next;
                cur.next = null;
                result[resultCtr++] = partHead;
                partHead = origNext;
                cur = origNext;
            } else {
                if (cur == null) break;
                cur = cur.next;
            }
        }
        return result;
    }

    // LC8
    public int myAtoi(String s) {
        char[] ca = s.toCharArray();
        boolean skipSpace = false;
        boolean checkSign = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (!skipSpace && c == ' ') continue;
            skipSpace = true;
            if (skipSpace && !checkSign) {
                checkSign = true;
                if (c != '+' && c != '-' && !Character.isDigit(c)) return 0;
                if (c == '-') {
                    sb.append('-');
                    continue;
                }
                if (c == '+') {
                    continue;
                }
            }
            if (!Character.isDigit(c)) break;
            sb.append(c);
        }
        if (sb.length() == 0) return 0;
        if (sb.toString().equals("-")) return 0;
        // 这里可以考虑:1)这里可以考虑和Integer.MAX/MIN .toString() 比较字符串字典序
        BigInteger b = new BigInteger(sb.toString());
        if (b.compareTo(new BigInteger("" + Integer.MAX_VALUE)) > 0) return Integer.MAX_VALUE;
        if (b.compareTo(new BigInteger("" + Integer.MIN_VALUE)) < 0) return Integer.MIN_VALUE;
        return Integer.parseInt(sb.toString());
    }

    // LC650 BFS
    public int minSteps(int n) {
        boolean[][] visited = new boolean[n * 2][n * 2];
        Deque<int[]> q = new LinkedList<>();
        // 当前个数 剪贴板个数
        q.offer(new int[]{1, 0});
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (p[0] == n) return layer;
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                // 复制
                if (p[0] <= n && !visited[p[0]][p[0]]) {
                    q.offer(new int[]{p[0], p[0]});
                }
                // 粘贴
                if (p[0] + p[1] <= n && !visited[p[0] + p[1]][p[0]]) {
                    // if (p[0] + p[1] == n) return layer + 1;  //少这步特判会快1ms
                    q.offer(new int[]{p[0] + p[1], p[1]});
                }
            }
        }
        return -1;
    }

    // LC758 LC616 就硬匹配 问你气不气
    public String boldWords(String[] words, String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        boolean[] mask = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (String w : words) {
                int idx = i;
                while ((idx = s.indexOf(w, idx)) != -1) {
                    for (int j = 0; j < w.length(); j++) {
                        mask[idx + j] = true;
                    }
                    idx += w.length();
                }
            }
        }
        StringBuilder result = new StringBuilder();
        int ptr = 0;
        int boldLen = 0;
        while (ptr < ca.length) {
            if (!mask[ptr]) {
                if (boldLen > 0) {
                    result.append("<b>");
                    result.append(s, ptr - boldLen, ptr);
                    result.append("</b>");
                    result.append(ca[ptr]);
                    boldLen = 0;
                } else {
                    result.append(ca[ptr]);
                }
            } else {
                boldLen++;
            }
            ptr++;
        }
        if (boldLen > 0) {
            result.append("<b>");
            result.append(s, ptr - boldLen, ptr);
            result.append("</b>");
        }
        return result.toString();
    }

    // LC1309
    public String freqAlphabets(String s) {
        StringBuilder sb = new StringBuilder();
        int ptr = 0;
        char[] ca = s.toCharArray();
        while (ptr < ca.length) {
            if (ptr + 2 < ca.length && ca[ptr + 2] == '#') {
                int idx = (ca[ptr] - '0') * 10 + (ca[ptr + 1] - '0') - 1;
                sb.append((char) ('a' + idx));
                ptr += 3;
            } else {
                sb.append((char) ('a' + ca[ptr] - '1'));
                ptr++;
            }
        }
        return sb.toString();
    }

    // LC1602
    public TreeNode54 findNearestRightNode(TreeNode54 root, TreeNode54 u) {
        Deque<TreeNode54> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                TreeNode54 p = q.poll();
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

class TreeNode54 {
    int val;
    TreeNode54 left;
    TreeNode54 right;

    TreeNode54() {
    }

    TreeNode54(int val) {
        this.val = val;
    }

    TreeNode54(int val, TreeNode54 left, TreeNode54 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Trie54 {
    TrieNode54 root = new TrieNode54();

    public void addWord(String word) {
        TrieNode54 cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) cur.children.put(c, new TrieNode54());
            cur = cur.children.get(c);
            cur.path++;
        }
        cur.end++;
    }

    public boolean removeWord(String word) {
        TrieNode54 target = getNode(word);
        if (target == null) return false;
        TrieNode54 cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children.get(c).path-- == 1) {
                cur.children.remove(c);
                return true;
            }
            cur = cur.children.get(c);
        }
        cur.end--;
        return true;
    }

    public boolean search(String word) {
        TrieNode54 target = getNode(word);
        return target.end > 0;
    }

    public boolean startsWith(String word) {
        return getNode(word) != null;
    }

    public void insert(String word) {
        addWord(word);
    }

    public int countWordsStartingWith(String prefix) {
        TrieNode54 target = getNode(prefix);
        if (target == null) return 0;
        return target.path;
    }

    public int countWordsEqualTo(String word) {
        TrieNode54 target = getNode(word);
        if (target == null) return 0;
        return target.end;
    }

    public void erase(String word) {
        removeWord(word);
    }

    private TrieNode54 getNode(String prefix) {
        TrieNode54 cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return null;
            cur = cur.children.get(c);
        }
        return cur;
    }
}

class TrieNode54 {
    Map<Character, TrieNode54> children = new HashMap<>();
    int end = 0;
    int path = 0;
}

class ListNode54 {
    int val;
    ListNode54 next;

    ListNode54() {
    }

    ListNode54(int val) {
        this.val = val;
    }

    ListNode54(int val, ListNode54 next) {
        this.val = val;
        this.next = next;
    }
}

// Interview 16.25 LRU
class LRUCache {
    int cap;
    Map<Integer, Node> m = new HashMap<>();
    Map<Node, Integer> reverse = new HashMap<>();
    Node head;
    Node tail;

    public LRUCache(int capacity) {
        this.cap = capacity;
        head = new Node(-1);
        tail = new Node(-1);
        head.next = head.prev = tail;
        tail.next = tail.prev = head;
    }

    public int get(int key) {
        if (!m.containsKey(key)) return -1;
        int result = m.get(key).val;
        // 将Node 移到头部
        Node victim = m.get(key);
        unlink(victim);
        Node origHeadNext = head.next;
        head.next = victim;
        origHeadNext.prev = victim;
        victim.next = origHeadNext;
        victim.prev = head;
        return result;
    }

    public void put(int key, int value) {
        if (m.containsKey(key)) {
            m.get(key).val = value;
        } else {
            if (m.size() == cap) {
                Node victim = tail.prev;
                int vKey = reverse.get(victim);
                unlink(victim);
                m.remove(vKey);
                reverse.remove(victim);
            }
            Node n = new Node(value);
            m.put(key, n);
            reverse.put(n, key);
        }
        get(key);
    }

    // Node 双向链表, 然后用Map来存指针
    class Node {
        int val;
        Node prev;
        Node next;

        public Node(int val) {
            this.val = val;
        }
    }

    private void unlink(Node n) {
        Node origPrev = n.prev, origNext = n.next;
        if (origNext != null) origNext.prev = origPrev;
        if (origPrev != null) origPrev.next = origNext;
        n.next = null;
        n.prev = null;
    }
}

// LC1622 ** 乘加合并, 可以用树状数组, 线段树, TBD
class Fancy {
    final long mod = 1000000007;
    List<long[]> op = new ArrayList<>(); // [加,乘]
    List<Integer> opIdx = new ArrayList<>();
    List<Integer> val = new ArrayList<>();
    boolean added = false;

    public Fancy() {

    }

    public void append(int v) {
        opIdx.add(op.size()); // 这里 op.size() 永远指向当前append值之后的第一个op (利用了size 和 最大下标之间差一的关系)
        val.add(v);
        added = true;
    }

    public void addAll(int inc) {
        if (added || op.isEmpty()) {
            op.add(new long[]{inc, 1});
        } else { // 合并加
            op.get(op.size() - 1)[0] = (op.get(op.size() - 1)[0] + inc) % mod;
        }
        added = false;
    }

    public void multAll(int m) {
        if (added || op.isEmpty()) {
            op.add(new long[]{0, m});
        } else { // 合并乘和加
            op.get(op.size() - 1)[0] = (op.get(op.size() - 1)[0] * m) % mod;
            op.get(op.size() - 1)[1] = (op.get(op.size() - 1)[1] * m) % mod;
        }
        added = false;
    }

    public int getIndex(int idx) {
        if (idx >= val.size()) return -1;
        long result = val.get(idx);
        for (int i = opIdx.get(idx); i < op.size(); i++) { // 从append之后的第一个op开始, 直到最后一个op
            result = result * op.get(i)[1] % mod;
            result = result + op.get(i)[0] % mod;
        }
        return (int) (result % mod);
    }
}

// LC359
class Logger {

    Map<String, Integer> m = new HashMap<>();

    public Logger() {

    }

    public boolean shouldPrintMessage(int timestamp, String message) {
        if (!m.containsKey(message) || timestamp >= m.get(message)) {
            m.put(message, timestamp + 10);
            return true;
        }
        return false;
    }
}

class MedianFinderMod { // 修改过的mf
    PriorityQueue<Integer> minPq = new PriorityQueue<>(); // 存大的半边
    PriorityQueue<Integer> maxPq = new PriorityQueue<>(Comparator.reverseOrder()); // 存小的半边, 数量要等于minPq 或 等于 minPq.size()+1
    long minSum = 0, maxSum = 0;

    /**
     * initialize your data structure here.
     */
    public MedianFinderMod() {

    }

    public void addNum(int num) {
        if (maxPq.isEmpty()) {
            maxPq.offer(num);
            maxSum += num;
        } else {
            if (num > maxPq.peek()) {
                minPq.offer(num);
                minSum += num;
            } else {
                maxPq.offer(num);
                maxSum += num;
            }
        }

        // 调整
        while (minPq.size() < maxPq.size()) {
            int victim = maxPq.poll();
            maxSum -= victim;
            minPq.offer(victim);
            minSum += victim;
        }
        while (minPq.size() > maxPq.size()) {
            int victim = minPq.poll();
            minSum -= victim;
            maxPq.offer(victim);
            maxSum += victim;
        }
    }

    public int findMedian() {
        if (maxPq.size() > minPq.size()) return maxPq.peek();
        return minPq.peek();
    }
}


// LC489 **
class Lc489 {

    Set<Pair<Integer, Integer>> visited = new HashSet<>();
    int[][] direction = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}; // 务必保证方向的顺序是顺时针方向 (左上右下)
    Robot r;

    public void cleanRoom(Robot robot) {
        this.r = robot;
        backtrack(0, 0, 0);
    }

    private void goBack() {
        r.turnRight();
        r.turnRight();
        r.move();
        r.turnRight();
        r.turnRight();
    }

    private void backtrack(int row, int col, int dirIdx) {
        visited.add(new Pair<>(row, col));
        r.clean();
        for (int i = 0; i < 4; i++) {
            int nd = (dirIdx + 1) % 4;
            int nr = row + direction[nd][0], nc = col + direction[nd][1];
            if (!visited.contains(new Pair<>(nr, nc)) && r.move()) {
                backtrack(nr, nc, nd);
                goBack();
            }
            r.turnRight(); // 顺时针转圈
        }
    }

    interface Robot {
        // Returns true if the cell in front is open and robot moves into the cell.
        // Returns false if the cell in front is blocked and robot stays in the current cell.
        public boolean move();

        // Robot will stay in the same cell after calling turnLeft/turnRight.
        // Each turn will be 90 degrees.
        public void turnLeft();

        public void turnRight();

        // Clean the current cell.
        public void clean();
    }
}