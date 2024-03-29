package moe.nyamori.test.historical;

import java.util.*;


class scratch_57 {
    public static void main(String[] args) {
        scratch_57 s = new scratch_57();
        long timing = System.currentTimeMillis();


        System.out.println(s.palindromePartition("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv", 6));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1278
    Integer[][] memo;

    public int palindromePartition(String s, int k) {
        int n = s.length();
        memo = new Integer[n + 1][k + 1];
        char[] ca = s.toCharArray();
        boolean[][] judge = new boolean[n][n];
        int[][] cost = new int[n][n];
        for (int i = 0; i < n; i++) Arrays.fill(cost[i], -1);
        for (int i = 0; i < n; i++) {
            judge[i][i] = true;
            cost[i][i] = 0;
        }
        // 初始化判定矩阵和代价矩阵
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left + len - 1 < n; left++) {
                int right = left + len - 1;
                if (len == 2) {
                    judge[left][right] = ca[left] == ca[right];

                } else {
                    judge[left][right] = ca[left] == ca[right] && judge[left + 1][right - 1];
                }

                if (judge[left][right]) {
                    cost[left][right] = 0;
                } else {
                    int lPtr = left, rPtr = right;
                    int tmpCost = 0;
                    while (lPtr < rPtr) {
                        if (ca[lPtr] != ca[rPtr]) {
                            tmpCost++;
                        }
                        if (lPtr + 1 < rPtr - 1 && cost[lPtr + 1][rPtr - 1] != -1) {
                            tmpCost += cost[lPtr + 1][rPtr - 1];
                            break;
                        }
                        lPtr++;
                        rPtr--;
                    }
                    cost[left][right] = tmpCost;
                }
            }
        }
        return helper(0, k, cost);
    }

    private int helper(int cur, int remain, int[][] cost) {
        if (cost.length - cur < remain) return Integer.MAX_VALUE / 2; // 如果剩下的字符个数不够分(每个字符作为一个回文串), 则视作无效答案, 返回极大值
        if (remain == 0 && cur < cost.length) return Integer.MAX_VALUE / 2;
        if (cur == cost.length) return 0;
        if (memo[cur][remain] != null) return memo[cur][remain];
        int result = Integer.MAX_VALUE / 2;
        for (int i = cur; i < cost.length; i++) {
            result = Math.min(result, cost[cur][i] + helper(i + 1, remain - 1, cost));
        }
        return memo[cur][remain] = result;
    }

    // LC245 LC244 LC243
    public int shortestWordDistance(String[] wordsDict, String word1, String word2) {
        int n = wordsDict.length;
        Map<String, TreeSet<Integer>> m = new HashMap<>();
        m.put(word1, new TreeSet<>());
        m.put(word2, new TreeSet<>());
        for (int i = 0; i < wordsDict.length; i++) {
            if (!m.containsKey(wordsDict[i])) continue;
            m.get(wordsDict[i]).add(i);
        }
        TreeSet<Integer> ts1 = m.get(word1), ts2 = m.get(word2);
        int result = Integer.MAX_VALUE;
        for (int w1Idx : ts1) {
            Integer lower = ts2.lower(w1Idx), higher = ts2.higher(w1Idx);
            if (lower != null) {
                result = Math.min(result, w1Idx - lower);
            }
            if (higher != null) {
                result = Math.min(result, higher - w1Idx);
            }
        }
        return result;
    }

    // JZOF II 047 LC814
    public TreeNode57 pruneTree(TreeNode57 root) {
        if (!subtreeHasOne(root)) return null;
        lc814Helper(root);
        return root;
    }

    private void lc814Helper(TreeNode57 root) {
        if (root == null) return;
        if (!subtreeHasOne(root.left)) root.left = null;
        else lc814Helper(root.left);
        if (!subtreeHasOne(root.right)) root.right = null;
        else lc814Helper(root.right);
    }

    private boolean subtreeHasOne(TreeNode57 root) {
        if (root == null) return false;
        if (root.val == 1) return true;
        return subtreeHasOne(root.left) || subtreeHasOne(root.right);
    }


    // LC1493 滑窗
    public int longestSubarray(int[] nums) {
        int result = 0, left = 0, right = 0, n = nums.length;
        int[] freq = new int[2];
        while (left < n && right < n) {
            freq[nums[right++]]++;
            while (freq[0] > 1) {
                freq[nums[left++]]--;
            }
            result = Math.max(result, freq[1]);
        }
        if (result == n) return result - 1;
        return result;
    }

    // LC340 经典滑窗
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        Map<Character, Integer> freq = new HashMap<>();
        int max = 0;
        for (int left = 0, right = 0; right < s.length(); right++) {
            freq.put(s.charAt(right), freq.getOrDefault(s.charAt(right), 0) + 1);
            while (left < s.length() && freq.size() > k) {
                if (freq.containsKey(s.charAt(left))) {
                    freq.put(s.charAt(left), freq.get(s.charAt(left)) - 1);
                    if (freq.get(s.charAt(left)) == 0) freq.remove(s.charAt(left));
                }
                left++;
            }
            max = Math.max(max, right - left + 1);
        }
        return max;
    }

    // Interview 17.18 参考LC76, 不同的是LC76是字符, 这里是整数, 所以要用Map查频, 或者像下面这样优化
    public int[] shortestSeq(int[] big, int[] small) {
        int[] freq = new int[small.length];
        Map<Integer, Integer> idxMap = new HashMap<>(small.length);
        for (int i = 0; i < small.length; i++) idxMap.put(small[i], i);
        int count = 0;
        int left = 0, right = 0;
        int[] result = new int[0];
        int min = Integer.MAX_VALUE;
        for (; right < big.length; right++) {
            if (!idxMap.containsKey(big[right])) continue;

            int rightIdx = idxMap.get(big[right]);
            freq[rightIdx]++;
            if (freq[rightIdx] == 1) {
                count++;
            }

            if (count != small.length) continue;

            // 缩减左边
            while (count == small.length) {
                if (idxMap.containsKey(big[left])) {
                    int leftIdx = idxMap.get(big[left]);
                    if (freq[leftIdx] == 1) {
                        // 左边不能再缩减了, 更新答案
                        int len = right - left + 1;
                        if (len < min) {
                            min = len;
                            result = new int[]{left, right};
                        }

                        count--;
                        freq[leftIdx] = 0;
                    } else if (freq[leftIdx] > 1) {
                        freq[leftIdx]--;
                    }
                }
                left++;
            }
        }
        return result;
    }

    class Lc158 {
        String prev = "";

        public int read(char[] buf, int n) {
            if (n <= prev.length()) {
                for (int i = 0; i < n; i++) {
                    buf[i] = prev.charAt(i);
                }
                prev = prev.substring(n);
                return n;
            }

            int ptr = prev.length();
            for (int i = 0; i < prev.length(); i++) {
                buf[i] = prev.charAt(i);
            }
            prev = "";

            char[] buf4 = new char[4];
            int thisReadLen = 0;
            while (ptr < n && ((thisReadLen = read4(buf4)) != 0)) {
                int buf4Ptr = 0;
                for (; buf4Ptr < thisReadLen && ptr < n; buf4Ptr++) {
                    buf[ptr++] = buf4[buf4Ptr];
                }
                if (ptr >= n && buf4Ptr < thisReadLen) {
                    StringBuilder sb = new StringBuilder();
                    while (buf4Ptr < thisReadLen) {
                        sb.append(buf4[buf4Ptr++]);
                    }
                    prev = sb.toString();
                }
            }
            return ptr;
        }

        private int read4(char[] buf) {
            return -1;
        }
    }

    // LC1368 Hard ** 学习建图思路, 本质Dijkstra
    // 稍加修改, 变成0-1 BFS, 不需要PQ
    // 此外还应该了解带负权边的单源最短路算法SPFA
    public int minCost(int[][] grid) {
        // 1 2 3 4  右左下上
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Deque<int[]> q = new LinkedList<>(); // [ r, c, cost ], 代价小的优先
        q.offer(new int[]{0, 0, 0});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int r = p[0], c = p[1], cost = p[2];
            if (r == m - 1 && c == n - 1) return cost;
            if (visited[r][c]) continue;
            visited[r][c] = true;
            for (int i = 0; i < 4; i++) {
                int nr = r + directions[i][0], nc = c + directions[i][1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || visited[nr][nc]) continue;
                if (i == grid[r][c] - 1) { // 注意grid里是1,2,3,4 要减一
                    q.offerFirst(new int[]{nr, nc, cost}); // 代价较小, 添加到队首确保小代价的先出队
                } else {
                    q.offerLast(new int[]{nr, nc, cost + 1}); // 代价较大, 添加到队尾确保后出队
                }
            }
        }
        return -1;
    }

    // JZOF58
    public String reverseWords(String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length / 2; i++) {
            char tmp = ca[i];
            ca[i] = ca[n - 1 - i];
            ca[n - 1 - i] = tmp;
        }
        List<String> list = new ArrayList<>();
        int idx = 0;
        while (idx < n) {
            while (idx < n && ca[idx] == ' ') idx++;
            if (idx >= n) break;
            int start = idx;
            while (idx < n && ca[idx] != ' ') idx++;
            int end = idx;
            StringBuilder sb = new StringBuilder(end - start);
            for (int i = end - 1; i >= start; i--) {
                sb.append(ca[i]);
            }
            list.add(sb.toString());
        }
        return String.join(" ", list);
    }

    // Interview 01.01
    public boolean isUnique(String astr) {
        int[] freq = new int[128];
        for (char c : astr.toCharArray()) {
            if (freq[c]++ > 0) return false;
        }
        return true;
    }

    // LC1419
    public int minNumberOfFrogs(String croakOfFrogs) {
        int[] formerIdx = new int[128];
        Arrays.fill(formerIdx, -1);
        formerIdx['r'] = 'c';
        formerIdx['o'] = 'r';
        formerIdx['a'] = 'o';
        formerIdx['k'] = 'a';
        int[] freq = new int[128];
        int result = 0, count = 0;
        for (char c : croakOfFrogs.toCharArray()) {
            switch (c) {
                case 'c':
                    freq[c]++;
                    count++;
                    break;
                case 'k':
                    if (freq[formerIdx[c]] == 0) return -1;
                    result = Math.max(result, count);
                    count--;
                    freq[formerIdx[c]]--;
                    break;
                default:
                    if (freq[formerIdx[c]] == 0) return -1;
                    freq[formerIdx[c]]--;
                    freq[c]++;
            }
        }
        if (Arrays.stream(freq).sum() != 0) return -1;
        return result;
    }

    // LC1366
    public String rankTeams(String[] votes) {
        int[][] freq = new int[128][26]; // freq[字母][排位] = 频率
        for (String s : votes) {
            for (int i = 0; i < s.length(); i++) {
                freq[s.charAt(i)][i]++;
            }
        }
        List<Character> rank = new ArrayList<>(votes[0].length());
        for (char c : votes[0].toCharArray()) rank.add(c);
        Collections.sort(rank, (o1, o2) -> {
            for (int i = 0; i < 26; i++) {
                if (freq[o1][i] == freq[o2][i]) continue;
                return freq[o2][i] - freq[o1][i];
            }
            return o1 - o2;
        });
        StringBuilder sb = new StringBuilder(rank.size());
        for (char c : rank) sb.append(c);
        return sb.toString();
    }

    // LC1218
    public int longestSubsequence(int[] arr, int difference) {
        final int offset = 20001;
        int[] idx = new int[40005];
        int result = 0;
        for (int i : arr) {
            idx[i + offset] = idx[i - difference + offset] + 1;
            result = Math.max(result, idx[i + offset]);
        }
        return result;
    }

    // LC1654 BFS
    public int minimumJumps(int[] forbidden, int a, int b, int x) {
        // 它可以 往前 跳恰好 a个位置（即往右跳）。
        // 它可以 往后跳恰好 b个位置（即往左跳）。
        // 它不能 连续 往后跳 2 次。
        // 它不能跳到任何forbidden数组中的位置。
        // 跳蚤可以往前跳 超过 它的家的位置，但是它 不能跳到负整数 的位置。

        final int LIMIT = 8000;
        boolean[][] visited = new boolean[LIMIT + 1][2];
        // int[][][] prev = new int[LIMIT + 1][2][];
        Set<Integer> forbid = new HashSet<>(forbidden.length);
        for (int i : forbidden) forbid.add(i);
        Deque<int[]> q = new LinkedList<>(); // [ 当前位置, 向后跳次数 ]
        q.offer(new int[]{0, 0});
        int layer = -1;
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                int cur = p[0], backwardCount = p[1];
                if (cur == x) {
                    // while (prev[cur][backwardCount] != null) {
                    //     System.err.println(cur + " " + backwardCount);
                    //     int origCur = cur;
                    //     cur = prev[origCur][backwardCount][0];
                    //     backwardCount = prev[origCur][backwardCount][1];
                    // }
                    return layer;
                }
                if (visited[cur][backwardCount]) continue;
                visited[cur][backwardCount] = true;

                if (cur + a <= LIMIT && !forbid.contains(cur + a)) {
                    q.offer(new int[]{cur + a, 0});
                    // prev[cur + a][0] = p;
                }
                if (cur - b >= 0 && backwardCount < 1 && !forbid.contains(cur - b)) {
                    q.offer(new int[]{cur - b, backwardCount + 1});
                    // prev[cur - b][backwardCount + 1] = p;
                }
            }
        }
        return -1;
    }

    // DFS
    // LC1654
    Long[][] lc1654Memo = new Long[8001][2];
    final int lc1654Limit = 8000;
    boolean[][] lc1654Visited = new boolean[8001][2];

    public int minimumJumpsDFS(int[] forbidden, int a, int b, int x) {
        // 它可以 往前 跳恰好 a个位置（即往右跳）。
        // 它可以 往后跳恰好 b个位置（即往左跳）。
        // 它不能 连续 往后跳 2 次。
        // 它不能跳到任何forbidden数组中的位置。
        // 跳蚤可以往前跳 超过 它的家的位置，但是它 不能跳到负整数 的位置。
        Set<Integer> forbid = new HashSet<>(forbidden.length);
        for (int i : forbidden) forbid.add(i);
        long result = lc1654Helper(0, x, forbid, a, b, 0);
        if (result == Integer.MAX_VALUE) return -1;
        return (int) result;
    }

    private long lc1654Helper(int cur, int target, Set<Integer> forbid, int forwardStep, int backwardStep, int backwardCount) {
        if (lc1654Memo[cur][backwardCount] != null) return lc1654Memo[cur][backwardCount];
        if (cur == target) return 0;
        if (lc1654Visited[cur][backwardCount]) return Integer.MAX_VALUE; // 防止成环
        lc1654Visited[cur][backwardCount] = true;
        long result = Integer.MAX_VALUE;
        if (cur + forwardStep <= lc1654Limit && !forbid.contains(cur + forwardStep)) {
            result = Math.min(result, 1 + lc1654Helper(cur + forwardStep, target, forbid, forwardStep, backwardStep, 0));
        }
        if (cur - backwardStep >= 0 && backwardCount < 1 && !forbid.contains(cur - backwardStep)) {
            result = Math.min(result, 1 + lc1654Helper(cur - backwardStep, target, forbid, forwardStep, backwardStep, backwardCount + 1));
        }
        lc1654Visited[cur][backwardCount] = false;
        return lc1654Memo[cur][backwardCount] = result;
    }


    // LC95
    public List<TreeNode57> generateTrees(int n) {
        List<TreeNode57> result = new ArrayList<>();
        Set<String> preOrder = new HashSet<>();
        for (int i = 1; i <= n; i++) {
            TreeNode57 root = new TreeNode57(i);
            boolean[] visited = new boolean[n + 1];
            visited[i] = true;
            lc95Helper(root, visited, n - 1, result, preOrder);
        }
        return result;
    }

    private void lc95Helper(TreeNode57 root, boolean[] visited, int leftCount, List<TreeNode57> result, Set<String> preOrder) {
        if (leftCount == 0) {
            StringBuilder sb = new StringBuilder();
            preOrderStr(root, sb);
            if (preOrder.contains(sb.toString())) return;
            preOrder.add(sb.toString());
            result.add(copyTree(root));
            return;
        }
        for (int i = 1; i <= visited.length - 1; i++) {
            if (!visited[i]) {
                insertToBST(root, i);
                visited[i] = true;
                lc95Helper(root, visited, leftCount - 1, result, preOrder);
                visited[i] = false;
                removeFromBST(root, i);
            }
        }
    }

    private void preOrderStr(TreeNode57 root, StringBuilder sb) {
        if (root == null) return;
        sb.append(root.val);
        preOrderStr(root.left, sb);
        preOrderStr(root.right, sb);
    }

    private void insertToBST(TreeNode57 root, int val) {
        // 左小右大
        if (val > root.val) {
            if (root.right == null) {
                root.right = new TreeNode57(val);
                return;
            } else {
                insertToBST(root.right, val);
            }
        } else if (val < root.val) {
            if (root.left == null) {
                root.left = new TreeNode57(val);
                return;
            } else {
                insertToBST(root.left, val);
            }
        }
    }

    private TreeNode57 removeFromBST(TreeNode57 root, int val) {
        if (root == null) return null;
        if (root.val == val) {
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            TreeNode57 minNode = findMinNode(root.right);
            root.val = minNode.val;
            root.right = removeFromBST(root.right, minNode.val);
        } else if (root.val < val) {
            root.right = removeFromBST(root.right, val);
        } else {
            root.left = removeFromBST(root.left, val);
        }
        return root;
    }

    private TreeNode57 findMinNode(TreeNode57 root) {
        while (root.left != null) {
            root = root.left;
        }
        return root;
    }

    private TreeNode57 copyTree(TreeNode57 root) {
        if (root == null) return null;
        TreeNode57 result = new TreeNode57(root.val);
        result.left = copyTree(root.left);
        result.right = copyTree(root.right);
        return result;
    }

    // LC777 **
    public boolean canTransform(String start, String end) {
        // L R 的相对位置是不会变的, 因为不存在L穿越R或反之
        // 去除所有X后 start 和 end 应该一样
        // L 不能到达L 的右侧
        // R 不能到达R 的左侧

        if (start.length() != end.length()) return false;
        if (!start.replaceAll("X", "").equals(end.replaceAll("X", ""))) return false;
        char[] cs = start.toCharArray(), ce = end.toCharArray();
        int n = start.length();

        // 正向遍历 找有无异常L
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (cs[i] == 'L') {
                while (ce[j] != 'L') j++;
                if (i < j) return false;
                j++;
            }
        }

        // 反向遍历 找有无异常R
        j = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (cs[i] == 'R') {
                while (ce[j] != 'R') j--;
                if (i > j) return false;
                j--;
            }
        }
        return true;
    }

    // Interview 17.17
    public int[][] multiSearch(String big, String[] smalls) {
        int n = smalls.length;
        int[][] result = new int[n][];
        for (int i = 0; i < n; i++) {
            if (smalls[i].length() == 0) {
                result[i] = new int[0];
                continue;
            }
            List<Integer> l = new ArrayList<>();
            int idx = 0;
            while (idx < big.length()) {
                idx = big.indexOf(smalls[i], idx);
                if (idx >= 0) l.add(idx);
                else break;
                idx += 1;
            }
            result[i] = l.stream().mapToInt(Integer::valueOf).toArray();
        }
        return result;
    }

    // LC361
    public int maxKilledEnemies(char[][] grid) {
        int m = grid.length, n = grid[0].length, max = 0;
        DSUArray57 dsuRow = new DSUArray57(m * n);
        DSUArray57 dsuCol = new DSUArray57(m * n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0') dsuRow.add(i * n + j);
                if (j - 1 >= 0 && grid[i][j - 1] == '0') dsuRow.merge(i * n + j, i * n + j - 1);
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[j][i] == '0') dsuCol.add(j * n + i);
                if (j - 1 >= 0 && grid[j - 1][i] == '0') dsuCol.merge(j * n + i, (j - 1) * n + i);
            }
        }

        Map<Integer, Integer> rowFatherVisited = new HashMap<>();
        Map<Integer, Integer> colFatherVisited = new HashMap<>();

        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0') {
                    int id = i * n + j;

                    int rowFather = dsuRow.find(id);
                    int colFather = dsuCol.find(id);
                    int rowResult = 0, colResult = 0;

                    if (rowFatherVisited.containsKey(rowFather)) {
                        rowResult = rowFatherVisited.get(rowFather);
                    } else {
                        Deque<int[]> q = new LinkedList<>();
                        for (int k = 0; k < 2; k++) {
                            q.offer(new int[]{i, j, k});
                        }
                        while (!q.isEmpty()) {
                            int[] p = q.poll();
                            int r = p[0], c = p[1], d = p[2];
                            if (grid[r][c] == 'E') rowResult++;
                            int nr = r + directions[d][0], nc = c + directions[d][1];
                            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] != 'W') {
                                q.offer(new int[]{nr, nc, d});
                            }
                        }
                        rowFatherVisited.put(rowFather, rowResult);
                    }

                    if (colFatherVisited.containsKey(colFather)) {
                        colResult = colFatherVisited.get(colFather);
                    } else {
                        Deque<int[]> q = new LinkedList<>();
                        for (int k = 2; k < 4; k++) {
                            q.offer(new int[]{i, j, k});
                        }
                        while (!q.isEmpty()) {
                            int[] p = q.poll();
                            int r = p[0], c = p[1], d = p[2];
                            if (grid[r][c] == 'E') colResult++;
                            int nr = r + directions[d][0], nc = c + directions[d][1];
                            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] != 'W') {
                                q.offer(new int[]{nr, nc, d});
                            }
                        }
                        colFatherVisited.put(colFather, colResult);
                    }

                    max = Math.max(max, colResult + rowResult);
                }
            }
        }
        return max;
    }

    // LC422
    public boolean validWordSquare(List<String> words) {
        int n = words.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < words.get(i).length(); j++) {
                if (j >= words.size()) return false;
                if (i >= words.get(j).length()) return false;
                if (words.get(i).charAt(j) != words.get(j).charAt(i)) return false;
            }
        }
        return true;
    }

    // JZOF 013
    public int movingCount(int m, int n, int k) {
        Deque<int[]> q = new LinkedList<>();
        q.offer(new int[]{0, 0});
        boolean[][] visited = new boolean[m][n];
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int result = 0;
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int r = p[0], c = p[1];
            if (visited[r][c]) continue;
            visited[r][c] = true;
            result++;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && check(nr, nc, k) && !visited[nr][nc]) {
                    q.offer(new int[]{nr, nc});
                }
            }
        }
        return result;
    }

    private boolean check(int x, int y, int bound) {
        return digitSum(x) + digitSum(y) <= bound;
    }

    private int digitSum(int num) {
        int result = 0;
        while (num != 0) {
            result += num % 10;
            num /= 10;
        }
        return result;
    }

    // LC1758
    public int minOperations(String s) {
        char[] ca = s.toCharArray();
        int zero = 0, one = 0;
        for (int i = 0; i < ca.length; i++) {
            if (i % 2 == 0) {
                zero += ca[i] == '0' ? 0 : 1;
                one += ca[i] == '1' ? 0 : 1;
            } else {
                zero += ca[i] == '1' ? 0 : 1;
                one += ca[i] == '0' ? 0 : 1;
            }
        }
        return Math.min(one, zero);
    }

    // LC1888 ** DP
    public int minFlips(String s) {
        int n = s.length();
        if (n == 1) return 0;
        char[] ca = s.toCharArray();
        // prefix[i][j] 表示将s的前i项变更为以j结尾的交替字符串所需要的操作步数
        // suffix[i][j] 表示将s的i...n-1项变更为以j开头的交替字符串所需要的操作步数
        int[][] prefix = new int[n][2], suffix = new int[n][2];
        prefix[0][0] = ca[0] == '0' ? 0 : 1;
        prefix[0][1] = ca[0] == '1' ? 0 : 1;

        for (int i = 1; i < n; i++) {
            prefix[i][0] = prefix[i - 1][1] + (ca[i] == '0' ? 0 : 1);
            prefix[i][1] = prefix[i - 1][0] + (ca[i] == '1' ? 0 : 1);
        }
        if (n % 2 == 0) {
            return Math.min(prefix[n - 1][0], prefix[n - 1][1]);
        }

        suffix[n - 1][0] = ca[n - 1] == '0' ? 0 : 1;
        suffix[n - 1][1] = ca[n - 1] == '1' ? 0 : 1;
        for (int i = n - 2; i >= 0; i--) {
            suffix[i][0] = suffix[i + 1][1] + (ca[i] == '0' ? 0 : 1);
            suffix[i][1] = suffix[i + 1][0] + (ca[i] == '1' ? 0 : 1);
        }

        int result = Math.min(prefix[n - 1][0], prefix[n - 1][1]);
        for (int i = 0; i < n - 1; i++) {
            result = Math.min(result, prefix[i][0] + suffix[i + 1][0]);
            result = Math.min(result, prefix[i][1] + suffix[i + 1][1]);
        }

        return result;
    }

    // LC367
    public boolean isPerfectSquare(int num) {
        if (num == 1) return true;
        int lo = 2, hi = 1 << 16;
        while (lo <= hi) {
            long mid = lo + (hi - lo) / 2;
            long product = mid * mid;
            if (product == num) return true;
            if (product < num) {
                lo = (int) mid + 1;
            } else if (product > num) {
                hi = (int) mid - 1;
            }
        }
        return false;
    }

    // LC998
    public TreeNode57 insertIntoMaxTree(TreeNode57 root, int val) {
        List<Integer> list = toList(root);
        list.add(val);
        return toTree(list);
    }

    private List<Integer> toList(TreeNode57 root) {
        if (root == null) return null;
        List<Integer> left = toList(root.left), right = toList(root.right);
        List<Integer> result = new ArrayList<>();
        result.add(root.val);
        if (left != null) {
            left.addAll(result);
            result = left;
        }
        if (right != null) {
            result.addAll(right);
        }
        return result;
    }

    // LC654
    private TreeNode57 toTree(List<Integer> list) {
        if (list.size() == 0) return null;
        if (list.size() == 1) return new TreeNode57(list.get(0));
        int maxIdx = 0;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(maxIdx) < list.get(i)) {
                maxIdx = i;
            }
        }
        TreeNode57 result = new TreeNode57(list.get(maxIdx));
        result.left = toTree(list.subList(0, maxIdx));
        result.right = toTree(list.subList(maxIdx + 1, list.size()));
        return result;
    }

    // LC1701
    public double averageWaitingTime(int[][] customers) {
        long curTime = customers[0][0];
        long[] totalTime = new long[customers.length];
        curTime = totalTime[0] = curTime + customers[0][1];
        for (int i = 1; i < customers.length; i++) {
            if (curTime < customers[i][0]) {
                curTime = customers[i][0];
            }
            curTime = totalTime[i] = curTime + customers[i][1];
        }
        long totalWaitingTime = 0;
        for (int i = 0; i < customers.length; i++) {
            totalWaitingTime += totalTime[i] - customers[i][0];
        }
        return (totalWaitingTime + 0d) / (customers.length + 0d);
    }

    // LC823 **
    Long[] lc823Memo;
    int[] lc823Arr;
    Map<Integer, Integer> lc823IdxMap;
    final long lc823Mod = 1000000007;

    public int numFactoredBinaryTrees(int[] arr) {
        int n = arr.length;
        lc823Memo = new Long[n + 1];
        Arrays.sort(arr);
        this.lc823Arr = arr;
        lc823IdxMap = new HashMap<>();
        for (int i = 0; i < n; i++) lc823IdxMap.put(arr[i], i);

        long result = 0;
        for (int i = 0; i < n; i++) {
            result += lc823Helper(i);
            result %= lc823Mod;
        }
        return (int) (result);
    }

    private long lc823Helper(int startIdx) {
        if (lc823Memo[startIdx] != null) return lc823Memo[startIdx];
        int val = lc823Arr[startIdx];
        long result = 1;
        for (int leftIdx = 0; leftIdx < startIdx; leftIdx++) {
            int left = lc823Arr[leftIdx];
            if (val % left == 0) {
                int right = val / left;
                if (lc823IdxMap.containsKey(right)) {
                    int rightIdx = lc823IdxMap.get(right);
                    // 注意是乘法原理
                    result += lc823Helper(leftIdx) * lc823Helper(rightIdx);
                    result %= lc823Mod;
                }
            }
        }
        return lc823Memo[startIdx] = result;
    }

    // LC42 Try Dijkstra
    public int trapDijk(int[] height) {
        int n = height.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        pq.offer(new int[]{0, height[0]});
        pq.offer(new int[]{n - 1, height[n - 1]});
        int[] water = new int[n];
        Arrays.fill(water, Integer.MAX_VALUE / 2);
        water[0] = height[0];
        water[n - 1] = height[n - 1];
        boolean[] visited = new boolean[n];
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int cur = p[0], w = p[1];
            if (visited[cur]) continue;
            visited[cur] = true;
            for (int next : new int[]{cur - 1, cur + 1}) {
                if (next >= 0 && next < n && !visited[next]) {
                    int nh = height[next];
                    water[next] = Math.max(height[next], Math.min(water[next], w));
                    pq.offer(new int[]{next, water[next]});
                }
            }
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            result += water[i] - height[i];
        }
        return result;
    }

    // LC42 接雨水I Try DSU
    public int trap(int[] height) {
        int n = height.length;
        DSUArray57 dsu = new DSUArray57(n + 1);
        int oobId = n, maxHeight = 0;
        dsu.add(oobId);
        Map<Integer, List<Integer>> m = new HashMap<>();
        for (int i = 0; i < n; i++) {
            maxHeight = Math.max(maxHeight, height[i]);
            m.putIfAbsent(height[i], new ArrayList<>());
            m.get(height[i]).add(i);
        }
        int visitedCount = 0, result = 0;
        for (int h = 0; h <= maxHeight; h++) {
            if (m.containsKey(h)) {
                for (int id : m.get(h)) {
                    dsu.add(id);
                    visitedCount++;
                    for (int next : new int[]{id - 1, id + 1}) {
                        if (next >= n || next < 0) {
                            dsu.merge(id, oobId);
                        } else if (dsu.contains(next)) {
                            dsu.merge(id, next);
                        }
                    }
                }
            }
            result += visitedCount - dsu.getSelfGroupSize(oobId) + 1;
        }
        return result;
    }

    // LC407 Hard ** 接雨水II
    // From Solution Dijkstra
    public int trapRainWaterDijk(int[][] heightMap) {
        int m = heightMap.length, n = heightMap[0].length, result = 0;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int[][] water = new int[m][n];
        boolean[][] visited = new boolean[m][n];
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[2]));

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    pq.offer(new int[]{i, j, heightMap[i][j]});
                    water[i][j] = heightMap[i][j];
                } else {
                    water[i][j] = Integer.MAX_VALUE / 2;
                }
            }
        }

        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int r = p[0], c = p[1], w = p[2];
            if (visited[r][c]) continue;
            visited[r][c] = true;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc]) {
                    int nh = heightMap[nr][nc];
                    water[nr][nc] = Math.max(nh, Math.min(w, water[nr][nc]));
                    pq.offer(new int[]{nr, nc, water[nr][nc]});
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result += water[i][j] - heightMap[i][j];
            }
        }

        return result;
    }

    // Try DSU
    public int trapRainWaterDSU(int[][] heightMap) {
        int m = heightMap.length, n = heightMap[0].length, maxHeight = 0;
        int outOfBoundId = m * n;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        DSUArray57 dsu = new DSUArray57(m * n + 1); // 将 m*n 视作界外单元格集合
        dsu.add(outOfBoundId);
        Map<Integer, List<int[]>> heightIdxMap = new HashMap<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxHeight = Math.max(maxHeight, heightMap[i][j]);
                heightIdxMap.putIfAbsent(heightMap[i][j], new ArrayList<>());
                heightIdxMap.get(heightMap[i][j]).add(new int[]{i, j});
            }
        }
        int visitedCount = 0, result = 0;
        for (int height = 0; height <= maxHeight; height++) {
            if (heightIdxMap.containsKey(height)) {
                for (int[] idx : heightIdxMap.get(height)) {
                    visitedCount++;
                    int r = idx[0], c = idx[1];
                    int id = r * n + c;
                    dsu.add(id);
                    for (int[] d : directions) {
                        int nr = r + d[0], nc = c + d[1];
                        int nid = nr * n + nc;
                        // 如果界外或者已经访问过
                        if (nr < 0 || nr >= m || nc < 0 || nc >= n) {
                            dsu.merge(id, outOfBoundId);
                        } else if (dsu.contains(nid)) {
                            dsu.merge(id, nid);
                        }
                    }
                }
            }
            result += visitedCount - (dsu.getSelfGroupSize(outOfBoundId) - 1); // 因为界外本身不占任何数量, 所以要减一
        }
        return result;
    }

    public int trapRainWater(int[][] heightMap) {
        // 由外往内 BFS
        int m = heightMap.length, n = heightMap[0].length, maxHeight = 0;
        int[][] water = new int[m][n], directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Deque<int[]> q = new LinkedList<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxHeight = Math.max(maxHeight, heightMap[i][j]);
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 将最外层的接水高度调整至自身高度, 并且加入队列, 由外而内BFS
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    water[i][j] = heightMap[i][j];
                    q.offer(new int[]{i, j});
                } else {
                    water[i][j] = maxHeight;
                }
            }
        }

        while (!q.isEmpty()) {
            int[] p = q.poll();
            int r = p[0], c = p[1];
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n) continue;
                // 如果当前位置比周围的接水高度要低, 并且周围位置的接水高度是高于本身高度的
                // 说明周围位置要降低自身的接水高度到与当前位置一样, 并且周围位置的下一个外层也需要相应调整, 故加入队列
                // 队列里每加入一个元素, 都可能导致整个water数组的调整, 故时间复杂度为O(m*m*n*n)
                if (water[r][c] < water[nr][nc] && water[nr][nc] > heightMap[nr][nc]) {
                    water[nr][nc] = Math.max(heightMap[nr][nc], water[r][c]);
                    q.offer(new int[]{nr, nc});
                }
            }
        }

        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result += water[i][j] - heightMap[i][j];
            }
        }
        return result;
    }

    // LCP44
    public int numColor(TreeNode57 root) {
        Set<Integer> s = new HashSet<>();
        Deque<TreeNode57> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode57 p = q.poll();
            s.add(p.val);
            if (p.left != null) q.offer(p.left);
            if (p.right != null) q.offer(p.right);
        }
        return s.size();
    }

    // LC1552 ** 二分 注意判定函数的思想
    public int maxDistance(int[] position, int m) {
        Arrays.sort(position);
        int lo = 1, hi = position[position.length - 1] - position[0] + 1;
        while (lo < hi) { // 满足条件的最大值
            int mid = lo + (hi - lo + 1) / 2;
            if (lc1552Check(position, mid, m)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    private boolean lc1552Check(int[] position, int distance, int bound) {
        int count = 1, prev = position[0];
        for (int i = 1; i < position.length; i++) {
            if (position[i] - prev < distance) continue;
            count++;
            prev = position[i];
        }
        return count >= bound;
    }

    // JZOF 19 LC10
    Boolean[][] lc10Memo;

    public boolean isMatch(String s, String p) {
        lc10Memo = new Boolean[s.length() + 1][p.length() + 1];
        return lc10Helper(0, 0, s.toCharArray(), p.toCharArray());
    }

    private boolean lc10Helper(int sIdx, int pIdx, char[] s, char[] p) {
        if (pIdx >= p.length) return sIdx >= s.length;
        if (lc10Memo[sIdx][pIdx] != null) return lc10Memo[sIdx][pIdx];

        // 单匹配
        boolean singleMatch = sIdx < s.length && (s[sIdx] == p[pIdx] || p[pIdx] == '.');

        // 多个匹配
        if (pIdx < p.length - 1 && p[pIdx + 1] == '*') {
            // 匹配0次 || 匹配多次
            return lc10Memo[sIdx][pIdx] = lc10Helper(sIdx, pIdx + 2, s, p) || (singleMatch && lc10Helper(sIdx + 1, pIdx, s, p));
        }
        return lc10Memo[sIdx][pIdx] = singleMatch && lc10Helper(sIdx + 1, pIdx + 1, s, p);
    }

    // LC1610 极坐标 滑动窗空 几何
    public int visiblePoints(List<List<Integer>> points, int angle, List<Integer> location) {
        int ox = location.get(0), oy = location.get(1);
        double gap = ((double) angle / 360d) * 2;
        List<Double> pointRadian = new ArrayList<>();
        int count = 0, result = 0;
        for (List<Integer> p : points) {
            int px = p.get(0) - ox, py = p.get(1) - oy;
            if (px == 0 && py == 0) {
                count++;
                continue;
            }
            double theta = Math.atan2(py, px) / Math.PI;
            pointRadian.add(theta);
        }
        if (pointRadian.size() == 0) return count; // 如果所有点都和原点重合, 没有任何点构成角度
        Collections.sort(pointRadian);
        List<Double> pass2 = new ArrayList<>(pointRadian);
        for (int i = 0; i < pass2.size(); i++) {
            pass2.set(i, pass2.get(i) + 2);
        }
        pointRadian.addAll(pass2);
        // 从 最小的点 开始, 一路滑窗
        double left = pointRadian.get(0);
        double right = left + gap;
        int leftIdx = 0, rightIdx = 0;
        while (rightIdx < pointRadian.size()) {
            if (pointRadian.get(rightIdx) > right) break;
            count++;
            rightIdx++;
        }
        result = Math.max(result, count);
        // 滑窗
        while (rightIdx < pointRadian.size()) {
            int sameLeftAngleCount = 1;
            while (leftIdx + 1 < pointRadian.size() && pointRadian.get(leftIdx + 1) == pointRadian.get(leftIdx)) {
                leftIdx++;
                sameLeftAngleCount++;
            }
            count -= sameLeftAngleCount;
            leftIdx++;
            left = pointRadian.get(leftIdx);
            right = left + gap;
            while (rightIdx < pointRadian.size()) {
                if (pointRadian.get(rightIdx) > right) break;
                count++;
                rightIdx++;
            }
            result = Math.max(result, count);
        }
        return result;
    }

    // LC469 ** 几何 叉积
    public boolean isConvex(List<List<Integer>> points) {
        long prev = 0;
        int n = points.size();
        for (int i = 0; i < n; i++) {
            int x0 = points.get(i).get(0), y0 = points.get(i).get(1);
            int x1 = points.get((i + 1) % n).get(0), y1 = points.get((i + 1) % n).get(1);
            int x2 = points.get((i + 2) % n).get(0), y2 = points.get((i + 2) % n).get(1);
            int dx1 = x1 - x0, dx2 = x2 - x0, dy1 = y1 - y0, dy2 = y2 - y0;
            long cur = dx1 * dy2 - dx2 * dy1; // ** 叉积公式, 正负代表z轴方向
            if (cur != 0) {
                if (cur * prev < 0) return false; // 判断前后的向量叉积是否同向
                prev = cur;
            }
        }
        return true;
    }

    // LC237
    public void deleteNode(ListNode57 node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    // LC1462
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        // prerequisites[k] : [i,j] , i before j
        // queries[k]: [i,j] is i before j
        boolean[][] reachable = new boolean[numCourses][numCourses];
        for (int[] pq : prerequisites) {
            reachable[pq[0]][pq[1]] = true;
        }
        for (int i = 0; i < numCourses; i++) {
            Deque<Integer> q = new LinkedList<>();
            boolean[] visited = new boolean[numCourses];
            q.offer(i);
            while (!q.isEmpty()) {
                int p = q.poll();
                if (visited[p]) continue;
                visited[p] = true;
                if (i != p) reachable[i][p] = true;
                for (int next = 0; next < numCourses; next++) {
                    if (reachable[p][next]) {
                        q.offer(next);
                    }
                }
            }
        }
        List<Boolean> result = new ArrayList<>(queries.length);
        for (int[] q : queries) {
            result.add(reachable[q[0]][q[1]]);
        }
        return result;
    }

    // LC1615
    public int maximalNetworkRank(int n, int[][] roads) {
        Map<Integer, Set<Integer>> m = new HashMap<>(n);
        int result = 0;
        for (int i = 0; i < n; i++) m.put(i, new HashSet<>());
        for (int[] r : roads) {
            m.get(r[0]).add(r[1]);
            m.get(r[1]).add(r[0]);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int cross = m.get(i).contains(j) ? 1 : 0;
                int rank = m.get(i).size() + m.get(j).size() - cross;
                result = Math.max(rank, result);
            }
        }
        return result;
    }

    // LC854 ** 很妙的DFS
    Map<String, Map<String, Integer>> lc854Memo = new HashMap<>();

    public int kSimilarity(String s1, String s2) {
        if (s1.equals("")) return 0;
        if (lc854Memo.containsKey(s1) && lc854Memo.get(s1).containsKey(s2)) return lc854Memo.get(s1).get(s2);
        lc854Memo.putIfAbsent(s1, new HashMap<>());
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) == s2.charAt(0)) {
                if (i == 0) {
                    result = Math.min(result, kSimilarity(s1.substring(1), s2.substring(1)));
                } else {
                    // 把 i 换到第一个来
                    result = Math.min(result, 1 + kSimilarity(s1.substring(1, i) + s1.charAt(0) + s1.substring(i + 1), s2.substring(1)));
                }
            }
        }
        lc854Memo.get(s1).put(s2, result);
        return result;
    }

    // LC1944 ** 单调栈
    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] result = new int[n];
        Deque<Integer> stack = new LinkedList<>(); // 递减栈
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty()) {
                result[i]++;
                if (heights[i] > heights[stack.peek()]) {
                    stack.pop();
                } else {
                    break;
                }
            }
            stack.push(i);
        }
        return result;
    }

    // LC944
    public int minDeletionSize(String[] strs) {
        int result = 0;
        outer:
        for (int i = 0; i < strs[0].length(); i++) {
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].charAt(i) < strs[j - 1].charAt(i)) {
                    result++;
                    continue outer;
                }
            }
        }
        return result;
    }

    // LC885
    public int[][] spiralMatrixIII(int rows, int cols, int rStart, int cStart) {
        int[][] result = new int[rows * cols][];
        int ctr = 0, total = rows * cols;
        int[][] directions = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int dIdx = 0;
        int r = rStart, c = cStart;
        result[ctr++] = new int[]{r, c};
        int steps = 1;
        while (ctr != total) {
            for (int i = 0; i < steps; i++) {
                r += directions[dIdx][0];
                c += directions[dIdx][1];
                if (r >= 0 && r < rows && c >= 0 && c < cols && ctr < total) {
                    result[ctr++] = new int[]{r, c};
                }
            }
            if (dIdx == 1 || dIdx == 3) steps++;
            dIdx = (dIdx + 1) % 4;
        }
        return result;
    }

    // LC1703 ** from solution 非常严格的数学证明 学不来
    // https://leetcode-cn.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/solution/de-dao-lian-xu-k-ge-1-de-zui-shao-xiang-lpa9i/
    public int minMoves(int[] nums, int k) {
        int sum = 0, n = nums.length, result = Integer.MAX_VALUE;
        for (int i : nums) sum += i;
        // 1. 找出所有1的位置
        int[] oneIdx = new int[sum];
        int ctr = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                oneIdx[ctr++] = i;
            }
        }

        // 2. 构造 g 函数, 构造g函数的累加函数(前缀和)
        int[] g = new int[sum];
        for (int i = 0; i < sum; i++) {
            g[i] = oneIdx[i] - i;
        }
        int[] gPrefix = new int[sum + 1];
        for (int i = 0; i < sum; i++) gPrefix[i + 1] = gPrefix[i] + g[i];

        // 3. 滑窗
        for (int i = 0; i + k <= g.length; i++) {
            int mid = (i + i + k - 1) / 2;
            int q = g[mid];
            int possible = (2 * (mid - i) - k + 1) * q + (gPrefix[i + k] - gPrefix[mid + 1]) - (gPrefix[mid] - gPrefix[i]);
            result = Math.min(result, possible);
        }
        return result;
    }

    // LC1576
    public String modifyString(String s) {
        StringBuilder sb = new StringBuilder();
        int n = s.length();
        char[] ca = s.toCharArray();
        outer:
        for (int i = 0; i < n; i++) {
            if (Character.isLetter(ca[i])) {
                sb.append(ca[i]);
                continue;
            }
            Set<Character> invalid = new HashSet<>(3);
            if (i - 1 >= 0) {
                invalid.add(sb.charAt(sb.length() - 1));
            }
            if (i + 1 < n && ca[i + 1] != '?') {
                invalid.add(ca[i + 1]);
            }
            for (int j = 0; j < 26; j++) {
                char target = (char) ('a' + j);
                if (!invalid.contains(target)) {
                    sb.append(target);
                    continue outer;
                }
            }
        }
        return sb.toString();
    }

    // LC1998 **
    public boolean gcdSort(int[] nums) {
        DSUArray57 dsu = new DSUArray57((int) 1e5);
        for (int i : nums) {
            if (dsu.contains(i)) continue;
            dsu.add(i);
            int factor = 2;
            int victim = i;
            // 分解质因数
            while (victim != 0) {
                while (victim % factor == 0) {
                    dsu.add(factor);
                    dsu.merge(factor, i);
                    victim /= factor;
                }
                factor++;
                if (factor > victim) break;
            }
        }

        int[] sorted = Arrays.copyOf(nums, nums.length);
        Arrays.sort(sorted);

        for (int i = 0; i < nums.length; i++) {
            if (sorted[i] != nums[i]) {
                if (!dsu.isConnected(sorted[i], nums[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    // LC575
    public int distributeCandies(int[] candyType) {
        int n = candyType.length, result = 0;
        Arrays.sort(candyType);
        for (int i = 0; i < n; ) {
            while (i + 1 < n && candyType[i] == candyType[i + 1]) i++;
            result++;
            if (result >= n / 2) return n / 2;
            i++;
        }
        return result;
    }

    // LC260 **
    public int[] singleNumber(int[] nums) {
        long xor = 0;
        for (long i : nums) xor ^= i;
        long lsb = xor & (-xor);
        long a = 0, b = 0;
        for (int i : nums) {
            if (((long) (i) & lsb) != 0) {
                a ^= i;
            } else {
                b ^= i;
            }
        }
        return new int[]{(int) a, (int) b};
    }

    // LC1911 **
    // from https://codeforces.com/contest/1420/submission/93658399
    public long maxAlternatingSum(int[] nums) {
        // 偶数下标之和减奇数下标之和
        int n = nums.length;
        long[][] dp = new long[n + 2][2];
        // dp[n][0/1] 表示选前n个数做子序列时候, 第n个数字下标作为偶/奇数时候的最大值
        for (int i = 0; i < n; i++) {
            dp[i + 1][0] = Math.max(dp[i][0]/*不选这个数*/, dp[i][1] + nums[i]/*选这个数, 从上一个奇数结尾的状态转移过来*/);
            dp[i + 1][1] = Math.max(dp[i][1], dp[i][0] - nums[i]);
        }
        return Math.max(dp[n][0], dp[n][1]);
    }

    // LC1102 **
    public int maximumMinimumPathDSU(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int start = 0, end = m * n - 1;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        DSUArray57 dsu = new DSUArray57(m * n + 2);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o[2]));
        for (int i = 1; i < end; i++) {
            pq.offer(new int[]{i / n, i % n, grid[i / n][i % n]});
        }
        dsu.add(start);
        dsu.add(end);
        while (!pq.isEmpty() && !dsu.isConnected(start, end)) {
            int[] p = pq.poll();
            int r = p[0], c = p[1], val = p[2];
            int id = r * n + c;
            dsu.add(id);
            for (int[] d : direction) {
                int nr = r + d[0], nc = c + d[1];
                int nid = nr * n + nc;
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && dsu.contains(nid)) {
                    dsu.merge(id, nid);
                    min = Math.min(min, val);
                }
            }
        }
        return min;
    }

    public int maximumMinimumPath(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        Set<Integer> possibleSet = new HashSet<>();
        possibleSet.add(min);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] <= min) {
                    possibleSet.add(grid[i][j]);
                }
            }
        }
        List<Integer> possibleList = new ArrayList<>(possibleSet);
        Collections.sort(possibleList);
        int lo = 0, hi = possibleList.size() - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            int victim = possibleList.get(mid);
            boolean[][] visited = new boolean[m][n];
            if (lc1102Helper(0, 0, grid, visited, victim)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return possibleList.get(lo);
    }

    private boolean lc1102Helper(int r, int c, int[][] grid, boolean[][] visited, int bound) {
        if (r == grid.length - 1 && c == grid[0].length - 1) return true;
        visited[r][c] = true;
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] d : direction) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && !visited[nr][nc] && grid[nr][nc] >= bound) {
                if (lc1102Helper(nr, nc, grid, visited, bound)) return true;
            }
        }
        return false;
    }


    // LC565
    public int arrayNesting(int[] nums) {
        int n = nums.length, max = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int count = 1, cur = i;
                visited[i] = true;
                while (!visited[nums[cur]]) {
                    visited[nums[cur]] = true;
                    count++;
                    cur = nums[cur];
                }
                max = Math.max(count, max);
            }
        }
        return max;
    }

    // LC1145 ** 贪心策略: 选x周围的三个节点, 统计两个子图节点数量
    Map<Integer, TreeNode57> valNodeMap = new HashMap<>();
    Map<TreeNode57, TreeNode57> fatherMap = new HashMap<>();

    public boolean btreeGameWinningMove(TreeNode57 root, int n, int x) {
        Deque<TreeNode57> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode57 p = q.poll();
            valNodeMap.put(p.val, p);
            n = Math.max(n, p.val);
            if (p.left != null) {
                fatherMap.put(p.left, p);
                q.offer(p.left);
            }
            if (p.right != null) {
                fatherMap.put(p.right, p);
                q.offer(p.right);
            }
        }

        TreeNode57 xNode = valNodeMap.get(x);
        TreeNode57[] choices = new TreeNode57[]{getFather(xNode), getLeft(xNode), getRight(xNode)};
        for (TreeNode57 y : choices) {
            if (y != null) {
                if (lc1145Helper(n, x, xNode, y)) return true;
            }
        }
        return false;
    }

    private boolean lc1145Helper(int n, int x, TreeNode57 rivalFirstChoice, TreeNode57 y) {
        Deque<TreeNode57> q = new LinkedList<>();
        // BFS, 统计邻接节点数量
        boolean[] visited = new boolean[n + 1];
        visited[x] = true;
        q.offer(y);
        int myCount = getTreeNodeCount(q, visited);

        Arrays.fill(visited, false);
        visited[y.val] = true;
        q.clear();
        q.offer(rivalFirstChoice);
        int rivalCount = getTreeNodeCount(q, visited);

        return rivalCount < myCount;
    }

    private int getTreeNodeCount(Deque<TreeNode57> q, boolean[] visited) {
        int count = 0;
        while (!q.isEmpty()) {
            TreeNode57 p = q.poll();
            if (visited[p.val]) continue;
            visited[p.val] = true;
            count++;
            TreeNode57 f = getFather(p), l = getLeft(p), r = getRight(p);
            if (f != null && !visited[f.val]) q.offer(f);
            if (l != null && !visited[l.val]) q.offer(l);
            if (r != null && !visited[r.val]) q.offer(r);
        }
        return count;
    }

    private TreeNode57 getFather(TreeNode57 root) {
        return fatherMap.get(root);
    }

    private TreeNode57 getLeft(TreeNode57 root) {
        return root.left;
    }

    private TreeNode57 getRight(TreeNode57 root) {
        return root.right;
    }


    // LC1983
    public int widestPairOfIndices(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int prefix1 = 0, prefix2 = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix1 += nums1[i];
            prefix2 += nums2[i];
            int diff = prefix1 - prefix2;
            if (map.containsKey(diff)) {
                result = Math.max(result, i - map.get(diff));
            } else {
                map.put(diff, i);
            }
        }
        return result;
    }

    // LC335 **
    public boolean isSelfCrossing(int[] distance) {
        for (int i = 3; i < distance.length; i++) {
            if (i >= 3
                    && distance[i] >= distance[i - 2]
                    && distance[i - 1] <= distance[i - 3])
                return true;
            else if (i >= 4
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 1] == distance[i - 3])
                return true;
            else if (i >= 5
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 5] + distance[i - 1] >= distance[i - 3]
                    && distance[i - 2] > distance[i - 4]
                    && distance[i - 3] > distance[i - 1])
                return true;
        }
        return false;
    }

    // LC869
    public boolean reorderedPowerOf2(int n) {
        if (n == 0) return false;
        List<Integer> power2List = new ArrayList<>(31);
        for (int i = 0; i < 31; i++) power2List.add(1 << i);
        int[][] freqList = new int[31][10];
        for (int i = 0; i < 31; i++) {
            int power2 = power2List.get(i);
            int[] freq = new int[10];
            while (power2 != 0) {
                freq[power2 % 10]++;
                power2 /= 10;
            }
            freqList[i] = freq;
        }
        int[] thisFreq = new int[10];
        int dummy = n;
        while (dummy != 0) {
            thisFreq[dummy % 10]++;
            dummy /= 10;
        }
        outer:
        for (int i = 0; i < 31; i++) {
            for (int j = 0; j < 10; j++) {
                if (freqList[i][j] != thisFreq[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    // JZOF II 086 LC131
    List<List<String>> lc131Result;
    List<String> lc131Tmp;

    public String[][] partition(String s) {
        lc131Result = new ArrayList<>();
        lc131Tmp = new ArrayList<>();
        int n = s.length();
        boolean[][] judge = new boolean[n][n];
        char[] ca = s.toCharArray();
        for (int i = 0; i < n; i++) judge[i][i] = true;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left + len - 1 < n; left++) {
                if (len == 2) {
                    judge[left][left + 1] = ca[left] == ca[left + 1];
                } else if (judge[left + 1][left + len - 1 - 1] && ca[left] == ca[left + len - 1]) {
                    judge[left][left + len - 1] = true;
                }
            }
        }
        lc131Helper(0, judge, s);
        String[][] resArr = new String[lc131Result.size()][];
        for (int i = 0; i < lc131Result.size(); i++) {
            resArr[i] = lc131Result.get(i).toArray(new String[lc131Result.get(i).size()]);
        }
        return resArr;
    }

    private void lc131Helper(int idx, boolean[][] judge, String s) {
        if (idx == judge.length) {
            lc131Result.add(new ArrayList<>(lc131Tmp));
            return;
        }
        for (int len = 1; idx + len - 1 < judge.length; len++) {
            if (judge[idx][idx + len - 1]) {
                lc131Tmp.add(s.substring(idx, idx + len));
                lc131Helper(idx + len, judge, s);
                lc131Tmp.remove(lc131Tmp.size() - 1);
            }
        }
    }


    // LC792 ** 桶思想
    public int numMatchingSubseqBucket(String s, String[] words) {
        int result = 0;
        Map<Character, List<List<Character>>> bucket = new HashMap<>();
        for (String w : words) {
            bucket.putIfAbsent(w.charAt(0), new LinkedList<>());
            List<Character> bucketItem = new LinkedList<>();
            for (char c : w.toCharArray()) bucketItem.add(c);
            bucket.get(w.charAt(0)).add(bucketItem);
        }
        for (char c : s.toCharArray()) {
            Set<Character> set = new HashSet<>(bucket.keySet());
            for (char key : set) {
                if (c != key) continue;
                List<List<Character>> items = bucket.get(key);
                ListIterator<List<Character>> it = items.listIterator();
                while (it.hasNext()) {
                    List<Character> seq = it.next();
                    it.remove();
                    seq.remove(0);
                    if (seq.size() == 0) result++;
                    else {
                        bucket.putIfAbsent(seq.get(0), new LinkedList<>());
                        if (seq.get(0) == key) {
                            it.add(seq);
                        } else {
                            bucket.get(seq.get(0)).add(seq);
                        }
                    }
                }
                if (bucket.get(key).size() == 0) bucket.remove(key);
            }
        }
        return result;
    }

    // LC1055 **
    public int shortestWay(String source, String target) {
        int tIdx = 0, result = 0;
        char[] cs = source.toCharArray(), ct = target.toCharArray();
        while (tIdx < ct.length) {
            int sIdx = 0;
            int pre = tIdx;
            while (tIdx < ct.length && sIdx < cs.length) {
                if (ct[tIdx] == cs[sIdx]) tIdx++;
                sIdx++;
            }
            if (tIdx == pre) return -1;
            result++;
        }
        return result;
    }


    // LC1689
    public int minPartitions(String n) {
        int max = 0;
        for (char c : n.toCharArray()) {
            max = Math.max(max, c - '0');
        }
        return max;
    }

    // LC1962
    public int minStoneSum(int[] piles, int k) {
        int[] freq = new int[10001];
        for (int i : piles) freq[i]++;
        int sum = 0;
        for (int i = 10000; i >= 0; i--) {
            if (freq[i] == 0) continue;
            if (k > 0) {
                int minusTime = Math.min(k, freq[i]);
                freq[i] -= minusTime;
                freq[i - i / 2] += minusTime;
                k -= minusTime;
            }
            sum += i * freq[i];
        }
        return sum;
    }

    // LC301 **
    Set<String> lc301Result = new HashSet<>();

    public List<String> removeInvalidParentheses(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        // 多余的左右括号个数, 注意右括号多余当且仅当左边左括号不够匹配的时候
        int leftToRemove = 0, rightToRemove = 0;
        for (char c : ca) {
            if (c == '(') leftToRemove++;
            else if (c == ')') {
                if (leftToRemove == 0) rightToRemove++;
                else leftToRemove--;
            }
        }
        lc301Helper(0, ca, leftToRemove, rightToRemove, 0, 0, new StringBuilder());
        return new ArrayList<>(lc301Result);
    }

    private void lc301Helper(int curIdx, char[] ca,
            /*待删的左括号数*/
                             int leftToRemove, int rightToRemove,
            /*已删的左括号数*/
                             int leftCount, int rightCount,
                             StringBuilder sb) {
        if (curIdx == ca.length) {
            if (leftToRemove == 0 && rightToRemove == 0) {
                lc301Result.add(sb.toString());
            }
            return;
        }

        char c = ca[curIdx];
        if (c == '(' && leftToRemove > 0) { // 无视当前左括号
            lc301Helper(curIdx + 1, ca, leftToRemove - 1, rightToRemove, leftCount, rightCount, sb);
        }
        if (c == ')' && rightToRemove > 0) { // 无视当前右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove - 1, leftCount, rightCount, sb);
        }

        sb.append(c);
        if (c != '(' && c != ')') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount, sb);
        } else if (c == '(') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount + 1, rightCount, sb);
        } else if (c == ')' && rightCount < leftCount) { // 只有当当前已选择的左括号比右括号多才在此步选右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount + 1, sb);
        }
        sb.deleteCharAt(sb.length() - 1);
    }

    // LC1957
    public String makeFancyString(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        int n = ca.length;
        for (int i = 0; i < n; ) {
            int curIdx = i;
            char cur = ca[i];
            while (i + 1 < n && ca[i + 1] == cur) i++;
            int count = Math.min(i - curIdx + 1, 2);
            for (int j = 0; j < count; j++) {
                sb.append(cur);
            }
            i++;
        }
        return sb.toString();
    }

    // LC1540
    public boolean canConvertString(String s, String t, int k) {
        if (s.length() != t.length()) return false;
        // 第i次操作(从1算) 可以将s种之前未被操作过的下标j(从1算)的char+i
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        List<Integer> shouldChangeIdx = new ArrayList<>();
        for (int i = 0; i < cs.length; i++) {
            if (cs[i] != ct[i]) shouldChangeIdx.add(i);
        }
        int[] minSteps = new int[shouldChangeIdx.size()];
        for (int i = 0; i < shouldChangeIdx.size(); i++) {
            char sc = cs[shouldChangeIdx.get(i)], tc = ct[shouldChangeIdx.get(i)];
            minSteps[i] = (tc - 'a' + 26 - (sc - 'a')) % 26;
        }
        int[] freq = new int[27];
        for (int i : minSteps) freq[i]++;
        int max = 0;
        for (int i = 1; i <= 26; i++) {
            max = Math.max(max, i + (freq[i] - 1) * 26);
        }
        return max <= k;
    }

    // LC266
    public boolean canPermutePalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int oddCount = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddCount++;
        return oddCount <= 1;
    }

    // LC409
    public int longestPalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int even = 0, oddMax = 0, odd = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 0) even += freq[i];
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddMax = Math.max(oddMax, freq[i]);
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) odd += freq[i] - 1;
        if (oddMax == 0) return even;
        return odd + even + 1;
    }

    // LC1266
    public int minTimeToVisitAllPoints(int[][] points) {
        int x = points[0][0], y = points[0][1];
        int result = 0;
        for (int i = 1; i < points.length; i++) {
            int nx = points[i][0], ny = points[i][1];
            int deltaX = Math.abs(nx - x), deltaY = Math.abs(ny - y);
            int slash = Math.min(deltaX, deltaY);
            int line = Math.max(deltaX, deltaY) - slash;
            result += line + slash;
            x = nx;
            y = ny;
        }
        return result;
    }

    // LC1416
    Integer[] lc1416Memo;

    public int numberOfArrays(String s, int k) {
        int n = s.length();
        lc1416Memo = new Integer[n + 1];
        return lc1416Helper(0, s, k);
    }

    private int lc1416Helper(int cur, String s, int k) {
        final long mod = 1000000007l;
        if (cur == s.length()) return 1;
        if (lc1416Memo[cur] != null) return lc1416Memo[cur];
        int len = 1;
        long result = 0;
        while (cur + len <= s.length()) {
            long num = Long.parseLong(s.substring(cur, cur + len));
            if (String.valueOf(num).length() != len) break;
            if (num > k) break;
            if (num < 1) break;
            result += lc1416Helper(cur + len, s, k);
            result %= mod;
            len++;
        }
        return lc1416Memo[cur] = (int) (result % mod);
    }

    // LC1844
    public String replaceDigits(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            if (i % 2 == 0) sb.append(ca[i]);
            else sb.append((char) (ca[i - 1] + (ca[i] - '0')));
        }
        return sb.toString();
    }
}

class TreeNode57 {
    int val;
    TreeNode57 left;
    TreeNode57 right;

    TreeNode57(int x) {
        val = x;
    }
}

class DSUArray57 {
    int[] father;
    int[] rank;
    int size;

    public DSUArray57(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray57() {
        this.size = 1 << 16;
        father = new int[1 << 16];
        rank = new int[1 << 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public boolean contains(int i) {
        if (i >= this.size || i < 0) return false;
        return father[i] != -1;
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (int i = 0; i < size; i++) {
            if (father[i] != -1) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
        }
        return result;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }

    public int getSelfGroupSize(int x) {
        return rank[find(x)];
    }

}

class ListNode57 {
    int val;
    ListNode57 next;

    ListNode57(int x) {
        val = x;
    }
}

// LC1320 Hard
class Lc1320 {
    static int[][] distance;
    static int[][] id;
    int min = Integer.MAX_VALUE;
    Integer[][] memo;
    char[] word;

    public int minimumDistance(String wordStr) {
        word = wordStr.toCharArray();
        memo = new Integer[word.length][31 * 31];
        // 遍历两个手指的初始位置
        for (int i = 'A'; i <= '^'; i++) {
            for (int j = i; j <= '^'; j++) {
                min = Math.min(min, helper(0, i, j));
            }
        }
        return min;
    }

    private int helper(int targetCharIdx, int curLeft, int curRight) {
        if (targetCharIdx == word.length) {
            return 0;
        }

        if (memo[targetCharIdx][id[curLeft][curRight]] != null)
            return memo[targetCharIdx][id[curLeft][curRight]];

        return memo[targetCharIdx][id[curLeft][curRight]] =
                Math.min(distance[curLeft][word[targetCharIdx]] + helper(targetCharIdx + 1, word[targetCharIdx], curRight),
                        distance[curRight][word[targetCharIdx]] + helper(targetCharIdx + 1, curLeft, word[targetCharIdx]));
    }

    static {
        int[][] alphabetIdx = new int[128][];
        distance = new int[128][128];
        id = new int[128][128];

        alphabetIdx['A'] = new int[]{0, 0};
        alphabetIdx['B'] = new int[]{0, 1};
        alphabetIdx['C'] = new int[]{0, 2};
        alphabetIdx['D'] = new int[]{0, 3};
        alphabetIdx['E'] = new int[]{0, 4};
        alphabetIdx['F'] = new int[]{0, 5};
        alphabetIdx['G'] = new int[]{1, 0};
        alphabetIdx['H'] = new int[]{1, 1};
        alphabetIdx['I'] = new int[]{1, 2};
        alphabetIdx['J'] = new int[]{1, 3};
        alphabetIdx['K'] = new int[]{1, 4};
        alphabetIdx['L'] = new int[]{1, 5};
        alphabetIdx['M'] = new int[]{2, 0};
        alphabetIdx['N'] = new int[]{2, 1};
        alphabetIdx['O'] = new int[]{2, 2};
        alphabetIdx['P'] = new int[]{2, 3};
        alphabetIdx['Q'] = new int[]{2, 4};
        alphabetIdx['R'] = new int[]{2, 5};
        alphabetIdx['S'] = new int[]{3, 0};
        alphabetIdx['T'] = new int[]{3, 1};
        alphabetIdx['U'] = new int[]{3, 2};
        alphabetIdx['V'] = new int[]{3, 3};
        alphabetIdx['W'] = new int[]{3, 4};
        alphabetIdx['X'] = new int[]{3, 5};
        alphabetIdx['Y'] = new int[]{4, 0};
        alphabetIdx['Z'] = new int[]{4, 1};
        alphabetIdx['['] = new int[]{4, 2};
        alphabetIdx['\\'] = new int[]{4, 3};
        alphabetIdx[']'] = new int[]{4, 4};
        alphabetIdx['^'] = new int[]{4, 5};

        for (int i = 'A'; i <= '^'; i++) {
            for (int j = i + 1; j <= '^'; j++) {
                distance[i][j] = distance[j][i] =
                        Math.abs(alphabetIdx[i][0] - alphabetIdx[j][0])
                                + Math.abs(alphabetIdx[i][1] - alphabetIdx[j][1]);
                id[i][j] = id[j][i] = 31 * (i - 'A') + j - 'A';
            }
        }
    }
}

// LC1724 Hard **
// From: https://leetcode-cn.com/problems/checking-existence-of-edge-length-limited-paths-ii/solution/javashuang-bai-gou-jian-bing-cha-ji-tong-c07a/
class DistanceLimitedPathsExist {
    int[][] dsu; // dsu[i][0] 存的是i的父亲(初始化为自身), dsu[i][1]存的是第一次连上两个分量的时间(时间戳)
    // 查询的时候走时间戳严格小于limit的并查集路径即可

    public DistanceLimitedPathsExist(int n, int[][] edgeList) {
        Arrays.sort(edgeList, Comparator.comparingInt(o -> o[2]));
        dsu = new int[n][2];
        for (int i = 0; i < n; i++) {
            dsu[i][0] = i;
            dsu[i][1] = 0;
        }
        for (int[] e : edgeList) {
            int a = e[0], b = e[1];
            while (dsu[a][0] != a) {
                a = dsu[a][0];
            }
            while (dsu[b][0] != b) {
                b = dsu[b][0];
            }
            if (a != b) {
                dsu[a][0] = b;
                dsu[a][1] = e[2];
            }
        }
    }

    public boolean query(int a, int b, int limit) {
        while (dsu[a][0] != a && dsu[a][1] < limit) {
            a = dsu[a][0];
        }
        while (dsu[b][0] != b && dsu[b][1] < limit) {
            b = dsu[b][0];
        }
        return a == b;
    }
}

// LC308
class NumMatrix {
    BIT57[] bitMtx;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        bitMtx = new BIT57[m];
        for (int i = 0; i < m; i++) {
            bitMtx[i] = new BIT57(n);
            for (int j = 0; j < n; j++) {
                bitMtx[i].update(j, matrix[i][j]);
            }
        }
    }

    public void update(int row, int col, int val) {
        bitMtx[row].set(col, val);
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        int result = 0;
        for (int i = row1; i <= row2; i++) {
            result += bitMtx[i].sumRange(col1, col2);
        }
        return result;
    }
}

class BIT57 {
    int[] tree;
    int len;

    public BIT57(int size) {
        this.len = size;
        this.tree = new int[size + 1];
    }

    public void set(int idx, int val) {
        int delta = val - get(idx);
        update(idx, delta);
    }

    public int get(int idx) {
        return sumRange(idx, idx);
    }


    public void update(int idx, int delta) {
        updateOneBased(idx + 1, delta);
    }

    public int sumRange(int start, int end) { // end inclusive
        return sum(end + 1) - sum(start);
    }

    private int sum(int idx) {
        int result = 0;
        while (idx > 0) {
            result += tree[idx];
            idx -= lowbit(idx);
        }
        return result;
    }

    private void updateOneBased(int idx, int delta) {
        while (idx <= len) {
            tree[idx] += delta;
            idx += lowbit(idx);
        }
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}