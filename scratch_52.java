import javafx.util.Pair;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.*;
import java.util.function.Function;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


//        System.out.println(s.splitArray(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 5));
//        System.out.println(s.containVirus(new int[][]{{0, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 0, 1, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}}));
        System.out.println(s.containVirus(new int[][]{{1}}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1221
    public int balancedStringSplit(String s) {
        char[] ca = s.toCharArray();
        int diff = 0, count = 0, ptr = 0;
        while (ptr != ca.length) {
            if (ca[ptr++] == 'L') diff++;
            else diff--;
            if (diff == 0) count++;
        }
        return count;
    }

    // LC749 The implementation is long.
    public int containVirus(int[][] isInfected) {
        int m = isInfected.length, n = isInfected[0].length;
        Function<int[], Boolean> checkLegalPos = new Function<int[], Boolean>() {
            @Override
            public Boolean apply(int[] pos) {
                return pos[0] >= 0 && pos[0] < m && pos[1] >= 0 && pos[1] < n;
            }
        };
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int totalCount = m * n, stillInfectCount = 0, wallCount = 0, isolatedCount = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                stillInfectCount += isInfected[i][j];
            }
        }

        while (stillInfectCount != 0) {
            DisjointSetUnion dsu = new DisjointSetUnion();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int idx = i * n + j;
                    if (isInfected[i][j] == 1) {
                        dsu.add(idx);
                        for (int[] dir : directions) {
                            int nr = i + dir[0], nc = j + dir[1];
                            int nIdx = nr * n + nc;
                            if (checkLegalPos.apply(new int[]{nr, nc}) && isInfected[nr][nc] == 1) {
                                dsu.add(nIdx);
                                dsu.merge(idx, nIdx);
                            }
                        }
                    }
                }
            }
            Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
            Set<Integer> victim = null;
            int maxCountInfect = -1;
            for (Set<Integer> virus : allGroups.values()) {
                boolean[][] visited = new boolean[m][n];
                int countInfect = 0;
                for (int idx : virus) {
                    int r = idx / n, c = idx % n;
                    for (int[] dir : directions) {
                        int nr = r + dir[0], nc = c + dir[1];
                        if (checkLegalPos.apply(new int[]{nr, nc})) {
                            if (!visited[nr][nc] && isInfected[nr][nc] == 0) {
                                visited[nr][nc] = true;
                                countInfect++;
                            }
                        }
                    }
                }
                if (countInfect > maxCountInfect) {
                    maxCountInfect = countInfect;
                    victim = virus;
                }
            }
            isolatedCount += victim.size();
            int wallCountDiff = 0;
            boolean[][] wallVisited = new boolean[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (!wallVisited[i][j] && isInfected[i][j] == 0) {
                        for (int[] dir : directions) {
                            int nr = i + dir[0], nc = j + dir[1];
                            int nIdx = nr * n + nc;
                            if (checkLegalPos.apply(new int[]{nr, nc}) && victim.contains(nIdx)) {
                                wallCountDiff++;
                            }
                        }
                        wallVisited[i][j] = true;
                    }
                }
            }
            wallCount += wallCountDiff;
            for (int idx : victim) {
                int r = idx / n, c = idx % n;
                isInfected[r][c] = 2; // 用2表示已被隔离
            }
            for (Set<Integer> virus : allGroups.values()) {
                if (virus != victim) {
                    for (int idx : virus) {
                        int r = idx / n, c = idx % n;
                        for (int[] dir : directions) {
                            int nr = r + dir[0], nc = c + dir[1];
                            if (checkLegalPos.apply(new int[]{nr, nc})) {
                                if (isInfected[nr][nc] == 0) {
                                    isInfected[nr][nc] = 1;
                                }
                            }
                        }
                    }
                }
            }
            stillInfectCount = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (isInfected[i][j] == 1) stillInfectCount++;
                }
            }
            if (stillInfectCount == totalCount - isolatedCount) return wallCount;
        }
        return wallCount;
    }

    // LC501
    Map<Integer, Integer> lc501Freq;

    public int[] findMode(TreeNode root) {
        lc501Freq = new HashMap<>();
        lc501Dfs(root);
        List<Integer> k = new ArrayList<>(lc501Freq.keySet());
        k.sort(Comparator.comparingInt(o -> -lc501Freq.get(o)));
        int maxFreq = lc501Freq.get(k.get(0));
        int end = 0;
        for (int i : k) {
            if (lc501Freq.get(i) != maxFreq) break;
            end++;
        }
        return k.subList(0, end).stream().mapToInt(Integer::valueOf).toArray();
    }

    private void lc501Dfs(TreeNode root) {
        if (root == null) return;
        lc501Freq.put(root.val, lc501Freq.getOrDefault(root.val, 0) + 1);
        lc501Dfs(root.left);
        lc501Dfs(root.right);
    }

    // LCP 12 TBD 参考 LC410
    public int minTime(int[] time, int m) {
        // m 天 完成 time.length 题, 按顺序做题, 每天耗时最长的一题可以不计入耗时, 求最长的一天的耗时
        int lo = 0, hi = Arrays.stream(time).sum();
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (lcp12Helper(time, mid) <= m) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int lcp12Helper(int[] nums, int segLen) {
        int count = 1, sum = 0, curMax = 0; // 多维护一个当前最大值, 判断是否大于segLen的时候先减去当前最大值
        for (int i : nums) {
            if (sum + i - Math.max(i, curMax) > segLen) {
                sum = i;
                curMax = i;
                count++;
            } else {
                sum += i;
                curMax = Math.max(curMax, i);
            }
        }
        return count;
    }

    // LC410 二分
    public int splitArrayBS(int[] nums, int m) {
        int n = nums.length;
        int lo = Arrays.stream(nums).max().getAsInt();
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        int hi = prefix[n];
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (lc410CountSeg(prefix, mid) <= m) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int lc410CountSeg(int[] prefix, int segLen) {
        int count = 0, ptr = 0;
        int len = prefix.length - 1;
        while (ptr < len) {
            int lo = ptr, hi = len;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                if (prefix[mid] <= prefix[ptr] + segLen) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            count++;
            ptr = lo;
        }
        return count;
    }

    // LC410 Hard Minmax, 极小化极大, DFS
    Integer[][] lc410Memo;

    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        lc410Memo = new Integer[n + 1][m + 1];
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        return lc410Helper(prefix, 0, m);
    }

    private int lc410Helper(int[] prefix, int begin, int leftSegNum) {
        if (leftSegNum == 1) {
            return prefix[prefix.length - 1] - prefix[begin];
        }

        if (lc410Memo[begin][leftSegNum] != null) return lc410Memo[begin][leftSegNum];
        int result = 0x3f3f3f3f;
        int len = prefix.length - 1;

        // 理想值: 即平均分配所有划分的和, 实际的最小值总是大于等于理想值, 所以可以找到小于等于理想值的第一个下标开始枚举
        int ideal = (prefix[len + 1] - prefix[begin]) / leftSegNum;
        int lo = begin, hi = len;

        // 找到小于等于ideal的第一个下标
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (prefix[mid] <= prefix[begin] + ideal) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        // 枚举下一子数组的开始下标i
        // 初始化 i = begin+1 是因为子数组的长度最少为1
        // i<= nums.length-(segNum-1): 保证剩下的子数组至少分配到有1个数
        // for (int i = begin+1; i <= len - (leftSegNum - 1); i++) {
        for (int i = lo; i <= len - (leftSegNum - 1); i++) {
            if (lo == 0) continue;
            int sum = prefix[i] - prefix[begin];
            int maxSum = Math.max(sum, lc410Helper(prefix, i, leftSegNum - 1));
            result = Math.min(result, maxSum);
        }

        return lc410Memo[begin][leftSegNum] = result;
    }

    // Interview 01.02
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int[] freq = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            freq[s1.charAt(i) - 'a']++;
            freq[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) if (freq[i] != 0) return false;
        return true;
    }

    // LC1263 Hard
    public int minPushBox(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> q = new LinkedList<>();
        boolean[][][][] visited = new boolean[m][n][m][n];
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int[] box = new int[]{-1, -1}, target = new int[]{-1, -1}, self = new int[]{-1, -1};
        Function<int[], Boolean> checkLegalPos = new Function<int[], Boolean>() {
            @Override
            public Boolean apply(int[] pos) {
                return pos[0] >= 0 && pos[0] < m && pos[1] >= 0 && pos[1] < n && grid[pos[0]][pos[1]] != '#';
            }
        };
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                switch (grid[i][j]) {
                    case 'S':
                        self = new int[]{i, j};
                        break;
                    case 'T':
                        target = new int[]{i, j};
                        break;
                    case 'B':
                        box = new int[]{i, j};
                    default:
                        continue;
                }
            }
        }
        // [selfRow, selfCol, boxRow, boxCol]
        int[] initState = new int[]{self[0], self[1], box[0], box[1]};
        q.offer(initState);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]][p[2]][p[3]]) continue;
                visited[p[0]][p[1]][p[2]][p[3]] = true;
                int selfRow = p[0], selfCol = p[1];
                int boxRow = p[2], boxCol = p[3];
                if (boxRow == target[0] && boxCol == target[1]) return layer;

                // 去到箱子的旁边推箱子
                // 1. 确定箱子四周是否是障碍物, 找出立足点和相对方向上的目标点
                // 2. 确定是否有路径到这些立足点, 此时应把箱子本身也视作障碍物
                // 3. 若有路到立足点, 向队列推入[箱子位置, 新箱子位置](玩家到了箱子的位置)

                // 立足点和箱子目标位置的delta_row,delta_col可以通过简单取负数得到(因为两个delta必有一个是0)
                List<Pair<int[], int[]>> legalStandPointTargetPosList = new ArrayList<>();
                for (int[] dir : directions) {
                    int[] standPoint = new int[]{boxRow + dir[0], boxCol + dir[1]};
                    int[] targetPos = new int[]{boxRow - dir[0], boxCol - dir[1]};
                    if (checkLegalPos.apply(standPoint) && checkLegalPos.apply(targetPos)) {
                        legalStandPointTargetPosList.add(new Pair<>(standPoint, targetPos));
                    }
                }

                for (Pair<int[], int[]> pair : legalStandPointTargetPosList) {
                    boolean[][] innerVisited = new boolean[m][n];
                    int[] innerTarget = pair.getKey();
                    int[] innerStartPoint = new int[]{selfRow, selfCol};
                    Deque<int[]> innerQ = new LinkedList<>();
                    boolean canReach = false;
                    // 这里可以用offer/poll 做bfs, 也可以用push/pop 做dfs
                    innerQ.offer(innerStartPoint);
                    while (!innerQ.isEmpty()) {
                        int[] innerP = innerQ.poll();
                        if (innerP[0] == innerTarget[0] && innerP[1] == innerTarget[1]) {
                            canReach = true;
                            break;
                        }
                        if (innerVisited[innerP[0]][innerP[1]]) continue;
                        innerVisited[innerP[0]][innerP[1]] = true;
                        for (int[] dir : directions) {
                            int nextRow = innerP[0] + dir[0], nextCol = innerP[1] + dir[1];
                            int[] next = new int[]{nextRow, nextCol};
                            if (checkLegalPos.apply(next) && !(nextRow == boxRow && nextCol == boxCol) && !innerVisited[nextRow][nextCol]) {
                                innerQ.offer(next);
                            }
                        }
                    }
                    if (canReach) {
                        // 若推得动, 则此时玩家位置变为原箱子位置, 箱子位置变为targetPos(即pair.getValue())
                        q.offer(new int[]{boxRow, boxCol, pair.getValue()[0], pair.getValue()[1]});
                    }
                }
            }
        }
        return -1;
    }

    // LCP10 Hard **
    public double minimalExecTime(TreeNode root) {
        double[] result = betterDfs(root);
        return result[1];
    }

    // 返回[任务总时间, 最短执行时间]
    private double[] betterDfs(TreeNode root) {
        if (root == null) return new double[]{0, 0};
        double[] left = betterDfs(root.left);
        double[] right = betterDfs(root.right);
        return new double[]{
                left[0] + right[0] + root.val,
                root.val + Math.max(Math.max(left[1], right[1]), (left[0] + right[0]) / 2d)
        };
    }

    // LC1103
    public int[] distributeCandies(int candies, int num_people) {
        int[] result = new int[num_people];
        int ptr = 0;
        while (candies != 0) {
            if (candies >= ptr + 1) {
                result[ptr % num_people] += ptr + 1;
                candies -= ptr + 1;
                ptr++;
            } else {
                result[ptr % num_people] += candies;
                break;
            }
        }
        return result;
    }

    // LC1719 Hard **

    // DFS
    Map<Integer, Set<Integer>> lc1719DfsMap;

    public int checkWaysDfs(int[][] pairs) {
        lc1719DfsMap = new HashMap<>();
        for (int[] p : pairs) {
            lc1719DfsMap.putIfAbsent(p[0], new HashSet<>());
            lc1719DfsMap.putIfAbsent(p[1], new HashSet<>());
            lc1719DfsMap.get(p[0]).add(p[1]);
            lc1719DfsMap.get(p[1]).add(p[0]);
        }
        int numEle = lc1719DfsMap.size();
        List<Integer> validNodes = new ArrayList<>(lc1719DfsMap.keySet());
        validNodes.sort(Comparator.comparingInt(o -> -lc1719DfsMap.get(o).size()));
        if (lc1719DfsMap.get(validNodes.get(0)).size() != numEle - 1) return 0;
        int rootNode = validNodes.get(0);
        return lc1719Dfs(rootNode);
    }

    private int lc1719Dfs(int root) {
        Set<Integer> subNode = new HashSet<>(lc1719DfsMap.get(root)); // root的children, 记subnode
        lc1719DfsMap.get(root).clear();
        for (int c : subNode) {
            lc1719DfsMap.get(c).remove(root);
        }
        boolean multi = false;
        List<Integer> subNodeList = new ArrayList<>(subNode);
        subNodeList.sort(Comparator.comparingInt(o -> -lc1719DfsMap.get(o).size()));
        for (int c : subNodeList) {
            // subNode 的 children
            for (int snc : lc1719DfsMap.get(c)) {
                if (!subNode.contains(snc)) return 0;
            }
            if (lc1719DfsMap.get(c).size() == subNode.size() - 1) { // -1是因为之前remove了root
                multi = true;
            }
            int result = lc1719Dfs(c);
            if (result == 0) return 0;
            if (result == 2) multi = true;
        }
        return multi ? 2 : 1;
    }

    public int checkWays(int[][] pairs) {
        final int maxSize = 501;
        int result = 1;
        int[] parent = new int[maxSize];
        Map<Integer, List<Integer>> mtx = new HashMap<>();
        for (int[] p : pairs) {
            mtx.putIfAbsent(p[0], new ArrayList<>());
            mtx.putIfAbsent(p[1], new ArrayList<>());
            mtx.get(p[0]).add(p[1]);
            mtx.get(p[1]).add(p[0]);
            parent[p[0]] = parent[p[1]] = -1;
        }
        int numEle = mtx.size();
        List<Integer> validEle = new ArrayList<>(mtx.keySet());
        validEle.sort(Comparator.comparingInt(o -> -mtx.get(o).size()));
        if (mtx.get(validEle.get(0)).size() != numEle - 1) return 0;
        boolean[] visited = new boolean[maxSize];
        for (int u : validEle) { // 按照关系数倒序排序, 按此顺序遍历后遍历到的只能是前面的子节点
            for (int v : mtx.get(u)) {
                if (mtx.get(u).size() == mtx.get(v).size()) result = 2;
                if (!visited[v]) {
                    // parent[v] 是 当前已知的v最近的父节点, 如果p[u]!=p[v],
                    // 说明v有比u更近的父节点(否则在p[v]未更新前, p[v]应该和p[u]拥有相同的最近父节点
                    // p[v]!=p[u] 说明v在不从属于u的一支中已被更新, 而u又是v一个可能的父节点, 说明v会存在两个入度, 此时无解
                    if (parent[u] != parent[v]) return 0;
                    parent[v] = u;
                }
            }
            visited[u] = true; // 将确定了所有可能孩子的u加入已访问
        }
        return result;
    }

    // LC798 Hard ** 差分数组 学习分析方法
    // https://leetcode-cn.com/problems/smallest-rotation-with-highest-score/solution/chai-fen-shu-zu-by-sssz-qdut/
    public int bestRotation(int[] nums) {
        // 数组向左平移轮转
        int n = nums.length;
        int[] diff = new int[n + 1];
        // [0,Math.min(i,i-arr[i])], [i+1,Math.min(i-arr[i]+arr.len),arr.len-1)] 为可以得分的范围
        for (int i = 0; i < n; i++) {
            diff[0]++;
            int r1 = Math.min(i, i - nums[i]) + 1;
            if (r1 >= 0)
                diff[r1]--;
            diff[i + 1]++;
            int r2 = Math.min(i - nums[i] + n, n - 1) + 1;
            if (r2 >= 0 && r2 <= n)
                diff[r2]--;
        }
        int accumulate = 0, max = 0, maxIdx = -1;
        for (int i = 0; i < n; i++) {
            accumulate += diff[i];
            if (accumulate > max) {
                max = accumulate;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Interview 17.14
    public int[] smallestK(int[] arr, int k) {
        quickSelect.topK(arr, arr.length - k);
        return Arrays.copyOfRange(arr, 0, k);
    }

    // Interview 08.08 全排列
    List<String> iv0808Result;

    public String[] permutation(String S) {
        iv0808Result = new ArrayList<>();
        iv0808Dfs(S.toCharArray(), 0);
        return iv0808Result.toArray(new String[iv0808Result.size()]);
    }

    private void iv0808Dfs(char[] ca, int cur) {
        if (cur == ca.length) {
            iv0808Result.add(new String(ca));
        }
        Set<Character> set = new HashSet<>();
        for (int i = cur; i < ca.length; i++) {
            if (!set.contains(ca[i])) {
                set.add(ca[i]);
                char tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
                iv0808Dfs(ca, cur + 1);
                tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
            }
        }
    }

    // LC905
    public int[] sortArrayByParity(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] % 2 == 1) {
                int tmp = nums[right];
                nums[right] = nums[left];
                nums[left] = tmp;
                left--;
                right--;
            }
            left++;
        }
        return nums;
    }

    // JZOF II 056 LC653
    List<Integer> lc653List = new ArrayList<>();

    public boolean findTarget(TreeNode root, int k) {
        lc653Inorder(root);
        int left = 0, right = lc653List.size() - 1;
        while (left < right) {
            if (lc653List.get(left) + lc653List.get(right) > k) {
                right--;
            } else if (lc653List.get(left) + lc653List.get(right) < k) {
                left++;
            } else {
                return true;
            }
        }
        return false;
    }

    private void lc653Inorder(TreeNode root) {
        if (root == null) return;
        lc653Inorder(root.left);
        lc653List.add(root.val);
        lc653Inorder(root.right);
    }

    // LC1611 Hard ** Reverse Gray Code
    public int minimumOneBitOperations(int n) {
        // Function<Integer, Integer> gray = orig -> orig ^ (orig >> 1);
        int orig = 0;
        while (n != 0) {
            orig ^= n;
            n >>= 1;
        }
        return orig;
    }

    // LC468
    public String validIPAddress(String IP) {
        final String NEITHER = "Neither", IPV4 = "IPv4", IPV6 = "IPv6";
        if (checkIpv4(IP)) return IPV4;
        if (checkIpv6(IP)) return IPV6;
        return NEITHER;
    }

    private boolean checkIpv4(String ip) {
        String[] groups = ip.split("\\.", -1);
        if (groups.length != 4) return false;
        for (String w : groups) {
            for (char c : w.toCharArray()) if (!Character.isDigit(c)) return false;
            if (w.length() > 3 || w.length() == 0) return false;
            if (w.charAt(0) == '0' && w.length() != 1) return false;
            if (Integer.valueOf(w) > 255 || Integer.valueOf(w) < 0) return false;
        }
        return true;
    }

    private boolean checkIpv6(String ip) {
        String[] groupsByTwoColon = ip.split("::", -1);
        if (groupsByTwoColon.length > 2) return false; // 注意LC题目这里要改成>1, 因为题目不允许双冒号表示
        String[] groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length > 8) return false;
        for (String w : groupsByOneColon) {
            if (w.length() > 4) return false;
            String wlc = w.toLowerCase();
            for (char c : wlc.toCharArray()) {
                if (!Character.isLetter(c) && !Character.isDigit(c)) return false;
                if (Character.isLetter(c) && c > 'f') return false;
            }
        }
        int numColon = groupsByOneColon.length - 1;
        if (numColon != 7) {
            StringBuilder sb = new StringBuilder();
            int remain = 7 - numColon + 1;
            for (int i = 0; i < remain; i++) {
                sb.append(":0");
            }
            sb.append(":");
            ip = ip.replaceFirst("::", sb.toString());
        }
        groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length != 8) return false;
        long ipv6Left = 0, ipv6Right = 0;
        for (int i = 0; i < 8; i++) {
            String w = groupsByOneColon[i];
            int part = 0;
            if (w.equals("")) {
                if (i == 7) return false;
            } else {
                part = Integer.parseInt(w, 16);
            }
            if (i < 4) {
                ipv6Left = (ipv6Left << 16) | part;
            } else {
                ipv6Right = (ipv6Right << 16) | part;
            }
        }
        return true;
    }

    // LC751 **
    public List<String> ipToCIDR(String ip, int n) {
        final int INT_MASK = 0xffffffff;
        long start = ipStrToInt(ip);
        long end = start + n - 1;
        List<String> result = new ArrayList<>();
        while (n > 0) {
            int numTrailingZero = Long.numberOfTrailingZeros(start);
            int mask = 0, bitsInCidr = 1;
            while (bitsInCidr < n && mask < numTrailingZero) {
                bitsInCidr <<= 1;
                mask++;
            }
            if (bitsInCidr > n) {
                bitsInCidr >>= 1;
                mask--;
            }
            result.add(longToIpStr(start) + "/" + (32 - mask));
            start += bitsInCidr;
            n -= bitsInCidr;
        }
        return result;
    }

    private int bitLength(long x) {
        if (x == 0) return 1;
        return Long.SIZE - Long.numberOfLeadingZeros(x);
    }

    private String longToIpStr(long ipLong) {
        return String.format("%d.%d.%d.%d", (ipLong >> 24) & 0xff, (ipLong >> 16) & 0xff, (ipLong >> 8) & 0xff, ipLong & 0xff);
    }

    private long ipStrToInt(String ip) {
        String[] split = ip.split("\\.");
        long result = 0;
        for (String s : split) {
            int i = Integer.valueOf(s);
            result = (result << 8) | i;
        }
        return result;
    }


    // LC356
    public boolean isReflected(int[][] points) {
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        for (int[] p : points) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
        }
        int midXTimes2 = (minX + maxX);
        Set<Pair<Integer, Integer>> all = new HashSet<>();
        for (int[] p : points) {
            all.add(new Pair<>(p[0], p[1]));
        }
        Set<Pair<Integer, Integer>> s = new HashSet<>();
        for (Pair<Integer, Integer> p : all) {
            if ((double) p.getKey() == ((double) midXTimes2 / 2)) continue;
            if (!s.remove(new Pair<>(p.getKey(), p.getValue()))) {
                s.add(new Pair<>(midXTimes2 - p.getKey(), p.getValue()));
            }
        }
        return s.size() == 0;
    }

    // LC336 **
    public List<List<Integer>> palindromePairs(String[] words) {
        Set<Pair<Integer, Integer>> result = new HashSet<>();
        int wLen = words.length;
        String[] rWords = new String[wLen];
        Map<String, Integer> rWordIdx = new HashMap<>();
        for (int i = 0; i < wLen; i++) {
            rWords[i] = new StringBuilder(words[i]).reverse().toString();
            rWordIdx.put(rWords[i], i);
        }
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            int len = cur.length();
            if (len == 0) continue;
            for (int j = 0; j <= len; j++) { // 注意边界, 为了取到空串, 截取长度可以去到len, 同时为了去重用到Set<Pair<>>
                if (checkPal(cur, j, len)) {
                    int leftId = rWordIdx.getOrDefault(cur.substring(0, j), -1);
                    if (leftId != -1 && leftId != i) {
                        result.add(new Pair<>(i, leftId));
                    }
                }
                if (checkPal(cur, 0, j)) {
                    int rightId = rWordIdx.getOrDefault(cur.substring(j), -1);
                    if (rightId != -1 && rightId != i) {
                        result.add(new Pair<>(rightId, i));
                    }
                }
            }
        }
        List<List<Integer>> listResult = new ArrayList<>(result.size());
        for (Pair<Integer, Integer> p : result) {
            listResult.add(Arrays.asList(p.getKey(), p.getValue()));
        }
        return listResult;
    }

    private boolean checkPal(String s, int startIdx, int endIdxExclude) {
        if (startIdx > endIdxExclude) return false;
        if (startIdx == endIdxExclude) return true;
        int len = endIdxExclude - startIdx;
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(startIdx + i) != s.charAt(endIdxExclude - 1 - i)) return false;
        }
        return true;
    }

    // LC747
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int[] idxMap = new int[101];
        for (int i = 0; i < nums.length; i++) {
            idxMap[nums[i]] = i;
        }
        Arrays.sort(nums);
        if (nums[nums.length - 1] >= nums[nums.length - 2] * 2) return idxMap[nums[nums.length - 1]];
        return -1;
    }

    // LC1224
    public int maxEqualFreq(int[] nums) {
        Map<Integer, Integer> numFreqMap = new HashMap<>();
        for (int i : nums) {
            numFreqMap.put(i, numFreqMap.getOrDefault(i, 0) + 1);
        }
        Map<Integer, Set<Integer>> freqIntSetMap = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : numFreqMap.entrySet()) {
            freqIntSetMap.putIfAbsent(e.getValue(), new HashSet<>());
            freqIntSetMap.get(e.getValue()).add(e.getKey());
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            // 当前元素
            int curEle = nums[i];

            // 看该不该删除

            // 情况1: freqMap.keySet.size > 2 此时删除哪个都没用
            if (freqIntSetMap.keySet().size() > 2) {
                continue;
            } else if (freqIntSetMap.keySet().size() == 2) {
                // 情况2: size == 2 时候, 看哪个set 的size ==1
                Iterator<Integer> it = freqIntSetMap.keySet().iterator();
                int freq1 = it.next(), freq2 = it.next();
                int smallFreq = freq1 < freq2 ? freq1 : freq2;
                int largeFreq = smallFreq == freq1 ? freq2 : freq1;
                Set<Integer> smallFreqSet = freqIntSetMap.get(smallFreq), largeFreqSet = freqIntSetMap.get(largeFreq);
                // 如果两个set都有超过一个元素, 则删除哪个元素都没用
                if (smallFreqSet.size() != 1 && largeFreqSet.size() != 1) {
                    continue;
                } else {
                    Set<Integer> oneEleSet = smallFreqSet.size() == 1 ? smallFreqSet : largeFreqSet;
                    Set<Integer> anotherSet = oneEleSet == smallFreqSet ? largeFreqSet : smallFreqSet;

                    int oneEle = oneEleSet.iterator().next();
                    int eleFreq = numFreqMap.get(oneEle);
                    int anotherFreq = eleFreq == smallFreq ? largeFreq : smallFreq;

                    // 情况1： 这个元素的当前频率是1
                    if (eleFreq == 1) return i + 1;
                        // 情况2: 当前元素的频率比另一个频率大1
                    else if (eleFreq == anotherFreq + 1) return i + 1;
                        // 特判一下 111 22 这种情况, 即两个freq的set的大小都是1
                        // 前面只判断了2不能删除, 没有判断1能不能删除, 此处补充判断一次
                    else if (anotherSet.size() == 1) {
                        if (anotherFreq == 1) return i + 1;
                        else if (anotherFreq == eleFreq + 1) return i + 1;
                    }
                    // 否则没办法 只能删除当前元素
                }
            }

            // 若没有找到该删除的 就删除当前元素
            int curFreq = numFreqMap.get(curEle);
            int nextFreq = curFreq - 1;
            numFreqMap.put(curEle, nextFreq);
            freqIntSetMap.get(curFreq).remove(nums[i]);
            if (freqIntSetMap.get(curFreq).size() == 0) freqIntSetMap.remove(curFreq);
            if (nextFreq != 0) {
                freqIntSetMap.putIfAbsent(nextFreq, new HashSet<>());
                freqIntSetMap.get(nextFreq).add(nums[i]);
            } else {
                numFreqMap.remove(nums[i]);
            }

        }
        return nums.length;
    }

    // JZOF 22
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // LC1705
    public int eatenApples(int[] apples, int[] days) {
        // pq 存数对 [i,j], i表示苹果数量, j表示过期时间
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        int n = apples.length;
        int result = 0;
        int i = 0;
        do {
            if (i < n) {
                if (apples[i] == 0 && days[i] == 0) {
                    ;
                } else if (apples[i] != 0) {
                    pq.offer(new int[]{apples[i], days[i] + i});
                }
            }
            if (!pq.isEmpty()) {
                int[] entry = null;
                do {
                    int[] p = pq.poll();
                    if (i >= p[1]) continue;
                    entry = p;
                    break;
                } while (!pq.isEmpty());
                if (entry != null) {
                    entry[0]--;
                    result++;
                    if (entry[0] > 0) pq.offer(entry);
                }
            }
            i++;
        } while (!pq.isEmpty() || i < n);
        return result;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Trie {
    Trie[] children = new Trie[26];
    boolean isEnd = false;

    public void addWord(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new Trie();
            }
            cur = cur.children[c - 'a'];
        }
        cur.isEnd = true;
    }

    public boolean startsWith(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return true;
    }

    public boolean search(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return cur.isEnd;
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

class quickSelect {
    static Random r = new Random();

    public static int topK(int[] arr, int topK) {
        return helper(arr, 0, arr.length - 1, topK);
    }

    private static Integer helper(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }
        int pivotVal = arr[start];
        int left = start, right = end;
        while (left < right) {
            while (left < right && arr[right] > pivotVal) {
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivotVal) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivotVal;
        if (left == arr.length - topK) return arr[left];
        Integer leftResult = helper(arr, start, left - 1, topK);
        if (leftResult != null) return leftResult;
        Integer rightResult = helper(arr, right + 1, end, topK);
        if (rightResult != null) return rightResult;
        return null;
    }
}

// LC635 TBD
class LogSystem {
    final SimpleDateFormat sdf = new SimpleDateFormat("yyyy:MM:dd:HH:mm:ss");
    final TreeSet<Integer> MAGIC_NUMBER = new TreeSet<Integer>() {{
        add(Calendar.YEAR);
        add(Calendar.MONTH);
        add(Calendar.DAY_OF_MONTH);
        add(Calendar.HOUR_OF_DAY);
        add(Calendar.MINUTE);
        add(Calendar.SECOND);
    }};

    TreeMap<Long, Integer> tm = new TreeMap<>();

    public void put(int id, String timestamp) {
        try {
            long ts = sdf.parse(timestamp).getTime();
            tm.put(ts, id);
        } catch (Exception e) {

        }
    }

    public List<Integer> retrieve(String start, String end, String granularity) {
        try {
            long sts = granHelper(start, granularity, false), ets = granHelper(end, granularity, true);
            List<Integer> result = new ArrayList<>();
            for (Map.Entry<Long, Integer> e : tm.subMap(sts, true, ets, false).entrySet()) {
                result.add(e.getValue());
            }
            return result;
        } catch (Exception e) {
            return null;
        }
    }

    private long granHelper(String timestamp, String gran, boolean isRight) throws ParseException {
        Date ts = sdf.parse(timestamp);
        int granMagic = granStrToMagic(gran);
        Calendar cal = Calendar.getInstance();
        cal.setTime(ts);
        for (int mn : MAGIC_NUMBER.tailSet(granMagic, false)) {
            if (mn <= Calendar.DAY_OF_MONTH) {
                if (mn == Calendar.MONTH) {
                    cal.set(mn, 0);
                } else {
                    cal.set(mn, 1);
                }
            } else {
                cal.set(mn, 0);
            }
        }
        if (isRight) {
            cal.add(granMagic, 1);
        }
        return cal.getTimeInMillis();
    }

    private int granStrToMagic(String gran) {
        switch (gran) {
            case "Year":
                return Calendar.YEAR;
            case "Month":
                return Calendar.MONTH;
            case "Day":
                return Calendar.DAY_OF_MONTH;
            case "Hour":
                return Calendar.HOUR_OF_DAY;
            case "Minute":
                return Calendar.MINUTE;
            case "Second":
                return Calendar.SECOND;
        }
        return -1;
    }

}

class DisjointSetUnion {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(int i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public int find(int i) {
        //先找到根 再压缩
        int root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            int origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (Integer i : father.keySet()) {
            int f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<Integer> s = new HashSet<Integer>();
        for (Integer i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}