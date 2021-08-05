import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.maxResultBottomUp(new int[]{100, -100, -300, -300, -300, -100, 100}, 4));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC940 **
    public int distinctSubseqII(String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        int[] dp = new int[n + 1];
        dp[0] = 1; // 空串
        int[] lastOccur = new int[26];
        Arrays.fill(lastOccur, -1);
        final int mod = 1000000007;
        for (int i = 0; i < n; i++) {
            dp[i + 1] = dp[i] * 2 % mod;
            if (lastOccur[ca[i] - 'a'] != -1) {
                dp[i + 1] -= dp[lastOccur[ca[i] - 'a']];
            }
            dp[i + 1] %= mod;
            lastOccur[ca[i] - 'a'] = i;
        }
        dp[n] = (dp[n] - 1 + mod) % mod; // -1 处理空串
        return dp[n];
    }

    // LC1696 单纯DP不行 求max是O(n), 加起来O(n^2)超时, 用TreeMap求max是O(log(n)), 总复杂度O(nlogn)
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[n - 1] = nums[n - 1];
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        tm.put(dp[n - 1], 1);
        int tmCounter = 1;
        for (int i = n - 2; i >= 0; i--) {
            int gain = tm.lastKey();
            if (tmCounter == k) {
                tm.put(dp[i + k], tm.get(dp[i + k]) - 1);
                if (tm.get(dp[i + k]) == 0) tm.remove(dp[i + k]);
                tmCounter--;
            }
            dp[i] = gain + nums[i];
            tm.put(dp[i], tm.getOrDefault(dp[i], 0) + 1);
            tmCounter++;
        }
        return dp[0];
    }

    // LC1696 TLE
    public int maxResultBottomUp(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            int gain = Integer.MIN_VALUE;
            for (int j = i + 1; j <= Math.min(n - 1, i + k); j++) {
                gain = Math.max(gain, dp[j]);
            }
            dp[i] = gain + nums[i];
        }
        return dp[0];
    }

    // LC1696 TLE
    Integer[] lc1696Memo;

    public int maxResultTopDown(int[] nums, int k) {
        lc1696Memo = new Integer[nums.length];
        return lc1696Helper(0, nums, k);
    }

    private int lc1696Helper(int cur, int[] nums, int k) {
        if (cur == nums.length - 1) return nums[nums.length - 1];
        if (lc1696Memo[cur] != null) return lc1696Memo[cur];
        int gain = Integer.MIN_VALUE;
        for (int i = cur + 1; i <= Math.min(cur + k, nums.length - 1); i++) {
            gain = Math.max(gain, lc1696Helper(i, nums, k));
        }
        return lc1696Memo[cur] = nums[cur] + gain;
    }

    // LC1780
    public boolean checkPowersOfThree(int n) {
        while (n != 0) {
            if (n % 3 == 2) return false;
            n /= 3;
        }
        return true;
    }

    // LC1523
    public int countOdds(int low, int high) {
        if (low % 2 == 1) low--;
        if (high % 2 == 1) high++;
        return (high - low) / 2;
    }

    // LC1782 ***
    public int[] countPairs(int n, int[][] edges, int[] queries) {
        int[] result = new int[queries.length];
        int[] deg = new int[n + 1];
        Map<Pair<Integer, Integer>, Integer> edgeCount = new HashMap<>();
        for (int[] e : edges) {
            int a = Math.min(e[0], e[1]), b = Math.max(e[0], e[1]);
            deg[a]++;
            deg[b]++;
            Pair<Integer, Integer> key = new Pair<>(a, b);
            edgeCount.put(key, edgeCount.getOrDefault(key, 0) + 1);
        }
        int[] sortedDeg = Arrays.copyOfRange(deg, 1, deg.length);
        Arrays.sort(sortedDeg);
        for (int i = 0; i < queries.length; i++) {
            // 容斥原理
            // c1: deg[a] + deg[b] - edgeCount(a,b) > q[i], ab存在边
            // c2: deg[a] + deg[b] > q[i], ab 存在边
            // c3: deg[a] + deg[b] > q[i], 对于所有点
            // result[i] = c1 + c3 - c2, c2被重复计算了
            int c1 = 0, c2 = 0, c3 = 0;
            for (Pair<Integer, Integer> edge : edgeCount.keySet()) {
                int a = edge.getKey(), b = edge.getValue();
                if (deg[a] + deg[b] - edgeCount.get(edge) > queries[i]) c1++;
                if (deg[a] + deg[b] > queries[i]) c2++;
            }
            int left = 0, right = n - 1;
            // 双指针求有序数组中和大于queries[i]的数对的个数
            while (left < n && right >= 0) {
                while (right > left && sortedDeg[left] + sortedDeg[right] <= queries[i]) {
                    left++;
                }
                if (right > left && sortedDeg[left] + sortedDeg[right] > queries[i]) {
                    c3 += right - left; // 求的是**数对**个数, 而不是两个数之间(含端点)共有多少个数, 所以不用+1
                }
                right--;
            }
            result[i] = c1 + c3 - c2;
        }
        return result;
    }

    // LC678 ** 两个栈
    public boolean checkValidString(String s) {
        char[] ca = s.toCharArray();
        Deque<Integer> left = new LinkedList<>(), star = new LinkedList<>();
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == '(') left.push(i);
            else if (ca[i] == '*') star.push(i);
            else {
                if (left.size() > 0) left.pop();
                else if (star.size() > 0) star.pop();
                else return false;
            }
        }
        if (left.size() > star.size()) return false;
        while (left.size() > 0 && star.size() > 0) {
            if (left.pop() > star.pop()) return false;
        }
        return true;
    }

    // LC761 ** 非常巧妙 看成括号对
    public String makeLargestSpecial(String s) {
        if (s.length() == 2) return s;
        Map<String, Integer> m = new HashMap<>();
        char[] ca = s.toCharArray();
        int prev = 0;
        int oneCount = 0;
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == '1') oneCount++;
            else {
                oneCount--;
                if (oneCount == 0) {
                    String magic = s.substring(prev, i + 1);
                    m.put(magic, m.getOrDefault(magic, 0) + 1);
                    prev = i + 1;
                }
            }
        }
        List<String> result = new ArrayList<>();
        for (String k : m.keySet()) {
            String kResult = k;
            if (k.length() > 2) {
                kResult = "1" + makeLargestSpecial(k.substring(1, k.length() - 1)) + "0";
            }
            for (int i = 0; i < m.get(k); i++) {
                result.add(kResult);
            }
        }
        result.sort(Comparator.reverseOrder());
        return String.join("", result);
    }

    // LC1605 ** 贪心
    public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
        int numRow = rowSum.length, numCol = colSum.length;
        int[][] result = new int[numRow][numCol];
        for (int i = 0; i < numRow; i++) {
            for (int j = 0; j < numCol; j++) {
                result[i][j] = Math.min(rowSum[i], colSum[j]);
                rowSum[i] -= result[i][j];
                colSum[j] -= result[i][j];
            }
        }
        return result;
    }

    // LC210 Topology
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        Deque<Integer> q = new LinkedList<>();
        List<List<Integer>> graph = new ArrayList<>(numCourses); // 拓扑排序算法中需要记录的出度表
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) graph.add(new ArrayList<>());
        for (int[] p : prerequisites) {
            inDegree[p[0]]++;
            graph.get(p[1]).add(p[0]);
        }
        for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) q.offer(i);
        while (!q.isEmpty()) {
            int p = q.poll();
            result.add(p);
            for (int next : graph.get(p)) {
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        for (int i = 0; i < numCourses; i++) if (inDegree[i] != 0) return new int[0];
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC802 Topology Sort
    public List<Integer> eventualSafeNodesTopologySort(int[][] graph) {
        List<Integer> result = new ArrayList<>();
        int n = graph.length;
        List<List<Integer>> reverseGraph = new ArrayList<>(n);
        int[] inDegree = new int[n];
        for (int i = 0; i < n; i++) reverseGraph.add(new LinkedList<>());
        for (int i = 0; i < n; i++) {
            int[] ithOutDegree = graph[i];
            for (int j : ithOutDegree) {
                reverseGraph.get(j).add(i);
                inDegree[i]++;
            }
        }
        Deque<Integer> zeroInDegreeQueue = new LinkedList<>();
        for (int i = 0; i < n; i++) if (inDegree[i] == 0) zeroInDegreeQueue.offer(i);
        while (!zeroInDegreeQueue.isEmpty()) {
            int i = zeroInDegreeQueue.poll();
            List<Integer> out = reverseGraph.get(i);
            for (int j : out) {
                inDegree[j]--;
                if (inDegree[j] == 0) {
                    zeroInDegreeQueue.offer(j);
                }
            }
        }
        for (int i = 0; i < n; i++) if (inDegree[i] == 0) result.add(i);
        return result;
    }

    // LC802 ** 三色算法 垃圾回收时候的判断有无依赖的一种算法
    int[] lc802Mark;
    final int UNVISITED = 0, IN_STACK = 1, SAFE = 2;

    public List<Integer> eventualSafeNodes(int[][] graph) {
        // graph[i] 为节点i的出度向量
        int n = graph.length;
        lc802Mark = new int[n];
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) { // 实际GC的三色算法中, 枚举的只是GC Root节点, 枚举完后, 如果有节点为UNVISITED, 则对其执行GC
            if (lc802Helper(i, graph)) {
                result.add(i);
            }
        }
        return result;
    }

    private boolean lc802Helper(int cur, int[][] graph) {
        if (lc802Mark[cur] != UNVISITED) {
            return lc802Mark[cur] == SAFE;
        }
        lc802Mark[cur] = IN_STACK;
        for (int next : graph[cur]) {
            if (!lc802Helper(next, graph)) {
                return false;
            }
        }
        lc802Mark[cur] = SAFE;
        return true;
    }

    // LC934
    int[][] lc934Directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int shortestBridge(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int ctr = 0;
        Set<Integer> s1 = new HashSet<>(), s2 = new HashSet<>();
        Set<Integer> curSet = s1;
        // DFS
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    lc934DfsHelper(grid, i, j, curSet);
                    ctr++;
                    if (ctr == 2) break;
                    curSet = s2;
                }
            }
            if (ctr == 2) break;
        }
        Set<Integer> smallSet = s1.size() > s2.size() ? s2 : s1;
        Set<Integer> largeSet = smallSet == s1 ? s2 : s1;
        Set<Integer> visited = new HashSet<>();
        // BFS
        int layer = -2;
        Deque<Integer> q = new LinkedList<>();
        for (int i : smallSet) {
            q.offer(i);
        }
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int p = q.poll();
                if (visited.contains(p)) continue;
                visited.add(p);
                if (largeSet.contains(p)) return layer;
                int row = p / grid[0].length, col = p % grid[0].length;
                for (int[] dir : lc934Directions) {
                    int newRow = row + dir[0], newCol = col + dir[1];
                    int newNum = newRow * grid[0].length + newCol;
                    if (newRow >= 0 && newRow < grid.length && newCol >= 0 && newCol < grid[0].length && !visited.contains(newNum)) {
                        q.offer(newNum);
                    }
                }
            }
        }
        return -1;
    }

    private void lc934DfsHelper(int[][] grid, int row, int col, Set<Integer> set) {
        grid[row][col] = -1;
        set.add(row * grid[0].length + col);
        for (int[] dir : lc934Directions) {
            int newRow = row + dir[0], newCol = col + dir[1];
            if (newRow >= 0 && newRow < grid.length && newCol >= 0 && newCol < grid[0].length && grid[newRow][newCol] == 1) {
                lc934DfsHelper(grid, newRow, newCol, set);
            }
        }
    }

    // LC1488
    public int[] avoidFlood(int[] rains) {
        int n = rains.length;
        int[] ans = new int[n];
        TreeSet<Integer> unrain = new TreeSet<>();
        Map<Integer, Integer> tbd = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (rains[i] == 0) unrain.add(i);
            else {
                ans[i] = -1;
                if (tbd.containsKey(rains[i])) {
                    if (unrain.isEmpty()) {
                        return new int[0];
                    }
                    int prevRainDay = tbd.get(rains[i]);
                    Integer ceiling = unrain.ceiling(prevRainDay);
                    if (ceiling == null) {
                        return new int[]{};
                    }
                    ans[ceiling] = rains[i];
                    unrain.remove(ceiling);
                }
                tbd.put(rains[i], i);
            }
        }
        for (int i : unrain) ans[i] = tbd.keySet().iterator().next();
        return ans;
    }

    // LC1300
    public int findBestValue(int[] arr, int target) {
        int n = arr.length;
        int[] prefix = new int[n + 1];
        Arrays.sort(arr);
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + arr[i - 1];
        int lo = 0, hi = arr[n - 1];
        while (lo < hi) { // 找value的下届
            int mid = lo + (hi - lo + 1) / 2;
            int idx = bsLargerOrEqualMin(arr, mid);
            while (idx < arr.length && arr[idx] == mid) idx++;
            if (idx > arr.length) break;
            // 此时idx及其之后的值都大于value
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * mid;
            if (curSum <= target) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int lowBound = lo;
        lo = 0;
        hi = arr[n - 1];
        while (lo < hi) { // 找value的上界
            int mid = lo + (hi - lo) / 2;
            int idx = bsLargerOrEqualMin(arr, mid);
            while (idx < arr.length && arr[idx] == mid) idx++;
            if (idx > arr.length) break;
            // 此时idx及其之后的值都大于value
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * mid;
            if (curSum < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int hiBound = lo;

        int result = lowBound, minDiff = Integer.MAX_VALUE;
        for (int i = lowBound; i <= hiBound; i++) {
            int idx = bsLargerOrEqualMin(arr, i);
            while (idx < arr.length && arr[idx] == i) idx++;
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * i;
            if (Math.abs(curSum - target) < minDiff) {
                minDiff = Math.abs(curSum - target);
                result = i;
            }
        }
        return result;
    }

    private int bsLessOrEqualMax(int[] arr, int target) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (arr[mid] <= target) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if (arr[lo] > target) return -1;
        return lo;
    }

    private int bsLargerOrEqualMin(int[] arr, int target) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if (arr[hi] < target) return -1;
        return hi;
    }

    // LC1553 ** 类比LCP20
    Map<Integer, Integer> lc1553Memo = new HashMap<>();

    public int minDays(int n) {
        return lc1553Helper(n);
    }

    private int lc1553Helper(int cur) {
        if (cur <= 1) return 1;
        if (cur == 2 || cur == 3) return 2;
        if (lc1553Memo.get(cur) != null) return lc1553Memo.get(cur);
        int result = cur;
        result = Math.min(result, 1 + lc1553Helper(cur / 2) + cur % 2);
        result = Math.min(result, 1 + lc1553Helper(cur / 3) + cur % 3);
        lc1553Memo.put(cur, result);
        return result;
    }

    // LCP 20 ** bottom up dfs
    Map<Long, Long> lcp20Map;
    final long lcp20Mod = 1000000007L;
    int lcp20Inc, lcp20Dec;
    int[] lcp20Jump, lcp20Cost;

    public int busRapidTransit(int target, int inc, int dec, int[] jump, int[] cost) {
        lcp20Map = new HashMap<>();
        lcp20Map.put(0l, 0l);
        lcp20Map.put(1l, (long) inc);
        lcp20Inc = inc;
        lcp20Cost = cost;
        lcp20Jump = jump;
        lcp20Dec = dec;
        return (int) (lcp20Helper(target) % lcp20Mod);
    }

    private long lcp20Helper(long cur) {
        if (lcp20Map.containsKey(cur)) return lcp20Map.get(cur);
        long result = cur * lcp20Inc;
        for (int i = 0; i < lcp20Jump.length; i++) {
            long remainder = cur % lcp20Jump[i];
            if (remainder == 0l) {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i]);
            } else {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i] + remainder * lcp20Inc);
                result = Math.min(result, lcp20Helper((cur / lcp20Jump[i]) + 1) + lcp20Cost[i] + (lcp20Jump[i] - remainder) * lcp20Dec);
            }
        }
        lcp20Map.put(cur, result);
        return result;
    }

    // LC1940 Prime Locked
    public List<Integer> longestCommomSubsequence(int[][] arrays) {
        List<Integer> result = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            int count = 0;
            for (int[] arr : arrays) {
                int bsResult = Arrays.binarySearch(arr, i);
                if (bsResult >= 0) count++;
            }
            if (count == arrays.length) result.add(i);
        }
        return result;
    }

    // LC1781
    public int beautySum(String s) {
        char[] ca = s.toCharArray();
        int[] freq = new int[26];
        int left = 0;
        int result = 0;
        while (left < s.length()) {
            freq = new int[26];
            int right = left;
            while (right < s.length()) {
                freq[ca[right++] - 'a']++;
                int[] j = lc1781Judge(freq);
                if (j[0] != -1) {
                    result += freq[j[1]] - freq[j[0]];
                }
            }
            left++;
        }
        return result;
    }

    private int[] lc1781Judge(int[] freq) {
        int min = Integer.MAX_VALUE, minIdx = -1, max = 0, maxIdx = -1;
        int notZeroCount = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] != 0) notZeroCount++;
            if (freq[i] > max) {
                max = freq[i];
                maxIdx = i;
            }
            if (freq[i] != 0 && freq[i] < min) {
                min = freq[i];
                minIdx = i;
            }
        }
        if (notZeroCount <= 1 || max == min) return new int[]{-1, -1};
        return new int[]{minIdx, maxIdx};
    }

    // LC417 **
    boolean[][] lc417P, lc417A;
    int[][] lc417Direction = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        int n = heights.length, m = heights[0].length;
        lc417A = new boolean[n][m];
        lc417P = new boolean[n][m];
        for (int i = 0; i < n; i++) {
            lc417Helper(heights, i, 0, lc417P);
            lc417Helper(heights, i, m - 1, lc417A);
        }
        for (int i = 0; i < m; i++) {
            lc417Helper(heights, 0, i, lc417P);
            lc417Helper(heights, n - 1, i, lc417A);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (lc417P[i][j] && lc417A[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }
        return result;
    }

    private boolean lc417IdxLegalJudge(int row, int col) {
        return (row >= 0 && row < lc417P.length && col >= 0 && col < lc417P[0].length);
    }

    private void lc417Helper(int[][] heights, int row, int col, boolean[][] judge) {
        if (judge[row][col]) return;
        judge[row][col] = true;
        for (int[] dir : lc417Direction) {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if (lc417IdxLegalJudge(newRow, newCol) && heights[newRow][newCol] >= heights[row][col]) {
                lc417Helper(heights, newRow, newCol, judge);
            }
        }
    }


    // LC611 **
    public int triangleNumber(int[] nums) {
        // nums.length <=1000
        // A + B > C
        int n = nums.length, result = 0;
        if (n <= 2) return 0;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int k = i;
            for (int j = i + 1; j < n; j++) {
                while (k + 1 < n && nums[k + 1] < nums[i] + nums[j]) {
                    k++;
                }
                result += Math.max(k - j, 0);
            }
        }
        return result;
    }

    // LC167
    public int[] twoSum(int[] numbers, int target) {
        int n = numbers.length;
        for (int i = 0; i < n; i++) {
            int tmp = target - numbers[i];
            int bsResult = Arrays.binarySearch(numbers, i + 1, n, tmp);
            if (bsResult >= 0) return new int[]{i + 1, bsResult + 1};
        }
        return new int[]{-1, -1};
    }

    // LC1823
    public int findTheWinner(int n, int k) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int i = 1; i <= n; i++) s.add(i);
        int cur = 1;
        while (s.size() > 1) {
            int ctr = 1;
            while (ctr < k) {
                Integer higher = s.higher(cur);
                if (higher == null) higher = s.first();
                cur = higher;
                ctr++;
            }
            Integer next = s.higher(cur);
            if (next == null) next = s.first();
            s.remove(cur);
            cur = next;
        }
        return s.first();
    }

    // LC1567 Solution DP
    public int getMaxLen(int[] nums) {
        int n = nums.length;
        int[] pos = new int[2], neg = new int[2];
        if (nums[0] > 0) pos[0] = 1;
        if (nums[0] < 0) neg[0] = 1;
        int result = pos[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                pos[i % 2] = pos[(i - 1) % 2] + 1;
                neg[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
            } else if (nums[i] < 0) {
                pos[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
                neg[i % 2] = pos[(i - 1) % 2] + 1;
            } else {
                pos[i % 2] = neg[i % 2] = 0;
            }
            result = Math.max(result, pos[i % 2]);
        }
        return result;
    }

    // LC1567 慢
    public int getMaxLenSimple(int[] nums) {
        int n = nums.length;
        int[] nextZero = new int[n];
        int[] negCount = new int[n];
        Arrays.fill(nextZero, -1);
        int nextZeroIdx = -1;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] == 0) {
                nextZeroIdx = i;
            }
            nextZero[i] = nextZeroIdx;
        }
        negCount[0] = nums[0] < 0 ? 1 : 0;
        for (int i = 1; i < n; i++) {
            negCount[i] = negCount[i - 1] + (nums[i] < 0 ? 1 : 0);
        }
        int result = 0;
        // 在下一个0来临之前, 找到最大的偶数个负数所在IDX 求长度
        for (int i = 0; i < n; i++) {
            if (nums[i] != 0) {
                int curNegCount = negCount[i];
                if (nums[i] < 0) curNegCount--;
                int start = i, end = -1;
                if (nextZero[i] == -1) end = n - 1;
                else end = nextZero[i] - 1;
                int j;
                for (j = end; j >= start; j--) {
                    if (negCount[j] % 2 == curNegCount % 2) break;
                }
                result = Math.max(result, j - i + 1);
            }
        }
        return result;
    }

    // LC198
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[3]; // 滚数组
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[(i + 3) % 3] = Math.max(dp[(i - 1 + 3) % 3], dp[(i - 2 + 3) % 3] + nums[i]);
        }
        return dp[(n - 1 + 3) % 3];
    }

    // LC740 ** 打家劫舍
    public int deleteAndEarn(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int[] sum = new int[max + 1];
        for (int i : nums) sum[i] += i;
        if (max == 1) return sum[1];
        if (max == 2) return Math.max(sum[1], sum[2]);
        int[] dp = new int[max + 1];
        dp[1] = sum[1];
        dp[2] = Math.max(sum[1], sum[2]);
        for (int i = 3; i <= max; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + sum[i]);
        }
        return dp[max];
    }

    // LC673 **
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        int[] dp = new int[n], count = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(count, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[i] <= dp[j]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
            }
        }
        int max = Arrays.stream(dp).max().getAsInt();
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == max) {
                result += count[i];
            }
        }
        return result;
    }
}

// Interview 03.06
class AnimalShelf {
    int seq = 0;
    // type 0-cat 1-dog
    final int CAT = 0, DOG = 1;
    Map<Integer, Integer> idSeqMap = new HashMap<>();
    Deque<Integer> catQueue = new LinkedList<>();
    Deque<Integer> dogQueue = new LinkedList<>();

    public AnimalShelf() {

    }

    public void enqueue(int[] a) {
        // a[0] = id, a[1] = type
        int sequence = getSeq();
        idSeqMap.put(a[0], sequence);
        if (a[1] == CAT) {
            catQueue.offer(a[0]);
        } else {
            dogQueue.offer(a[0]);
        }
    }

    public int[] dequeueAny() {
        if (catQueue.isEmpty() && dogQueue.isEmpty()) {
            return new int[]{-1, -1};
        } else if (catQueue.isEmpty() && !dogQueue.isEmpty()) {
            return dequeueDog();
        } else if (!catQueue.isEmpty() && dogQueue.isEmpty()) {
            return dequeueCat();
        } else if (idSeqMap.get(catQueue.peek()) < idSeqMap.get(dogQueue.peek())) {
            return dequeueCat();
        } else {
            return dequeueDog();
        }

    }

    public int[] dequeueDog() {
        if (dogQueue.isEmpty()) return new int[]{-1, -1};
        int polledDogId = dogQueue.poll();
        idSeqMap.remove(polledDogId);
        return new int[]{polledDogId, DOG};
    }

    public int[] dequeueCat() {
        if (catQueue.isEmpty()) return new int[]{-1, -1};
        int polledCatId = catQueue.poll();
        idSeqMap.remove(polledCatId);
        return new int[]{polledCatId, CAT};
    }

    private int getSeq() {
        return seq++;
    }
}

// LC478
class Solution {
    double x_center;
    double y_center;
    double radius;

    public Solution(double radius, double x_center, double y_center) {
        this.x_center = x_center;
        this.y_center = y_center;
        this.radius = radius;
    }

    public double[] randPoint() {
        double len = Math.sqrt(Math.random()) * radius; // 注意开方 , 参考solution
        double theta = Math.random() * Math.PI * 2;

        double x = len * Math.sin(theta) + x_center;
        double y = len * Math.cos(theta) + y_center;
        return new double[]{x, y};
    }
}

// JZOF 59
class KthLargest {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        for (int i : nums) {
            add(i);
        }
    }

    public int add(int val) {
        if (pq.size() < k) {
            pq.offer(val);
        } else {
            if (val > pq.peek()) {
                pq.poll();
                pq.offer(val);
            }
        }
        return pq.peek();
    }
}

class quickSort {

    static Random r = new Random();

    public static void sort(int[] arr) {
        helper(arr, 0, arr.length - 1);
    }

    private static void helper(int[] arr, int start, int end) {
        if (start >= end) return;
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
        helper(arr, start, left - 1);
        helper(arr, right + 1, end);
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