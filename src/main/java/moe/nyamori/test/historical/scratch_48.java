package moe.nyamori.test.historical;

import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;

class scratch_48 {
    public static void main(String[] args) {
        scratch_48 s = new scratch_48();
        long timing = System.currentTimeMillis();

        // [3,4],[2,3],[1,2]
        System.out.println(s.getMinSwaps("00123", 1));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1452 Tips: Two pointer, O(n^2 *m), n=favoriteCompanies.size(), m = companySet.size()
    // 但我这个好像就是朴素思路, 复杂度一样?
    // 不进行事先哈希成idx更慢 但复杂度一样 代码比较简洁
    public List<Integer> peopleIndexes(List<List<String>> favoriteCompanies) {
        List<Integer> result = new ArrayList<>();
        List<Set<String>> hashedFC = new ArrayList<>(favoriteCompanies.size());
        for (List<String> f : favoriteCompanies) {
            Set<String> tmp = new HashSet<>(f.size());
            for (String c : f) tmp.add(c);
            hashedFC.add(tmp);
        }
        for (int i = 0; i < hashedFC.size(); i++) {
            int count = 0;
            for (int j = 0; j < hashedFC.size(); j++) {
                if (i != j) {// 判断 i 是不是 j的子集, 如果i中有元素j没有, 则i不是j的子集
                    boolean flag = false; // 假设是, 若不是立即break;
                    for (String ele : hashedFC.get(i)) {
                        if (!hashedFC.get(j).contains(ele)) {
                            flag = true;
                            break;
                        }
                    }
                    if (flag) count++;
                }
            }
            if (count == hashedFC.size() - 1) result.add(i);
        }
        return result;
    }

    // LC1850 ** 交换的是相邻位数字, 贪心策略见Solution
    public int getMinSwaps(String num, int k) {
        int n = num.length();
        char[] ca = num.toCharArray();
        char[] orig = Arrays.copyOfRange(ca, 0, n);
        for (int i = 0; i < k; i++) {
            ca = getNextPermutation(ca);
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (ca[i] != orig[i]) {
                int j;
                for (j = i; ca[i] != orig[j]; j++) ;
                result += j - i;
                while (j != i) {
                    arraySwap(orig, j, j - 1);
                    j--;
                }
            }
        }
        return result;
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

    // LC332 可以先对机场按照字典序排序 也可以参考Solution使用优先队列
    public List<String> findItinerary(List<List<String>> tickets) {
        List<String> result = new ArrayList<>();
        Set<String> airportSet = new HashSet<>();
        for (List<String> t : tickets) {
            airportSet.addAll(t);
        }
        int n = airportSet.size();
        List<String> airportList = new ArrayList<>(airportSet);
        Collections.sort(airportList);
        Map<String, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < airportList.size(); i++) idxMap.put(airportList.get(i), i);
        int[][] reachable = new int[n][n];
        for (List<String> t : tickets) {
            reachable[idxMap.get(t.get(0))][idxMap.get(t.get(1))]++;
        }
        Deque<Integer> stack = new LinkedList<>();
        lc332Helper(idxMap.get("JFK"), stack, reachable);
        while (!stack.isEmpty()) result.add(airportList.get(stack.pop()));
        return result;
    }

    private void lc332Helper(int cur, Deque<Integer> stack, int[][] reachable) {
        for (int i = 0; i < reachable.length; i++) {
            if (reachable[cur][i] != 0) {
                reachable[cur][i]--;
                lc332Helper(i, stack, reachable);
            }
        }
        stack.push(cur);
    }

    // LC581 Stack
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int start = 0, end = -1;
        int min = nums[n - 1], max = nums[0];
        for (int i = 0, j = n - 1; i < n; i++, j--) {
            if (nums[i] < max) {
                end = i;
            } else {
                max = nums[i];
            }

            if (nums[j] > min) {
                start = j;
            } else {
                min = nums[j];
            }
        }
        return end - start + 1;
    }

    // LC581
    public int findUnsortedSubarraySort(int[] nums) {
        int n = nums.length;
        int[] orig = Arrays.copyOfRange(nums, 0, nums.length);
        Arrays.sort(nums);
        int left = 0, right = n - 1;
        while (left < n) {
            if (orig[left] != nums[left]) break;
            left++;
        }
        while (right >= 0) {
            if (orig[right] != nums[right]) break;
            right--;
        }
        if (left == n && right == -1) return 0;
        return right - left + 1;
    }

    // LC753 Hierholzer算法, 找欧拉通路 **
    public String crackSafe(int n, int k) {
        boolean[] visited = new boolean[(int) Math.pow(k, n)];
        Deque<Character> stack = new LinkedList<>();
        visited[0] = true;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) sb.append("0");
        if (k == 1) return sb.toString();
        lc753Dfs(sb.toString(), visited, stack, k, n);
        stack.pop();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.toString();
    }

    private void lc753Dfs(String cur, boolean[] visited, Deque<Character> stack, int k, int n) {
        String prefix = cur.substring(1, n);
        for (int i = 0; i < k; i++) {
            String next = prefix + i;
            int nextIdx = Integer.valueOf(next, k);
            if (!visited[nextIdx]) {
                visited[nextIdx] = true;
                lc753Dfs(next, visited, stack, k, n);
            }
        }
        stack.push(cur.charAt(cur.length() - 1));
    }

    // LC481 Kolakoski 数列
    public int magicalString(int n) {
        // S = 1 22 1
        // sh = 1 2
        if (n == 0) return 0;
        if (n <= 3) return 1;
        int[] seq = new int[n + 2];
        seq[1] = 1;
        seq[2] = 2;
        seq[3] = 2;
        int mainIdx = 3; // 从1算
        int shadowIdx = 2;
        int count = 1;
        while (mainIdx <= n) {
            if (seq[shadowIdx] == 1 || (seq[shadowIdx] == 2 && seq[mainIdx] == seq[mainIdx - 1])) {
                seq[mainIdx + 1] = seq[mainIdx] == 1 ? 2 : 1;
                shadowIdx++;
            } else {
                seq[mainIdx + 1] = seq[mainIdx];
            }
            if (seq[mainIdx] == 1) count++;
            mainIdx++;
        }
        return count;
    }

    // LC436
    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        int[] result = new int[n];
        Map<Integer, Integer> origIdxMap = new HashMap<>(n);
        for (int i = 0; i < n; i++) {
            origIdxMap.put(intervals[i][0], i);
        }
        Arrays.fill(result, -1);
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        for (int i = 0; i < n; i++) {
            int target = intervals[i][1];
            int lo = i, hi = n - 1;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (intervals[mid][0] >= target) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if (intervals[lo][0] < target) continue;
            result[origIdxMap.get(intervals[i][0])] = origIdxMap.get(intervals[lo][0]);
        }
        return result;
    }

    // LC743 Dijkstra
    public int networkDelayTime(int[][] times, int n, int k) {
        int[] distance = new int[n + 1];
        boolean[] visited = new boolean[n + 1];
        Arrays.fill(distance, -1);
        Map<Integer, Map<Integer, Integer>> m = new HashMap<>();
        for (int[] t : times) {
            m.putIfAbsent(t[0], new HashMap<>());
            m.get(t[0]).put(t[1], t[2]);
        }
        if (!m.containsKey(k)) return -1;
        for (int key : m.get(k).keySet()) {
            distance[key] = m.get(k).get(key);
        }
        distance[k] = 0;
        visited[k] = true;
        for (int i = 1; i <= n; i++) {
            if (i != k) {
                int curMinIdx = -1;
                int curMinLen = Integer.MAX_VALUE;
                for (int j = 1; j <= n; j++) {
                    if (!visited[j] && distance[j] != -1 && curMinLen > distance[j]) {
                        curMinIdx = j;
                        curMinLen = distance[j];
                    }
                }
                if (curMinIdx == -1) break;
                visited[curMinIdx] = true;
                if (m.containsKey(curMinIdx)) {
                    for (int j = 1; j <= n; j++) {
                        if (!visited[j]) {
                            if (m.get(curMinIdx).containsKey(j)
                                    && (distance[j] == -1 || distance[j] > curMinLen + m.get(curMinIdx).get(j))) {
                                distance[j] = curMinLen + m.get(curMinIdx).get(j);
                            }
                        }
                    }
                }
            }
        }
        int result = Integer.MIN_VALUE;
        for (int i = 1; i <= n; i++) {
            if (distance[i] == -1) return -1;
            result = Math.max(result, distance[i]);
        }
        return result;
    }


    // LC1337
    public int[] kWeakestRows(int[][] mat, int k) {
        int m = mat.length, n = mat[0].length;
        int[] result = new int[k];
        // <idx,sum>
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(
                (o1, o2) -> o2.getValue() == o1.getValue() ? o2.getKey() - o1.getKey() : o2.getValue() - o1.getValue()
        );
        for (int i = 0; i < mat.length; i++) {
            int[] arr = mat[i];
            int j = 0, sum = 0;
            while (j < n && arr[j++] == 1) sum++;
            pq.offer(new Pair<>(i, sum));
        }
        int ctr = 0;
        while (ctr < k) {
            result[ctr++] = pq.poll().getKey();
        }
        return result;
    }

    // LC987
    public List<List<Integer>> verticalTraversal(TreeNode8 root) {
        Map<TreeNode8, Integer> colMap = new HashMap<>();
        Map<TreeNode8, Integer> rowMap = new HashMap<>();
        Map<Integer, List<TreeNode8>> rColMap = new HashMap<>();
        Deque<TreeNode8> q = new LinkedList<>();
        rowMap.put(root, 0);
        colMap.put(root, 0);
        rColMap.put(0, new ArrayList<>());
        rColMap.get(0).add(root);
        q.offer(root);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                TreeNode8 p = q.poll();
                if (p.left != null) {
                    colMap.put(p.left, colMap.get(p) - 1);
                    rowMap.put(p.left, layer + 1);
                    rColMap.putIfAbsent(colMap.get(p) - 1, new ArrayList<>());
                    rColMap.get(colMap.get(p) - 1).add(p.left);
                    q.offer(p.left);
                }
                if (p.right != null) {
                    colMap.put(p.right, colMap.get(p) + 1);
                    rowMap.put(p.right, layer + 1);
                    rColMap.putIfAbsent(colMap.get(p) + 1, new ArrayList<>());
                    rColMap.get(colMap.get(p) + 1).add(p.right);
                    q.offer(p.right);
                }
            }
        }
        List<Integer> colList = new ArrayList<>(rColMap.keySet());
        Collections.sort(colList);
        List<List<Integer>> result = new ArrayList<>(colList.size());
        for (int i : colList) {
            List<TreeNode8> thisCol = rColMap.get(i);
            thisCol.sort((o1, o2) -> rowMap.get(o1) == rowMap.get(o2) ? o1.val - o2.val : rowMap.get(o1) - rowMap.get(o2));
            List<Integer> thisColResult = new ArrayList<>(thisCol.size());
            for (TreeNode8 t : thisCol) thisColResult.add(t.val);
            result.add(thisColResult);
        }
        return result;
    }

    // LC462
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        if (nums.length % 2 == 1) {
            int result = 0;
            int mid = nums[nums.length / 2];
            for (int i : nums) {
                result += Math.abs(i - mid);
            }
            return result;
        } else {
            int right = nums[nums.length / 2], left = nums[(nums.length / 2) - 1];
            int lr = 0, rr = 0;
            for (int i : nums) {
                lr += Math.abs(i - left);
                rr += Math.abs(i - right);
            }
            return Math.min(lr, rr);
        }
    }

    // LC1837
    public int sumBase(int n, int k) {
        int i = n, result = 0;
        while (i != 0) {
            result += i % k;
            i /= k;
        }
        return result;
    }

    // LC133
    class lc133 {
        class Node {
            public int val;
            public List<Node> neighbors;

            public Node() {
                val = 0;
                neighbors = new ArrayList<Node>();
            }

            public Node(int _val) {
                val = _val;
                neighbors = new ArrayList<Node>();
            }

            public Node(int _val, ArrayList<Node> _neighbors) {
                val = _val;
                neighbors = _neighbors;
            }
        }


        class Solution {
            Map<Node, Node> origToClone = new HashMap<>();

            public Node cloneGraph(Node node) {
                if (node == null) return null;
                if (origToClone.containsKey(node)) {
                    return origToClone.get(node);
                }
                Node clone = new Node(node.val);
                origToClone.put(node, clone);
                for (Node ne : node.neighbors) {
                    clone.neighbors.add(cloneGraph(ne));
                }
                return clone;
            }

        }
    }

    // JZOF 32
    public List<List<Integer>> levelOrder(TreeNode8 root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Deque<TreeNode8> q = new LinkedList<>();
        q.offer(root);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            List<Integer> thisLine = new ArrayList<>(qSize);
            for (int i = 0; i < qSize; i++) {
                TreeNode8 p = q.poll();
                thisLine.add(p.val);
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
            result.add(thisLine);
        }
        return result;
    }

    // LC124
    int lc124Result;

    public int maxPathSum(TreeNode8 root) {
        lc124Result = Integer.MIN_VALUE;
        lc124Helper(root);
        return lc124Result;
    }

    private int lc124Helper(TreeNode8 root) {
        if (root == null) return 0;
        int leftGain = Math.max(0, lc124Helper(root.left));
        int rightGain = Math.max(0, lc124Helper(root.right));

        int sum = root.val + leftGain + rightGain;
        lc124Result = Math.max(lc124Result, sum);

        return root.val + Math.max(leftGain, rightGain);
    }

    // LC45 Greedy
    public int jumpGreedy(int[] nums) {
        int n = nums.length;
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        int cur = 0;
        while (end < n) {
            maxPosition = Math.max(maxPosition, cur + nums[cur]);
            if (end == cur) {
                end = maxPosition;
                steps++;
            }
            cur++;
        }
        return steps;
    }

    // LC45 DP O(n^2) AC
    public int jump(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                // 看能不能用j中转, 即i在j的可达范围内
                int left = Math.max(0, j - nums[j]);
                int right = Math.min(n - 1, j + nums[j]);
                boolean reachable = i >= left && i <= right;
                if (reachable) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1];
    }

    // LC45 TLE Dijkstra O(n^2)
    public int jumpDijkstra(int[] nums) {
        if (nums.length == 1) return 0;
        int n = nums.length;
        int[] matrix = new int[n];
        Arrays.fill(matrix, -1);
        for (int j = 1; j <= nums[0]; j++) {
            if (j < n) {
                matrix[j] = 1;
            }
        }
        if (matrix[n - 1] == 1) return 1;

        // Dijkstra?
        Set<Integer> visited = new HashSet<>();
        visited.add(0);
        for (int i = 1; i < n; i++) {
            int curMinIdx = 0;
            int curMinLen = Integer.MAX_VALUE;
            for (int j = 1; j < n; j++) {
                if (!visited.contains(j) && matrix[j] != -1 && curMinLen > matrix[j]) {
                    curMinIdx = j;
                    curMinLen = matrix[j];
                }
            }
            visited.add(curMinIdx);
            for (int j = 1; j < n; j++) {
                if (!visited.contains(j)) {
                    int left = Math.max(0, curMinIdx - nums[curMinIdx]);
                    int right = Math.min(n - 1, curMinIdx + nums[curMinIdx]);
                    boolean a = j >= left && j <= right;
                    boolean b = matrix[j] == -1;
                    boolean c = matrix[j] > matrix[curMinIdx] + 1;
                    if (a && (b || c)) {
                        matrix[j] = matrix[curMinIdx] + 1;
                    }
                }
            }
        }
        return matrix[n - 1];
    }

    // LC787 DFS 不存在环 TLE
    int lc787DfsResult;
    int[] lc787DfsMinCost;
    int[] lc787DfsMaxStop;

    public int findCheapestPriceDFS(int n, int[][] flights, int src, int dst, int k) {
        lc787DfsResult = Integer.MAX_VALUE;
        int[][] reachable = new int[n][n];
        lc787DfsMinCost = new int[n];
        lc787DfsMaxStop = new int[n];
        Arrays.fill(lc787DfsMinCost, Integer.MAX_VALUE);
        for (int[] r : reachable) Arrays.fill(r, -1);
        for (int[] f : flights) {
            reachable[f[0]][f[1]] = f[2];
        }
        lc787DfsHelper(reachable, src, k + 1, 0, dst);
        return lc787DfsResult == Integer.MAX_VALUE ? -1 : lc787DfsResult;
    }

    private void lc787DfsHelper(int[][] reachable, int cur, int limit, int price, int dst) {
        if (cur == dst) {
            lc787DfsResult = Math.min(lc787DfsResult, price);
            return;
        }
        if (limit > 0) {
            for (int i = 0; i < reachable.length; i++) {
                int minCostToi = lc787DfsMinCost[i], costFromCurToI = price + reachable[cur][i];
                if (reachable[cur][i] != -1) {
                    // **剪枝
                    if (costFromCurToI > lc787DfsResult) continue;

                    if (minCostToi > costFromCurToI) {
                        lc787DfsHelper(reachable, i, limit - 1, costFromCurToI, dst);
                        lc787DfsMinCost[i] = costFromCurToI;
                        lc787DfsMaxStop[i] = limit - 1;
                    } else if (lc787DfsMaxStop[i] < limit - 1) {
                        lc787DfsHelper(reachable, i, limit - 1, costFromCurToI, dst);
                    }
                }
            }
        }
    }


    // LC787  ** From Solution
    // 剪枝参考: https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/solution/dijkstraji-bai-100yong-hu-jie-jue-guan-f-hpmn/
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        // Dijkstra
        int[][] reachable = new int[n][n];
        for (int[] r : reachable) Arrays.fill(r, -1);
        for (int[] f : flights) {
            reachable[f[0]][f[1]] = f[2];
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        pq.offer(new int[]{k + 1, 0, src});
        // [a,b,c] a: 跳数限制 b: 价格 c:目标站点。按价格排序 保证队首是价格最低的

        int[] minCostTo = new int[n]; // 到i最小的价格, 初始化为极大值
        int[] maxStopsTo = new int[n]; // 到i最大的可经停站数, 初始化为0
        Arrays.fill(minCostTo, Integer.MAX_VALUE);

        while (!pq.isEmpty()) {
            int[] polled = pq.poll();
            int stop = polled[0], costToNext = polled[1], next = polled[2];
            if (next == dst) return costToNext;
            if (stop > 0) {
                for (int i = 0; i < n; i++) {
                    if (reachable[next][i] != -1) {
                        int minCostToI = minCostTo[i], costFromNextToI = reachable[next][i];
                        if (costToNext + costFromNextToI < minCostToI) { // 经过中间点i后价格更小的加入bfs
                            pq.offer(new int[]{stop - 1, costToNext + costFromNextToI, i});
                            minCostTo[i] = costToNext + costFromNextToI;
                            maxStopsTo[i] = stop - 1;
                        } else if (maxStopsTo[i] < stop - 1) { // 经过中间点i后可经停的站数更多的加入bfs
                            pq.offer(new int[]{stop - 1, costToNext + costFromNextToI, i});
                        }
                    }
                }
            }
        }
        return -1;
    }

    // LC1392
    public String longestPrefix(String s) {
        // 哈希
        final int mod = 1000000007;
        final int base = 31; // 选一个质数
        long prefix = 0, suffix = 0, mul = 1;
        int happy = 0;
        int n = s.length();
        for (int i = 1; i < n; i++) {
            prefix = (prefix * base + s.charAt(i - 1)) % mod;
            suffix = (suffix + (s.charAt(n - i) - 'a') * mul) % mod;
            if (prefix == suffix) happy = i;
            mul = (mul * base) % mod;
        }
        // 建议 用10进制帮助思考
        return s.substring(0, happy);
    }

    // LC576
    int[][] lc576Directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    final long lc576mod = 1000000007;
    int lc576MaxMove;
    Integer[][][] lc576Memo;

    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        lc576Memo = new Integer[m][n][maxMove];
        lc576MaxMove = maxMove;
        int result = lc576Helper(startRow, startColumn, 0);
        return result;
    }

    private int lc576Helper(int row, int col, int curMove) {
        if (row < 0 || row >= lc576Memo.length || col < 0 || col >= lc576Memo[0].length) return 0;
        if (curMove >= lc576MaxMove) return 0;
        if (lc576Memo[row][col][curMove] != null) return lc576Memo[row][col][curMove];
        long result = 0;
        if (row == 0) result++;
        if (row == lc576Memo.length - 1) result++;
        if (col == 0) result++;
        if (col == lc576Memo[0].length - 1) result++;
        for (int[] dir : lc576Directions) {
            result = (result + lc576Helper(row + dir[0], col + dir[1], curMove + 1)) % lc576mod;
        }
        return lc576Memo[row][col][curMove] = (int) (result);
    }

    // Interview 04.05
    long[] itv0405Memo;
    int itv0405Counter;

    public boolean isValidBST(TreeNode8 root) {
        itv0405Memo = new long[2];
        itv0405Counter = 0;
        itv0405Memo[1] = Long.MIN_VALUE;
        return isValidBstHelper(root);
    }

    private boolean isValidBstHelper(TreeNode8 root) {
        if (root == null) return true;
        boolean result = isValidBstHelper(root.left);
        itv0405Memo[(itv0405Counter++) % 2] = root.val;
        if (root.val <= itv0405Memo[(itv0405Counter) % 2]) return false;
        result = result && isValidBstHelper(root.right);
        return result;
    }

    // LC1498 **
    public int numSubseq(int[] nums, int target) {
        final long mod = 1000000007;
        int n = nums.length;
        long result = 0;
        Arrays.sort(nums);
        int left = 0, right = n - 1;
        while (left <= right && left < n) {
            while (right >= 0 && nums[left] + nums[right] > target) {
                right--;
            }
            if (right < left) break;
            if (left == right && nums[left] + nums[right] > target) break;
            result = (result + quickPowerMod(2, right - left, mod)) % mod;
            left++;
        }
        return (int) (result % mod);
    }

    private long quickPowerMod(long a, long b, long m) {
        long result = 1;
        long base = a;
        while (b > 0) {
            if ((b & 1l) == 1) {
                result = (result * base) % m;
            }
            base = (base * base) % m;
            b >>= 1;
        }
        return result;
    }

    // LC383
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] freq = new int[26];
        for (char c : ransomNote.toCharArray()) freq[c - 'a']--;
        for (char c : magazine.toCharArray()) freq[c - 'a']++;
        for (int i : freq) if (i < 0) return false;
        return true;
    }

    // LC859
    public boolean buddyStrings(String s, String goal) {
        if (s.length() != goal.length()) return false;
        int n = s.length();
        char[] sa = s.toCharArray(), ga = goal.toCharArray();
        int[] sf = new int[26], gf = new int[26];
        int diff = 0;
        for (int i = 0; i < n; i++) {
            sf[sa[i] - 'a']++;
            gf[ga[i] - 'a']++;
            if (sa[i] != ga[i]) diff++;
        }
        boolean twoOrMoreSameLetterFlag = false;
        for (int i = 0; i < 26; i++) {
            if (sf[i] != gf[i]) return false;
            if (sf[i] > 1) twoOrMoreSameLetterFlag = true;
        }
        if (diff != 2) return diff == 0 ? twoOrMoreSameLetterFlag : false;
        return true;
    }

    // LC313 ** DP
    public int nthSuperUglyNumberDP(int n, int[] primes) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        int[] pointer = new int[primes.length];
        Arrays.fill(pointer, 1);
        for (int i = 2; i <= n; i++) {
            int next = Integer.MAX_VALUE;
            List<Integer> idx = new ArrayList<>();
            for (int j = 0; j < primes.length; j++) {
                int tmp = dp[pointer[j]] * primes[j];
                if (tmp < next) {
                    idx = new ArrayList<>();
                    idx.add(j);
                    next = tmp;
                } else if (tmp == next) {
                    idx.add(j);
                }
            }
            dp[i] = next;
            for (int j : idx) pointer[j]++;
        }
        return dp[n];
    }

    // LC313
    public int nthSuperUglyNumber(int n, int[] primes) {
        Set<Long> visited = new HashSet<>();
        PriorityQueue<Long> pq = new PriorityQueue<>();
        visited.add(1L);
        pq.offer(1L);
        long result = 1;
        for (int i = 0; i < n; i++) {
            long p = pq.poll();
            result = p;
            for (int j : primes) {
                long tmp = j * p;
                if (visited.add(tmp)) {
                    pq.offer(tmp);
                }
            }
        }
        return (int) result;
    }

    // LC1104
    public List<Integer> pathInZigZagTree(int label) {
        List<Integer> result = new LinkedList<>();
        // 获取层数, 从0算
        int floor = Integer.SIZE - Integer.numberOfLeadingZeros(label) - 1;
        // 每一层的个数
        int floorNum = 1 << floor;
        int floorStart = 1 << floor;
        int floorEnd = floorStart + floorNum - 1;
        if (floor % 2 == 1) {
            label = floorEnd - label + floorStart;
        }
        int cur = label;
        while (cur != 0) {
            result.add(0, cur);
            cur >>= 1;
        }
        result = new ArrayList<>(result);
        for (int i = 1; i < result.size(); i += 2) {
            floor = i;
            floorNum = 1 << floor;
            floorStart = 1 << floor;
            floorEnd = floorStart + floorNum - 1;
            result.set(i, floorEnd - result.get(i) + floorStart);
        }
        return result;
    }

    // LC808 ** 看公式推导
    // https://leetcode-cn.com/problems/soup-servings/solution/zhao-zhong-jie-shi-yi-xia-dong-tai-gui-hua-zhong-d/
    public double soupServings(int n) {
        if (n >= 500 * 25) return 1d;
        n = n / 25 + (n % 25 == 0 ? 0 : 1);
        Double[][] memo = new Double[n + 1][n + 1];
        return lc808Helper(n, n, memo);
    }

    private double lc808Helper(int i, int j, Double[][] memo) {
        if (i <= 0) {
            if (j <= 0) return 0.5d;
            else return 1d;
        }
        if (j <= 0) return 0d;
        if (memo[i][j] != null) return memo[i][j];
        return memo[i][j] = 0.25d * (
                lc808Helper(i - 4, j, memo)
                        + lc808Helper(i - 3, j - 1, memo)
                        + lc808Helper(i - 2, j - 2, memo)
                        + lc808Helper(i - 1, j - 3, memo)
        );
    }

    // LC539
    public int findMinDifference(List<String> timePoints) {
        int round = 1440;
        List<Integer> minutes = new ArrayList<>(timePoints.size());
        for (String s : timePoints) {
            Integer[] tmp = Arrays.stream(s.split(":")).map(Integer::valueOf).toArray(Integer[]::new);
            minutes.add(tmp[0] * 60 + tmp[1]);
        }
        Collections.sort(minutes);
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < minutes.size(); i++) {
            result = Math.min(result, (minutes.get((i + 1) % minutes.size()) - minutes.get(i) + round) % round);
        }
        return result;
    }

    // LC1175
    public int numPrimeArrangements(int n) {
        final long mod = 1000000007;
        List<Integer> primeList = new ArrayList<>();
        Set<Integer> notPrime = new HashSet<>();
        notPrime.add(1);
        for (int i = 2; i <= n; i++) {
            if (!notPrime.contains(i)) {
                primeList.add(i);
            }
            for (int j = 0; j < primeList.size(); j++) {
                if (i * primeList.get(j) > n) break;
                notPrime.add(i * primeList.get(j));
                if (i % primeList.get(j) == 0) break;
            }
        }
        // 全排列
        int primeNum = primeList.size();
        int notPrimeNum = n - primeNum;
        // 尝试Lambda, 发现要递归只能这么写 (汗
        Function<Integer, Long> f = new Function<Integer, Long>() {
            @Override
            public Long apply(Integer i) {
                return i <= 1 ? 1L : ((i * this.apply(i - 1)) % mod);
            }
        };
        long pnf = f.apply(primeNum);
        long npnf = f.apply(notPrimeNum);
        return (int) (((pnf % mod) * (npnf % mod)) % mod);
    }

    // LC1566 from solution
    public boolean containsPattern(int[] arr, int m, int k) {
        int n = arr.length;
        for (int l = 0; l <= n - m * k; l++) {
            int offset;
            for (offset = 0; offset < m * k; offset++) {
                if (arr[l + offset] != arr[l + offset % m]) {
                    break;
                }
            }
            if (offset == m * k) {
                return true;
            }
        }
        return false;
    }

    // LC893
    public int numSpecialEquivGroups(String[] words) {
        // 对word奇偶查频, 0奇数, 1偶数
        int n = words.length;
        int[][] freq;
        Map<String, Set<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            freq = new int[2][26];
            char[] cArr = words[i].toCharArray();
            for (int j = 0; j < cArr.length; j++) {
                freq[j % 2][cArr[j] - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 26; k++) {
                    if (freq[j][k] != 0) {
                        sb.append((char) ('a' + k));
                        sb.append(freq[j][k]);
                    }
                }
                sb.append(",");
            }
            String hash = sb.toString();
            map.putIfAbsent(hash, new HashSet<>());
            map.get(hash).add(i);
        }
        return map.size();
    }

    // LC704
    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) return mid;
            if (nums[mid] > target) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return -1;
    }

    // LCP28
    public int purchasePlans(int[] nums, int target) {
        int n = nums.length;
        final long mod = 1000000007;
        long result = 0;
        Arrays.sort(nums);
        int right = n - 1;
        for (int i = 0; i < n; i++) {
            while (right >= 0 && nums[right] + nums[i] > target) right--;
            if (right <= i) break;
            result = (result + (right - i)) % mod;
        }
        return (int) result;
    }

    // LC1441
    public List<String> buildArray(int[] target, int n) {
        int cur = 1;
        int ctr = 0;
        List<String> result = new ArrayList<>();
        while (ctr != target.length) {
            while (cur != target[ctr]) {
                result.add("Push");
                result.add("Pop");
                cur++;
            }
            result.add("Push");
            cur++;
            ctr++;
        }
        return result;
    }

    // LC847 **
    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        Deque<lc847State> q = new LinkedList<>();
        int[][] distance = new int[1 << n][n];
        for (int[] row : distance) Arrays.fill(row, n * n);
        int allMask = (1 << n) - 1;
        for (int i = 0; i < n; i++) {
            q.offer(new lc847State(1 << i, i));
            distance[1 << i][i] = 0;
        }
        while (!q.isEmpty()) {
            lc847State p = q.poll();
            int d = distance[p.mask][p.head];
            if (p.mask == allMask) return d;
            for (int child : graph[p.head]) {
                int newMask = p.mask | (1 << child);
                if (d + 1 < distance[newMask][child]) {
                    distance[newMask][child] = d + 1;
                    q.offer(new lc847State(newMask, child));
                }
            }
        }
        return Integer.MAX_VALUE;
    }

    class lc847State {
        int mask, head;

        public lc847State(int mask, int head) {
            this.mask = mask;
            this.head = head;
        }
    }

    // LC807
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] top = new int[n], left = new int[m];
        int result = 0;
        Arrays.fill(top, Integer.MIN_VALUE);
        Arrays.fill(left, Integer.MIN_VALUE);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                left[i] = Math.max(left[i], grid[i][j]);
                top[j] = Math.max(top[j], grid[i][j]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] < left[i] && grid[i][j] < top[j]) {
                    result += Math.min(left[i] - grid[i][j], top[j] - grid[i][j]);
                }
            }
        }
        return result;
    }

    // LC199
    public List<Integer> rightSideView(TreeNode8 root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Deque<TreeNode8> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qSize = q.size();
            for (int i = 1; i <= qSize; i++) {
                TreeNode8 p = q.poll();
                if (i == qSize) result.add(p.val);
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
        }
        return result;
    }

    // LC109
    public TreeNode8 sortedListToBST(ListNode37 head) {
        if (head == null) return null;
        if (head.next == null) return new TreeNode8(head.val);
        ListNode37 fast = head;
        ListNode37 slow = head;
        ListNode37 prev = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        prev.next = null;
        TreeNode8 root = new TreeNode8(slow.val);
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }

    // LC129
    long lc129Sum = 0;

    public int sumNumbers(TreeNode8 root) {
        lc129Helper(root, 0);
        return (int) lc129Sum;
    }

    private void lc129Helper(TreeNode8 root, long cur) {
        if (root == null) return;
        cur = cur * 10 + root.val;
        if (root.left == null && root.right == null) {
            lc129Sum += cur;
            return;
        }
        lc129Helper(root.left, cur);
        lc129Helper(root.right, cur);
    }

    // LC106
    public TreeNode8 buildTree(int[] inorder, int[] postorder) {
        // 后序遍历 根在最后
        if (inorder.length == 0) return null;
        TreeNode8 root = new TreeNode8(postorder[postorder.length - 1]);

        int i = 0;
        for (; i < inorder.length; i++) {
            if (inorder[i] == postorder[postorder.length - 1]) {
                break;
            }
        }
        // i 是 root在inorder中的位置, 所以i为左子树的长,
        root.left = buildTree(Arrays.copyOfRange(inorder, 0, i), Arrays.copyOfRange(postorder, 0, i));
        root.right = buildTree(Arrays.copyOfRange(inorder, i + 1, inorder.length), Arrays.copyOfRange(postorder, i, postorder.length - 1));
        return root;
    }

    // LC863 **
    Map<TreeNode8, Integer> lc863Distance;
    Map<TreeNode8, TreeNode8> lc863Parent;
    List<Integer> lc863Result;

    public List<Integer> distanceK(TreeNode8 root, TreeNode8 target, int k) {
        lc863Distance = new HashMap<>();
        lc863Result = new ArrayList<>();
        lc863Parent = new HashMap<>();
        Deque<TreeNode8> q = new LinkedList<>();
        q.offer(root);
        // BFS
        while (!q.isEmpty()) {
            TreeNode8 p = q.poll();
            if (p.left != null) {
                q.offer(p.left);
                lc863Parent.put(p.left, p);
            }
            if (p.right != null) {
                q.offer(p.right);
                lc863Parent.put(p.right, p);
            }
        }
        lc863Helper(target, 0, k, null);
        return lc863Result;
    }

    private void lc863Helper(TreeNode8 root, int curDistance, int targetDistance, TreeNode8 source) {
        if (root == null) return;
        lc863Distance.put(root, curDistance);
        if (curDistance == targetDistance) lc863Result.add(root.val);
        if (root.left != source)
            lc863Helper(root.left, curDistance + 1, targetDistance, root);
        if (root.right != source)
            lc863Helper(root.right, curDistance + 1, targetDistance, root);
        if (lc863Parent.get(root) != source)
            lc863Helper(lc863Parent.get(root), curDistance + 1, targetDistance, root);
    }


    // LC1334 Floyd 算法模板
    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        int[][] distance = new int[n][n];
        for (int i = 0; i < n; i++) Arrays.fill(distance[i], Integer.MAX_VALUE / 2);
        for (int[] e : edges) distance[e[0]][e[1]] = distance[e[1]][e[0]] = e[2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (distance[j][i] != Integer.MAX_VALUE / 2 && distance[i][k] != Integer.MAX_VALUE / 2) {
                        distance[k][j] = distance[j][k] = Math.min(distance[j][k], distance[j][i] + distance[i][k]);
                    }
                }
            }
        }
        int minFreq = Integer.MAX_VALUE / 2;
        int minIdx = -1;
        for (int i = 0; i < n; i++) {
            int freq = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && distance[i][j] != Integer.MAX_VALUE / 2 && distance[i][j] <= distanceThreshold) {
                    freq++;
                }
            }
            if (freq <= minFreq) {
                minFreq = freq;
                minIdx = i;
            }
        }
        return minIdx;
    }

    // LC187
    public List<String> findRepeatedDnaSequences(String s) {
        if (s.length() <= 10) return new ArrayList<>();
        // A C G T 00 01 10 11 2*10=20 < 32
        int hash = 0;
        int allMask = (1 << 20) - 1;
        int[] alphabet = new int[26];
        char[] reverseAlphabet = {'A', 'C', 'G', 'T'};
        char[] cArr = s.toCharArray();
        Set<Integer> set = new HashSet<>();
        Set<Integer> result = new HashSet<>();
        List<String> strResult;
        alphabet['A' - 'A'] = 0b00;
        alphabet['C' - 'A'] = 0b01;
        alphabet['G' - 'A'] = 0b10;
        alphabet['T' - 'A'] = 0b11;
        for (int i = 0; i < 10; i++) {
            hash = (hash << 2) | alphabet[cArr[i] - 'A'];
        }
        set.add(hash);
        for (int i = 10; i < cArr.length; i++) {
            hash = allMask & ((hash << 2) | alphabet[cArr[i] - 'A']);
            if (set.contains(hash)) result.add(hash);
            set.add(hash);
        }
        int twoBitMask = 0b11;
        strResult = new ArrayList<>(result.size());
        for (int i : result) {
            StringBuilder sb = new StringBuilder();
            for (int j = 1; j <= 10; j++) {
                sb.append(reverseAlphabet[twoBitMask & (i >> (20 - j * 2))]);
            }
            strResult.add(sb.toString());
        }
        return strResult;
    }

    // LC997
    public int findJudge(int n, int[][] trust) {
        int[] trustFrom = new int[n + 1];
        int[] trustTo = new int[n + 1];
        for (int[] t : trust) {
            trustFrom[t[0]]++;
            trustTo[t[1]]++;
        }
        for (int i = 1; i <= n; i++) {
            if (trustFrom[i] == 0 && trustTo[i] == n - 1) return i;
        }
        return -1;
    }

    // Interview 04.10
    public boolean checkSubTree(TreeNode8 t1, TreeNode8 t2) {
        Deque<TreeNode8> q = new LinkedList<>();
        q.offer(t1);
        while (!q.isEmpty()) {
            TreeNode8 p = q.poll();
            if (checkSubTreeHelper(p, t2)) {
                return true;
            }
            if (p.left != null) q.offer(p.left);
            if (p.right != null) q.offer(p.right);
        }
        return false;
    }

    private boolean checkSubTreeHelper(TreeNode8 root, TreeNode8 target) {
        if (root == null && target == null) return true;
        else if (root == null || target == null) return false;
        if (root.val != target.val) {
            return false;
        }
        return checkSubTreeHelper(root.left, target.left) && checkSubTreeHelper(root.right, target.right);
    }

    // LC1948 **
    static class Lc1948 {
        class Node {
            String serial = "";
            // 用TreeMap保证序列化后的出现顺序唯一
            TreeMap<String, Node> children = new TreeMap<>();
        }

        public List<List<String>> deleteDuplicateFolder(List<List<String>> paths) {
            Node root = new Node();
            Map<Node, List<String>> nodeListStringMap = new HashMap<>();
            List<List<String>> result = new LinkedList<>();
            for (List<String> p : paths) {
                Node cur = root;
                for (int i = 0; i < p.size(); i++) {
                    cur.children.putIfAbsent(p.get(i), new Node());
                    cur = cur.children.get(p.get(i));
                }
                nodeListStringMap.put(cur, p);
            }
            getSerial(root);
            // BFS
            Deque<Node> q = new LinkedList<>();
            Map<String, Integer> serialCountMap = new HashMap<>();
            q.offer(root);
            while (!q.isEmpty()) {
                Node p = q.poll();
                serialCountMap.put(p.serial, serialCountMap.getOrDefault(p.serial, 0) + 1);
                for (String s : p.children.keySet()) {
                    q.offer(p.children.get(s));
                }
            }
            // DFS
            Deque<Node> stack = new LinkedList<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                Node p = stack.pop();
                if (!p.serial.equals("()") && serialCountMap.get(p.serial) > 1) {
                    continue;
                }
                if (nodeListStringMap.containsKey(p)) {
                    result.add(nodeListStringMap.get(p));
                }
                for (String s : p.children.keySet()) {
                    stack.push(p.children.get(s));
                }
            }
            return result;
        }

        private void getSerial(Node node) {
            if (node.children.size() == 0) {
                node.serial = "()";
                return;
            }
            for (String s : node.children.keySet()) {
                getSerial(node.children.get(s));
            }
            for (String s : node.children.keySet()) {
                node.serial += "(" + s + node.children.get(s).serial + ")";
            }
        }

    }

    // LC1943 **
    public List<List<Long>> splitPainting(int[][] segments) {
        Map<Integer, Long> diff = new HashMap<>();
        for (int[] intv : segments) {
            diff.putIfAbsent(intv[0], 0L);
            diff.putIfAbsent(intv[1], 0L);
            diff.put(intv[0], diff.get(intv[0]) + intv[2]);
            diff.put(intv[1], diff.get(intv[1]) - intv[2]);
        }
        List<Integer> keyArr = new ArrayList<>(diff.keySet());
        Collections.sort(keyArr);
        long prev = 0, accumulative = 0;
        List<List<Long>> result = new ArrayList<>();
        for (int i : keyArr) {
            if (accumulative != 0) result.add(Arrays.asList(prev, (long) i, accumulative));
            accumulative += diff.get(i);
            prev = i;
        }
        return result;
    }

    // LC657
    public boolean judgeCircle(String moves) {
        int[] count = new int[2];
        for (char c : moves.toCharArray()) {
            if (c == 'U') count[0]++;
            else if (c == 'D') count[0]--;
            else if (c == 'L') count[1]++;
            else if (c == 'R') count[1]--;
        }
        return count[0] == 0 && count[1] == 0;
    }

    // LCP 01
    public int game(int[] guess, int[] answer) {
        int result = 0;
        for (int i = 0; i < 3; i++) if (guess[i] == answer[i]) result++;
        return result;
    }

    // LC1487
    public String[] getFolderNames(String[] names) {
        Map<String, Integer> m = new HashMap<>();
        int n = names.length;
        String[] result = new String[n];
        for (int i = 0; i < n; i++) {
            if (m.containsKey(names[i])) {
                int count = m.get(names[i]);
                while (m.containsKey(names[i] + '(' + count + ')')) {
                    count++;
                }
                result[i] = names[i] + '(' + count + ')';

                m.put(names[i], count + 1);
                m.put(result[i], 1);
            } else {
                result[i] = names[i];
                m.put(names[i], 1);
            }
        }
        return result;
    }

    // LC671
    TreeSet<Integer> lc671Ts;

    public int findSecondMinimumValue(TreeNode8 root) {
        lc671Ts = new TreeSet<>();
        lc671Helper(root);
        if (lc671Ts.size() <= 1) return -1;
        return lc671Ts.last();
    }

    private void lc671Helper(TreeNode8 root) {
        if (root == null) return;
        if (lc671Ts.contains(root.val)) {
            ;
        } else if (lc671Ts.size() < 2) {
            lc671Ts.add(root.val);
        } else {
            if (root.val < lc671Ts.last()) {
                lc671Ts.remove(lc671Ts.last());
                lc671Ts.add(root.val);
            }
        }
        lc671Helper(root.left);
        lc671Helper(root.right);
    }

    // LC1713 **
    public int minOperations(int[] target, int[] arr) {
        Map<Integer, Integer> targetValIdxMap = new HashMap<>();
        for (int i = 0; i < target.length; i++) targetValIdxMap.put(target[i], i);
        List<Integer> arrList = new ArrayList<>(arr.length);
        for (int i = 0; i < arr.length; i++) {
            if (targetValIdxMap.containsKey(arr[i])) arrList.add(targetValIdxMap.get(arr[i]));
        }
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : arrList) {
            Integer ceiling = ts.ceiling(i);
            if (ceiling != null) {
                ts.remove(ceiling);
            }
            ts.add(i);
        }
        return target.length - ts.size();
    }

    // LC1743
    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, List<Integer>> m = new HashMap<>();
        for (int[] p : adjacentPairs) {
            m.putIfAbsent(p[0], new ArrayList<>(2));
            m.putIfAbsent(p[1], new ArrayList<>(2));
            m.get(p[0]).add(p[1]);
            m.get(p[1]).add(p[0]);
        }

        int end = -1;
        for (int i : m.keySet()) {
            if (m.get(i).size() == 1) {
                end = i;
                break;
            }
        }
        int[] result = new int[adjacentPairs.length + 1];
        result[0] = end;
        result[1] = m.get(result[0]).get(0);
        for (int i = 2; i < result.length; i++) {
            List<Integer> prevAdj = m.get(result[i - 1]);
            result[i] = result[i - 2] == prevAdj.get(0) ? prevAdj.get(1) : prevAdj.get(0);
        }
        return result;
    }

    // Interview 16.19
    public int[] pondSizes(int[][] land) {
        int[][] directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        int totalRow = land.length, totalCol = land[0].length;
        DisjointSetUnion48 dsu = new DisjointSetUnion48();
        for (int i = 0; i < totalRow; i++) {
            for (int j = 0; j < totalCol; j++) {
                int curId = getMatrixId(i, j, totalRow, totalCol);
                if (land[i][j] == 0) {
                    dsu.add(curId);
                    for (int[] dir : directions) {
                        int targetRow = i + dir[0];
                        int targetCol = j + dir[1];
                        if (checkPoint(targetRow, targetCol, totalRow, totalCol) && land[targetRow][targetCol] == 0) {
                            dsu.add(getMatrixId(targetRow, targetCol, totalRow, totalCol));
                            dsu.merge(curId, getMatrixId(targetRow, targetCol, totalRow, totalCol));
                        }
                    }
                }
            }
        }
        Map<Integer, Set<Integer>> groups = dsu.allGroups();
        List<Integer> result = new ArrayList<>(groups.size());
        for (int i : groups.keySet()) {
            result.add(groups.get(i).size());
        }
        Collections.sort(result);
        int[] res = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            res[i] = result.get(i);
        }
        return res;
    }

    private boolean checkPoint(int targetRow, int targetCol, int totalRow, int totalCol) {
        return !(targetRow >= totalRow || targetRow < 0 || targetCol >= totalCol || targetCol < 0);
    }

    private int getMatrixId(int targetRow, int targetCol, int totalRow, int totalCol) {
        return targetRow * totalCol + targetCol;
    }

    // LC500
    public String[] findWords(String[] words) {
        List<String> result = new ArrayList<>(words.length);
        String[] kb = new String[]{"qwertyuiop", "asdfghjkl", "zxcvbnm"};
        boolean[][] kbCheck = new boolean[3][26];
        int kbCtr = 0;
        for (String s : kb) {
            for (char c : s.toCharArray()) {
                kbCheck[kbCtr][c - 'a'] = true;
            }
            kbCtr++;
        }
        for (String word : words) {
            boolean flag = false;
            for (int i = 0; i < 3; i++) {
                boolean flag2 = true;
                for (char c : word.toLowerCase().toCharArray()) {
                    if (!kbCheck[i][c - 'a']) {
                        flag2 = false;
                        break;
                    }
                }
                if (flag2) {
                    result.add(word);
                    break;
                }
            }
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1893
    public boolean isCovered(int[][] ranges, int left, int right) {
        boolean[] check = new boolean[51];
        int min = 51, max = -1;
        for (int[] r : ranges) {
            min = Math.min(min, r[0]);
            max = Math.max(max, r[1]);
            for (int i = r[0]; i <= r[1]; i++) {
                check[i] = true;
            }
        }
        if (left < min || right > max) return false;
        for (int i = left; i <= right; i++) if (!check[i]) return false;
        return true;
    }

    // LC713
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int left = 0, prod = 1, result = 0, n = nums.length;
        for (int right = 0; right < n; right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            result += right - left + 1;
        }
        return result;
    }

    // LC209
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        int left = 0, right = 0;
        int min = Integer.MAX_VALUE;
        while (left <= right && right != n) {
            if (prefix[right + 1] - prefix[left] < target) {
                right++;
                continue;
            } else {
                if (right - left + 1 < min) {
                    min = right - left + 1;
                }
                left++;
            }
        }
        if (min == Integer.MAX_VALUE) return 0;
        return min;
    }

    // LC5
    public String longestPalindrome(String s) {
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        boolean[][] dp = new boolean[n][n];
        // dp[i][j] 表示 [i,j]是不是回文串
        // dp[i][j] = true iff dp[i+1][j-1]==true && cArr[i]==cArr[j]
        for (int i = 0; i < n; i++) dp[i][i] = true;
        int max = 1;
        int maxLeft = 0, maxRight = 0;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left < n; left++) {
                int right = left + len - 1;
                if (right >= n) break;
                if (cArr[left] == cArr[right]) {
                    if (len == 2) dp[left][right] = true;
                    else {
                        dp[left][right] = dp[left + 1][right - 1];
                    }
                }
                if (len > max && dp[left][right]) {
                    maxLeft = left;
                    maxRight = right;
                }
            }
        }
        return s.substring(maxLeft, maxRight + 1);
    }

    // LC3
    public int lengthOfLongestSubstring(String s) {
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        Map<Character, Integer> lastOccur = new HashMap<>();
        int left = 0, right = 0;
        int max = 0;
        while (right != n) {
            if (lastOccur.containsKey(cArr[right])) {
                int end = lastOccur.get(cArr[right]);
                for (int i = left; i <= end; i++) {
                    lastOccur.remove(cArr[i]);
                }
                left = end + 1;
            }
            lastOccur.put(cArr[right], right);
            max = Math.max(max, right - left + 1);
            right++;
        }
        return max;
    }

    // LC720 You sure this is EASY???
    int lc720MaxLen = 0;
    String lc720Result = "";
    StringBuilder lc720Sb = new StringBuilder();

    public String longestWord(String[] words) {
        Trie48 trie = new Trie48();
        for (String word : words) trie.addWord(word);
        TrieNode48 root = trie.root;
        lc720Helper(root);
        return lc720Result;
    }

    private void lc720Helper(TrieNode48 root) {
        for (char c : root.children.keySet()) {
            if (root.children.get(c).isEnd) {
                lc720Sb.append(c);
                if (lc720Sb.length() > lc720MaxLen) {
                    lc720MaxLen = lc720Sb.length();
                    lc720Result = lc720Sb.toString();
                } else if (lc720Sb.length() == lc720MaxLen && lc720Sb.toString().compareTo(lc720Result) < 0) {
                    lc720Result = lc720Sb.toString();
                }
                lc720Helper(root.children.get(c));
                lc720Sb.deleteCharAt(lc720Sb.length() - 1);
            }
        }
    }

    // LC968 **
    enum lc968Status {
        NO_NEED,
        NEED,
        HAS_CAMERA
    }

    int lc968Result;

    public int minCameraCover(TreeNode8 root) {
        lc968Result = 0;
        if (lc968Helper(root) == lc968Status.NEED) lc968Result++;
        return lc968Result;
    }

    private lc968Status lc968Helper(TreeNode8 root) {
        if (root == null) {
            return lc968Status.NO_NEED;
        }
        lc968Status left = lc968Helper(root.left), right = lc968Helper(root.right);
        // 如果子结点中有一个需要相机, 则该节点需要放置相机, 否则子节点会不被覆盖
        if (left == lc968Status.NEED || right == lc968Status.NEED) {
            lc968Result++;
            return lc968Status.HAS_CAMERA;
        }
        // 如果子节点中有一个拥有相机, 则父节点被覆盖, 不需要相机, 否则需要相机
        return (left == lc968Status.HAS_CAMERA || right == lc968Status.HAS_CAMERA) ? lc968Status.NO_NEED : lc968Status.NEED;
    }

    // LC866
    public int primePalindrome(int n) {
        // 前 663,961 个素数的回文素数表, 借用nthPrime打出来, 超过这个数就耗时太长了
        Integer[] table = {2, 3, 5, 7, 11, 101, 131, 151, 181, 191, 313, 353, 373, 383, 727, 757, 787, 797, 919, 929, 10301, 10501, 10601, 11311, 11411, 12421, 12721, 12821, 13331, 13831, 13931, 14341, 14741, 15451, 15551, 16061, 16361, 16561, 16661, 17471, 17971, 18181, 18481, 19391, 19891, 19991, 30103, 30203, 30403, 30703, 30803, 31013, 31513, 32323, 32423, 33533, 34543, 34843, 35053, 35153, 35353, 35753, 36263, 36563, 37273, 37573, 38083, 38183, 38783, 39293, 70207, 70507, 70607, 71317, 71917, 72227, 72727, 73037, 73237, 73637, 74047, 74747, 75557, 76367, 76667, 77377, 77477, 77977, 78487, 78787, 78887, 79397, 79697, 79997, 90709, 91019, 93139, 93239, 93739, 94049, 94349, 94649, 94849, 94949, 95959, 96269, 96469, 96769, 97379, 97579, 97879, 98389, 98689, 1003001, 1008001, 1022201, 1028201, 1035301, 1043401, 1055501, 1062601, 1065601, 1074701, 1082801, 1085801, 1092901, 1093901, 1114111, 1117111, 1120211, 1123211, 1126211, 1129211, 1134311, 1145411, 1150511, 1153511, 1160611, 1163611, 1175711, 1177711, 1178711, 1180811, 1183811, 1186811, 1190911, 1193911, 1196911, 1201021, 1208021, 1212121, 1215121, 1218121, 1221221, 1235321, 1242421, 1243421, 1245421, 1250521, 1253521, 1257521, 1262621, 1268621, 1273721, 1276721, 1278721, 1280821, 1281821, 1286821, 1287821, 1300031, 1303031, 1311131, 1317131, 1327231, 1328231, 1333331, 1335331, 1338331, 1343431, 1360631, 1362631, 1363631, 1371731, 1374731, 1390931, 1407041, 1409041, 1411141, 1412141, 1422241, 1437341, 1444441, 1447441, 1452541, 1456541, 1461641, 1463641, 1464641, 1469641, 1486841, 1489841, 1490941, 1496941, 1508051, 1513151, 1520251, 1532351, 1535351, 1542451, 1548451, 1550551, 1551551, 1556551, 1557551, 1565651, 1572751, 1579751, 1580851, 1583851, 1589851, 1594951, 1597951, 1598951, 1600061, 1609061, 1611161, 1616161, 1628261, 1630361, 1633361, 1640461, 1643461, 1646461, 1654561, 1657561, 1658561, 1660661, 1670761, 1684861, 1685861, 1688861, 1695961, 1703071, 1707071, 1712171, 1714171, 1730371, 1734371, 1737371, 1748471, 1755571, 1761671, 1764671, 1777771, 1793971, 1802081, 1805081, 1820281, 1823281, 1824281, 1826281, 1829281, 1831381, 1832381, 1842481, 1851581, 1853581, 1856581, 1865681, 1876781, 1878781, 1879781, 1880881, 1881881, 1883881, 1884881, 1895981, 1903091, 1908091, 1909091, 1917191, 1924291, 1930391, 1936391, 1941491, 1951591, 1952591, 1957591, 1958591, 1963691, 1968691, 1969691, 1970791, 1976791, 1981891, 1982891, 1984891, 1987891, 1988891, 1993991, 1995991, 1998991, 3001003, 3002003, 3007003, 3016103, 3026203, 3064603, 3065603, 3072703, 3073703, 3075703, 3083803, 3089803, 3091903, 3095903, 3103013, 3106013, 3127213, 3135313, 3140413, 3155513, 3158513, 3160613, 3166613, 3181813, 3187813, 3193913, 3196913, 3198913, 3211123, 3212123, 3218123, 3222223, 3223223, 3228223, 3233323, 3236323, 3241423, 3245423, 3252523, 3256523, 3258523, 3260623, 3267623, 3272723, 3283823, 3285823, 3286823, 3288823, 3291923, 3293923, 3304033, 3305033, 3307033, 3310133, 3315133, 3319133, 3321233, 3329233, 3331333, 3337333, 3343433, 3353533, 3362633, 3364633, 3365633, 3368633, 3380833, 3391933, 3392933, 3400043, 3411143, 3417143, 3424243, 3425243, 3427243, 3439343, 3441443, 3443443, 3444443, 3447443, 3449443, 3452543, 3460643, 3466643, 3470743, 3479743, 3485843, 3487843, 3503053, 3515153, 3517153, 3528253, 3541453, 3553553, 3558553, 3563653, 3569653, 3586853, 3589853, 3590953, 3591953, 3594953, 3601063, 3607063, 3618163, 3621263, 3627263, 3635363, 3643463, 3646463, 3670763, 3673763, 3680863, 3689863, 3698963, 3708073, 3709073, 3716173, 3717173, 3721273, 3722273, 3728273, 3732373, 3743473, 3746473, 3762673, 3763673, 3765673, 3768673, 3769673, 3773773, 3774773, 3781873, 3784873, 3792973, 3793973, 3799973, 3804083, 3806083, 3812183, 3814183, 3826283, 3829283, 3836383, 3842483, 3853583, 3858583, 3863683, 3864683, 3867683, 3869683, 3871783, 3878783, 3893983, 3899983, 3913193, 3916193, 3918193, 3924293, 3927293, 3931393, 3938393, 3942493, 3946493, 3948493, 3964693, 3970793, 3983893, 3991993, 3994993, 3997993, 3998993, 7014107, 7035307, 7036307, 7041407, 7046407, 7057507, 7065607, 7069607, 7073707, 7079707, 7082807, 7084807, 7087807, 7093907, 7096907, 7100017, 7114117, 7115117, 7118117, 7129217, 7134317, 7136317, 7141417, 7145417, 7155517, 7156517, 7158517, 7159517, 7177717, 7190917, 7194917, 7215127, 7226227, 7246427, 7249427, 7250527, 7256527, 7257527, 7261627, 7267627, 7276727, 7278727, 7291927, 7300037, 7302037, 7310137, 7314137, 7324237, 7327237, 7347437, 7352537, 7354537, 7362637, 7365637, 7381837, 7388837, 7392937, 7401047, 7403047, 7409047, 7415147, 7434347, 7436347, 7439347, 7452547, 7461647, 7466647, 7472747, 7475747, 7485847, 7486847, 7489847, 7493947, 7507057, 7508057, 7518157, 7519157, 7521257, 7527257, 7540457, 7562657, 7564657, 7576757, 7586857, 7592957, 7594957, 7600067, 7611167, 7619167, 7622267, 7630367, 7632367, 7644467, 7654567, 7662667, 7665667, 7666667, 7668667, 7669667, 7674767, 7681867, 7690967, 7693967, 7696967, 7715177, 7718177, 7722277, 7729277, 7733377, 7742477, 7747477, 7750577, 7758577, 7764677, 7772777, 7774777, 7778777, 7782877, 7783877, 7791977, 7794977, 7807087, 7819187, 7820287, 7821287, 7831387, 7832387, 7838387, 7843487, 7850587, 7856587, 7865687, 7867687, 7868687, 7873787, 7884887, 7891987, 7897987, 7913197, 7916197, 7930397, 7933397, 7935397, 7938397, 7941497, 7943497, 7949497, 7957597, 7958597, 7960697, 7977797, 7984897, 7985897, 7987897, 7996997, 9002009, 9015109, 9024209, 9037309, 9042409, 9043409, 9045409, 9046409, 9049409, 9067609, 9073709, 9076709, 9078709, 9091909, 9095909, 9103019, 9109019, 9110119, 9127219, 9128219, 9136319, 9149419, 9169619, 9173719, 9174719, 9179719, 9185819, 9196919, 9199919, 9200029, 9209029, 9212129, 9217129, 9222229, 9223229, 9230329, 9231329, 9255529, 9269629, 9271729, 9277729, 9280829, 9286829, 9289829, 9318139, 9320239, 9324239, 9329239, 9332339, 9338339, 9351539, 9357539, 9375739, 9384839, 9397939, 9400049, 9414149, 9419149, 9433349, 9439349, 9440449, 9446449, 9451549, 9470749, 9477749, 9492949, 9493949, 9495949, 9504059, 9514159, 9526259, 9529259, 9547459, 9556559, 9558559, 9561659, 9577759, 9583859, 9585859, 9586859, 9601069, 9602069, 9604069, 9610169, 9620269, 9624269, 9626269, 9632369, 9634369, 9645469, 9650569, 9657569, 9670769, 9686869, 9700079, 9709079, 9711179, 9714179, 9724279, 9727279, 9732379, 9733379, 9743479, 9749479, 9752579, 9754579, 9758579, 9762679, 9770779, 9776779, 9779779, 9781879, 9782879, 9787879, 9788879, 9795979, 9801089, 9807089, 9809089, 9817189, 9818189, 9820289, 9822289, 9836389, 9837389, 9845489, 9852589, 9871789, 9888889, 9889889, 9896989, 9902099, 9907099, 9908099, 9916199, 9918199, 9919199, 9921299, 9923299, 9926299, 9927299, 9931399, 9932399, 9935399, 9938399, 9957599, 9965699, 9978799, 9980899, 9981899, 9989899, 100030001};
        TreeSet<Integer> ts = new TreeSet<>(Arrays.asList(table));
        return ts.ceiling(n);
    }

    private void printPrimePalindromeFrom1e7to2e8() {
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973, 10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211, 10223, 10243, 10247, 10253, 10259, 10267, 10271, 10273, 10289, 10301, 10303, 10313, 10321, 10331, 10333, 10337, 10343, 10357, 10369, 10391, 10399, 10427, 10429, 10433, 10453, 10457, 10459, 10463, 10477, 10487, 10499, 10501, 10513, 10529, 10531, 10559, 10567, 10589, 10597, 10601, 10607, 10613, 10627, 10631, 10639, 10651, 10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729, 10733, 10739, 10753, 10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853, 10859, 10861, 10867, 10883, 10889, 10891, 10903, 10909, 10937, 10939, 10949, 10957, 10973, 10979, 10987, 10993, 11003, 11027, 11047, 11057, 11059, 11069, 11071, 11083, 11087, 11093, 11113, 11117, 11119, 11131, 11149, 11159, 11161, 11171, 11173, 11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261, 11273, 11279, 11287, 11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383, 11393, 11399, 11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489, 11491, 11497, 11503, 11519, 11527, 11549, 11551, 11579, 11587, 11593, 11597, 11617, 11621, 11633, 11657, 11677, 11681, 11689, 11699, 11701, 11717, 11719, 11731, 11743, 11777, 11779, 11783, 11789, 11801, 11807, 11813, 11821, 11827, 11831, 11833, 11839, 11863, 11867, 11887, 11897, 11903, 11909, 11923, 11927, 11933, 11939, 11941, 11953, 11959, 11969, 11971, 11981, 11987, 12007, 12011, 12037, 12041, 12043, 12049, 12071, 12073, 12097, 12101, 12107, 12109, 12113, 12119, 12143, 12149, 12157, 12161, 12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251, 12253, 12263, 12269, 12277, 12281, 12289, 12301, 12323, 12329, 12343, 12347, 12373, 12377, 12379, 12391, 12401, 12409, 12413, 12421, 12433, 12437, 12451, 12457, 12473, 12479, 12487, 12491, 12497, 12503, 12511, 12517, 12527, 12539, 12541, 12547, 12553, 12569, 12577, 12583, 12589, 12601, 12611, 12613, 12619, 12637, 12641, 12647, 12653, 12659, 12671, 12689, 12697, 12703, 12713, 12721, 12739, 12743, 12757, 12763, 12781, 12791, 12799, 12809, 12821, 12823, 12829, 12841, 12853, 12889, 12893, 12899, 12907, 12911, 12917, 12919, 12923, 12941, 12953, 12959, 12967, 12973, 12979, 12983, 13001, 13003, 13007, 13009, 13033, 13037, 13043, 13049, 13063, 13093, 13099, 13103, 13109, 13121, 13127, 13147, 13151, 13159, 13163, 13171, 13177, 13183, 13187, 13217, 13219, 13229, 13241, 13249, 13259, 13267, 13291, 13297, 13309, 13313, 13327, 13331, 13337, 13339, 13367, 13381, 13397, 13399, 13411, 13417, 13421, 13441, 13451, 13457, 13463, 13469, 13477, 13487, 13499, 13513, 13523, 13537, 13553, 13567, 13577, 13591, 13597, 13613, 13619, 13627, 13633, 13649, 13669, 13679, 13681, 13687, 13691, 13693, 13697, 13709, 13711, 13721, 13723, 13729, 13751, 13757, 13759, 13763, 13781, 13789, 13799, 13807, 13829, 13831, 13841, 13859, 13873, 13877, 13879, 13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933, 13963, 13967, 13997, 13999, 14009, 14011, 14029, 14033, 14051, 14057, 14071, 14081, 14083, 14087, 14107};
        // 回文根
        int proot = 1000;
        for (; proot <= 2000; proot++) {
            // 事实: 不存在偶数位数的回文素数, 即所有回文素数都是奇数位数的, 所以在回文根1000-9999的8位回文整数中不存在回文素数
            // 为此, 在两4位回文根中间插入0~9, 构成9位回文素数
            for (int j = 0; j < 10; j++) {
                String spr = String.valueOf(proot);
                spr = spr + String.valueOf(j) + new StringBuilder(spr).reverse().toString();
                int p = Integer.valueOf(spr);
                boolean flag = true;
                for (int i : primes) {
                    if (p % i == 0) {
                        flag = false;
                        break;
                    }
                }
                if (flag) System.out.println(p);
            }
        }

    }

    // LC846 LC1296
    public boolean isNStraightHand(int[] hand, int groupSize) {
        if (hand.length % groupSize != 0) return false;
        TreeMap<Integer, Integer> m = new TreeMap<>();
        for (int i : hand) m.put(i, m.getOrDefault(i, 0) + 1);
        while (!m.isEmpty()) {
            int firstKey = m.firstKey();
            for (int i = 0; i < groupSize; i++) {
                if (!m.containsKey(firstKey + i)) {
                    return false;
                }
                m.put(firstKey + i, m.get(firstKey + i) - 1);
                if (m.get(firstKey + i) == 0) m.remove(firstKey + i);
            }
        }
        return true;
    }

    // LC1302
    public int deepestLeavesSum(TreeNode8 root) {
        Deque<TreeNode8> q = new LinkedList<>();
        q.offer(root);
        List<TreeNode8> thisLayer = new LinkedList<>();
        while (!q.isEmpty()) {
            thisLayer.clear();
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                TreeNode8 p = q.poll();
                thisLayer.add(p);
                if (p.left != null) {
                    q.offer(p.left);
                }
                if (p.right != null) {
                    q.offer(p.right);
                }
            }
        }
        int sum = 0;
        for (TreeNode8 t : thisLayer) {
            sum += t.val;
        }
        return sum;
    }

    // LC138
    class LC138 {

        public Node copyRandomList(Node head) {
            Node dummy = new Node(-1);
            dummy.next = head;
            Node cur = head;
            Map<Node, Node> m = new HashMap<>();
            m.put(null, null);
            while (cur != null) {
                Node t = new Node(cur.val);
                m.put(cur, t);
                cur = cur.next;
            }

            cur = head;
            while (cur != null) {
                m.get(cur).next = m.get(cur.next);
                m.get(cur).random = m.get(cur.random);
                cur = cur.next;
            }

            return m.get(head);
        }

        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }

    }

    // LC1051
    public int heightChecker(int[] heights) {
        int[] freq = new int[101];
        for (int i : heights) freq[i]++;
        int result = 0;
        for (int i = 1, j = 0; i <= 100; i++) {
            while (freq[i]-- != 0) {
                if (heights[j++] != i) result++;
            }
        }
        return result;
    }

    // LC805 **
    public boolean splitArraySameAverage(int[] nums) {
        Arrays.sort(nums); // 避免出现{18,0,16,2}的被hack情况
        int n = nums.length;
        long sum = 0;
        for (int i : nums) sum += i;
        long[] arr = new long[n];
        for (int i = 0; i < n; i++) {
            arr[i] = ((long) nums[i]) * ((long) n) - sum;
        }
        long[] left = new long[n / 2];
        long[] right = new long[n - left.length];
        for (int i = 0; i < left.length; i++) {
            left[i] = arr[i];
        }
        for (int i = left.length; i < n; i++) {
            right[i - left.length] = arr[i];
        }

        // 找0
        Map<Integer, Set<Integer>> leftSumMaskMap = new HashMap<>();
        for (int subset = (1 << left.length) - 1; subset != 0; subset--) {
            int tmp = 0;
            for (int i = 0; i < left.length; i++) {
                if (((subset >> i) & 1) == 1) {
                    tmp += arr[i];
                }
            }
            leftSumMaskMap.putIfAbsent(tmp, new HashSet<>());
            leftSumMaskMap.get(tmp).add(subset);
        }
        for (int subset = (1 << right.length) - 1; subset != 0; subset--) {
            int tmp = 0;
            for (int i = 0; i < right.length; i++) {
                if (((subset >> i) & 1) == 1) {
                    tmp += arr[i + left.length];
                }
            }
            if (leftSumMaskMap.containsKey(-tmp)) {
                for (int mask : leftSumMaskMap.get(-tmp)) {
                    if (Integer.bitCount(mask) + Integer.bitCount(subset) != n) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // LC1799 **
    int lc1799Result;
    int[][] lc1799GcdCache;
    Integer[] lc1799Memo;

    public int maxScore(int[] nums) {
        int n = nums.length / 2;
        int allMask = (1 << (2 * n)) - 1;
        lc1799Result = (1 + n) * n / 2;
        lc1799GcdCache = new int[n * 2][n * 2];
        lc1799Memo = new Integer[1 << (nums.length)];
        for (int i = 0; i < n * 2; i++) {
            for (int j = i + 1; j < n * 2; j++) {
                lc1799GcdCache[i][j] = lc1799GcdCache[j][i] = gcd(nums[i], nums[j]);
            }
        }
        return lc1799Helper(nums, 0, allMask);
    }

    // 注意DFS应该携带什么信息, 不应该携带什么信息, 不要把当前状态(比如这里的score)放进函数入参, 而应该动态计算, 动态更新 (见highlight)
    private int lc1799Helper(int[] nums, int curMask, int allMask) {
        if (lc1799Memo[curMask] != null) return lc1799Memo[curMask];
        lc1799Memo[curMask] = 0;
        int selectable = allMask ^ curMask;
        for (int subset = selectable; subset != 0; subset = (subset - 1) & selectable) {
            if (Integer.bitCount(subset) == 2) {
                int[] select = new int[2];
                int ctr = 0;
                for (int i = 0; i < nums.length; i++) {
                    if (((subset >> i) & 1) == 1) {
                        select[ctr++] = i;
                    }
                    if (ctr == 2) break;
                }
                int newMask = subset ^ curMask;
                lc1799Memo[curMask] = Math.max(lc1799Memo[curMask],
                        lc1799Helper(nums, newMask, allMask) + lc1799GcdCache[select[0]][select[1]] * Integer.bitCount(newMask) / 2); // Highlight
            }
        }
        return lc1799Memo[curMask];
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC844
    public boolean backspaceCompare(String s, String t) {
        Deque<Character> ss = new LinkedList<>();
        Deque<Character> ts = new LinkedList<>();
        for (char c : s.toCharArray()) {
            if (c != '#') {
                ss.push(c);
            } else {
                if (!ss.isEmpty()) ss.pop();
            }
        }
        for (char c : t.toCharArray()) {
            if (c != '#') {
                ts.push(c);
            } else {
                if (!ts.isEmpty()) ts.pop();
            }
        }
        if (ss.size() != ts.size()) return false;
        while (!ss.isEmpty()) {
            if (ss.pollLast() != ts.pollLast()) return false;
        }
        return true;
    }

    // LC650 **
    public int minSteps(int n) {
        if (n == 1) return 0;
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                if (i % j == 0) {
                    dp[i] = dp[j] + dp[i / j];
                }
            }
        }
        return dp[n];
    }

    // LC1255
    int lc1255Max;

    public int maxScoreWords(String[] words, char[] letters, int[] score) {
        lc1255Max = 0;
        int[] usable = new int[26];
        for (char c : letters) {
            usable[c - 'a']++;
        }
        boolean[] canAdd = new boolean[words.length];
        for (int i = 0; i < words.length; i++) {
            int[] tmp = new int[26];
            System.arraycopy(usable, 0, tmp, 0, 26);
            boolean flag = true;
            for (char c : words[i].toCharArray()) {
                tmp[c - 'a']--;
                if (tmp[c - 'a'] < 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) canAdd[i] = true;
        }
        List<String> addableWords = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (canAdd[i]) addableWords.add(words[i]);
        }
        int[] addableScores = new int[addableWords.size()];
        for (int i = 0; i < addableWords.size(); i++) {
            for (char c : addableWords.get(i).toCharArray()) {
                addableScores[i] += score[c - 'a'];
            }
        }

        lc1255Backtrack(0, 0, usable, addableWords, addableScores);

        return lc1255Max;
    }

    private void lc1255Backtrack(int curIdx, int curScore, int[] curUsable, List<String> addableWords,
                                 int[] addableScores) {
        if (curIdx == addableWords.size()) {
            lc1255Max = Math.max(lc1255Max, curScore);
            return;
        }
        for (int i = curIdx; i < addableWords.size(); i++) {
            subCurWordFreq(curUsable, addableWords, i);
            if (!isCanUse(curUsable)) {
                addBackCurWordFreq(curUsable, addableWords, i);
                lc1255Backtrack(i + 1, curScore, curUsable, addableWords, addableScores);
            } else {
                lc1255Backtrack(i + 1, curScore + addableScores[i], curUsable, addableWords, addableScores);
                addBackCurWordFreq(curUsable, addableWords, i);
            }
        }
    }

    private boolean isCanUse(int[] curUsable) {
        for (int j = 0; j < 26; j++) {
            if (curUsable[j] < 0) {
                return false;
            }
        }
        return true;
    }

    private void addBackCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']++;
        }
    }

    private void subCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']--;
        }
    }

    // LC1277
    public int countSquares(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1]; // dp[i][j] 表示以matrix[i][j]为右下角的最大正方形边长
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    dp[i + 1][j + 1] = 0;
                } else {
                    dp[i + 1][j + 1] = 1 + Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]);
                }
                result += dp[i + 1][j + 1];
            }
        }
        return result;
    }

    // LC147
    public ListNode37 insertionSortList(ListNode37 head) {
        if (head == null) return head;
        ListNode37 dummy = new ListNode37();
        dummy.next = head;
        ListNode37 lastSorted = head, cur = head.next;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode37 prev = dummy;
                while (prev.next.val <= cur.val) {
                    prev = prev.next;
                }
                lastSorted.next = cur.next;
                cur.next = prev.next;
                prev.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    // JSOF 52 LC160
    public ListNode37 getIntersectionNode(ListNode37 headA, ListNode37 headB) {
        if (headA == null || headB == null) return null;
        ListNode37 aPtr = headA, bPtr = headB;
        int aLen = 0, bLen = 0;
        while (aPtr != null && aPtr.next != null) {
            aLen++;
            aPtr = aPtr.next;
        }
        while (bPtr != null && bPtr.next != null) {
            bLen++;
            bPtr = bPtr.next;
        }
        if (aPtr != bPtr) return null;
        ListNode37 fast = aLen > bLen ? headA : headB;
        ListNode37 slow = fast == headA ? headB : headA;
        int aheadStep = Math.abs(aLen - bLen);
        while (aheadStep != 0) {
            fast = fast.next;
            aheadStep--;
        }
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}

class ListNode48 {
    int val;
    ListNode48 next;

    ListNode48() {
    }

    ListNode48(int val) {
        this.val = val;
    }

    ListNode48(int val, ListNode48 next) {
        this.val = val;
        this.next = next;
    }
}

class TreeNode48 {
    int val;
    TreeNode48 left;
    TreeNode48 right;

    TreeNode48() {
    }

    TreeNode48(int val) {
        this.val = val;
    }

    TreeNode48(int val, TreeNode48 left, TreeNode48 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Trie48 {

    public TrieNode48 root;

    public Trie48() {
        this.root = new TrieNode48();
    }

    public boolean addWord(String word) {
        TrieNode48 cur = root;
        for (char c : word.toCharArray()) {
            cur.children.putIfAbsent(c, new TrieNode48());
            cur = cur.children.get(c);
        }
        if (cur.isEnd) return false;
        return cur.isEnd = true;
    }

    public boolean beginWith(String prefix) {
        TrieNode48 cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return true;
    }

    public boolean search(String word) {
        TrieNode48 cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return cur.isEnd;
    }

}

class TrieNode48 {
    Map<Character, TrieNode48> children;
    boolean isEnd;

    public TrieNode48() {
        children = new HashMap<>();
        isEnd = false;
    }
}

class DisjointSetUnion48 {
    Map<Integer, Integer> parent = new HashMap<>();

    public boolean add(int i) {
        if (parent.containsKey(i)) return false;
        parent.put(i, i);
        return true;
    }

    // 找最终父节点
    public int find(int i) {
        int cur = i;
        while (parent.get(cur) != cur) {
            cur = parent.get(cur);
        }
        int finalParent = cur;
        cur = i;
        // 路径压缩
        while (parent.get(cur) != finalParent) {
            int origParent = parent.get(cur);
            parent.put(cur, finalParent);
            cur = origParent;
        }
        return finalParent;
    }

    public boolean merge(int i, int j) {
        int ip = find(i);
        int jp = find(j);
        if (ip == jp) return false;
        parent.put(ip, jp);
        return true;
    }

    public boolean isConnect(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> allGroups() {
        for (int i : parent.keySet()) {
            find(i);
        }
        Map<Integer, Set<Integer>> result = new HashMap<>();
        for (int i : parent.keySet()) {
            result.putIfAbsent(parent.get(i), new HashSet<>());
            result.get(parent.get(i)).add(i);
        }
        return result;
    }
}