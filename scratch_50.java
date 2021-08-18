import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.shortestAlternatingPaths(5,
                new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}},
                new int[][]{{1, 2}, {2, 3}, {3, 1}}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // Interview 01.08
    public void setZeroes(int[][] matrix) {
        boolean[] rowMark = new boolean[matrix.length], colMark = new boolean[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    rowMark[i] = true;
                    colMark[j] = true;
                }
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            if (rowMark[i]) {
                for (int j = 0; j < matrix[0].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < matrix[0].length; j++) {
            if(colMark[j]){
                for (int i = 0; i < matrix.length; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    // LC1338
    public int minSetSize(int[] arr) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i : arr) freq.put(i, freq.getOrDefault(i, 0) + 1);
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -freq.getOrDefault(o, 0)));
        for (int i : freq.keySet()) {
            pq.offer(i);
        }
        int left = arr.length, half = arr.length / 2, result = 0;
        while (left > half) {
            left -= freq.get(pq.poll());
            result++;
        }
        return result;
    }

    // Interview 05.01
    public int insertBits(int N, int M, int i, int j) {
        int mask = 0;
        for (int k = i; k <= j; k++) {
            mask |= 1 << k;
        }
        mask = ~mask;
        N &= mask;
        M <<= i;
        M &= ~mask;
        N |= M;
        return N;
    }

    // LC1129 ** 可能有环 有平行边
    public int[] shortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges) {
        final int RED = 0, BLUE = 1;
        int[] result = new int[n];
        Arrays.fill(result, -1);
        result[0] = 0;
        List<List<Integer>> redOutEdge = new ArrayList<>(n), blueOutEdge = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            redOutEdge.add(new ArrayList<>());
            blueOutEdge.add(new ArrayList<>());
        }
        boolean[] redVisit = new boolean[n], blueVisit = new boolean[n];
        for (int[] re : red_edges) {
            redOutEdge.get(re[0]).add(re[1]);
        }
        for (int[] be : blue_edges) {
            blueOutEdge.get(be[0]).add(be[1]);
        }
        if (redOutEdge.get(0) == null && blueOutEdge.get(0) == null) return result;

        Deque<int[]> q = new LinkedList<>();
        for (int next : redOutEdge.get(0)) {
            q.offer(new int[]{next, RED});
        }
        for (int next : blueOutEdge.get(0)) {
            q.offer(new int[]{next, BLUE});
        }
        int layer = 0;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                // 注意剪枝时机!!!
                if (blueVisit[p[0]] && redVisit[p[0]]) continue;
                if (p[1] == RED && redVisit[p[0]]) continue;
                if (p[1] == BLUE && blueVisit[p[0]]) continue;
                if (result[p[0]] == -1) {
                    result[p[0]] = layer;
                } else {
                    result[p[0]] = Math.min(result[p[0]], layer);
                }
                if (p[1] == RED) {
                    redVisit[p[0]] = true;
                    for (int next : blueOutEdge.get(p[0])) {
                        q.offer(new int[]{next, BLUE});
                    }
                } else {
                    blueVisit[p[0]] = true;
                    for (int next : redOutEdge.get(p[0])) {
                        q.offer(new int[]{next, RED});
                    }
                }
            }
        }
        return result;
    }

    // LC1387
    Map<Integer, Integer> lc1387Memo;

    public int getKth(int lo, int hi, int k) {
        lc1387Memo = new HashMap<>();
        List<Integer> arr = new ArrayList<>(hi - lo + 1);
        for (int i = lo; i <= hi; i++) {
            arr.add(i);
        }
        arr.sort((o1, o2) -> lc1387Weight(o1) == lc1387Weight(o2) ? o1 - o2 : lc1387Weight(o1) - lc1387Weight(o2));
        return arr.get(k - 1);
    }

    private int lc1387Weight(int num) {
        if (num == 1) return 0;
        if (lc1387Memo.containsKey(num)) return lc1387Memo.get(num);
        int result = num % 2 == 1 ? lc1387Weight(3 * num + 1) + 1 : lc1387Weight(num / 2) + 1;
        lc1387Memo.put(num, result);
        return result;
    }


    // LC1513
    public int numSub(String s) {
        long oneCount = 0, mod = 1000000007;
        long result = 0;
        for (char c : s.toCharArray()) {
            if (c == '1') {
                oneCount++;
            } else {
                result += oneCount * (oneCount + 1) / 2;
                result %= mod;
                oneCount = 0;
            }
        }
        if (oneCount != 0) {
            result += oneCount * (oneCount + 1) / 2;
            result %= mod;
        }
        return (int) result;
    }

    // LC1709
    public int[] largestSubarrayBetter(int[] nums, int k) {
        int n = nums.length;
        int maxInt = 0, ptr = n - k, maxStartPoint = 0;
        while (ptr >= 0) {
            if (nums[ptr] > maxInt) {
                maxStartPoint = ptr;
                maxInt = nums[ptr];
            }
            ptr--;
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1709 没有利用数字不重复的条件
    public int[] largestSubarray(int[] nums, int k) {
        int maxStartPoint = 0, n = nums.length;
        for (int i = 1; i < n - k + 1; i++) {
            for (int j = 0; j < k; j++) {
                if (nums[maxStartPoint + j] < nums[i + j]) {
                    maxStartPoint = i;
                } else if (nums[maxStartPoint + j] == nums[i + j]) {
                    continue;
                } else {
                    break;
                }
            }
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1039 ** 几何
    Integer[][] lc1039Memo;

    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        lc1039Memo = new Integer[n + 1][n + 1];
        return lc1039Helper(values, 0, n - 1);
    }

    private int lc1039Helper(int[] values, int start, int end) {
        if (start + 1 == end) return 0;
        if (lc1039Memo[start][end] != null) return lc1039Memo[start][end];
        int result = Integer.MAX_VALUE;
        for (int i = start + 1; i < end; i++) {
            result = Math.min(result, lc1039Helper(values, start, i) + lc1039Helper(values, i, end) + values[start] * values[end] * values[i]);
        }
        return lc1039Memo[start][end] = result;
    }

    // LC128
    // https://bbs.byr.cn/n/article/Talking/6295267
    public int longestConsecutive(int[] nums) {
        Set<Integer> s = new HashSet<>();
        int result = 0;
        for (int e : nums) {
            s.add(e);
        }
        for (int i : s) {
            if (!s.contains(i - 1)) {
                int l = 1;
                while (s.contains(i + 1)) {
                    l++;
                    i++;
                }
                result = Math.max(result, l);
            }
        }
        return result;
    }


    // LC1775
    public int minOperations(int[] nums1, int[] nums2) {
        if (nums1.length * 6 < nums2.length || nums2.length * 6 < nums1.length) return -1;
        int origSum1 = Arrays.stream(nums1).sum(), origSum2 = Arrays.stream(nums2).sum();
        if (origSum1 == origSum2) return 0;
        int[] inc = new int[6], dec = new int[6];
        for (int i : nums1) {
            inc[6 - i]++;
            dec[i - 1]++;
        }
        for (int i : nums2) {
            dec[6 - i]++;
            inc[i - 1]++;
        }
        inc[0] = dec[0] = 0;
        int result = 0;
        int delta = origSum1 - origSum2;
        if (delta > 0) { // nums1 should decrease
            for (int i = 5; i >= 1; i--) {
                while (dec[i] > 0) {
                    result++;
                    dec[i]--;
                    delta -= i;
                    if (delta <= 0) return result;
                }
            }
        } else {
            for (int i = 5; i >= 1; i--) {
                while (inc[i] > 0) {
                    result++;
                    inc[i]--;
                    delta += i;
                    if (delta >= 0) return result;
                }
            }
        }
        return -1;
    }

    // LC552 HARD
    public int checkRecord(int n) {
        final int mod = 1000000007;
        long[] dp = new long[Math.max(4, n + 1)];
        dp[0] = 1;
        dp[1] = 2;
        dp[2] = 4;
        dp[3] = 7;
        for (int i = 4; i <= n; i++) {
            dp[i] = (2 * dp[i - 1] - dp[i - 4] + mod) % mod;
        }
        long result = dp[n];
        for (int i = 1; i <= n; i++) {
            result += (dp[i - 1] * dp[n - i]) % mod;
        }
        return (int) (result % mod);
    }

    // LC551
    public boolean checkRecord(String s) {
        int lCount = 0, aCount = 0;
        for (char c : s.toCharArray()) {
            if (c == 'L') {
                lCount++;
            } else {
                lCount = 0;
            }
            if (lCount >= 3) return false;
            if (c == 'A') {
                aCount++;
            }
            if (aCount >= 2) return false;
        }
        return true;
    }

    // LC1389
    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (index[i] >= result.size()) {
                int targetSize = (index[i] + 1) - result.size();
                for (int j = 0; j < targetSize; j++) {
                    result.add(-1);
                }
                result.set(index[i], nums[i]);
            } else {
                result.add(index[i], nums[i]);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC1390
    public int sumFourDivisors(int[] nums) {
        int result = 0;
        for (int n : nums) result += lc1390Helper(n);
        return result;
    }

    private int lc1390Helper(int n) {
        if (n <= 5) return 0;
        int sqrt = (int) Math.sqrt(n);
        Set<Integer> s = new HashSet<>();
        for (int i = 1; i <= sqrt; i++) {
            if (n % i == 0) {
                s.add(i);
                s.add(n / i);
            }
            if (s.size() > 4) return 0;
        }
        if (s.size() != 4) return 0;
        return s.stream().reduce((a, b) -> a + b).get();
    }

    // JZOF 06
    public int[] reversePrint(ListNode head) {
        // 倒置链表
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode prev = null, cur = head;
        int count = 0;
        while (cur != null) {
            count++;
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }
        int[] result = new int[count];
        count = 0;
        cur = prev;
        while (cur != null) {
            result[count++] = cur.val;
            cur = cur.next;
        }
        return result;
    }

    // JZOF II 090
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[1], nums[0]);
        // ROB ZERO, then n-1 can't be robbed
        dp[0] = dp[1] = nums[0];
        for (int i = 2; i < n - 1; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int noZero = dp[n - 2];
        Arrays.fill(dp, 0);
        // Not rob zero, then n-1 can be robbed
        dp[1] = nums[1];
        dp[2] = Math.max(nums[1], nums[2]);
        for (int i = 3; i < n; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int zero = dp[n - 1];
        return Math.max(zero, noZero);
    }

    public TreeNode LCA(TreeNode root, TreeNode p, TreeNode q) {
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        parent.put(root, null);
        Deque<TreeNode> queue = new LinkedList<>();
        Set<TreeNode> visited = new HashSet<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode poll = queue.poll();
            if (poll.left != null) {
                parent.put(poll.left, poll);
                queue.offer(poll.left);
            }
            if (poll.right != null) {
                parent.put(poll.right, poll);
                queue.offer(poll.right);
            }
        }
        while (p != null) {
            visited.add(p);
            p = parent.get(p);
        }
        while (q != null) {
            if (visited.contains(q)) return q;
            q = parent.get(q);
        }
        return null;
    }

    // JZOF 68
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode result = root;
        while (true) {
            if (p.val > result.val && q.val > result.val) {
                result = result.right;
            } else if (p.val < result.val && q.val < result.val) {
                result = result.left;
            } else {
                break;
            }
        }
        return result;
    }

    // LC1338
    Map<TreeNode, Long> nodeSumMap = new HashMap<>();

    public int maxProduct(TreeNode root) {
        final int mod = 1000000007;
        lc1338Helper(root);
        long result = 0;
        long total = nodeSumMap.get(root);
        for (TreeNode node : nodeSumMap.keySet()) {
            result = Math.max(result, (total - nodeSumMap.get(node)) * nodeSumMap.get(node));
        }
        return (int) (result % mod);
    }

    private long lc1338Helper(TreeNode root) {
        if (root == null) return 0;
        long result = root.val + lc1338Helper(root.left) + lc1338Helper(root.right);
        nodeSumMap.put(root, result);
        return result;
    }

    // LC733
    int[][] lc733Directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int origColor = image[sr][sc];
        boolean[][] visited = new boolean[image.length][image[0].length];
        lc733Helper(image, sr, sc, origColor, newColor, visited);
        return image;
    }

    private void lc733Helper(int[][] image, int x, int y, int origColor, int newColor, boolean[][] visited) {
        if (x < 0 || x >= image.length || y < 0 || y >= image[0].length || visited[x][y]) {
            return;
        }
        if (image[x][y] == origColor) {
            image[x][y] = newColor;
            visited[x][y] = true;
            for (int[] dir : lc733Directions) {
                lc733Helper(image, x + dir[0], y + dir[1], origColor, newColor, visited);
            }
        }
    }

    // JZOF 64 **
    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    // LC1282
    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, List<Integer>> sizeCountMap = new HashMap<>();
        for (int i = 0; i < groupSizes.length; i++) {
            sizeCountMap.putIfAbsent(groupSizes[i], new ArrayList<>());
            sizeCountMap.get(groupSizes[i]).add(i);
        }
        for (int gs : sizeCountMap.keySet()) {
            List<Integer> users = sizeCountMap.get(gs);
            int cur = 0;
            while (cur != users.size()) {
                result.add(users.subList(cur, cur + gs));
                cur += gs;
            }
        }
        return result;
    }

    // LC1684
    public int countConsistentStrings(String allowed, String[] words) {
        int result = 0, mask = 0;
        for (char c : allowed.toCharArray()) mask |= 1 << (c - 'a');
        for (String w : words) {
            int wm = 0;
            for (char c : w.toCharArray()) wm |= 1 << (c - 'a');
            if ((wm & mask) == wm) result++;
        }
        return result;
    }

    // LC526 **
    boolean[] lc526Visited;
    int lc526Result;

    public int countArrangement(int n) {
        lc526Visited = new boolean[n + 1];
        lc526Backtrack(1, n);
        return lc526Result;
    }

    public void lc526Backtrack(int index, int n) {
        if (index == n + 1) {
            lc526Result++;
            return;
        }
        for (int i = 1; i <= n; i++) {
            if (!lc526Visited[i] && (i % index == 0 || index % i == 0)) {
                lc526Visited[i] = true;
                lc526Backtrack(index + 1, n);
                lc526Visited[i] = false;
            }
        }
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

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}