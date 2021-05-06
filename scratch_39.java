import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

//        System.err.println(

        s.sortColors(new int[]{2});

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC324
    public void wiggleSort(int[] nums) {
        // TBD
        return;
    }

    // LC75 3 way partition
    public void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {

            if (right < i) break;
            if (nums[i] == 0) {
                // 注意亦或运算时候如果两数相同会置零
                if (nums[i] != nums[left]) {
                    // swap i,left
                    nums[left] ^= nums[i];
                    nums[i] ^= nums[left];
                    nums[left] ^= nums[i];
                }
                left++;
            } else if (nums[i] == 2) {
                if (nums[i] != nums[right]) {
                    nums[right] ^= nums[i];
                    nums[i] ^= nums[right];
                    nums[right] ^= nums[i];
                }
                right--;
                i--;
            }
        }
        return;
    }

    // LC289 O(1) Space
    public void gameOfLife(int[][] board) {
        // LIFE: 1) 0,1 -> DEAD 2) 2,3 -> LIFE 3) 4,5,6,7,8 ->DEAD
        // DEAD: 1) 3 -> LIFE 2) 0,1,2,4,5,6,7,8 -> DEAD
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = board[i][j] == 0 ? -1 : 1;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int ctr = gameOfLifeLiveCounter(i, j, board);
                if (board[i][j] > 0) board[i][j] = 1 + ctr;
                else if (board[i][j] < 0) board[i][j] = -1 - ctr;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] < 0) {
                    if (board[i][j] == -4) board[i][j] = 1;
                    else board[i][j] = 0;
                } else if (board[i][j] > 0) {
                    if (board[i][j] == 3 || board[i][j] == 4) board[i][j] = 1;
                    else board[i][j] = 0;
                }
            }
        }
        return;
    }

    // LIVE : +1, DEAD: -1
    private int gameOfLifeLiveCounter(int m, int n, int[][] board) {
        int ctr = 0;
        for (int i = m - 1; i <= m + 1; i++) {
            if (i < 0 || i >= board.length) continue;
            for (int j = n - 1; j <= n + 1; j++) {
                if (j < 0 || j >= board[0].length) continue;
                if (i == m && j == n) continue;
                if (board[i][j] > 0) ctr++;
            }
        }
        return ctr;
    }

    // LC242
    public boolean isAnagram(String s, String t) {
        int[] freq = new int[26];
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            freq[c - 'a']--;
        }
        for (int i : freq) {
            if (i != 0) return false;
        }
        return true;
    }

    // LC240 双指针
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length - 1;
        int n = 0;
        while (m >= 0 && n < matrix[0].length) {
            if (matrix[m][n] > target) {
                m--;
            } else if (matrix[m][n] < target) {
                n++;
            } else {
                return true;
            }
        }
        return false;
    }

    // LC239 Solution
    public int[] maxSlidingWindowSolution(int[] nums, int k) {
        int n = nums.length;
        PriorityQueue<int[]> pq = new PriorityQueue<>((pair1, pair2) -> pair1[0] != pair2[0] ? pair2[0] - pair1[0] : pair2[1] - pair1[1]);
        for (int i = 0; i < k; i++) {
            pq.offer(new int[]{nums[i], i});
        }
        int[] ans = new int[n - k + 1];
        ans[0] = pq.peek()[0];
        for (int i = k; i < n; i++) {
            pq.offer(new int[]{nums[i], i});
            while (pq.peek()[1] <= i - k) {
                pq.poll();
            }
            ans[i - k + 1] = pq.peek()[0];
        }
        return ans;
    }

    // LC239
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] result = new int[nums.length - k + 1];
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i < k; i++) {
            tm.put(nums[i], tm.getOrDefault(nums[i], 0) + 1);
        }
        result[0] = tm.lastKey();
        for (int i = k; i < nums.length; i++) {
            tm.put(nums[i - k], tm.get(nums[i - k]) - 1);
            if (tm.get(nums[i - k]) == 0) {
                tm.remove(nums[i - k]);
            }
            tm.put(nums[i], tm.getOrDefault(nums[i], 0) + 1);
            result[i - k + 1] = tm.lastKey();
        }
        return result;
    }

    // LC238
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        result[0] = 1;
        for (int i = 1; i < n; i++) {
            result[i] = result[i - 1] * nums[i - 1];
        }
        int r = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] = result[i] * r;
            r = r * nums[i];
        }
        return result;
    }


    // LC1473 Hard
    public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
        // Define dp[i][j][k] as the minimum cost where we have k neighborhoods in the first i houses and the i-th house is painted with the color j.
        // houses[i] != -1, 则不需要涂漆, 设第i-1的颜色为j_0, 枚举j:
        //                 1) 如果j=!house[i], 则 dp[i][j][k]=inf
        //             否则 2) 若j==j_0, 则和(i-1)同属一个街区, 可从k转移来:    dp[i][j][k] = dp[i-1][j_0][k] iff houses[i]==j
        //                    若j!=j_0, 则和(i-1)不同属于一个街区, 从k-1转移来: dp[i][j][k] = min dp[i-1][j][k-1] iff houses[i]==j
        // houses[i] == -1, 则需要涂漆, 设第i-1的颜色为j_0, 同样枚举j:
        //                 1) 若j!=j_0, 则和(i-1)不是同一个街区, 从k-1转移到k: dp[i][j][k] = min dp[i-1][j][k-1] + cost[i][j] iff houses[i]==0
        //                 2) 若j==j_0, 则和(i-1)是同一个街区, 可从k转移来:    dp[i][j][k] = min dp[i-1][j][k] + cost[i][j] iff j==j_0

        assert m == houses.length;
        assert n == cost[0].length;

        int numHouse = m;
        int numColor = n;
        final int INF = Integer.MAX_VALUE / 2;
        for (int i = 0; i < numHouse; i++) {
            houses[i]--;
        }

        int[][][] dp = new int[numHouse][numColor][target];
        for (int i = 0; i < numHouse; i++) {
            for (int j = 0; j < numColor; j++) {
                Arrays.fill(dp[i][j], INF);
            }
        }

        for (int i = 0; i < numHouse; i++) {
            for (int j = 0; j < numColor; j++) {
                if (houses[i] != -1 && j != houses[i]) {
                    continue;
                }

                for (int k = 0; k < target; k++) {
                    for (int j0 = 0; j0 < numColor; j0++) {
                        if (j == j0) {
                            if (i == 0) {
                                if (k == 0) {
                                    dp[i][j][k] = 0;
                                }
                            } else {
                                dp[i][j][k] = Math.min(dp[i][j][k], dp[i - 1][j][k]);
                            }
                        } else if (i > 0 && k > 0) {
                            dp[i][j][k] = Math.min(dp[i][j][k], dp[i - 1][j0][k - 1]);
                        }
                    }
                    if (dp[i][j][k] != INF && houses[i] == -1) {
                        dp[i][j][k] += cost[i][j];
                    }
                }
            }
        }
        int result = INF;
        for (int j = 0; j < numColor; j++) {
            result = Math.min(result, dp[numHouse - 1][j][target - 1]);
        }
        return result == INF ? -1 : result;
    }

    // LC740 另一种打家劫舍
    public int deleteAndEarn(int[] nums) {
        int n = nums.length;
        int maxVal = Arrays.stream(nums).max().getAsInt();
        int[] sum = new int[maxVal + 1];
        for (int i : nums) {
            sum[i] += i;
        }
        if (sum.length == 2) return sum[1];
        if (sum.length == 3) return Math.max(sum[1], sum[2]);
        int[] dp = new int[maxVal + 1];
        dp[1] = sum[1];
        dp[2] = Math.max(dp[1], sum[2]);
        for (int i = 2; i <= maxVal; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + sum[i]);
        }
        return dp[maxVal];
    }

    // LC1720
    public int[] decode(int[] encoded, int first) {
        int[] result = new int[encoded.length + 1];
        result[0] = first;
        for (int i = 0; i < encoded.length; i++) {
            result[i + 1] = encoded[i] ^ result[i];
        }
        return result;
    }

    // LC690
    public int getImportance(List<Employee> employees, int id) {
        Set<Integer> visit = new HashSet<>();
        Map<Integer, Employee> allEmployees = new HashMap<>();
        for (Employee e : employees) {
            allEmployees.put(e.id, e);
        }
        Deque<Integer> q = new LinkedList<>();
        int result = 0;
        q.add(id);
        while (!q.isEmpty()) {
            int tmpId = q.poll();
            if (visit.contains(tmpId)) {
                continue;
            }
            result += allEmployees.get(tmpId).importance;
            visit.add(tmpId);
            for (int sub : allEmployees.get(tmpId).subordinates) {
                q.offer(sub);
            }
        }
        return result;
    }


    // LC279 O(n^1.5) Time O(n) Space
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i; // 至多需要n个1
        }
        int maxSquareRoot = ((int) (Math.sqrt(n)));
        for (int i = 2; i <= maxSquareRoot; i++) {
            dp[i * i] = 1;
        }
        for (int i = 5; i <= n; i++) {
            for (int j = (int) (Math.sqrt(i)); j >= 1; j--) {
                dp[i] = Math.min(dp[i], dp[j * j] + dp[i - j * j]);
            }
        }

        return dp[n];
    }

    // LC236
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        Set<TreeNode> visited = new HashSet<>();
        Deque<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        parent.put(root, null);
        // 层次遍历 取得所有父子关系
        while (!queue.isEmpty()) {
            TreeNode tmp = queue.poll();
            if (tmp.left != null) {
                parent.put(tmp.left, tmp);
                queue.offer(tmp.left);
            }
            if (tmp.right != null) {
                parent.put(tmp.right, tmp);
                queue.offer(tmp.right);
            }
        }
        while (p != null) {
            visited.add(p);
            p = parent.get(p);
        }
        while (q != null) {
            if (visited.contains(q)) {
                return q;
            }
            q = parent.get(q);
        }
        return null;
    }

    // LC218 Heap Solution
    public List<List<Integer>> getSkyline(int[][] buildings) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
        for (int[] building : buildings) {
            pq.offer(new int[]{building[0], -building[2]});
            pq.offer(new int[]{building[1], building[2]});
        }

        List<List<Integer>> res = new ArrayList<>();

        TreeMap<Integer, Integer> heights = new TreeMap<>((a, b) -> b - a);
        heights.put(0, 1);
        int left = 0, formerMaxHeight = 0;
        while (!pq.isEmpty()) {
            int[] arr = pq.poll();
            if (arr[1] < 0) {
                heights.put(-arr[1], heights.getOrDefault(-arr[1], 0) + 1);
            } else {
                heights.put(arr[1], heights.get(arr[1]) - 1);
                if (heights.get(arr[1]) == 0) heights.remove(arr[1]);
            }
            int maxHeight = heights.keySet().iterator().next();
            if (maxHeight != formerMaxHeight) {
                left = arr[0];
                formerMaxHeight = maxHeight;
                res.add(Arrays.asList(left, maxHeight));
            }
        }

        return res;
    }

    // LC403 Try Queue
    public boolean canCrossQueue(int[] stones) {
        int n = stones.length;
        for (int i = 1; i < n; i++) {
            if (stones[i] - stones[i - 1] > i) {
                return false;
            }
        }
        Map<Integer, Integer> idxMap = new HashMap<>();
        Set<Pair<Integer, Integer>> pairSet = new HashSet<>();
        for (int i = 0; i < n; i++) {
            idxMap.put(stones[i], i);
        }
        Deque<Pair<Integer, Integer>> q = new LinkedList<>();
        q.add(new Pair<>(0, 0));
        while (!q.isEmpty()) {
            Pair<Integer, Integer> tmp = q.poll();
            int i = tmp.getKey();
            int j = tmp.getValue();
            for (int k = j - 1; k <= j + 1; k++) {
                if (k > 0 && k < n && idxMap.containsKey(stones[i] + k)) {
                    Pair<Integer, Integer> tmpPair = new Pair<>(idxMap.get(stones[i] + k), k);
                    if (pairSet.add(tmpPair)) {
                        if (tmpPair.getKey() == n - 1) return true; // 剪枝
                        q.offer(tmpPair);
                    }
                }
            }
        }
        return false;
    }

    // LC403
    public boolean canCross(int[] stones) {
        int n = stones.length;
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            idxMap.put(stones[i], i);
        }
        boolean[][] dp = new boolean[n][n];
        // dp[i][j] 表示能否通过j步到达第编号为i的石头
        // 转移方程: dp[a][b] = true, if  dp[idxMap.get(stones[a]-b)][k]==true 且 b==k-1 or b==k+1 or b==k
        // 反过来说 如果dp[i][j] ==true, 且 0<(j-1|j|j+1)<n, 且idxMap.containsKey(stones[i] + (j-1|j|j+1)) 则 dp[  idxMap.get( stones[i] + (j-1|j|j+1) ) ][(j-1|j|j+1)] = true
        dp[0][0] = true;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dp[i][j]) {
                    for (int k = j - 1; k <= j + 1; k++) {
                        if (k > 0 && k < n && idxMap.containsKey(stones[i] + k)) {
                            dp[idxMap.get(stones[i] + k)][k] = true;
                            if (idxMap.get(stones[i] + k) == n - 1) return true; // 剪枝
                        }
                    }
                }
            }
        }
//        for (int i = 0; i < n; i++) {
//            if(dp[n-1][i]) return true;
//        }
        return false;
    }


    // LC403 TLE
    public boolean canCrossTLE(int[] stones) {
        Set<Long> stoneIdxSet = new HashSet<>();
        for (int i : stones) {
            stoneIdxSet.add((long) i);
        }
        long totalLen = stones[stones.length - 1] + 1;
//        boolean[][] dp = new boolean[totalLen + 1][totalLen + 1]; // dp[i][j] 表示能否通过上一个位置跳j步到达第i个位置
        Map<Long, Map<Long, Boolean>> dpMap = new HashMap<>();
        // 转移方程 dp[a][j+1] =  dp[a][j+1] | (dp[i][j] ? i+j+1==a : false)
        //         dp[b][j-1] = dp[b][j-1] | (dp[m][j] ? m+j-1==b : false)
        //         dp[c][j]   =  dp[c][j] | (dp[n][j]? n+j == b : false)
        dpMap.put(0l, new HashMap<>());
        dpMap.get(0l).put(0l, true);
//        dp[0][0] = true;
        for (long i = 0; i <= totalLen; i++) {
            for (long j = 0; j <= totalLen; j++) {
                if (dpMap.containsKey(i) && dpMap.get(i).containsKey(j) && dpMap.get(i).get(j) == true) {
                    for (long k = j - 1; k <= j + 1; k++) {
                        if (k > 0 && k <= totalLen && i + k <= totalLen && stoneIdxSet.contains(i + k)) {
                            dpMap.putIfAbsent(i + k, new HashMap<>());
                            dpMap.get(i + k).put(k, true);
                        }
                    }
                }
            }
        }
        if (!dpMap.containsKey(totalLen - 1)) return false;
        for (Map.Entry<Long, Boolean> entry : dpMap.get(totalLen - 1).entrySet()) {
            if (entry.getValue() == true) return true;
        }


        return false;
    }

    // LC234 O(1) Space O(n) Time
    public boolean isPalindrome(ListNode head) {
        // get mid
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode middleDummy = new ListNode();
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
            if (fast != null) {
                fast = fast.next;
            }
        }
        middleDummy.next = slow;

        // reverse the right part
        ListNode prev = null;
        ListNode cur = slow;
        while (cur != null) {
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }

        // judge
        ListNode end = prev;
        ListNode first = dummy.next;
        boolean result = true;
        while (end != null) {
            if (end.val != first.val) {
                result = false;
                break;
            }
            end = end.next;
            first = first.next;
        }

        // recover
        cur = prev; // end
        prev = null;
        while (cur != null) {
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }

        return result;
    }

    // LC230
    int lc230Ctr = 0;
    int lc230Result = -1;

    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return lc230Result;
    }

    private void inorder(TreeNode root, int k) {
        if (root.left != null) inorder(root.left, k);
        lc230Ctr++;
        if (lc230Ctr == k) {
            lc230Result = root.val;
            return;
        }
        if (root.right != null) inorder(root.right, k);
    }

    // LC633
    public boolean judgeSquareSum(int c) {
        for (long a = 0; a * a <= c; a++) {
            double b = Math.sqrt(c - a * a);
            if (b == (int) b) {
                return true;
            }
        }
        return false;
    }

    // LC217
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i : nums) {
            if (!s.add(i)) return true;
        }
        return false;
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

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}


class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};


