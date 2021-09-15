import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.IntConsumer;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.mostVisited(7, new int[]{1, 3, 5, 7}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1560
    public List<Integer> mostVisited(int n, int[] rounds) {
        int[] diff = new int[n + 1];
        diff[rounds[0] - 1]++;
        diff[rounds[0]]--;
        for (int i = 0; i < rounds.length - 1; i++) {
            int start = rounds[i] - 1, end = rounds[i + 1] - 1;
            if (start < end) {
                diff[start + 1]++;
                diff[end + 1]--;
            } else {
                diff[start + 1]++;
                diff[n]--;
                diff[0]++;
                diff[end + 1]--;
            }
        }
        int[] count = new int[n];
        count[0] = diff[0];
        for (int i = 1; i < n; i++) {
            count[i] = count[i - 1] + diff[i];
        }
        int max = 0;
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (count[i] > max) {
                result.clear();
                result.add(i + 1);
                max = count[i];
            } else if (count[i] == max) {
                result.add(i + 1);
            }
        }
        return result;
    }

    // LC1395 O(n^2+nlog(n)) Time O(n) Space
    public int numTeams(int[] rating) {
        // 离散化
        int n = rating.length;
        int[] orig = new int[n];
        System.arraycopy(rating, 0, orig, 0, n);
        Arrays.sort(rating);
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            idxMap.put(rating[i], i);
        }
        for (int i = 0; i < n; i++) {
            orig[i] = idxMap.get(orig[i]);
        }
        BIT bit = new BIT(n + 1);
        int[] numLarget = new int[n];
        // 记录后面有多少个比自己大的数
        for (int i = n - 1; i >= 0; i--) {
            numLarget[i] = (int) bit.sumRange(orig[i] + 1, n);
            bit.update(orig[i], 1);
        }
        bit = new BIT(n + 1);
        // 记录前面有多少个比自己大的数
        int[] numLargerReverse = new int[n];
        for (int i = 0; i < n; i++) {
            numLargerReverse[i] = (int) bit.sumRange(orig[i] + 1, n);
            bit.update(orig[i], 1);
        }
        int result = 0;
        // 正向 升序
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (orig[j] > orig[i]) {
                    result += numLarget[j];
                }
            }
        }

        // 反向 逆序
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                if (orig[j] > orig[i]) {
                    result += numLargerReverse[j];
                }
            }
        }
        return result;
    }

    // JZOF II 018 LC125
    public boolean isPalindrome(String s) {
        if (s.equals("")) return true;
        char[] ca = s.toCharArray();
        int left = 0, right = ca.length - 1;
        while (left <= right) {
            if (!Character.isDigit(ca[left]) && !Character.isLetter(ca[left])) {
                left++;
                continue;
            }
            if (!Character.isDigit(ca[right]) && !Character.isLetter(ca[right])) {
                right--;
                continue;
            }
            if (Character.isLetter(ca[left]) && Character.isUpperCase(ca[left]))
                ca[left] = Character.toLowerCase(ca[left]);
            if (Character.isLetter(ca[right]) && Character.isUpperCase(ca[right]))
                ca[right] = Character.toLowerCase(ca[right]);
            if (ca[left] != ca[right]) return false;
            left++;
            right--;
        }
        return true;
    }

    // LC935
    class Lc935 {
        final int[][] keyboard = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {-1, 0, -1}};
        final int[][] directions = {{-2, -1}, {-1, -2}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {1, 2}, {2, 1}};
        final int mod = 1000000007;
        Integer[][][] memo;

        public int knightDialer(int n) {
            memo = new Integer[4][3][n + 1];
            long result = 0;
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 3; col++) {
                    if (keyboard[row][col] != -1) {
                        result = (result + dfs(row, col, n - 1)) % mod;
                    }
                }
            }
            return (int) (result % mod);
        }

        private int dfs(int row, int col, int leftSteps) {
            if (leftSteps == 0) return 1;
            if (memo[row][col][leftSteps] != null) return memo[row][col][leftSteps];
            int result = 0;
            for (int[] dir : directions) {
                int nr = row + dir[0], nc = col + dir[1];
                if (check(nr, nc)) {
                    result = (result + dfs(nr, nc, leftSteps - 1)) % mod;
                }
            }
            return memo[row][col][leftSteps] = result;
        }

        private boolean check(int row, int col) {
            return row >= 0 && row < 4 && col >= 0 && col < 3 && keyboard[row][col] != -1;
        }
    }

    // LC163 **
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        lower--;
        upper++;
        List<String> result = new ArrayList<>();
        int n = nums.length;
        if (n == 0) {
            lc163Helper(lower, upper, result);
            return result;
        }
        lc163Helper(lower, nums[0], result);
        for (int i = 1; i < n; i++) lc163Helper(nums[i - 1], nums[i], result);
        lc163Helper(nums[n - 1], upper, result);
        return result;
    }

    private void lc163Helper(int l, int r, List<String> result) {
        if (l + 1 >= r) return;
        if (l + 2 == r) {
            result.add("" + (l + 1));
        } else {
            result.add("" + (l + 1) + "->" + (r - 1));
        }
    }

    // Interview 10.09 双指针
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0) return false;
        if (matrix.length == 1 && matrix[0].length == 0) return false;
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == target) return true;
            else if (matrix[row][col] < target) {
                row++;
            } else {
                col--;
            }
        }
        return false;
    }

    public boolean searchMatrixBS(int[][] matrix, int target) {
        if (matrix.length == 0) return false;
        if (matrix.length == 1 && matrix[0].length == 0) return false;
        int lo = 0, hi = matrix.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (matrix[mid][0] <= target) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if (matrix[lo][0] > target) return false;
        for (int i = 0; i <= lo; i++) {
            int[] row = matrix[i];
            int innerLo = 0, innerHi = matrix[0].length - 1;
            while (innerLo <= innerHi) {
                int mid = innerLo + (innerHi - innerLo) / 2;
                if (row[mid] == target) return true;
                else if (matrix[i][mid] > target) {
                    innerHi = mid - 1;
                } else {
                    innerLo = mid + 1;
                }
            }
        }
        return false;
    }

    // JZOF II 061 LC373 原来O(mnlogk)的方法现在会超时 以下为O(mk)的方法 **
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        int[] idx = new int[m]; // minNums2IdxForNums1  , e.g. idx[0] = 0
        List<List<Integer>> result = new ArrayList<>(k);
        int loopingUpperBound = 1;
        while (result.size() < k) {
            int cur = 0; // nums1 中的下标
            for (int j = 0; j < loopingUpperBound; j++) {
                if (idx[j] == n) continue; // 在nums2中已经没有可用的下标了
                if (idx[cur] == n || nums1[cur] + nums2[idx[cur]] > nums1[j] + nums2[idx[j]]) { // 获得当前最小组合的下标
                    cur = j;
                }
            }
            if (idx[cur] == n) break;
            result.add(Arrays.asList(nums1[cur], nums2[idx[cur]]));
            idx[cur]++;
            if (cur == loopingUpperBound - 1) {
                loopingUpperBound = Math.min(loopingUpperBound + 1, nums1.length);
            }
        }
        return result;
    }

    // Interview 02.07
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        // 判断是否有共同终点
        ListNode pa = headA, pb = headB;
        int lenA = 1, lenB = 1;
        while (pa.next != null) {
            pa = pa.next;
            lenA++;
        }
        while (pb.next != null) {
            pb = pb.next;
            lenB++;
        }
        if (pa != pb) return null;
        int diff = Math.abs(lenA - lenB);
        ListNode fast = lenA > lenB ? headA : headB;
        ListNode slow = fast == headA ? headB : headA;
        while (diff != 0) {
            fast = fast.next;
            diff--;
        }
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    // LC162
    public int findPeakElement(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        if (n == 2) return nums[0] > nums[1] ? 0 : 1;
        int lo = 0, hi = n - 1;
        int result = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid == n - 1) {
                if (nums[n - 1] > nums[n - 2]) return n - 1;
                hi = mid - 1;  // 折返
            } else if (mid == 0) {
                if (nums[0] > nums[1]) return 0;
                lo = mid + 1;  // 折返
            } else {
                if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
                    result = mid;
                    break;
                } else if (nums[mid] < nums[mid + 1]) { // 继续在右侧找
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return result;
    }

    // LC699
    public List<Integer> fallingSquares(int[][] positions) {
        List<Integer> result = new ArrayList<>();
        // 离散化
        Map<Integer, Integer> idxMap = new HashMap<>();
        Set<Integer> coordinates = new HashSet<>();
        for (int[] p : positions) {
            coordinates.add(p[0]);
            coordinates.add(p[0] + p[1] - 1);
        }
        List<Integer> coordList = new ArrayList<>(coordinates);
        Collections.sort(coordList);
        for (int i = 0; i < coordList.size(); i++) {
            idxMap.put(coordList.get(i), i);
        }
        int[] maxHeight = new int[coordList.size()];
        int curMax = 0;
        for (int[] p : positions) {
            int left = idxMap.get(p[0]), right = idxMap.get(p[0] + p[1] - 1);
            // 找 [left, right]之间的最大值
            int max = 0;
            for (int i = left; i <= right; i++) {
                max = Math.max(max, maxHeight[i]);
            }
            for (int i = left; i <= right; i++) {
                maxHeight[i] = max + p[1];
            }
            // 更新全局最大值
            curMax = Math.max(curMax, max + p[1]);
            // 添加**全局最大值**到结果 (不是区间最大值!)
            result.add(curMax);
        }
        return result;
    }

    // LC524
    public String findLongestWord(String s, List<String> dictionary) {
        char[] ca = s.toCharArray();
        String result = "";
        for (String word : dictionary) {
            char[] cw = word.toCharArray();
            int pc = 0, pw = 0;
            while (pc < ca.length && pw < cw.length) {
                if (ca[pc] == cw[pw]) {
                    pc++;
                    pw++;
                } else {
                    pc++;
                }
            }
            if (pw == cw.length) {
                if (word.length() > result.length()) {
                    result = word;
                } else if (word.length() == result.length()) {
                    if (word.compareTo(result) < 0) {
                        result = word;
                    }
                }
            }
        }
        return result;
    }

    // JZOF II 094 LC132
    public int minCut(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        // 判定数组
        boolean[][] judge = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            judge[i][i] = true;
        }
        for (int len = 1; len <= n; len++) {
            for (int left = 0; left < n; left++) {
                if (left + len >= n) break;
                if (ca[left] == ca[left + len]) {
                    if (len == 1) judge[left][left + len] = true;
                    else {
                        judge[left][left + len] = judge[left + 1][left + len - 1];
                    }
                }
            }
        }

        int[] dp = new int[n];
        Arrays.fill(dp, n - 1);
        for (int i = 0; i < n; i++) {
            if (judge[0][i]) dp[i] = 0;
            else {
                for (int j = 0; j < i; j++) {
                    if (judge[j + 1][i]) {
                        dp[i] = Math.min(dp[i], 1 + dp[j]);
                    }
                }
            }
        }
        Integer[] memo = new Integer[n];

        return lc132Helper(judge, n - 1, memo);
    }

    private int lc132Helper(boolean[][] judge, int end, Integer[] memo) {
        if (judge[0][end] == true) return 0;
        if (memo[end] != null) return memo[end];
        // 切割
        int result = judge.length - 1;
        for (int i = 0; i < end; i++) {
            if (judge[i + 1][end]) {
                result = Math.min(result, 1 + lc132Helper(judge, i, memo));
            }
        }
        return memo[end] = result;
    }

    // LC695 JZOF II 105
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Function<int[], Boolean> check = i -> i[0] >= 0 && i[0] < m && i[1] >= 0 && i[1] < n;
        DisjointSetUnion<Integer> dsu = new DisjointSetUnion<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    int cur = i * n + j;
                    dsu.add(cur);
                    for (int[] dir : directions) {
                        int nr = i + dir[0], nc = j + dir[1];
                        int next = nr * n + nc;
                        if (check.apply(new int[]{nr, nc}) && grid[nr][nc] == 1) {
                            dsu.add(next);
                            dsu.merge(cur, next);
                        }
                    }
                }
            }
        }
        Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
        int maxGroupSize = 0;
        for (Set<Integer> s : allGroups.values()) {
            maxGroupSize = Math.max(maxGroupSize, s.size());
        }
        return maxGroupSize;
    }

    // LC1063 Hard
    public int validSubarrays(int[] nums) {
        // NSE
        int n = nums.length;
        int[] nse = new int[n];
        Arrays.fill(nse, -1);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                nse[stack.pop()] = i;
            }
            stack.push(i);
        }
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (nse[i] == -1) {
                result += n - i;
            } else {
                result += nse[i] - i;
            }
        }
        return result;
    }

    // LC490
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        int m = maze.length, n = maze[0].length;
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int[] p = q.poll();
            if (visited[p[0]][p[1]]) continue;
            visited[p[0]][p[1]] = true;
            if (p[0] == destination[0] && p[1] == destination[1]) return true;

            // 上下左右 边缘都是墙壁

            // 上
            int upWallIdx = 0;
            for (int i = p[0]; i >= 0; i--) {
                if (maze[i][p[1]] == 1) {
                    upWallIdx = i + 1;
                    break;
                }
            }
            q.offer(new int[]{upWallIdx, p[1]});

            // 下
            int downWallIdx = m - 1;
            for (int i = p[0]; i < m; i++) {
                if (maze[i][p[1]] == 1) {
                    downWallIdx = i - 1;
                    break;
                }
            }
            q.offer(new int[]{downWallIdx, p[1]});

            // 左
            int leftWallIdx = 0;
            for (int i = p[1]; i >= 0; i--) {
                if (maze[p[0]][i] == 1) {
                    leftWallIdx = i + 1;
                    break;
                }
            }
            q.offer(new int[]{p[0], leftWallIdx});


            int rightWallIdx = n - 1;
            for (int i = p[1]; i < n; i++) {
                if (maze[p[0]][i] == 1) {
                    rightWallIdx = i - 1;
                    break;
                }
            }
            q.offer(new int[]{p[0], rightWallIdx});
        }
        return false;
    }

    // LC1273
    Map<Integer, Integer> lc1273ParentMap;
    Map<Integer, Set<Integer>> lc1273ChildrenMap;
    Integer[] lc1273SubTreeSum;
    int[] lc1273Value;

    public int deleteTreeNodes(int nodes, int[] parent, int[] value) {
        lc1273ParentMap = new HashMap<>();
        lc1273ChildrenMap = new HashMap<>();
        this.lc1273Value = value;
        lc1273SubTreeSum = new Integer[nodes];
        for (int i = 0; i < nodes; i++) {
            lc1273ParentMap.put(i, parent[i]);
            lc1273ChildrenMap.putIfAbsent(i, new HashSet<>());
            lc1273ChildrenMap.putIfAbsent(parent[i], new HashSet<>());
            lc1273ChildrenMap.get(parent[i]).add(i);
        }
        lc1273Helper(lc1273ChildrenMap.get(-1).iterator().next());
        Deque<Integer> toRemove = new LinkedList<>();
        for (int i = 0; i < nodes; i++) {
            if (lc1273SubTreeSum[i] == 0) {
                toRemove.offer(i);
            }
        }
        while (!toRemove.isEmpty()) {
            int n = toRemove.poll();
            if (lc1273ChildrenMap.containsKey(n)) {
                for (int child : lc1273ChildrenMap.get(n)) {
                    toRemove.offer(child);
                }
            }
            lc1273ChildrenMap.remove(n);
            if (lc1273ChildrenMap.containsKey(lc1273ParentMap.get(n)))
                lc1273ChildrenMap.get(lc1273ParentMap.get(n)).remove(n);
        }
        int result = 0;
        for (Set<Integer> s : lc1273ChildrenMap.values()) {
            result += s.size();
        }
        return result;
    }

    private int lc1273Helper(int idx) {
        if (lc1273SubTreeSum[idx] != null) return lc1273SubTreeSum[idx];
        lc1273SubTreeSum[idx] = lc1273Value[idx];
        for (int child : lc1273ChildrenMap.get(idx)) {
            lc1273SubTreeSum[idx] += lc1273Helper(child);
        }
        return lc1273SubTreeSum[idx];
    }

    // LC1779
    public int nearestValidPoint(int x, int y, int[][] points) {
        int minIdx = -1, minDistance = Integer.MAX_VALUE;
        for (int i = 0; i < points.length; i++) {
            int[] p = points[i];
            if (p[0] == x || p[1] == y) {
                int dis = Math.abs(p[0] - x) + Math.abs(p[1] - y);
                if (dis < minDistance) {
                    minIdx = i;
                    minDistance = dis;
                }
            }
        }
        return minIdx;
    }

    // LC1564
    public int maxBoxesInWarehouse(int[] boxes, int[] warehouse) {
        Arrays.sort(boxes);
        // Next Smaller Element
        int n = warehouse.length;
        int min = warehouse[0];
        for (int i = 0; i < n; i++) {
            if (warehouse[i] >= min) {
                warehouse[i] = min;
            } else {
                min = warehouse[i];
            }
        }
        int ptr = n - 1, result = 0;
        for (int i = 0; i < boxes.length; i++) {
            if (ptr < 0) break;
            while (ptr >= 0 && warehouse[ptr] < boxes[i]) ptr--;
            if (ptr-- >= 0) result++;
        }
        return result;
    }

    // LC1580 O(nlog(n))
    public int maxBoxesInWarehouseII(int[] boxes, int[] warehouse) {
        Arrays.sort(boxes);
        int n = warehouse.length;
        int min = Arrays.stream(warehouse).min().getAsInt();
        int leftMost = -1, rightMost = -1;
        int curMin = warehouse[0];
        for (int i = 0; i < n; i++) {
            if (warehouse[i] != min) {
                if (warehouse[i] >= curMin) {
                    warehouse[i] = curMin;
                } else {
                    curMin = warehouse[i];
                }
            } else {
                leftMost = i;
                break;
            }
        }

        curMin = warehouse[n - 1];
        for (int i = n - 1; i >= 0; i--) {
            if (warehouse[i] != min) {
                if (warehouse[i] >= curMin) {
                    warehouse[i] = curMin;
                } else {
                    curMin = warehouse[i];
                }
            } else {
                rightMost = i;
                break;
            }
        }

        for (int i = leftMost; i < rightMost; i++) {
            warehouse[i] = min;
        }

        int leftPtr = rightMost, rightPtr = rightMost + 1;
        int result = 0;

        for (int i = 0; i < boxes.length; i++) {
            if (leftPtr < 0 && rightPtr >= n) break;

            // 比较左右两边哪一侧的格子的绝对值差最小

            int shadowLeftPtr = leftPtr, shadowRightPtr = rightPtr;

            while (shadowLeftPtr >= 0 && warehouse[shadowLeftPtr] < boxes[i]) shadowLeftPtr--;
            while (shadowRightPtr < n && warehouse[shadowRightPtr] < boxes[i]) shadowRightPtr++;

            int leftDistance = -1;
            if (shadowLeftPtr >= 0) leftDistance = warehouse[shadowLeftPtr] - boxes[i];
            int rightDistance = -1;
            if (shadowRightPtr < n) rightDistance = warehouse[shadowRightPtr] - boxes[i];

            if (leftDistance == -1 && rightDistance == -1) break;
            if ((leftDistance == -1 && rightDistance != -1) || (leftDistance != -1 && rightDistance != -1 && leftDistance > rightDistance)) {
                rightPtr = ++shadowRightPtr;
                result++;
            } else if ((leftDistance != -1 && rightDistance == -1) || (leftDistance != -1 && rightDistance != -1 && leftDistance <= rightDistance)) {
                leftPtr = --shadowLeftPtr;
                result++;
            }

        }
        return result;

    }

    // LC1283 二分
    public int smallestDivisor(int[] nums, int threshold) {
        int lo = 1, hi = Integer.MAX_VALUE;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sumDivide(nums, mid) <= threshold) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int sumDivide(int[] nums, int divider) {
        int result = 0;
        for (int i : nums) result += divideUpper(i, divider);
        return result;
    }

    private int divideUpper(int a, int b) {
        if (b == 0) throw new ArithmeticException("Zero divider!");
        if (a == 0) return 0;
        if (a % b == 0) return a / b;
        return a / b + 1;
    }


    // LC1415
    class Lc1415 {
        int kth = 0;
        int targetTh, len;
        String result;
        char[] valid = {'a', 'b', 'c'};

        public String getHappyString(int n, int k) {
            targetTh = k;
            len = n;
            backtrack(new StringBuilder());
            if (result == null) return "";
            return result;
        }

        private void backtrack(StringBuilder cur) {
            if (cur.length() == len) {
                if (++kth == targetTh) {
                    result = cur.toString();
                }
                return;
            }
            for (char c : valid) {
                if ((cur.length() > 0 && cur.charAt(cur.length() - 1) != c) || cur.length() == 0) {
                    cur.append(c);
                    backtrack(cur);
                    cur.deleteCharAt(cur.length() - 1);
                }
            }
        }
    }

    // LC536
    public TreeNode str2tree(String s) {
        if (s.equals("")) return null;
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        boolean number = true, left = false, right = false;
        TreeNode root = new TreeNode();
        int val = -1, pair = 0;
        int startOfLeft = -1, endOfLeft = -1, startOfRight = -1, endOfRight = -1;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (number && (c == '(' || c == ')')) {
                number = false;
                val = Integer.valueOf(sb.toString());
                root.val = val;
                left = true;
                startOfLeft = i;
            }
            if (number) {
                sb.append(c);
            } else if (left) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfLeft = i;
                        TreeNode leftNode = str2tree(s.substring(startOfLeft + 1, endOfLeft));
                        root.left = leftNode;
                        startOfRight = i + 1;
                        right = true;
                        left = false;
                    }
                }
            } else if (right) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfRight = i;
                        TreeNode rightNode = str2tree(s.substring(startOfRight + 1, endOfRight));
                        root.right = rightNode;
                        right = false;
                    }
                }
            }
        }
        if (number) {
            val = Integer.valueOf(sb.toString());
            root.val = val;
        }
        return root;
    }

    // LC1608
    public int specialArray(int[] nums) {
        int[] count = new int[1001];
        for (int i : nums) count[i]++;
        int ctr = 0;
        for (int i = 1000; i >= 0; i--) {
            ctr += count[i];
            if (ctr == i) return i;
        }
        return -1;
    }

    // LC447
    public int numberOfBoomerangs(int[][] points) {
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int[] pi = points[i];
            Map<Integer, Integer> m = new HashMap<>();
            for (int j = 0; j < points.length; j++) {
                if (i != j) {
                    int[] pj = points[j];
                    int distance = (pi[0] - pj[0]) * (pi[0] - pj[0]) + (pi[1] - pj[1]) * (pi[1] - pj[1]);
                    m.put(distance, m.getOrDefault(distance, 0) + 1);
                }
            }
            for (int e : m.keySet()) {
                result += m.get(e) * (m.get(e) - 1);
            }
        }
        return result;
    }

    // LC1955 ** DP
    public int countSpecialSubsequences(int[] nums) {
        int i0 = 0, i1 = 0, i2 = 0;
        final int mod = 1000000007;
        for (int i : nums) {
            switch (i) {
                case 0:
                    i0 = ((i0 * 2) + 1) % mod;
                    break;
                case 1:
                    i1 = (((i1 * 2) % mod) + i0) % mod;
                    break;
                case 2:
                    i2 = (((i2 * 2) % mod) + i1) % mod;
                    break;
                default:
                    continue;
            }
        }
        return i2;
    }

    // LC600 ** 数位DP
    public int findIntegers(int n) {
        if (n == 0) return 1;
        int[] dp = new int[32];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < 32; i++) { // fib???
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        int prev = 0, result = 0;
        int len = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        for (int i = len; i >= 1; i--) {
            int cur = (n >> (i - 1)) & 1;
            if (cur == 1) {
                result += dp[i];
            }
            if (cur == 1 && prev == 1) break;
            prev = cur;
            if (i == 1) result++;
        }
        return result;
    }


    // LC898 ** 看题解
    // https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/solution/zi-shu-zu-an-wei-huo-cao-zuo-by-leetcode/
    public int subarrayBitwiseORs(int[] arr) {
        Set<Integer> result = new HashSet<>();
        Set<Integer> cur = new HashSet<>();
        for (int i : arr) {
            Set<Integer> tmp = new HashSet<>();
            for (int j : cur) { // 最多有32个数 (1的个数是递增的) ???
                tmp.add(i | j);
            }
            tmp.add(i); // 记得加上自身(长度为1)
            cur = tmp;
            result.addAll(cur);
        }
        return result.size();
    }

    // LC248
    public int strobogrammaticInRange(String low, String high) {
        int count = 0;
        for (int i = low.length(); i <= high.length(); i++) {
            List<String> result = findStrobogrammatic(i);
            if (i > low.length() && i < high.length()) {
                count += result.size();
                continue;
            }
            for (String s : result) {
                if (bigIntCompare(s, low) >= 0 && bigIntCompare(s, high) <= 0) {
                    count++;
                }
            }
        }
        return count;
    }

    private int bigIntCompare(String a, String b) {
        if (a.equals(b)) return 0;
        if (a.length() < b.length()) return -1;
        if (a.length() > b.length()) return 1;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) > b.charAt(i)) return 1;
            if (a.charAt(i) < b.charAt(i)) return -1;
        }
        return 0;
    }

    // LC247
    int[] validDigit = {0, 1, 6, 8, 9};
    int[] symmetryDigit = {0, 1, 8};
    List<String> lc247Result;

    public List<String> findStrobogrammatic(int n) {
        lc247Result = new ArrayList<>();
        if (n == 1) return Arrays.asList("0", "1", "8");
        lc247Helper(new StringBuilder(), n);
        return lc247Result;
    }

    private void lc247Helper(StringBuilder sb, int total) {
        if (sb.length() == total / 2) {
            if (sb.charAt(0) == '0') return;
            if (total % 2 == 1) {
                for (int i : symmetryDigit) {
                    String r = sb.toString() + i + getReverse(sb);
                    lc247Result.add(r);
                }
            } else {
                String r = sb + getReverse(sb);
                lc247Result.add(r);
            }
            return;
        }
        for (int i : validDigit) {
            if (i == 0 && sb.length() == 0) continue;
            sb.append(i);
            lc247Helper(sb, total);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    private String getReverse(StringBuilder input) {
        StringBuilder sb = new StringBuilder();
        for (int i = input.length() - 1; i >= 0; i--) {
            if (input.charAt(i) == '6') {
                sb.append('9');
            } else if (input.charAt(i) == '9') {
                sb.append('6');
            } else {
                sb.append(input.charAt(i));
            }
        }
        return sb.toString();
    }

    // LC246
    public boolean isStrobogrammatic(String num) {
        int[] notValid = {2, 3, 4, 5, 7};
        char[] ca = num.toCharArray();
        for (int i = 0; i <= ca.length / 2; i++) {
            char c = ca[i];
            for (int j : notValid) if (c - '0' == j) return false;
            if (c == '6') {
                if (ca[ca.length - 1 - i] != '9') return false;
            } else if (c == '9') {
                if (ca[ca.length - 1 - i] != '6') return false;
            } else {
                if (ca[ca.length - 1 - i] != c) return false;
            }
        }
        return true;
    }

    // LC1953 Hint: 只和最大时间有关
    public long numberOfWeeks(int[] milestones) {
        long sum = 0;
        long max = Long.MIN_VALUE;
        for (int i : milestones) {
            sum += i;
            max = Math.max(max, i);
        }
        long remain = sum - max;
        max = Math.min(remain + 1, max);
        return remain + max;
    }

    // LC249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<>();
        Map<Integer, Map<String, Integer>> m = new HashMap<>();
        for (String s : strings) {
            m.putIfAbsent(s.length(), new HashMap<>());
            Map<String, Integer> inner = m.get(s.length());
            inner.put(s, inner.getOrDefault(s, 0) + 1);
        }

        for (Map<String, Integer> s : m.values()) {
            while (!s.isEmpty()) {
                String w = s.keySet().iterator().next();
                // 构造
                List<String> list = new ArrayList<>();
                char[] ca = w.toCharArray();
                for (int i = 0; i < 26; i++) {
                    for (int j = 0; j < ca.length; j++) {
                        ca[j] = (char) (((ca[j] - 'a' + 1) % 26) + 'a');
                    }
                    String built = new String(ca);
                    if (s.containsKey(built)) {
                        int count = s.get(built);
                        s.remove(built);
                        for (int j = 0; j < count; j++)
                            list.add(built);
                    }
                }
                result.add(list);
            }
        }
        return result;
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

// JZOF II 30 LC380
class RandomizedSet {

    Map<Integer, Integer> idxMap = new HashMap<>();
    List<Integer> entities = new ArrayList<>();

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {

    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified element.
     */
    public boolean insert(int val) {
        if (idxMap.containsKey(val)) return false;
        idxMap.put(val, entities.size());
        entities.add(val);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!idxMap.containsKey(val)) return false;
        int lastEntity = entities.get(entities.size() - 1);
        int targetIdx = idxMap.get(val);
        entities.set(targetIdx, lastEntity);
        idxMap.put(lastEntity, targetIdx);
        idxMap.remove(val);
        entities.remove(entities.size() - 1);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        int idx = (int) (Math.random() * entities.size());
        return entities.get(idx);
    }
}

class DisjointSetUnion<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
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
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
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

    public boolean isConnected(T i, T j) {
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}


// LC1825 Hard
class MKAverage {
    final int MK_M, MK_K;
    Deque<Integer> dataStream = new LinkedList<>();
    TreeMap<Integer, Integer> tm = new TreeMap<>();

    public MKAverage(int m, int k) {
        MK_M = m;
        MK_K = k;
    }

    public void addElement(int num) {
        dataStream.offer(num);
        tm.put(num, tm.getOrDefault(num, 0) + 1);
        while (dataStream.size() > MK_M) {
            tm.put(dataStream.peekFirst(), tm.get(dataStream.peekFirst()) - 1);
            if (tm.get(dataStream.peekFirst()) == 0) {
                tm.remove(dataStream.peekFirst());
            }
            dataStream.poll();
        }
    }

    public int calculateMKAverage() {
        if (dataStream.size() < MK_M) return -1;
        int leftBound = -1, rightBound = -1;
        int origLeftBoundValue = -1, origRightBoundValue = -1;
        int ctr = 0;
        Iterator<Integer> it = tm.navigableKeySet().iterator();
        while (ctr <= MK_K && it.hasNext()) {
            int firstKey = it.next();
            if (ctr + tm.get(firstKey) <= MK_K) {
                ctr += tm.get(firstKey);
            } else {
                ctr += tm.get(firstKey);
                leftBound = firstKey;
                origLeftBoundValue = tm.get(firstKey);
                tm.put(firstKey, ctr - MK_K);
                break;
            }
        }
        ctr = 0;
        it = tm.descendingKeySet().iterator();
        while (ctr <= MK_K && it.hasNext()) {
            int lastKey = it.next();
            if (ctr + tm.get(lastKey) <= MK_K) {
                ctr += tm.get(lastKey);
            } else {
                ctr += tm.get(lastKey);
                rightBound = lastKey;
                origRightBoundValue = tm.get(rightBound);
                tm.put(lastKey, ctr - MK_K);
                break;
            }
        }

        int sum = 0;
        NavigableMap<Integer, Integer> subMap = tm.subMap(leftBound, true, rightBound, true);

        for (int key : subMap.keySet()) {
            sum += key * subMap.get(key);
        }
        tm.put(rightBound, origRightBoundValue);
        tm.put(leftBound, origLeftBoundValue);
        return sum / (MK_M - 2 * MK_K);
    }
}

// LC346
class MovingAverage {
    final int SIZE;
    double sum = 0d;
    int[] q;
    int last = -1;
    int count = 0;


    /**
     * Initialize your data structure here.
     */
    public MovingAverage(int size) {
        SIZE = size;
        q = new int[size];
        last = -1;
    }

    public double next(int val) {
        sum += val;
        if (count < SIZE) {
            count++;
        } else {
            sum -= q[(last + 1) % SIZE];
        }
        q[(++last) % SIZE] = val;
        return sum / count;
    }
}


// LC1117 ** 多线程
class H2O {

    int count = 0;
    ArrayBlockingQueue<Integer> hq = new ArrayBlockingQueue<>(2);
    ArrayBlockingQueue<Integer> oq = new ArrayBlockingQueue<>(1);

    public H2O() {

    }

    public void hydrogen(Runnable releaseHydrogen) throws InterruptedException {
        hq.put(1);
        releaseHydrogen.run();
        count++;

        if (count >= 3) {
            count = 0;
            hq.clear();
            oq.clear();
        }

        // releaseHydrogen.run() outputs "H". Do not change or remove this line.
//        releaseHydrogen.run();
    }

    public void oxygen(Runnable releaseOxygen) throws InterruptedException {
        oq.put(1);
        releaseOxygen.run();
        count++;
        if (count >= 3) {
            count = 0;
            hq.clear();
            oq.clear();
        }

        // releaseOxygen.run() outputs "O". Do not change or remove this line.
//        releaseOxygen.run();
    }
}

// LC1114
class Foo {

    ReentrantLock lock = new ReentrantLock();
    Condition firstReady = lock.newCondition();
    Condition secondReady = lock.newCondition();
    Condition thirdReady = lock.newCondition();
    boolean secondAllow = false;
    boolean thirdAllow = false;

    public Foo() {

    }

    public void first(Runnable printFirst) throws InterruptedException {

        lock.lock();
        try {
            // printFirst.run() outputs "first". Do not change or remove this line.
            printFirst.run();
            secondAllow = true;
            firstReady.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public void second(Runnable printSecond) throws InterruptedException {
        lock.lock();
        try {
            // printSecond.run() outputs "second". Do not change or remove this line.
            while (!secondAllow) {
                firstReady.await();
            }
            printSecond.run();
            thirdAllow = true;
            secondReady.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public void third(Runnable printThird) throws InterruptedException {
        lock.lock();
        try {
            while (!thirdAllow) {
                secondReady.await();
            }

            // printThird.run() outputs "third". Do not change or remove this line.
            printThird.run();
            secondAllow = false;
            thirdAllow = false;

        } finally {
            lock.unlock();
        }
    }
}

// LC1188 阻塞队列 参考了jdk8 ArrayBlockingQueue的实现
class BoundedBlockingQueue {

    ReentrantLock lock = new ReentrantLock();
    Condition notEmpty = lock.newCondition();
    Condition notFull = lock.newCondition();
    int size = 0;
    int[] queue;
    int len;
    int takeIdx = 0;
    int putIdx = 0;

    public BoundedBlockingQueue(int capacity) {
        len = capacity;
        queue = new int[len];
    }

    public void enqueue(int element) throws InterruptedException {
        lock.lock();
        try {
            while (size == len) {
                notFull.await();
            }
            insert(element);
        } finally {
            lock.unlock();
        }
    }

    public int dequeue() throws InterruptedException {
        lock.lock();
        try {
            while (size == 0) {
                notEmpty.await();
            }

            return extract();
        } finally {
            lock.unlock();
        }
    }

    private int extract() {
        int result = queue[takeIdx];
        queue[takeIdx] = 0;
        takeIdx = inc(takeIdx);
        size--;
        notFull.signalAll();
        return result;
    }

    private void insert(int ele) {
        queue[putIdx] = ele;
        putIdx = inc(putIdx);
        size++;
        notEmpty.signalAll();

    }

    public int size() {
        return size;
    }

    private int inc(int idx) {
        return (++idx == len) ? 0 : idx;
    }

    private int dec(int idx) {
        return (idx == 0 ? len : idx) - 1;
    }
}

// LC1195
class FizzBuzz {

    Object lock = new Object();
    int cur = 1;

    private int n;

    public FizzBuzz(int n) {
        this.n = n;
    }

    // printFizz.run() outputs "fizz". // MOD 3 = 0
    public void fizz(Runnable printFizz) throws InterruptedException {
        synchronized (lock) {
            while (cur <= n) {
                if (cur % 3 == 0 && cur % 5 != 0) {
                    printFizz.run();
                    cur++;
                    lock.notifyAll();
                } else {
                    lock.wait();
                }
            }
        }
    }

    // printBuzz.run() outputs "buzz". // MOD 5 = 0
    public void buzz(Runnable printBuzz) throws InterruptedException {
        synchronized (lock) {
            while (cur <= n) {
                if (cur % 3 != 0 && cur % 5 == 0) {
                    printBuzz.run();
                    cur++;
                    lock.notifyAll();
                } else {
                    lock.wait();
                }
            }
        }
    }

    // printFizzBuzz.run() outputs "fizzbuzz". // MOD 15 = 0
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
        synchronized (lock) {
            while (cur <= n) {
                if (cur % 3 == 0 && cur % 5 == 0) {
                    printFizzBuzz.run();
                    cur++;
                    lock.notifyAll();
                } else {
                    lock.wait();
                }
            }
        }
    }

    // printNumber.accept(x) outputs "x", where x is an integer. // ELSE
    public void number(IntConsumer printNumber) throws InterruptedException {
        synchronized (lock) {
            while (cur <= n) {
                if (cur % 3 != 0 && cur % 5 != 0) {
                    printNumber.accept(cur);
                    cur++;
                    lock.notifyAll();
                } else {
                    lock.wait();
                }
            }
        }
    }
}

// LC1242
interface HtmlParser {
    List<String> getUrls(String url);
}

class HP implements HtmlParser {

    @Override
    public List<String> getUrls(String url) {
        LockSupport.parkNanos(3000 * 1000);
        List<String> s = Arrays.asList(
                "http：//leetcode.com",
                "http：//leetcode.com/",
                "http：//leetcode.com/b",
                "http：//leetcode.com/c",
                "http：//leetcode.com/d",
                "http：//leetcode.com/e",
                "http：//leetcode.com/f",
                "http：//leetcode.com/g");
        Collections.shuffle(s);
        return s.subList(0, 5);
    }
}

// LC1242 TLE
class Lc1242 {
    ConcurrentHashMap<String, Object> set = new ConcurrentHashMap<>();

    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        ForkJoinPool fjp = new ForkJoinPool(32);

        Task t = new Task(startUrl, htmlParser);
        ForkJoinTask<Void> submit = fjp.submit(t);
        try {
            submit.get();
            return new ArrayList<>(set.keySet());
        } catch (ExecutionException | InterruptedException e) {
            return new ArrayList<>();
        } finally {
            fjp.shutdown();
        }
    }

    class Task extends RecursiveAction {
        String startUrl;
        HtmlParser htmlParser;

        public Task(String startUrl, HtmlParser htmlParser) {
            this.startUrl = startUrl;
            this.htmlParser = htmlParser;
        }

        @Override
        protected void compute() {
            if (set.containsKey(startUrl)) return;
            set.put(startUrl, new Object());
            List<Task> taskList = new ArrayList<>();

            List<String> nextLevel = htmlParser.getUrls(startUrl);
            for (String next : nextLevel) {
                if (!set.containsKey(next) && isSameHost(next, startUrl)) {
                    Task nt = new Task(next, htmlParser);
                    taskList.add(nt);
                    nt.fork();
                }
            }
            for (Task t : taskList) {
                t.join();
            }
        }
    }

    private String getHostName(String url) {
        int thirdSlashIdx = url.indexOf('/', 7);
        return url.substring(7, thirdSlashIdx <= 0 ? url.length() : thirdSlashIdx);
    }

    private boolean isSameHost(String a, String b) {
        return getHostName(a).equals(getHostName(b));
    }
}

// LC1279
class TrafficLight {

    ReentrantLock lock = new ReentrantLock();
    boolean roadOneGreen = true;

    public TrafficLight() {

    }


    // 1,2 -> a/1
    // 3,4 -> b/2
    public void carArrived(
            int carId,           // ID of the car
            int roadId,          // ID of the road the car travels on. Can be 1 (road A) or 2 (road B)
            int direction,       // Direction of the car
            Runnable turnGreen,  // Use turnGreen.run() to turn light to green on current road
            Runnable crossCar    // Use crossCar.run() to make car cross the intersection
    ) {
        lock.lock();
        try {
            if ((roadId == 1 && !roadOneGreen) || (roadId == 2 && roadOneGreen)) {
                roadOneGreen = !roadOneGreen;
                turnGreen.run();
            }
            crossCar.run();
        } finally {
            lock.unlock();
        }
    }
}

// LC1115
class FooBar {

    boolean barPrinted = true;
    boolean fooPrinted = false;
    ReentrantLock lock = new ReentrantLock();
    Condition fooDone = lock.newCondition();
    Condition barDone = lock.newCondition();

    private int n;

    public FooBar(int n) {
        this.n = n;
    }

    public void foo(Runnable printFoo) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            lock.lock();
            try {
                while (!barPrinted) {
                    barDone.await();
                }
                printFoo.run();
                fooPrinted = true;
                barPrinted = false;
                fooDone.signalAll();
            } finally {
                lock.unlock();
            }
        }
    }

    public void bar(Runnable printBar) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            lock.lock();
            try {
                while (!fooPrinted) {
                    fooDone.await();
                }
                printBar.run();
                fooPrinted = false;
                barPrinted = true;
                barDone.signalAll();
            } finally {
                lock.unlock();
            }
        }
    }
}

// LC1226
class DiningPhilosophers {

    ReentrantLock[] forks = new ReentrantLock[5];

    public DiningPhilosophers() {
        for (int i = 0; i < 5; i++) {
            forks[i] = new ReentrantLock();
        }
    }

    // call the run() method of any runnable to execute its code
    public void wantsToEat(int philosopher,
                           Runnable pickLeftFork,
                           Runnable pickRightFork,
                           Runnable eat,
                           Runnable putLeftFork,
                           Runnable putRightFork) throws InterruptedException {
        int left = philosopher, right = (philosopher + 1) % 5;
        ReentrantLock lockLeft = forks[left], lockRight = forks[right];
        lockLeft.lock();
        try {
            lockRight.lock();
            try {
                pickLeftFork.run();
                pickRightFork.run();
                eat.run();
                putLeftFork.run();
                putRightFork.run();
            } finally {
                lockRight.unlock();
            }
        } finally {
            lockLeft.unlock();
        }
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
        next = null;
    }
}

class BIT {
    long[] tree;
    int len;

    public BIT(int len) {
        this.len = len;
        this.tree = new long[len + 1];
    }

    public BIT(int[] arr) {
        this.len = arr.length;
        this.tree = new long[len + 1];
        for (int i = 0; i < arr.length; i++) {
            int oneBasedIdx = i + 1;
            tree[oneBasedIdx] += arr[i];
            int nextOneBasedIdx = oneBasedIdx + lowbit(oneBasedIdx);
            if (nextOneBasedIdx <= len) tree[nextOneBasedIdx] += tree[oneBasedIdx];
        }
    }

    public void set(int idxZeroBased, long val) {
        long delta = val - get(idxZeroBased);
        update(idxZeroBased, delta);
    }

    public long get(int idxZeroBased) {
        return sumOneBased(idxZeroBased + 1) - sumOneBased(idxZeroBased);
    }

    public void update(int idxZeroBased, long delta) {
        updateOneBased(idxZeroBased + 1, delta);
    }

    public long sumRange(int left, int right) {
        return sumOneBased(right + 1) - sumOneBased(left);
    }

    public void updateOneBased(int idxOneBased, long delta) {
        while (idxOneBased <= len) {
            tree[idxOneBased] += delta;
            idxOneBased += lowbit(idxOneBased);
        }
    }

    public long sumOneBased(int idxOneBased) {
        int sum = 0;
        while (idxOneBased > 0) {
            sum += tree[idxOneBased];
            idxOneBased -= lowbit(idxOneBased);
        }
        return sum;
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}
