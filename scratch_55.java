import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


//        System.out.println(s.numSquarefulPerms(new int[]{13949, 64411, 26920, 5204, 2177, 23617, 44128, 3455, 47315, 40706, 45874, 22858}));
//        System.out.println(s.numSquarefulPerms(new int[]{1, 17, 8}));
        System.out.println(s.numSquarefulPerms(new int[]{1, 1, 8, 1, 8}));
//        System.out.println(s.numSquarefulPerms(new int[]{5, 11, 5, 4, 5}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1673 长度为K的字典序最小的子序列 单调栈 **
    public int[] mostCompetitive(int[] nums, int k) {
        if (k == nums.length) return nums;
        if (k == 1) return new int[]{Arrays.stream(nums).min().getAsInt()};
        int n = nums.length;
        Deque<Integer> stack = new LinkedList<>(); // 单调递增栈
        int removable = n - k; // 最多只能移除 n-k个, 否则不足k个数字; 移除即出栈
        for (int i = 0; i < n; i++) {
            while (removable > 0 && !stack.isEmpty() && nums[i] < stack.peek()) {
                stack.pop();
                removable--;
            }
            stack.push(nums[i]);
        }
        while (stack.size() > k) stack.pop();
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[k - 1 - i] = stack.pop();
        }
        return result;
    }

    // LC1669
    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode dummy = new ListNode(-1);
        dummy.next = list1;
        if (a == 0) dummy.next = list2;
        ListNode prev = null, cur = list1;
        ListNode firstHalfEnd = new ListNode(-1), secondHalfStart = new ListNode(-1);
        int idx = 0;
        for (int i = 0; i <= b; i++) {
            if (idx == a) {
                firstHalfEnd = prev;
            }
            if (idx == b) {
                secondHalfStart = cur.next;
            }
            prev = cur;
            cur = cur.next;
            idx++;
        }

        firstHalfEnd.next = list2;
        cur = list2;
        while (cur.next != null) cur = cur.next;
        cur.next = secondHalfStart;
        return dummy.next;
    }

    // LC996
    int lc996Result = 0;

    public int numSquarefulPerms(int[] nums) {
        lc996Helper(nums, 0);
        return lc996Result;
    }

    // 朴素全排列
    private void lc996Helper(int[] arr, int cur) {
        if (cur == arr.length - 1) {
            if (lc996Check(arr)) lc996Result++;
            return;
        }
        if (cur > 1) {
            int sum = arr[cur - 1] + arr[cur - 2];
            int sqrt = (int) Math.sqrt(sum);
            if (sqrt * sqrt != sum) return;
        }
        Set<Integer> set = new HashSet<>();
        for (int i = cur; i < arr.length; i++) {
            if (!set.add(arr[i])) continue;
            int tmp = arr[cur];
            arr[cur] = arr[i];
            arr[i] = tmp;
            lc996Helper(arr, cur + 1);
            tmp = arr[cur];
            arr[cur] = arr[i];
            arr[i] = tmp;
        }
    }

    private boolean lc996Check(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int sum = arr[i] + arr[i - 1];
            int sqrt = (int) Math.sqrt(sum);
            if (sqrt * sqrt != sum) return false;
        }
        return true;
    }

    private boolean nextPerm(int[] arr) {
        int right = arr.length - 2;
        while (right >= 0 && arr[right] >= arr[right + 1]) {
            right--;
        }
        if (right < 0) return false;
        if (right >= 0) {
            int left = arr.length - 1;
            while (left >= 0 && arr[right] >= arr[left]) {
                left--;
            }
            arrSwap(arr, left, right);
        }
        arrReverse(arr, right + 1, arr.length - 1);
        return true;
    }

    private void arrSwap(int[] arr, int a, int b) {
        if (arr[a] != arr[b]) {
            arr[a] ^= arr[b];
            arr[b] ^= arr[a];
            arr[a] ^= arr[b];
        }
    }

    private void arrReverse(int[] arr, int a, int b) {
        for (int i = 0; i < (b - a + 1) / 2; i++) {
            arrSwap(arr, a + i, b - i);
        }
    }

    // LC164 Try Radix sort
    public int maximumGap(int[] nums) {
        int n = nums.length;
        if (n < 2) return 0;
        int max = Integer.MIN_VALUE;
        for (int i : nums) max = Math.max(max, i);
        int maxDigit = Integer.SIZE - Integer.numberOfLeadingZeros(max);
        int[] buf = new int[n];
        int count0 = 0;
        for (int i = 0; i < maxDigit; i++) {
            count0 = 0;
            for (int j : nums) {
                if (((j >> i) & 1) == 0) count0++;
            }
            int idx0 = 0, idx1 = count0;
            for (int j : nums) {
                if (((j >> i) & 1) == 0) {
                    buf[idx0++] = j;
                } else {
                    buf[idx1++] = j;
                }
            }
            int[] tmp = nums;
            nums = buf;
            buf = tmp;
        }
        int result = 0;
        for (int i = 1; i < n; i++) {
            result = Math.max(result, nums[i] - nums[i - 1]);
        }
        return result;
    }


    // LC1773
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        int matchIdx = -1, result = 0;
        switch (ruleKey) {
            case "type":
                matchIdx = 0;
                break;
            case "color":
                matchIdx = 1;
                break;
            case "name":
                matchIdx = 2;
                break;
            default:
        }
        for (List<String> i : items) {
            if (i.get(matchIdx).equals(ruleValue)) {
                result++;
            }
        }
        return result;
    }

    // JZOF II 069 LC852 **
    public int peakIndexInMountainArray(int[] arr) {
        int n = arr.length;
        int lo = 1, hi = n - 2, max = 1; // 搜索范围不包括两个端点
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] > arr[mid + 1]) {
                max = mid; // 在一个右肩位置, 往左继续搜索
                hi = mid - 1;
            } else {
                lo = mid + 1; // 否则在左肩位置, 或者平原地带, 往右搜索
            }
        }
        return max;
    }

    // JZOF 04 LC240
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        int col = n - 1, row = 0;
        while (col >= 0 && row < m) {
            if (matrix[row][col] == target) return true;
            if (matrix[row][col] < target) row++;
            else col--;
        }
        return false;
    }

    // LC386
    public List<Integer> lexicalOrder(int n) {
        List<Integer> result = new ArrayList<>(n);
        for (int i = 1; i <= 9; i++) {
            if (i > n) break;
            result.add(i);
            lc386Helper(i, n, result);
        }
        return result;
    }

    private void lc386Helper(int prefix, int n, List<Integer> result) {
        prefix *= 10;
        for (int i = 0; i <= 9; i++) {
            if (prefix + i > n) break;
            result.add(prefix + i);
            lc386Helper(prefix + i, n, result);
        }
    }

    // LC1898
    public int maximumRemovals(String s, String p, int[] removable) {
        int lo = 0, hi = removable.length;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (check(s, p, removable, mid)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    private boolean check(String s, String p, int[] removable, int k) {
        if (s.length() - k < p.length()) return false;
        Set<Integer> idxSet = new HashSet<>(k);
        for (int i = 0; i < k; i++) idxSet.add(removable[i]);
        int sptr = 0, pptr = 0;
        while (pptr != p.length()) {
            if (sptr == s.length()) return false;
            if (idxSet.contains(sptr)) {
                sptr++;
                continue;
            }
            if (s.charAt(sptr) == p.charAt(pptr)) {
                sptr++;
                pptr++;
            } else {
                sptr++;
            }
        }
        return true;
    }

    // LC1958
    public boolean checkMove(char[][] board, int rMove, int cMove, char color) {
        int m = board.length, n = board[0].length;
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
        // 8个方向
        for (int[] d : directions) {
            char middle, endpoint;
            int nr = rMove + d[0], nc = cMove + d[1];
            // 如果这个方向无路可走
            if (nr < 0 || nr >= m || nc < 0 || nc >= n) {
                continue;
            }
            middle = board[nr][nc];
            if (middle == '.') {
                continue;
            }
            endpoint = color;
            if (middle == endpoint) {
                continue;
            }
            int steps = 2;
            while (nr + d[0] >= 0 && nr + d[0] < m && nc + d[1] >= 0 && nc + d[1] < n) {
                nr += d[0];
                nc += d[1];
                steps++;
                if (board[nr][nc] != middle) break;
            }
            if (steps < 3) {
                continue;
            }
            if (board[nr][nc] != endpoint) {
                continue;
            }
            return true;
        }
        return false;
    }

    // LC2009 Hard
    public int minOperationsLc2009(int[] nums) {
        int n = nums.length, result = Integer.MAX_VALUE;
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i : nums) tm.put(i, null);
        int ctr = 0;
        for (int i : tm.keySet()) {
            tm.put(i, ctr++);
        }
        Iterator<Map.Entry<Integer, Integer>> it = tm.entrySet().iterator();

        while (it.hasNext()) {
            Map.Entry<Integer, Integer> entry = it.next();
            int next = entry.getKey();
            // 上界
            int upperBound;
            if (Integer.MAX_VALUE - n + 1 < next) {
                upperBound = Integer.MAX_VALUE;
            } else {
                upperBound = next + n - 1;
            }

            int left = tm.get(next);
            Integer right = tm.floorKey(upperBound);
            if (right == null) right = tm.size();
            else right = tm.get(right) + 1;

            result = Math.min(result, n - (right - left));
        }
        return result;
    }

    // Interview 08.10
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int[][] directions = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int m = image.length, n = image[0].length, origColor = image[sr][sc];
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        q.offer(new int[]{sr, sc});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            if (visited[p[0]][p[1]]) continue;
            visited[p[0]][p[1]] = true;
            image[p[0]][p[1]] = newColor;
            for (int[] d : directions) {
                int nr = p[0] + d[0], nc = p[1] + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc] && image[nr][nc] == origColor) {
                    q.offer(new int[]{nr, nc});
                }
            }
        }
        return image;
    }

    // LC1463 Hard
    Integer[][][] lc1463Memo;

    public int cherryPickup(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        lc1463Memo = new Integer[n + 1][n + 1][m + 1];
        return lc1463Helper(0, n - 1, 0, grid);
    }

    private int lc1463Helper(int r1Col, int r2Col, int row, int[][] grid) {
        if (row == grid.length) return Integer.MIN_VALUE;
        if (lc1463Memo[r1Col][r2Col][row] != null) return lc1463Memo[r1Col][r2Col][row];
        int result = 0;
        for (int i = -1; i < 2; i++) {
            int r1NextCol = r1Col + i;
            if (r1NextCol >= 0 && r1NextCol < grid[0].length) {
                for (int j = -1; j < 2; j++) {
                    int r2NextCol = r2Col + j;
                    if (r2NextCol >= 0 && r2NextCol < grid[0].length) {
                        result = Math.max(result, lc1463Helper(r1NextCol, r2NextCol, row + 1, grid));
                    }
                }
            }
        }
        if (r1Col == r2Col) result += grid[row][r1Col];
        else result += grid[row][r1Col] + grid[row][r2Col];
        return lc1463Memo[r1Col][r2Col][row] = result;
    }

    // LC1351 O(m+n)
    public int countNegatives(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int col = n - 1, row = 0;
        int result = 0;
        while (col >= 0 && row < m) {
            while (col >= 0 && row < m && grid[row][col] >= 0) {
                row++;
            }
            if (row == m) break;
            // now the current row is negative
            result += m - row;
            col--;
        }
        return result;
    }

    // LC2000
    public String reversePrefix(String word, char ch) {
        char[] ca = word.toCharArray();
        int targetIdx = -1;
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == ch) {
                targetIdx = i;
                break;
            }
        }
        if (targetIdx == -1) return word;
        for (int i = 0; i <= (targetIdx / 2); i++) {
            char tmp = ca[i];
            ca[i] = ca[targetIdx - i];
            ca[targetIdx - i] = tmp;
        }
        return new String(ca);
    }

    // LC1999
    int lc1999Result = Integer.MAX_VALUE;

    public int findInteger(int k, int digit1, int digit2) {
        lc1999Helper(0, digit1, digit2, k);
        if (lc1999Result == Integer.MAX_VALUE) return -1;
        return lc1999Result;
    }

    private void lc1999Helper(int cur, int d1, int d2, int k) {
        if (cur > k && cur % k == 0) {
            lc1999Result = Math.min(lc1999Result, cur);
            return;
        }
        if (cur >= 1e9) return;

        if (cur == 0) {
            if (d1 != 0) {
                lc1999Helper(d1, d1, d2, k);
            }
            if (d2 != 0) {
                lc1999Helper(d2, d1, d2, k);
            }
        } else {
            if (cur <= Integer.MAX_VALUE / 10) {
                lc1999Helper(cur * 10 + d1, d1, d2, k);
                lc1999Helper(cur * 10 + d2, d1, d2, k);
            }
        }
    }

    // JZOF II 022 LC142
    public ListNode detectCycle(ListNode head) {
        // a b c
        // b+c = 环长
        // slow fast 在 b 相遇
        // fast 在环上走了 (b+c)*k + b 路程, slow在环上走了 b 路程
        // 又 (b+c)*k+b+a = 2*(b+a)
        // 得 b+a = (b+c)*k, a = (b+c)*k-b = c + (k-1)(b+c)
        ListNode slow = head, fast = head;
        while (fast != null) {
            fast = fast.next;
            if (fast != null) fast = fast.next;
            else return null;
            slow = slow.next;

            if (fast == slow) {
                ListNode chase = head;
                while (slow != chase) {
                    slow = slow.next;
                    chase = chase.next;
                }
                return chase;
            }
        }
        return null;
    }

    // LC1700 O(n) time O(1) space
    public int countStudents(int[] students, int[] sandwiches) {
        int n = students.length;
        int[] statistic = new int[2];
        for (int i : students) statistic[i]++;
        int count = 0;
        while (count < n && statistic[sandwiches[count]] > 0) {
            statistic[sandwiches[count]]--;
            count++;
        }
        return n - count;
    }

    // LC1590
    public int minSubarray(int[] nums, int mod) {
        int n = nums.length;
        long sum = 0;
        for (int i : nums) sum += i;
        if (sum % mod == 0) return 0;
        long sumSoFar = 0;
        int target = (int) (sum % mod);
        int result = Integer.MAX_VALUE;
        Map<Integer, Integer> remainderSumMap = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            if (nums[i - 1] == target) return 1;
            sumSoFar = sumSoFar + nums[i - 1];
            int remainder = (int) (sumSoFar % mod);
            if (remainder == target) {
                result = Math.min(result, i);
            }
            int key = ((remainder - target + mod) % mod);
            Integer victim;
            if ((victim = remainderSumMap.get(key)) != null) {
                result = Math.min(result, i - victim);
            }
            remainderSumMap.put(remainder, i);
        }
        if (result == Integer.MAX_VALUE || result == n) return -1;
        return result;
    }

    // LC1554
    public boolean differByOne(String[] dict) {
        Set<Integer> set31 = new HashSet<>(), set29 = new HashSet<>();
        for (String w : dict) {
            char[] ca = w.toCharArray();
            for (int i = 0; i < ca.length; i++) {
                char tmp = ca[i];
                ca[i] = '*';
                int hash31 = wordHash(ca, 31), hash29 = wordHash(ca, 29);
                boolean flag31 = set31.add(hash31), flag29 = set29.add(hash29);
                if (!flag31 && !flag29) {
                    return true;
                }
                ca[i] = tmp;
            }
        }
        return false;
    }

    private int wordHash(char[] ca, int base) {
        final int mod = 1000000007; // *号用30代替?
        long result = 0;
        long accu = 1;
        for (char c : ca) {
            if (c != '*') {
                result += accu * (c - 'a');
                result %= mod;
            } else {
                result += (base - 1) * accu;
                result %= mod;
            }
            accu *= base;
            accu %= mod;
        }
        return (int) result % mod;
    }

    // LC29 **
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }

        boolean positive = (dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0);
        int result = 0;
        dividend = -Math.abs(dividend); // 唯一会溢出的情况是 dividend 取到了Integer.MIN_VAL, 已经在上面处理
        divisor = -Math.abs(divisor);
        while (dividend <= divisor) { // 当分子大于分母
            int tmp = divisor; // 快速试乘
            int ctr = 1;
            while (dividend - tmp <= tmp) { // 分子减分母大于分母时
                tmp <<= 1;
                ctr <<= 1;
            }
            result += ctr;
            dividend -= tmp;
        }
        return positive ? result : -result;
    }

    // LC441 注意精度
    public int arrangeCoins(int n) {
        if (n == 0) return 0;
        int lo = 1, hi = (int) Math.sqrt(2d * (n + 0d)) + 1;
        while (lo < hi) {
            int mid = (hi + lo + 1) / 2;
            long result = (1l + mid) * (mid + 0l) / 2l;
            if (result <= n) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    // LC1065
    public int[][] indexPairs(String text, String[] words) {
        List<int[]> result = new ArrayList<>();
        for (String w : words) {
            int startIdx = 0;
            while ((startIdx = text.indexOf(w, startIdx)) != -1) {
                result.add(new int[]{startIdx, startIdx + w.length() - 1});
                startIdx++;
            }
        }
        Collections.sort(result, (o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o1[0] - o2[0]);
        return result.toArray(new int[result.size()][]);
    }

    // LC436
    public int[] findRightInterval(int[][] intervals) {
        Map<Integer, Integer> idxMap = new HashMap<>();
        TreeSet<int[]> ts = new TreeSet<>(Comparator.comparingInt(o -> o[0]));
        int[] result = new int[intervals.length];
        Arrays.fill(result, -1);
        for (int i = 0; i < intervals.length; i++) {
            idxMap.put(intervals[i][0], i);
            ts.add(intervals[i]);
        }
        for (int i = 0; i < intervals.length; i++) {
            int[] ceil = ts.ceiling(new int[]{intervals[i][1], -1});
            if (ceil != null) {
                result[i] = idxMap.get(ceil[0]);
            }
        }
        return result;
    }

    // LC228
    public List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList<>();
        if (nums.length == 0) return result;
        int prev = nums[0], startVal = nums[0], n = nums.length;
        for (int i = 1; i < n; i++) {
            if (nums[i] == prev + 1) {
                prev = nums[i];
            } else {
                if (startVal != prev) {
                    result.add("" + startVal + "->" + prev);
                } else {
                    result.add("" + prev);
                }
                startVal = prev = nums[i];
            }
        }
        if (startVal != prev) {
            result.add("" + startVal + "->" + prev);
        } else {
            result.add("" + prev);
        }
        return result;
    }

    // LC1883 Hard ** 精度问题 转换为整数运算
    Long[][] lc1883Memo;

    public int minSkips(int[] dist, int speed, int hoursBefore) {
        int n = dist.length;
        lc1883Memo = new Long[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            if (lc1883Helper(n, i, dist, speed) <= (hoursBefore + 0l) * (speed + 0l)) return i;
        }
        return -1;
    }

    private long lc1883Helper(int roadsRemain, int countSkip, int[] dist, long speed) { // 返回的是耗时
        if (roadsRemain == 0) {
            return 0;
        }
        if (lc1883Memo[roadsRemain][countSkip] != null) return lc1883Memo[roadsRemain][countSkip];
        long result = Long.MAX_VALUE / 2;
        if (roadsRemain > countSkip) { // 间隔和道路数量一一对应, 不应该出现间隔数量比道路多/一样多的情况
            result = Math.min(result, ((lc1883Helper(roadsRemain - 1, countSkip, dist, speed) + dist[roadsRemain - 1] - 1) / speed + 1) * speed);
        }
        if (countSkip != 0) {
            result = Math.min(result, lc1883Helper(roadsRemain - 1, countSkip - 1, dist, speed) + dist[roadsRemain - 1]);
        }
        return lc1883Memo[roadsRemain][countSkip] = result;
    }

    // LC1652
    public int[] decrypt(int[] code, int k) {
        if (k == 0) return new int[code.length];
        int n = code.length;
        int[] sums = new int[n];
        for (int i = 0; i < n; i++) {
            if (k > 0) {
                for (int j = 1; j <= k; j++) {
                    sums[i] += code[(i + j) % n];
                }
            } else {
                for (int j = 1; j <= (-k); j++) {
                    sums[i] += code[(i - j + n) % n];
                }
            }
        }
        return sums;
    }

    // LC1827
    public int minOperations(int[] nums) {
        int result = 0, prev = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] <= prev) {
                result += (prev + 1 - nums[i]);
                nums[i] = prev + 1;
            }
            prev = nums[i];
        }
        return result;
    }

    // LC1557
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        int[] indegree = new int[n];
        for (List<Integer> e : edges) {
            indegree[e.get(1)]++;
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0) {
                result.add(i);
            }
        }
        return result;
    }

    // LC1377
    public double frogPosition(int n, int[][] edges, int t, int target) {
        List<List<Integer>> mtx = new ArrayList<>();
        Deque<Integer> q = new LinkedList<>();
        double[] possibility = new double[n + 1];
        boolean[] visited = new boolean[n + 1];
        for (int i = 0; i <= n; i++) {
            mtx.add(new ArrayList<>());
        }
        for (int[] e : edges) {
            mtx.get(e[0]).add(e[1]);
            mtx.get(e[1]).add(e[0]);
        }
        q.offer(1);
        possibility[1] = 1d;
        while (!q.isEmpty() && t-- != 0) {
            int qs = q.size();
            double[] nextPos = new double[n + 1];
            for (int i = 0; i < qs; i++) {
                int p = q.poll();
                visited[p] = true; // 这里不能根据是否曾经访问剪枝, 因为有可能一直停留在本位置
                int canJumpCount = 0;
                for (int next : mtx.get(p)) {
                    if (visited[next]) continue;
                    canJumpCount++;
                }
                if (canJumpCount == 0) {
                    nextPos[p] += possibility[p];
                    q.offer(p); // 一直停留在本位置
                    continue;
                }
                for (int next : mtx.get(p)) {
                    if (!visited[next]) {
                        nextPos[next] += possibility[p] * ((1 + 0d) / (canJumpCount + 0d));
                        q.offer(next);
                    }
                }
            }
            possibility = nextPos;
        }
        return possibility[target];
    }

    // LC1306
    public boolean canReach(int[] arr, int start) {
        if (arr[start] == 0) return true;
        int len = arr.length;
        boolean[] visited = new boolean[len];
        Deque<Integer> q = new LinkedList<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int p = q.poll();
            if (visited[p]) continue;
            visited[p] = true;
            int forward = p + arr[p];
            int backward = p - arr[p];
            if (forward < len && !visited[forward]) {
                if (arr[forward] == 0) return true;
                q.offer(forward);
            }
            if (backward >= 0 && !visited[backward]) {
                if (arr[backward] == 0) return true;
                q.offer(backward);
            }
        }
        // 实际上是BFS过程中不判断, 最后循环判断耗时更短
        // for (int i = 0; i < len; i++) {
        //     if (arr[i] == 0 && visited[i]) return true;
        // }
        return false;
    }

    // LC1253
    public List<List<Integer>> reconstructMatrix(int upper, int lower, int[] colsum) {
        if (upper > colsum.length || lower > colsum.length || Arrays.stream(colsum).sum() != upper + lower) {
            return new ArrayList<>();
        }
        List<Integer> up = new ArrayList<>(), low = new ArrayList<>();
        for (int i : colsum) {
            if (i == 2) {
                upper--;
                lower--;
                up.add(1);
                low.add(1);
                if (upper < 0 || lower < 0) {
                    return new ArrayList<>();
                }
            } else if (i == 1) {
                if (upper > lower) {
                    upper--;
                    up.add(1);
                    low.add(0);
                } else {
                    lower--;
                    up.add(0);
                    low.add(1);
                }
            } else {
                up.add(0);
                low.add(0);
            }
        }
        return Arrays.asList(up, low);
    }

    // LC1784
    public boolean checkOnesSegment(String s) {
        int seg = 0;
        char prev = '\0';
        for (char c : s.toCharArray()) {
            if (prev == '\0' || prev == '0') {
                if (c == '1') {
                    seg++;
                    if (seg > 1) return false;
                }
            }
            prev = c;
        }
        return true;
    }

    // LC1706
    public int[] findBall(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] result = new int[grid[0].length];
        Arrays.fill(result, -1);
        // 1:左上-右下 -1:右上-左下
        // 在一个格子中的位置: 上-0 下-1
        // (#,#) 表示(格子状态,上下状态)
        //  1,0: 看右边: 如果右边是1, 通行, 变为(1,1); 如果右边是-1或碰壁, 卡死
        //  1,1: 看下边: 如果下边是1, 通行, 变为(1,0); 如果下边是-1, 变为(-1,0); 如果当前行为最后一行, 返回列号
        // -1,0: 看左边: 如果左边是-1, 通行, 变为(-1,1); 如果左边是1或者碰壁, 卡死
        // -1,1: 看下边: 如果下边是1, 通行, 变为(1,0); 如果下边是-1, 通行, 变为(-1,0); 如果当前行为最后一行, 返回列号
        loop:
        for (int i = 0; i < n; i++) {
            int col = i, row = 0;
            int upDown = 0;
            int gridStatus = grid[0][i];
            while (row < m) {
                if (gridStatus == 1) {
                    if (upDown == 0) { // look right
                        if (col + 1 >= n) continue loop;
                        if (grid[row][col + 1] == -1) continue loop;
                        col = col + 1;
                        upDown = 1;
                    } else if (upDown == 1) { // look down
                        if (row == m - 1) {
                            result[i] = col;
                            break;
                        } else {
                            gridStatus = grid[row + 1][col];
                            row++;
                            upDown = 0;
                        }
                    }
                } else if (gridStatus == -1) {
                    if (upDown == 0) { // look left
                        if (col - 1 < 0) continue loop;
                        if (grid[row][col - 1] == 1) continue loop;
                        col = col - 1;
                        upDown = 1;
                    } else if (upDown == 1) { // look down
                        if (row == m - 1) {
                            result[i] = col;
                            break;
                        } else {
                            gridStatus = grid[row + 1][col];
                            row++;
                            upDown = 0;
                        }
                    }
                }
            }
        }
        return result;
    }

    // LC988
    StringBuilder lc988Sb;
    String lc988Result;

    public String smallestFromLeaf(TreeNode root) {
        lc988Result = ((char) ('z' + 1) + "");
        lc988Sb = new StringBuilder();
        lc1883Helper(root);
        return lc988Result;
    }

    private void lc1883Helper(TreeNode root) {
        lc988Sb.append((char) ('a' + root.val));
        if (root.left == null && root.right == null) {
            String tmp = new StringBuilder(lc988Sb).reverse().toString();
            if (tmp.compareTo(lc988Result) < 0) {
                lc988Result = tmp;
            }
            lc988Sb.deleteCharAt(lc988Sb.length() - 1);
            return;
        }
        if (root.left != null) {
            lc1883Helper(root.left);
        }
        if (root.right != null) {
            lc1883Helper(root.right);
        }
        lc988Sb.deleteCharAt(lc988Sb.length() - 1);
    }

    // LC187
    public List<String> findRepeatedDnaSequences(String s) {
        if (s.length() < 10) return new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        Set<String> result = new HashSet<>();
        char[] ca = s.toCharArray();
        // A C G T 0 1 2 3
        int[] alphabet = new int[256];
        alphabet['A'] = 0;
        alphabet['C'] = 1;
        alphabet['G'] = 2;
        alphabet['T'] = 3;
        int hash = 0, fullMask = (1 << 20) - 1;
        for (int i = 0; i < 10; i++) {
            hash = (hash << 2) | (alphabet[ca[i]]);
        }
        set.add(hash);
        for (int i = 10; i < ca.length; i++) {
            hash = ((hash << 2) | (alphabet[ca[i]])) & fullMask;
            if (set.contains(hash)) {
                result.add(s.substring(i - 10 + 1, i + 1));
            } else {
                set.add(hash);
            }
        }
        return new ArrayList<>(result);
    }

    // LC414
    public int thirdMax(int[] nums) {
        int count = 0;
        long[] max = new long[3];
        Arrays.fill(max, Long.MIN_VALUE);
        for (int i : nums) {
            if (count == 0) {
                max[count++] = i;
            } else if (count == 1) {
                if (max[0] == i) continue;
                if (max[0] < i) {
                    max[1] = max[0];
                    max[0] = i;
                    count++;
                } else {
                    max[1] = i;
                    count++;
                    continue;
                }
            } else if (count == 2) {
                if (max[0] == i || max[1] == i) continue;
                if (max[0] < i) {
                    max[2] = max[1];
                    max[1] = max[0];
                    max[0] = i;
                    count++;
                    continue;
                } else if (max[1] < i) {
                    max[2] = max[1];
                    max[1] = i;
                    count++;
                    continue;
                } else {
                    max[2] = i;
                    count++;
                    continue;
                }
            } else {
                if (max[0] == i || max[1] == i || max[2] == i) continue;
                if (max[0] < i) {
                    max[2] = max[1];
                    max[1] = max[0];
                    max[0] = i;
                    continue;
                } else if (max[1] < i) {
                    max[2] = max[1];
                    max[1] = i;
                    continue;
                } else if (max[2] < i) {
                    max[2] = i;
                    continue;
                }
            }
        }
        if (count == 3) return (int) max[2];
        return (int) max[0];
    }

    // LC10
    Boolean[][] lc10Memo;

    public boolean isMatchLc10(String s, String p) {
        lc10Memo = new Boolean[s.length() + 1][p.length() + 1];

        return lc10Helper(s.toCharArray(), p.toCharArray(), 0, 0);
    }

    private boolean lc10Helper(char[] sa, char[] pa, int sIdx, int pIdx) {
        if (pIdx >= pa.length) return sIdx >= sa.length;
        if (lc10Memo[sIdx][pIdx] != null) return lc10Memo[sIdx][pIdx];
        // 单匹配
        boolean singleMatch = sIdx < sa.length && (sa[sIdx] == pa[pIdx] || pa[pIdx] == '.');

        // 多个匹配
        if (pIdx < pa.length - 1 && pa[pIdx + 1] == '*') {
            // 匹配0次 || 匹配多次
            return lc10Memo[sIdx][pIdx] = lc10Helper(sa, pa, sIdx, pIdx + 2) || (singleMatch && lc10Helper(sa, pa, sIdx + 1, pIdx));
        }
        return lc10Memo[sIdx][pIdx] = singleMatch && lc10Helper(sa, pa, sIdx + 1, pIdx + 1);
    }

    // LC44 **
    class Lc44Memo2 {
        // LC44
        Boolean[][] memo;

        public boolean isMatch(String s, String p) {
            memo = new Boolean[s.length() + 1][p.length() + 1];
            return helper(s.toCharArray(), p.toCharArray(), 0, 0);
        }

        private boolean helper(char[] sa, char[] pa, int sIdx, int pIdx) {
            if (pIdx >= pa.length) return sIdx >= sa.length;
            if (memo[sIdx][pIdx] != null) return memo[sIdx][pIdx];
            if (sIdx >= sa.length) {
                for (int i = pIdx; i < pa.length; i++) {
                    if (pa[i] != '*') return memo[sIdx][pIdx] = false;
                }
                return memo[sIdx][pIdx] = true;
            }
            if (pa[pIdx] == '?') {
                return memo[sIdx][pIdx] = helper(sa, pa, sIdx + 1, pIdx + 1);
            }
            if (pa[pIdx] == '*') {
                for (int i = sIdx; i <= sa.length; i++) {
                    if (helper(sa, pa, i, pIdx + 1)) {
                        return memo[sIdx][pIdx] = true;
                    }
                }
            }
            return memo[sIdx][pIdx] = sa[sIdx] == pa[pIdx] && helper(sa, pa, sIdx + 1, pIdx + 1);
        }
    }


    class Lc44Memo1 {
        Boolean[][] lc44Memo;

        public boolean isMatch(String s, String p) {
            lc44Memo = new Boolean[s.length() + 1][p.length() + 1];
            return lc44Helper(s.toCharArray(), p.toCharArray(), s.length(), p.length());
        }

        private boolean lc44Helper(char[] sa, char[] pa, int sIdx, int pIdx) {
            if (sIdx == 0 && pIdx == 0) return true;
            if (sIdx != 0 && pIdx == 0) return false;
            if (lc44Memo[sIdx][pIdx] != null) return lc44Memo[sIdx][pIdx];
            if (sIdx == 0 && pIdx != 0) {
                for (int i = pIdx; i >= 1; i--) {
                    if (pa[i - 1] != '*') return lc44Memo[sIdx][pIdx] = false;
                }
                return lc44Memo[sIdx][pIdx] = true;
            }
            if (pa[pIdx - 1] == '?' || pa[pIdx - 1] == sa[sIdx - 1])
                return lc44Memo[sIdx][pIdx] = lc44Helper(sa, pa, sIdx - 1, pIdx - 1);
            if (pa[pIdx - 1] == '*') {
                return lc44Memo[sIdx][pIdx] = lc44Helper(sa, pa, sIdx, pIdx - 1) || lc44Helper(sa, pa, sIdx - 1, pIdx);
            }
            return lc44Memo[sIdx][pIdx] = false;
        }
    }


    // LC445
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode h1 = reverseList(l1), h2 = reverseList(l2);
        ListNode p1 = h1, p2 = h2;
        int carry = 0;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (p1 != null && p2 != null) {
            int next = carry + p1.val + p2.val;
            carry = next / 10;
            ListNode nextNode = new ListNode(next % 10);
            cur.next = nextNode;
            cur = cur.next;
            p1 = p1.next;
            p2 = p2.next;
        }
        while (p1 != null) {
            int next = carry + p1.val;
            carry = next / 10;
            ListNode nextNode = new ListNode(next % 10);
            cur.next = nextNode;
            cur = cur.next;
            p1 = p1.next;
        }
        while (p2 != null) {
            int next = carry + p2.val;
            carry = next / 10;
            ListNode nextNode = new ListNode(next % 10);
            cur.next = nextNode;
            cur = cur.next;
            p2 = p2.next;
        }
        if (carry != 0) {
            cur.next = new ListNode(carry);
            cur = cur.next;
        }
        return reverseList(dummy.next);
    }

    private ListNode reverseList(ListNode head) {
        if (head == null) return null;
        ListNode cur = head, prev = null;
        while (cur != null) {
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }
        return prev;
    }


    // LC253
    public int minMeetingRooms(int[][] intervals) {
        final int OPEN = 0, CLOSE = 1;
        List<int[]> events = new ArrayList<>();
        for (int[] i : intervals) {
            events.add(new int[]{i[0], OPEN});
            events.add(new int[]{i[1], CLOSE});
        }
        Collections.sort(events, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        });
        int max = 0, active = 0;
        for (int[] e : events) {
            if (e[1] == OPEN) {
                active++;
                max = Math.max(active, max);
            } else {
                active--;
            }
        }
        return max;
    }

    // LC1448
    int lc1448Result = 0;

    public int goodNodes(TreeNode root) {
        lc1448Helper(root, Integer.MIN_VALUE);
        return lc1448Result;
    }

    private void lc1448Helper(TreeNode root, int curMax) {
        if (root == null) return;
        if (root.val >= curMax) lc1448Result++;
        lc1448Helper(root.left, Math.max(root.val, curMax));
        lc1448Helper(root.right, Math.max(root.val, curMax));
    }

    // LC482
    public String licenseKeyFormatting(String s, int k) {
        LinkedList<Character> q = new LinkedList<>();
        for (char c : s.toCharArray()) {
            if (c == '-') continue;
            if (Character.isLowerCase(c)) c = Character.toUpperCase(c);
            q.offer(c);
        }
        StringBuilder sb = new StringBuilder(((q.size() / k) + 1) * (k + 1));
        int remain = q.size() % k;
        int parts = q.size() / k;
        if (q.size() % k != 0) {
            for (int i = 0; i < remain; i++) {
                sb.append(q.pollFirst());
            }
            if (parts != 0) sb.append('-');
        }
        parts = q.size() / k;
        for (int i = 0; i < parts; i++) {
            for (int j = 0; j < k; j++) {
                sb.append(q.pollFirst());
            }
            if (i != parts - 1) sb.append('-');
        }
        return sb.toString();
    }

    // LC174
    Integer[][] lc174Memo;

    public int calculateMinimumHP(int[][] dungeon) {
        lc174Memo = new Integer[dungeon.length + 1][dungeon[0].length + 1];
        return lc174Helper(0, 0, dungeon) + 1;
    }

    private int lc174Helper(int row, int col, int[][] dungeon) { // 到r,c的时候至少要有多少血
        if (lc174Memo[row][col] != null) return lc174Memo[row][col];
        if (row == dungeon.length - 1 && col == dungeon[0].length - 1) {
            if (dungeon[row][col] >= 0) return lc174Memo[row][col] = 0;
            return lc174Memo[row][col] = -dungeon[row][col];
        }
        int down = Integer.MAX_VALUE / 2, right = Integer.MAX_VALUE / 2;
        if (row + 1 < dungeon.length) {
            down = lc174Helper(row + 1, col, dungeon) - dungeon[row][col];
        }
        if (col + 1 < dungeon[0].length) {
            right = lc174Helper(row, col + 1, dungeon) - dungeon[row][col];
        }
        int min = Math.min(down, right);
        if (min < 0) {
            return lc174Memo[row][col] = 0;
        }
        return lc174Memo[row][col] = min;
    }

    // LC77
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) == k) {
                result.add(getCombine(mask));
            }
        }
        return result;
    }

    private List<Integer> getCombine(int mask) {
        List<Integer> result = new ArrayList<>(Integer.bitCount(mask));
        for (int i = 0; i < Integer.SIZE; i++) {
            if (((mask >> i) & 1) == 1) {
                result.add(i + 1);
            }
        }
        return result;
    }

    // LC40 给的数字有重复 选择不可重复 可选个数100 无法位运算枚举 哈希超时
    List<List<Integer>> lc40Result;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<int[]> freq = new ArrayList<>();
        for (int i = 0; i < candidates.length; i++) {
            int fs = freq.size();
            if (fs == 0 || candidates[i] != freq.get(fs - 1)[0]) {
                freq.add(new int[]{candidates[i], 1});
            } else {
                freq.get(fs - 1)[1]++;
            }
        }
        lc40Result = new ArrayList<>();
        // curIdx 是在freq种的idx
        lc40Helper(candidates, 0, new ArrayList<>(), target, freq);
        return lc40Result;
    }

    private void lc40Helper(int[] candidates, int curIdx, List<Integer> selected, int remain, List<int[]> freq) {
        if (remain == 0) {
            lc40Result.add(new ArrayList<>(selected));
            return;
        }
        if (curIdx == freq.size() || remain < freq.get(curIdx)[0]) return;
        int mostSelect = Math.min(freq.get(curIdx)[1], remain / freq.get(curIdx)[0]);
        for (int i = 1; i <= mostSelect; i++) {
            selected.add(freq.get(curIdx)[0]);
            lc40Helper(candidates, curIdx + 1, selected, remain - i * freq.get(curIdx)[0], freq);
        }
        for (int i = 1; i <= mostSelect; i++) {
            selected.remove(selected.size() - 1);
        }
        // 不选
        lc40Helper(candidates, curIdx + 1, selected, remain, freq);
    }


    // LC39 可重复
    List<List<Integer>> lc39Result;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        lc39Result = new ArrayList<>();
        lc39Helper(candidates, 0, new ArrayList<>(), target);
        return lc39Result;
    }

    private void lc39Helper(int[] candidates, int curIdx, List<Integer> selected, int remain) {
        if (remain == 0) {
            lc39Result.add(new ArrayList<>(selected));
            return;
        }
        for (int i = curIdx; i < candidates.length; i++) {
            int c = candidates[i];
            if (c <= remain) {
                selected.add(c);
                lc39Helper(candidates, i, selected, remain - c);
                selected.remove(selected.size() - 1);
            }
        }
    }

    // LC166
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        long num = Math.abs(numerator), den = Math.abs(0l + denominator);
        String left = String.valueOf(num / den);
        if ((0l + numerator) * (0l + denominator) < 0l) left = "-" + left;
        long remainder = num % den;
        if (remainder == 0l) return left;
        left += ".";
        Map<Long, Integer> map = new HashMap<>();
        StringBuilder sb = new StringBuilder(left);
        while (remainder != 0) {
            if (map.containsKey(remainder)) {
                sb.insert(map.get(remainder), "(");
                sb.append(")");
                break;
            }
            map.put(remainder, sb.length());
            remainder *= 10;
            sb.append(remainder / den);
            remainder %= den;
        }
        return sb.toString();
    }

    // LC405
    public String toHex(int num) {
        if (num == 0) return "0";
        char[] hex = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
        StringBuilder result = new StringBuilder();
        // int -> 4byte ,1byte = 8bit = 2*4bit
        for (int i = 1; i <= 8; i++) {
            int offset = i * 4;
            int this4bit = (num >> (32 - offset)) & 0x0f;
            if (result.length() == 0 && this4bit == 0) continue;
            result.append(hex[this4bit]);
        }
        return result.toString();
    }

    // LC1057
    public int[] assignBikesI(int[][] workers, int[][] bikes) {
        int nw = workers.length, nb = bikes.length;
        int[][] distance = new int[nw][nb];
        int[] result = new int[nw];
        boolean[] visitedBike = new boolean[nb];
        boolean[] visitedWorker = new boolean[nb];
        TreeSet<int[]> ts = new TreeSet<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (distance[o1[0]][o1[1]] == distance[o2[0]][o2[1]]) {
                    if (o1[0] == o2[0]) {
                        return o1[1] - o2[1];
                    }
                    return o1[0] - o2[0];
                }
                return distance[o1[0]][o1[1]] - distance[o2[0]][o2[1]];
            }
        });
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < nb; j++) {
                distance[i][j] = Math.abs(workers[i][0] - bikes[j][0]) + Math.abs(workers[i][1] - bikes[j][1]);
            }
        }
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < nb; j++) {
                ts.add(new int[]{i, j});
            }
        }
        Iterator<int[]> it = ts.iterator();
        while (it.hasNext()) {
            int[] next = it.next();
            if (visitedBike[next[1]]) {
                it.remove();
                continue;
            }
            if (visitedWorker[next[0]]) {
                it.remove();
                continue;
            }
            result[next[0]] = next[1];
            visitedWorker[next[0]] = true;
            visitedBike[next[1]] = true;
        }
        return result;
    }

    // LC1066
    public int assignBikes(int[][] workers, int[][] bikes) {
        int nw = workers.length, nb = bikes.length;
        // dp[mask][mask]
        int[][] dp = new int[1 << nw][1 << nb];

        for (int mw = 0; mw < 1 << nw; mw++) {
            Arrays.fill(dp[mw], Integer.MAX_VALUE / 2);
        }
        dp[0][0] = 0;

        for (int mw = 1; mw < 1 << nw; mw++) {
            for (int mb = 1; mb < 1 << nb; mb++) {
                if (Integer.bitCount(mw) > Integer.bitCount(mb)) continue;
                for (int w = 0; w < nw; w++) {
                    if (((mw >> w) & 1) == 1) {
                        int parentWorkerMask = mw ^ (1 << w);
                        for (int b = 0; b < nb; b++) {
                            if (((mb >> b) & 1) == 1) {
                                int parentBikeMask = mb ^ (1 << b);
                                int distance = Math.abs(workers[w][0] - bikes[b][0]) + Math.abs(workers[w][1] - bikes[b][1]);
                                dp[mw][mb] = Math.min(dp[mw][mb], dp[parentWorkerMask][parentBikeMask] + distance);
                            }
                        }
                    }
                }
            }
        }
        int min = Integer.MAX_VALUE / 2;
        for (int i = 0; i < 1 << nb; i++) {
            min = Math.min(min, dp[(1 << nw) - 1][i]);
        }
        return min;
    }

    // JZOF II 055
    class BSTIterator {
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode cur;

        public BSTIterator(TreeNode root) {
            cur = root;
        }

        public int next() { // 先序遍历
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            int result = cur.val;
            cur = cur.right;
            return result;
        }

        public boolean hasNext() {
            return cur != null || !stack.isEmpty();
        }

    }

    // JZOF 26
    public boolean isSubStructure(TreeNode a, TreeNode b) {
        // 空树不是任何树的子结构
        if (a == null || b == null) return false;
        return lc40Helper(a, b) || isSubStructure(a.left, b) || isSubStructure(a.right, b);
    }

    private boolean lc40Helper(TreeNode a, TreeNode b) {
        if (b == null) return true;
        if (a == null || a.val != b.val) return false;
        return lc40Helper(a.left, b.left) && lc40Helper(a.right, b.right);
    }

    // LC1024 DP
    public int videoStitching(int[][] clips, int time) {
        int[] dp = new int[time + 1]; // 表示当前下标能覆盖到的最远距离
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i = 1; i <= time; i++) {
            for (int[] c : clips) {
                if (c[0] < i && i <= c[1]) { // 如果i在该片段的覆盖范围内 (注意点还是线)
                    dp[i] = Math.min(dp[i], 1 + dp[c[0]]);
                }
            }
        }
        return dp[time] == Integer.MAX_VALUE / 2 ? -1 : dp[time];
    }


    // LC45
    Integer[] lc45Memo;

    public int jump(int[] nums) {
        lc45Memo = new Integer[nums.length + 1];
        return lc45Helper(0, nums);
    }

    private int lc45Helper(int curIdx, int[] nums) {
        if (curIdx >= nums.length - 1) return 0;
        if (lc45Memo[curIdx] != null) return lc45Memo[curIdx];
        int min = Integer.MAX_VALUE / 2; // 防溢出
        for (int i = 1; i <= nums[curIdx]; i++) {
            min = Math.min(min, 1 + lc45Helper(curIdx + i, nums));
        }
        return lc45Memo[curIdx] = min;
    }

    // LC1326
    public int minTapsGreedy(int n, int[] ranges) {
        int[] land = new int[n]; // 表示覆盖范围内最远覆盖到的土地下标
        for (int i = 0; i < n; i++) {
            int l = Math.max(i - ranges[i], 0);
            int r = Math.min(i + ranges[i], n);
            for (int j = l; j < r; j++) { // 最多两百次, 视作常数
                land[j] = Math.max(land[j], r); // 更新范围内最远覆盖到的土地下标
            }
        }
        int ctr = 0, cur = 0;
        while (cur < n) {
            if (land[cur] == 0) return -1; // 如果有土地未被覆盖到
            cur = land[cur];
            ctr++;
        }
        return ctr;
    }

    public int minTaps(int n, int[] ranges) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i <= n; i++) {
            if (ranges[i] == 0) continue;
            tm.put(Math.max(i - ranges[i], 0), Math.min(Math.max(tm.getOrDefault(i - ranges[i], Integer.MIN_VALUE), i + ranges[i]), n));
        }
        int result = Integer.MAX_VALUE;
        loop:
        for (Map.Entry<Integer, Integer> i : tm.entrySet()) { // 从i开始
            if (i.getKey() > 0) break;
            LinkedList<Map.Entry<Integer, Integer>> candidateQueue = new LinkedList<>();
            candidateQueue.add(i);
            while (candidateQueue.getLast().getValue() < n) {
                Map.Entry<Integer, Integer> last = candidateQueue.getLast();
                NavigableMap<Integer, Integer> intersect = tm.subMap(last.getKey(), false, last.getValue(), true);
                if (intersect.isEmpty()) break loop;
                Map.Entry<Integer, Integer> candidate = null;
                int rightMost = last.getValue();
                for (Map.Entry<Integer, Integer> j : intersect.entrySet()) {
                    if (j.getValue() > rightMost) {
                        candidate = j;
                        rightMost = j.getValue();
                    }
                }
                if (candidate == null) break;
                candidateQueue.add(candidate);
            }
            if (candidateQueue.getLast().getValue() < n) break;
            result = Math.min(result, candidateQueue.size());
            if (result == 1) return 1;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
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

// LC1472
class BrowserHistory {
    LinkedList<String> history;
    int ptr = -1;

    public BrowserHistory(String homepage) {
        history = new LinkedList<>();
        history.add(homepage);
        ptr++;
    }

    public void visit(String url) {
        if (ptr != history.size() - 1) {
            int times = history.size() - 1 - ptr;
            for (int i = 0; i < times; i++) {
                history.removeLast();
            }
        }
        history.add(url);
        ptr = history.size() - 1;
    }

    public String back(int steps) {
        if (ptr - steps < 0) {
            ptr = 0;
            return history.getFirst();
        }
        ptr -= steps;
        return history.get(ptr);
    }

    public String forward(int steps) {
        if (ptr + steps >= history.size()) {
            ptr = history.size() - 1;
            return history.getLast();
        }
        ptr += steps;
        return history.get(ptr);
    }
}

// LC1628

class Lc1628 {
    class TreeBuilder {
        Node buildTree(String[] postfix) {
            Deque<Node> stack = new LinkedList<>();
            for (String op : postfix) {
                NodeType type;
                NodeImpl cur;
                switch (op) {
                    case "+":
                        type = NodeType.ADD;
                        break;
                    case "-":
                        type = NodeType.SUBTRACT;
                        break;
                    case "*":
                        type = NodeType.MULTIPLY;
                        break;
                    case "/":
                        type = NodeType.DIVIDE;
                        break;
                    default:
                        type = NodeType.NUMBER;
                }
                if (type == NodeType.NUMBER) {
                    cur = new NodeImpl(NodeType.NUMBER, Integer.parseInt(op));
                } else {
                    cur = new NodeImpl(type);
                    cur.right = stack.pop();
                    cur.left = stack.pop();
                }
                stack.push(cur);
            }
            return stack.pop();
        }
    }


    enum NodeType {
        NUMBER,
        ADD,
        SUBTRACT,
        DIVIDE,
        MULTIPLY
    }

    class NodeImpl extends Node {
        final NodeType type;
        final int val;
        Node left;
        Node right;

        public NodeImpl(NodeType type, int val) {
            this.type = type;
            this.val = val;
        }

        public NodeImpl(NodeType type) {
            this.type = type;
            this.val = 0;
        }

        @Override
        public int evaluate() {
            switch (type) {
                case NUMBER:
                    return val;
                case ADD:
                    return left.evaluate() + right.evaluate();
                case SUBTRACT:
                    return left.evaluate() - right.evaluate();
                case DIVIDE:
                    return left.evaluate() / right.evaluate();
                case MULTIPLY:
                    return left.evaluate() * right.evaluate();
                default:
                    throw new AssertionError();
            }
        }
    }


    abstract class Node {
        public abstract int evaluate();
        // define your fields here
    }
}

// JZOF II 064 LC676
class MagicDictionary {
    Map<Integer, List<String>> map = new HashMap<>();

    /**
     * Initialize your data structure here.
     */
    public MagicDictionary() {

    }

    public void buildDict(String[] dictionary) {
        for (String word : dictionary) {
            map.putIfAbsent(word.length(), new ArrayList<>());
            map.get(word.length()).add(word);
        }
    }

    public boolean search(String searchWord) {
        int len = searchWord.length();
        if (!map.containsKey(len)) return false;
        for (String word : map.get(len)) {
            int count = 0;
            for (int i = 0; i < len; i++) {
                if (searchWord.charAt(i) != word.charAt(i)) count++;
                if (count > 1) break;
            }
            if (count == 1) return true;
        }
        return false;
    }
}

// LC352
class SummaryRanges {
    TreeSet<Pair<Integer, Integer>> leftSide, rightSide;

    public SummaryRanges() {
        leftSide = new TreeSet<>(Comparator.comparingInt(Pair::getKey));
        rightSide = new TreeSet<>(Comparator.comparingInt(Pair::getValue));
    }

    public void addNum(int val) {
        Pair<Integer, Integer> p = new Pair<>(val, val);
        Pair<Integer, Integer> lsf = leftSide.floor(p), rsc = rightSide.ceiling(p); // 左侧小于等于val的最大值, 右侧大于等于val的最小值, 以期合并
        if (lsf != null && rsc != null && lsf.getValue() + 1 == val && rsc.getKey() - 1 == val) {
            // 如 [1,2] [3] [4,5] -> [1,5]
            leftSide.remove(lsf);
            leftSide.remove(rsc);
            rightSide.remove(lsf);
            rightSide.remove(rsc);
            Pair<Integer, Integer> merged = new Pair<>(lsf.getKey(), rsc.getValue());
            leftSide.add(merged);
            rightSide.add(merged);
        } else if (lsf != null && lsf.getValue() + 1 == val) {
            leftSide.remove(lsf);
            rightSide.remove(lsf);
            Pair<Integer, Integer> merged = new Pair<>(lsf.getKey(), val);
            leftSide.add(merged);
            rightSide.add(merged);
        } else if (rsc != null && rsc.getKey() - 1 == val) {
            leftSide.remove(rsc);
            rightSide.remove(rsc);
            Pair<Integer, Integer> merged = new Pair<>(val, rsc.getValue());
            leftSide.add(merged);
            rightSide.add(merged);
        } else if (lsf != null && val <= lsf.getValue() && rsc != null && val >= rsc.getKey()) {
            ;
        } else {
            leftSide.add(p);
            rightSide.add(p);
        }
    }

    public int[][] getIntervals() {
        int[][] result = new int[leftSide.size()][];
        int ctr = 0;
        for (Pair<Integer, Integer> p : leftSide) {
            result[ctr++] = new int[]{p.getKey(), p.getValue()};
        }
        return result;
    }
}

// LC715
class RangeModule {
    TreeMap<Integer, Integer> tm = new TreeMap<>();

    public RangeModule() {
    }

    public void addRange(int left, int right) {
        Iterator<Map.Entry<Integer, Integer>> iterator = tm.tailMap(left).entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> e = iterator.next();
            if (right < e.getValue()) break;
            left = Math.min(left, e.getValue());
            right = Math.max(right, e.getKey());
            iterator.remove();
        }
        tm.put(right, left);
    }

    public boolean queryRange(int left, int right) {
        Map.Entry<Integer, Integer> higherEntry = tm.higherEntry(left);
        return higherEntry != null && higherEntry.getValue() <= left && right <= higherEntry.getKey();
    }

    public void removeRange(int left, int right) {
        addRange(left, right);
        Iterator<Map.Entry<Integer, Integer>> iterator = tm.tailMap(left).entrySet().iterator();
        int leftBorder = Integer.MAX_VALUE, rightBorder = Integer.MIN_VALUE;
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> e = iterator.next();
            if (e.getValue() > right) break;
            leftBorder = Math.min(leftBorder, e.getValue());
            rightBorder = Math.max(rightBorder, e.getKey());
            iterator.remove();
        }
        if (leftBorder < left) {
            tm.put(left, leftBorder);
        }
        if (rightBorder > right) {
            tm.put(rightBorder, right);
        }
    }
}

// LC1476
class SubrectangleQueries {

    List<int[]> history = new ArrayList<>();
    int[][] rectangle;

    public SubrectangleQueries(int[][] rectangle) {
        this.rectangle = rectangle;
    }

    public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
        history.add(new int[]{row1, col1, row2, col2, newValue});
    }

    public int getValue(int row, int col) {
        for (int i = history.size() - 1; i >= 0; i--) {
            int[] h = history.get(i);
            if (h[0] <= row && h[1] <= col && h[2] >= row && h[3] >= col) return h[4];
        }
        return rectangle[row][col];
    }
}

// JZOF 41 LC295
class MedianFinder {
    // 约定: 最大堆的堆顶存中位数
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
    PriorityQueue<Integer> minHeap = new PriorityQueue<>(Comparator.naturalOrder());

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {

    }

    public void addNum(int num) {
        if (maxHeap.size() == 0) {
            maxHeap.offer(num);
        } else {
            if (num > maxHeap.peek()) {
                minHeap.offer(num);
            } else {
                maxHeap.offer(num);
            }
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }

        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) return maxHeap.peek();
        return (maxHeap.peek() + minHeap.peek() + 0d) / 2d;
    }
}