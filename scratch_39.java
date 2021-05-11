import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        int[] a = new int[]{3, 2, 3};
        System.err.println(
                s.decode(new int[]{6, 5, 4, 6})
        );
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1734
    public int[] decode(int[] encoded) {
        // decoded.length = n, n%2 = 1, 是前n个正整数的排列
        // encoded[i] = decoded[i] ^ decoded[i+1]
        int n = encoded.length + 1;
        int total = 0;
        for (int i = 1; i <= n; i++) {
            total ^= i;
        }
        int odd = 0;
        for (int i = 1; i < (n - 1); i += 2) {
            odd ^= encoded[i];
        }
        int first = odd ^ total;
        int[] result = new int[n];
        result[0] = first;
        for (int i = 1; i < n; i++) {
            result[i] = encoded[i - 1] ^ result[i - 1];
        }
        return result;
    }

    // LC872 DFS
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> leaf1 = new ArrayList<>(200);
        List<Integer> leaf2 = new ArrayList<>(200);
        Deque<TreeNode> q1 = new LinkedList<>();
        Deque<TreeNode> q2 = new LinkedList<>();
        q1.push(root1);
        q2.push(root2);
        while (!q1.isEmpty()) {
            TreeNode tmp = q1.pop();
            if (tmp.left != null) q1.push(tmp.left);
            if (tmp.right != null) q1.push(tmp.right);
            if (tmp.left == null && tmp.right == null) leaf1.add(tmp.val);
        }
        while (!q2.isEmpty()) {
            TreeNode tmp = q2.pop();
            if (tmp.left != null) q2.push(tmp.left);
            if (tmp.right != null) q2.push(tmp.right);
            if (tmp.left == null && tmp.right == null) leaf2.add(tmp.val);
        }
        if (leaf1.size() != leaf2.size()) return false;
        for (int i = 0; i < leaf1.size(); i++) {
            if (leaf1.get(i) != leaf2.get(i)) return false;
        }
        return true;
    }

    // LC1482
    public int minDays(int[] bloomDay, int m, int k) {
        int n = bloomDay.length;
        if (m * k > n) return -1;
        int min = Arrays.stream(bloomDay).min().getAsInt();
        int max = Arrays.stream(bloomDay).max().getAsInt();
        int left = min, right = max;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (minDaysHelper(bloomDay, m, k, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean minDaysHelper(int[] bloomDay, int m, int k, int nthDay) {
        int bq = 0;
        int tmpCtr = 0;
        for (int i = 0; i < bloomDay.length; i++) {
            if (bloomDay[i] <= nthDay) {
                tmpCtr++;
                if (tmpCtr == k) {
                    bq++;
                    tmpCtr = 0;
                }
            } else {
                tmpCtr = 0;
            }
        }
        return bq >= m;
    }

    // LC1723
    Set<Pair<Integer, Integer>> visitedStatus = new HashSet<>();
    int[] lc1723Cache;
    int[] lc1723MaxTimeMemo;
    int lc1723MaxStatus;
    int lc1723Answer = Integer.MAX_VALUE;

    public int minimumTimeRequired(int[] jobs, int k) {
        lc1723MaxStatus = (1 << jobs.length) - 1;
        lc1723Cache = new int[lc1723MaxStatus + 1];
        for (int i = 0; i <= lc1723MaxStatus; i++) {
            for (int j = 0; j < jobs.length; j++) {
                if (((i >> j) & 1) == 1) {
                    lc1723Cache[i] += jobs[j];
                }
            }
        }
        int[][] dp = new int[k + 1][lc1723MaxStatus + 1];
        for (int i = 1; i <= k; i++) {
            Arrays.fill(dp[i], Integer.MAX_VALUE / 2);
        }
        for (int i = 0; i <= lc1723MaxStatus; i++) {
            dp[1][i] = lc1723Cache[i];
        }

        for (int i = 2; i <= k; i++) {
            for (int j = 0; j <= lc1723MaxStatus; j++) {
                for (int subset = j; subset != 0; subset = (subset - 1) & j) {
                    // j-subset : 关于subset关于j的补集
                    dp[i][j] = Math.min(dp[i][j], Math.max(dp[i - 1][j - subset], lc1723Cache[subset]));
                }
            }
        }
        return dp[k][lc1723MaxStatus];
    }

    // LC1323
    public int maximum69Number(int num) {
        int orig = num;
        int tenPow = 1;
        int max = orig;
        while (num != 0) {
            // 翻转一位
            int digit = num % 10;
            if (digit == 6) {
                max = orig + 3 * tenPow;
            }
            num /= 10;
            tenPow *= 10;
        }
        return max;
    }

    // LC454
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int result = 0;
        Map<Integer, Integer> map12 = new HashMap<>();
        for (int a : nums1) {
            for (int b : nums2) {
                map12.put(a + b, map12.getOrDefault(a + b, 0) + 1);
            }
        }
        for (int c : nums3) {
            for (int d : nums4) {
                if (map12.containsKey(-c - d)) {
                    result += map12.get(-c - d);
                }
            }
        }
        return result;
    }

    // LC412
    public List<String> fizzBuzz(int n) {
        List<String> l = new ArrayList<>(n);
        for (int i = 1; i <= n; i++) {
            if (i % 3 != 0 && i % 5 != 0) {
                l.add(String.valueOf(i));
            } else if (i % 3 != 0 && i % 5 == 0) {
                l.add("Buzz");
            } else if (i % 3 == 0 && i % 5 != 0) {
                l.add("Fizz");
            } else {
                l.add("FizzBuzz");
            }
        }
        return l;
    }

    // JZOF51 HARD
    public int reversePairs(int[] nums) {
        return mergeSort(nums);
    }

    private int mergeSort(int[] arr) {
        int[] mock = new int[arr.length];
        return mergeSort(arr, mock, 0, arr.length - 1);
    }

    // 对数组进行归并排序
    private int mergeSort(int[] real, int[] mock, int start, int end) {
        if (start < end) {
            int mid = start + (end - start) / 2;
            int leftCount = mergeSort(real, mock, start, mid);
            int rightCount = mergeSort(real, mock, mid + 1, end);
            int mergeCount = mergeSortHelper(real, mock, start, mid, end);
            return leftCount + rightCount + mergeCount;
        }
        return 0;
    }

    private int mergeSortHelper(int[] real, int[] mock, int start, int mid, int end) {
        int leftPointer = start, rightPointer = mid + 1, mockPointer = start;
        int count = 0;
        // 目标: 将排好序的放入mock中
        // 假设[start,mid] [mid+1,end]已经排好序
        while (leftPointer <= mid && rightPointer <= end) {
            if (real[leftPointer] <= real[rightPointer]) {
                mock[mockPointer++] = real[leftPointer++];
                count += (rightPointer - (mid + 1));
            } else {
                mock[mockPointer++] = real[rightPointer++];
            }
        }
        while (leftPointer <= mid) {
            mock[mockPointer++] = real[leftPointer++];
            count += (rightPointer - (mid + 1));
        }
        while (rightPointer <= end) {
            mock[mockPointer++] = real[rightPointer++];
        }
        // 将mock的数放回real中
        for (int i = start; i <= end; i++) {
            real[i] = mock[i];
        }
        return count;
    }


    // PRIMEOJ 1002
    public int primeoj1002(int[][] money) {
        int totalVotes = 0;
        for (int[] i : money) {
            totalVotes += i[0];
        }
        int[] dp = new int[totalVotes + 1];
        // dp[i][j] 表示贿赂前i个群友得到j张选票的最小花费
        // dp[i][j] = Math.min(dp[i-1][j], dp[i-1][j - money[i][0] ] + money[i][1])

        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i = 1; i <= money.length; i++) {
            for (int j = totalVotes; j >= money[i - 1][0]; j--) {
                dp[j] = Math.min(dp[j], dp[j - money[i - 1][0]] + money[i - 1][1]);
            }
        }

        int result = Integer.MAX_VALUE;
        for (int i = (totalVotes + 1) / 2; i <= totalVotes; i++) {
            result = Math.min(dp[i], result);
        }
        return result;
    }

    // LC371 **
    public int getSum(int a, int b) {
        if (a == 0) return b;
        if (b == 0) return a;
        while (b != 0) {
            int tmp = a ^ b;
            b = (a & b) << 1;
            a = tmp;
        }
        return a;
    }

    // LC350 Sort
    public int[] intersectStream(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int[] ints = new int[nums1.length + nums2.length];
        int idx = 0, idx1 = 0, idx2 = 0;
        while (idx1 < nums1.length && idx2 < nums2.length) {
            int n1 = nums1[idx1], n2 = nums2[idx2];
            if (n1 < n2) {
                idx1++;
            } else if (n2 < n1) {
                idx2++;
            } else {
                ints[idx] = n1;
                idx1++;
                idx2++;
                idx++;
            }
        }
        return Arrays.copyOfRange(ints, 0, idx);
    }

    // LC350
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> m1 = new HashMap<>(), m2 = new HashMap<>();
        for (int i : nums1) {
            m1.put(i, m1.getOrDefault(i, 0) + 1);
        }
        for (int i : nums2) {
            m2.put(i, m2.getOrDefault(i, 0) + 1);
        }
        Set<Integer> m1KSCopy = new HashSet<>(m1.keySet());
        Set<Integer> ints = new HashSet<>();
        for (int i : m2.keySet()) {
            if (!m1KSCopy.add(i)) {
                ints.add(i);
            }
        }
        List<Integer> result = new LinkedList<>();
        for (int i : ints) {
            int count = Math.min(m1.get(i), m2.get(i));
            for (int j = 0; j < count; j++) {
                result.add(i);
            }
        }
        int[] ans = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            ans[i] = result.get(i);
        }
        return ans;
    }

    // LC334
    public boolean increasingTriplet(int[] nums) {
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : nums) {
            Integer ceil = ts.ceiling(i);
            if (ceil != null) {
                ts.remove(ceil);
            }
            ts.add(i);
            if (ts.size() >= 3) return true;
        }
        return false;
    }

    // LC329 Hard
    final int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int rowNum = matrix.length;
        int colNum = matrix[0].length;
        Integer[][] memo = new Integer[rowNum][colNum];
        int result = 0;
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                result = Math.max(result, lipDfs(memo, i, j, matrix));
            }
        }
        return result;
    }

    private int lipDfs(Integer[][] memo, int row, int col, int[][] matrix) {
        if (memo[row][col] != null) {
            return memo[row][col];
        }
        memo[row][col] = 1;
        for (int[] dir : directions) {
            int newRow = row + dir[0], newCol = col + dir[1];
            if (newRow >= 0 && newRow < matrix.length && newCol >= 0 && newCol < matrix[0].length && matrix[newRow][newCol] > matrix[row][col]) {
                memo[row][col] = Math.max(memo[row][col], lipDfs(memo, newRow, newCol, matrix) + 1);
            }
        }
        return memo[row][col];
    }

    // LC328
    public ListNode oddEvenList(ListNode head) {
        ListNode first = head;
        if (first == null || first.next == null) return head;
        ListNode second = head.next;
        ListNode secondDummy = new ListNode(-1);
        secondDummy.next = second;
        while (first != null && second != null) {
            first.next = second.next;
            if (second.next == null) break;
            second.next = second.next.next;

            first = first.next;
            second = second.next;
        }
        first.next = secondDummy.next;

        return head;
    }

    // LC1486
    public int xorOperation(int n, int start) {
        // nums[i] = start + 2*i
        // n==nums.length
        int result = start;
        for (int i = 1; i < n; i++) {
            result ^= (start + 2 * i);
        }
        return result;
    }

    // LC324
    public void wiggleSort(int[] nums) {
        Arrays.sort(nums);
        int[] leftHalf = Arrays.copyOfRange(nums, 0, (nums.length + 1) / 2);
        int[] rightHalf = Arrays.copyOfRange(nums, (nums.length + 1) / 2, nums.length);
        int leftPointer = leftHalf.length - 1;
        int rightPointer = rightHalf.length - 1;
        for (int i = 0; i < nums.length; i++) {
            if (i % 2 == 0) {
                nums[i] = leftHalf[leftPointer--];
            } else {
                nums[i] = rightHalf[rightPointer--];
            }
        }
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


// LC380 O(1) getRandom
class RandomizedSet {
    Map<Integer, Integer> idxMap;
    List<Integer> entities;
    Random random = new Random();

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {
        idxMap = new HashMap<>();
        entities = new ArrayList<>();
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

        int idx = idxMap.get(val);
        int lastEntity = entities.get(entities.size() - 1);
        // swap entities[idx] entities[last]
        entities.set(idx, lastEntity);
        idxMap.put(lastEntity, idx);
        idxMap.remove(val);
        entities.remove(entities.size() - 1);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        return entities.get(random.nextInt(entities.size()));
    }
}

// LC384
class Solution {
    int[] origArr;

    Random random = new Random();

    public Solution(int[] nums) {
        this.origArr = nums;
    }

    /**
     * Resets the array to its original configuration and return it.
     */
    public int[] reset() {
        return origArr;
    }

    /**
     * Returns a random shuffling of the array.
     */
    public int[] shuffle() {
        int size = origArr.length;
        int[] shuffled = Arrays.copyOf(origArr, size);
        int idx = size;
        while (idx > 0) {
            int randomIdx = random.nextInt(idx--);
            // swap shuffled[randomIdx] shuffled[idx]
            if (shuffled[randomIdx] != shuffled[idx]) {
                shuffled[randomIdx] ^= shuffled[idx];
                shuffled[idx] ^= shuffled[randomIdx];
                shuffled[randomIdx] ^= shuffled[idx];
            }
        }
        return shuffled;
    }
}