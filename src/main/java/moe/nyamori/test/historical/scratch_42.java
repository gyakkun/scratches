package moe.nyamori.test.historical;


import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class scratch_42 {
    public static void main(String[] args) {
        scratch_42 s = new scratch_42();
        long timing = System.currentTimeMillis();
        int[] arr = new int[]{1};
        System.err.println(s.wiggleMaxLengthON2(arr));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC376 O(n) time Solution
    public int wiggleMaxLength(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return n;
        }
        int up = 1, down = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) {
                up = Math.max(up, down + 1);
            } else if (nums[i] < nums[i - 1]) {
                down = Math.max(up + 1, down);
            }
        }
        return Math.max(up, down);
    }

    // LC376 O(n*n) time
    public int wiggleMaxLengthON2(int[] nums) {
        int ans = 1;
        int n = nums.length;
        lc376Status[][] dp = new lc376Status[n][n];
        // dp[i][j] 表示 nums[i,j]之间最长摆动摆动序列的长度

        for (int i = 0; i < n - 1; i++) {
            if (nums[i] != nums[i + 1]) {
                dp[i][i + 1] = new lc376Status(2, nums[i] - nums[i + 1] < 0);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 2; j < n; j++) {
                if (dp[i][j - 1] != null) {
                    Boolean status = null;
                    if (nums[j - 1] - nums[j] < 0) {
                        status = true;
                    } else if (nums[j - 1] - nums[j] > 0) {
                        status = false;
                    }

                    if (status != null && status != dp[i][j - 1].status) {
                        dp[i][j] = new lc376Status(dp[i][j - 1].len + 1, status);
                    } else {
                        dp[i][j] = new lc376Status(dp[i][j - 1].len, dp[i][j - 1].status);
                    }
                    ans = Math.max(ans, dp[i][j].len);
                }
            }
        }
        return ans;
    }

    class lc376Status {
        int len;
        Boolean status; // upForTrue, downForFalse, equal for null

        public lc376Status(int len, boolean upForTrue) {
            this.len = len;
            this.status = upForTrue;
        }
    }

    // LC1739 Hard 二分
//    Map<Long, Long> lc1739Memo;
    public int minimumBoxes(int n) {
//        lc1739Memo = new HashMap<>();
        long left = 1, right = n;
        while (left < right) { // 二分查找第一层应该放多少个, 第一层放的个数越多, 能放下的总数越多
            // 找出第一个total>=n时候的mid
            long mid = left + (right - left) / 2;
            long total = lc1739TotalHelper(mid);
            if (total >= n) {
                right = mid;
            } else if (total < n) {
                left = mid + 1;
            }
        }

        return (int) left;
    }

    private long lc1739Helper(long total) { // 这一层放total个盒子的时候, 最多有多少个上面可以放盒子？
        // (1+n)*n / 2 <= total
        // n*n + n - 2*total <= 0
        long n = (long) ((-1.0d + Math.sqrt(1.0d + 8 * total)) / (2.0)); // n的下界
        long remain = total - ((1 + n) * n) / 2;
        if (remain <= 1) {
            return (n * (n - 1) / 2);
        } else {
            return (n * (n - 1) / 2) + (remain - 1);
        }
    }

    private long lc1739TotalHelper(long firstFloorCount) { // 第一层摆放firstFloorCount个的时候, 最多总共可以摆放多少个？
//        if(lc1739Memo.containsKey(firstFloorCount)) return lc1739Memo.get(firstFloorCount);
        long sum = firstFloorCount;
        long thisFloorCount = firstFloorCount;
        while (thisFloorCount >= 3) {
            long nextFloorCount = lc1739Helper(thisFloorCount);
            sum += nextFloorCount;
            thisFloorCount = nextFloorCount;
        }
//        lc1739Memo.put(firstFloorCount, sum);
        return sum;
    }

    // LC1760 Solution 二分
    public int minimumSize(int[] nums, int maxOperations) {
        int left = 1, right = Arrays.stream(nums).max().getAsInt();
        int ans = -1;
        while (left <= right) { // 二分查找每个袋子的最大球个数y, y越大, 总操作数sum越小, 是一个单调的关系
            int mid = left + (right - left) / 2;
            long sum = 0;
            for (int i : nums) {
                sum += (i - 1) / mid;
            }
            if (sum <= maxOperations) { // 总操作数小于最大操作数, 说明y太大, 还可能有更小的值, 往左边找
                ans = mid;
                right = mid - 1;
            } else { // 总操作数太多, 说明y太小, 需要往右边找
                left = mid + 1;
            }
        }

        return ans;
    }

    // LC63
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid[0][0] == 1) return 0;
        int[][] dp = new int[obstacleGrid.length][obstacleGrid[0].length];
        dp[0][0] = 1;
        for (int i = 1; i < obstacleGrid.length; i++) {
            if (obstacleGrid[i][0] != 1 && dp[i - 1][0] == 1) {
                dp[i][0] = 1;
            }
        }
        for (int i = 1; i < obstacleGrid[0].length; i++) {
            if (obstacleGrid[0][i] != 1 && dp[0][i - 1] == 1) {
                dp[0][i] = 1;
            }
        }
        for (int i = 1; i < obstacleGrid.length; i++) {
            for (int j = 1; j < obstacleGrid[0].length; j++) {
                if (obstacleGrid[i][j] != 1) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[obstacleGrid.length - 1][obstacleGrid[0].length - 1];

    }

    // LC51 LC52 N Queens
    List<List<String>> lc51Result;

    public List<List<String>> solveNQueens(int n) {
        lc51Result = new ArrayList<>();
        lc51Helper(n, 0, 0, new ArrayList<>(n), 0);
        return lc51Result;
    }

    private void lc51Helper(int n, int curRow, int curCol, List<Integer> curBoard, int status) {
        if (curRow == n) {
            if (lc51check(curBoard, n)) {
                lc51Result.add(lc51ToListString(curBoard, n));
            }
            return;
        }
        for (int i = curCol; i < n; i++) {
            if (curBoard.size() != 0 && (i == curBoard.get(curBoard.size() - 1) + 1 || i == curBoard.get(curBoard.size() - 1) - 1)) {
                continue;
            }
            if (((status >> i) & 1) == 1) {
                continue;
            }
            int newStatus = ((1 << i) ^ status);
            curBoard.add(i);
            lc51Helper(n, curRow + 1, 0, curBoard, newStatus);
            curBoard.remove(curBoard.size() - 1);
        }
    }

    private boolean lc51check(List<Integer> curBoard, int n) {
        // 同一列
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (curBoard.get(i) == curBoard.get(j)) return false;
            }
        }
        // 对角线
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int aRow = i, aCol = curBoard.get(i);
                int bRow = j, bCol = curBoard.get(j);
                while (aRow >= 0 && aRow < n && aCol >= 0 && aCol < n) {
                    if (aRow == bRow && aCol == bCol) return false;
                    aRow++;
                    aCol++;
                }
                aRow = i;
                aCol = curBoard.get(i);
                while (aRow >= 0 && aRow < n && aCol >= 0 && aCol < n) {
                    if (aRow == bRow && aCol == bCol) return false;
                    aRow--;
                    aCol--;
                }
                aRow = i;
                aCol = curBoard.get(i);
                while (aRow >= 0 && aRow < n && aCol >= 0 && aCol < n) {
                    if (aRow == bRow && aCol == bCol) return false;
                    aRow++;
                    aCol--;
                }
                aRow = i;
                aCol = curBoard.get(i);
                while (aRow >= 0 && aRow < n && aCol >= 0 && aCol < n) {
                    if (aRow == bRow && aCol == bCol) return false;
                    aRow--;
                    aCol++;
                }
            }
        }
        return true;
    }

    private List<String> lc51ToListString(List<Integer> curIntegerBoard, int n) {
        List<String> result = new ArrayList<>(curIntegerBoard.size());
        for (int i : curIntegerBoard) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sb.append('.');
                } else {
                    sb.append('Q');
                }
            }
            result.add(sb.toString());
        }
        return result;
    }

    // LC1190
    public String reverseParentheses(String s) {
        Deque<StringBuffer> sbStack = new LinkedList<>();
        sbStack.push(new StringBuffer());
        char[] cArr = s.toCharArray();
        int n = s.length();
        for (int i = 0; i < n; i++) {
            if (cArr[i] == '(') {
                sbStack.push(new StringBuffer());
            } else if (cArr[i] == ')') {
                StringBuffer tmp = sbStack.pop();
                sbStack.peek().append(tmp.reverse());
            } else {
                sbStack.peek().append(cArr[i]);
            }
        }
        return sbStack.peek().toString();
    }

    // LC1787 TLE
    public int minChanges(int[] nums, int k) {
        int n = nums.length;
        int[][] mArr = new int[k][1 << 10];
        int[] eachGroupCtr = new int[k];
        Arrays.fill(eachGroupCtr, 0);
        for (int i = 0; i < n; i++) {
            mArr[i % k][nums[i]]++;
            eachGroupCtr[i % k]++;
        }
        Integer[][] memo = new Integer[k][1 << 10];
        return minChangesHelper(k - 1, 0, k, memo, eachGroupCtr, mArr);
    }

    private int minChangesHelper(int i, int mask, int k, Integer[][] memo, int[] eachGroupCtr, int[][] mArr) {
        if (i >= 0 && mask >= 0 && memo[i][mask] != null) return memo[i][mask];
        if (i < 0 && mask == 0) return 0;
        if (i < 0) return Integer.MAX_VALUE / 2;
        int max = 1 << 10;

        int t2 = Integer.MAX_VALUE / 2;
        int t1 = Integer.MAX_VALUE / 2;

        for (int j = 0; j < max; j++) {
            if (mArr[i][j] != 0) {
                t1 = Math.min(t1, minChangesHelper(i - 1, mask ^ j, k, memo, eachGroupCtr, mArr) - mArr[i][j]);
            } else {
                t2 = Math.min(t2, minChangesHelper(i - 1, mask ^ j, k, memo, eachGroupCtr, mArr));
            }
        }
        int result = eachGroupCtr[i] + Math.min(t1, t2);
        memo[i][mask] = result;
        return result;
    }

    // LC1707 TBD Trie 位运算
    public int[] maximizeXor(int[] nums, int[][] queries) {
        int[] result = new int[queries.length];
        return result;
    }

    // LC739 单调栈
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] nge = new int[n];
        Arrays.fill(nge, -1);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                nge[stack.pop()] = i;
            }
            stack.push(i);
        }
        for (int i = 0; i < n; i++) {
            if (nge[i] != -1) {
                nge[i] = nge[i] - i;
            } else {
                nge[i] = 0;
            }
        }
        return nge;
    }

    // LC647 求回文子串的个数
    public int countSubstrings(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int ctr = 0;

        for (int len = 0; len < n; len++) {
            for (int i = 0; i + len < n; i++) {
                int j = i + len;
                if (len == 0) {
                    dp[i][j] = true;
                } else if (len == 1) {
                    dp[i][j] = s.charAt(i) == s.charAt(j);
                } else {
                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
                }
                if (dp[i][j]) ctr++;
            }
        }
        return ctr;
    }

    // LC810 Solution 精妙
    public boolean xorGame(int[] nums) {
        if (nums.length % 2 == 0) {
            return true;
        }
        int xorSum = 0;
        for (int i : nums) {
            xorSum ^= i;
        }
        return xorSum == 0;
    }

    // LC664 Recursive
    public int strangePrinter(String s) {
        int n = s.length();
        Integer[][] memo = new Integer[n][n];
        return strangePrinterHelper(s, 0, n - 1, memo);
    }

    private int strangePrinterHelper(String s, int i, int j, Integer[][] memo) {
        if (i > j) return 0;
        if (memo[i][j] != null) {
            return memo[i][j];
        }
        if (s.charAt(i) == s.charAt(j)) {
            if (i == j) {
                memo[i][j] = 1;
            } else {
                memo[i][j] = strangePrinterHelper(s, i, j - 1, memo);
            }
            return memo[i][j];
        }
        int result = Integer.MAX_VALUE;
        for (int k = i; k < j; k++) { // 注意这里是怎么枚举 [i,i] [i,i+1] ... [j-1,j], [j,j]的, 留意k的范围和下面的传参
            result = Math.min(strangePrinterHelper(s, i, k, memo) + strangePrinterHelper(s, k + 1, j, memo), result);
        }
        memo[i][j] = result;
        return memo[i][j];
    }

    // LC664 Solution
    public int strangePrinterSolution(String s) {
        int n = s.length();
        char[] cArr = s.toCharArray();
        int[][] dp = new int[n][n];
        // dp[i][j] 表示区间[i,j]打印需要的最小次数
        // 1) c[i] == c[j], dp[i][j] = dp[i][j-1]
        // 2) c[i] != c[j], dp[i][j] = 枚举 k:[i,j), 取最小的dp[i][k]+dp[k+1][j]
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (cArr[i] == cArr[j]) {
                    dp[i][j] = j - i == 0 ? 1 : dp[i][j - 1];
                } else {
                    dp[i][j] = Integer.MAX_VALUE;
                    for (int k = i; k < j; k++) {
                        dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]);
                    }
                }
            }
        }
        return dp[0][n - 1];
    }

    // LC621 ** Solution
    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> freq = new HashMap<Character, Integer>();
        // 最多的执行次数
        int maxExec = 0;
        for (char ch : tasks) {
            int exec = freq.getOrDefault(ch, 0) + 1;
            freq.put(ch, exec);
            maxExec = Math.max(maxExec, exec);
        }

        // 具有最多执行次数的任务数量
        int maxCount = 0;
        Set<Map.Entry<Character, Integer>> entrySet = freq.entrySet();
        for (Map.Entry<Character, Integer> entry : entrySet) {
            int value = entry.getValue();
            if (value == maxExec) {
                ++maxCount;
            }
        }

        return Math.max((maxExec - 1) * (n + 1) + maxCount, tasks.length);
    }

    // LC617
    public TreeNode42 mergeTrees(TreeNode42 root1, TreeNode42 root2) {
        if (root1 == null && root2 == null) return null;
        if (root1 == null && root2 != null) {
            return root2;
        }
        if (root1 != null && root2 == null) {
            return root1;
        }
        int result = root1.val + root2.val;
        root1.val = result;
        TreeNode42 r1L = mergeTrees(root1.left, root2.left);
        TreeNode42 r1R = mergeTrees(root1.right, root2.right);
        root1.left = r1L;
        root1.right = r1R;
        return root1;
    }

    // LC581 Stack O(n) Time 单调栈思想
    public int findUnsortedSubarrayStack(int[] nums) {
        Deque<Integer> stack = new LinkedList<>();
        stack.push(0);
        int minLeft = nums.length - 1;
        for (int i = 1; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[i] < nums[stack.peek()]) {
                minLeft = Math.min(stack.pop(), minLeft);
            }
            stack.push(i);
        }
        if (minLeft == nums.length - 1) return 0;
        stack.clear();
        int maxRight = 0;
        stack.push(nums.length - 1);
        for (int i = nums.length - 2; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                maxRight = Math.max(stack.pop(), maxRight);
            }
            stack.push(i);
        }
        return nums.length - (minLeft + (nums.length - 1 - maxRight));
    }

    // LC581 O(n*log(n)) Time
    public int findUnsortedSubarray(int[] nums) {
        int[] orig = new int[nums.length];
        System.arraycopy(nums, 0, orig, 0, nums.length);
        Arrays.sort(nums);
        int leftIdx = -1;
        int rightIdx = nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != orig[i]) {
                leftIdx = i;
                break;
            }
        }
        if (leftIdx == -1) {
            return 0;
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] != orig[i]) {
                rightIdx = i;
                break;
            }
        }
        return nums.length - (leftIdx + (nums.length - 1 - rightIdx));
    }

    // LC543
    int lc543ans = 0;

    public int diameterOfBinaryTree(TreeNode42 root) {
        lc543ans = 1;
        lc543Recursive(root);
        return lc543ans - 1;
    }

    // 记录遍历左右子树经过节点数的最大值, 节点数-1 = 路径长度
    private int lc543Recursive(TreeNode42 root) {
        if (root == null) {
            return 0;
        }
        int left = lc543Recursive(root.left);
        int right = lc543Recursive(root.right);
        lc543ans = Math.max(lc543ans, left + right + 1); // +1 为加上root本身这个节点
        return Math.max(left, right) + 1; // 返回的是左右子树节点数的最大值加上自己这个节点
    }

    // LC538 **
    int lc538Sum = 0;

    public TreeNode42 convertBST(TreeNode42 root) {
        if (root != null) {
            convertBST(root.right);
            lc538Sum += root.val;
            root.val = lc538Sum;
            convertBST(root.left);
        }
        return root;
    }


    // LC461
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }

    // LC438 O(S*Sigma(C)) time, Sigma(C) represents character set length
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new LinkedList<>();
        int[] pFreq = new int[26];
        for (char c : p.toCharArray()) {
            pFreq[c - 'a']++;
        }
        int pLen = p.length();
        char[] sArr = s.toCharArray();
        int[] freq = new int[26];
        boolean[] changeSet = new boolean[26];
        for (int i = 0; i < sArr.length; i++) {
            if (i < (pLen - 1)) {
                freq[sArr[i] - 'a']++;
                changeSet[sArr[i] - 'a'] = true;
            } else {
                freq[sArr[i] - 'a']++;
                changeSet[sArr[i] - 'a'] = true;
                if (i >= pLen) {
                    freq[sArr[i - pLen] - 'a']--;
                    changeSet[sArr[i - pLen] - 'a'] = true;
                }
                boolean flag = true;
                for (int j = 0; j < 26; j++) {
                    if (changeSet[j]) {
                        if (freq[j] != pFreq[j]) {
                            flag = false;
                            break;
                        }
                    }
                }
                if (flag) {
                    result.add(i - pLen + 1);
                    Arrays.fill(changeSet, false);
                }
            }
        }
        return result;
    }

    // LC406 **
    public int[][] reconstructQueue(int[][] people) {
        // people[i][0] = h : 身高h
        // people[i][1] = k : 前面有k个人身高大于等于自己
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        });
        int[][] result = new int[people.length][];
        for (int i = 0; i < people.length; i++) {
            int numOfEmpty = 0; // 统计从左往右第k+i个空位置
            int[] p = people[i];
            int ki = p[1];
            int j = 0;
            for (; j < people.length; j++) {
                if (result[j] == null) numOfEmpty++;
                if (numOfEmpty == ki + 1) {
                    break;
                }
            }
            result[j] = p;
        }
        return result;
    }

    // LC718 最长公共子串 Longest Common Sub-array  / Substring
    public int findLength(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        int maxLen = 0;
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > maxLen) {
                        maxLen = dp[i][j];
                    }
                }
            }
        }
        return maxLen;
    }

    // LC1035 同LC1143 最长公共子序列 Longest Common Subsequence
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        // dp[i][j] 表示nums1前i个数字和nums2前j个数字最多可以组成多少条不相交的线
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[nums1.length][nums2.length];
    }


    Random quickAlgorithmRandom = new Random();

    // Quick Select For topK
    public Integer quickSelect(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int rIdx = quickAlgorithmRandom.nextInt(end - start) + start;
        if (arr[start] != arr[rIdx]) {
            arr[start] ^= arr[rIdx];
            arr[rIdx] ^= arr[start];
            arr[start] ^= arr[rIdx];
        }
        int pivot = arr[start];
        int left = start;
        int right = end;
        while (left < right) {
            while (left < right && arr[right] > pivot) { // 找到第一个比pivot小的数
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivot) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivot;
        if (left == (arr.length - topK)) {
            return arr[left];
        }
        Integer leftResult = quickSelect(arr, start, left - 1, topK);
        Integer rightResult = quickSelect(arr, right + 1, end, topK);
        if (leftResult != null) {
            return leftResult;
        }
        if (rightResult != null) {
            return rightResult;
        }
        return null;
    }


    // Quick Sort
    public void quickSort(Integer[] arr, int start, int end, boolean isRandom) {
        if (start >= end) return;
        if (isRandom) {
            int rIdx = quickAlgorithmRandom.nextInt(end - start) + start;
            if (arr[start] != arr[rIdx]) {
                arr[start] ^= arr[rIdx];
                arr[rIdx] ^= arr[start];
                arr[start] ^= arr[rIdx];
            }
        }

        int pivot = arr[start];
        int left = start;
        int right = end;
        while (left < right) {
            while (left < right && arr[right] > pivot) { // 找到第一个比pivot小的数
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivot) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivot;
        quickSort(arr, start, left - 1, isRandom);
        quickSort(arr, right + 1, end, isRandom);

    }


    // LC394
    public String decodeString(String s) {
        char[] arr = s.toCharArray();
        int len = s.length();
        int idx = 0;
        Deque<Integer> stackNum = new LinkedList<>();
        Deque<StringBuffer> stackSb = new LinkedList<>();
        stackNum.push(1);
        stackSb.push(new StringBuffer());
        int num = 0;
        for (; idx < len; idx++) {
            if (Character.isDigit(arr[idx])) {
                num = num * 10 + (arr[idx] - '0');
            } else if (arr[idx] == '[') {
                stackNum.push(num);
                num = 0;
                stackSb.push(new StringBuffer());
            } else if (arr[idx] == ']') {
                StringBuffer top = stackSb.pop();
                int ctr = stackNum.pop();
                for (int i = 0; i < ctr; i++) {
                    stackSb.peek().append(top);
                }
            } else {
                stackSb.peek().append(arr[idx]);
            }
        }
        return stackSb.peek().toString();
    }

    // LC301 Solution
    Set<String> lc301result;
    char[] lc301CharArr;
    int lc301Len;

    public List<String> removeInvalidParentheses(String s) {
        lc301result = new HashSet<>();
        lc301CharArr = s.toCharArray();
        lc301Len = s.length();
        // 待删除的左括号数和右括号数
        int leftRemove = 0;
        int rightRemove = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                leftRemove++;
            } else if (c == ')') {
                if (leftRemove == 0) {
                    rightRemove++;
                } else {
                    leftRemove--;
                }
            }
        }
        StringBuffer sb = new StringBuffer();
        lc301backtrack(leftRemove, rightRemove, sb, 0, 0, 0);
        return new ArrayList<>(lc301result);
    }

    private void lc301backtrack(int leftRemove, int rightRemove, StringBuffer sb, int curIdx, int leftCtr, int rightCtr) {
        if (curIdx == lc301Len) {
            if (leftRemove == 0 && rightRemove == 0) {
                lc301result.add(sb.toString());
            }
            return;
        }

        char c = lc301CharArr[curIdx];
        if (c == '(' && leftRemove > 0) {
            lc301backtrack(leftRemove - 1, rightRemove, sb, curIdx + 1, leftCtr, rightCtr);
        }
        if (c == ')' && rightRemove > 0) {
            lc301backtrack(leftRemove, rightRemove - 1, sb, curIdx + 1, leftCtr, rightCtr);
        }

        sb.append(c);
        if (c != '(' && c != ')') {
            lc301backtrack(leftRemove, rightRemove, sb, curIdx + 1, leftCtr, rightCtr);
        } else if (c == '(') {
            lc301backtrack(leftRemove, rightRemove, sb, curIdx + 1, leftCtr + 1, rightCtr);
        } else if (rightCtr < leftCtr) {
            lc301backtrack(leftRemove, rightRemove, sb, curIdx + 1, leftCtr, rightCtr + 1);
        }
        sb.deleteCharAt(sb.length() - 1);
    }

    // LC114 展开 先序遍历顺序
    public void flatten(TreeNode42 root) {
        if (root == null || (root.left == null && root.right == null)) return;
        Deque<TreeNode42> stack = new LinkedList<>();
        List<TreeNode42> inorder = new ArrayList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode42 tmp = stack.pop();
            inorder.add(tmp);
            if (tmp.right != null) {
                stack.push(tmp.right);
            }
            if (tmp.left != null) {
                stack.push(tmp.left);
            }
        }
        for (int i = 0; i < inorder.size() - 1; i++) {
            inorder.get(i).right = inorder.get(i + 1);
            inorder.get(i).left = null;
        }
    }

    // LC692 O(n) Space O(n*log(k))time
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> freq = new HashMap<>();
        for (String word : words) {
            freq.put(word, freq.getOrDefault(word, 0) + 1);
        }
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue() == o2.getValue() ? o2.getKey().compareTo(o1.getKey()) : o1.getValue() - o2.getValue();
            }
        });
        for (Map.Entry<String, Integer> entry : freq.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        List<String> result = new ArrayList<>(k);
        while (!pq.isEmpty()) {
            result.add(pq.poll().getKey());
        }
        Collections.reverse(result);
        return result;
    }

    // LC31 **
    public void nextPermutation(int[] nums) {
        int right = nums.length - 2;
        while (right >= 0 && nums[right] >= nums[right + 1]) {
            right--;
        }
        if (right >= 0) {
            int left = nums.length - 1;
            while (left >= 0 && nums[right] >= nums[left]) {
                left--;
            }
            arraySwap(nums, left, right);
        }
        arrayReverse(nums, right + 1, nums.length - 1);
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        result.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
    }

    private void arraySwap(int[] arr, int i, int j) {
        if (i < 0 || i >= arr.length || j < 0 || j >= arr.length) return;
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    private void arrayReverse(int[] arr, int i, int j) {
        if (i < 0 || i >= arr.length || j < 0 || j >= arr.length || j <= i) return;
        int mid = (i + j + 1) / 2;
        for (int k = i; k < mid; k++) {
            arraySwap(arr, k, j - (k - i));
        }
    }

    // LC85
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) return 0;
        int[][] prefix = new int[matrix.length + 1][matrix[0].length + 1];
        int max = 0;
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    prefix[i][j] = prefix[i][j - 1] + 1;
                }
            }
        }
        for (int i = 0; i <= matrix.length; i++) {
            for (int j = 0; j <= matrix[0].length; j++) {
                int width = prefix[i][j];
                for (int k = i; k >= 0; k--) {
                    int height = i - k + 1;
                    width = Math.min(width, prefix[k][j]);
                    max = Math.max(max, height * width);
                }
            }
        }
        return max;
    }

    // LC221
    public int maximalSquare(char[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int max = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                    }
                    if (dp[i][j] > max) {
                        max = dp[i][j];
                    }
                }
            }
        }
        return max * max;
    }

    // LC557
    public String reverseWords(String s) {
        String[] arr = s.split(" ");
        for (int i = 0; i < arr.length; i++) {
            String word = arr[i];
            arr[i] = new StringBuffer(word).reverse().toString();
        }
        return String.join(" ", arr);
    }

    // LC292
    public boolean canWinNim(int n) {
        return n % 4 != 0;
    }

    // LC235 二叉搜索树
    public TreeNode42 lowestCommonAncestor(TreeNode42 root, TreeNode42 p, TreeNode42 q) {
        TreeNode42 anc = root;
        while (true) {
            if (p.val > anc.val && q.val > anc.val) {
                anc = anc.right;
            } else if (p.val < anc.val && q.val < anc.val) {
                anc = anc.left;
            } else {
                break;
            }
        }
        return anc;
    }

    // LC231
    public boolean isPowerOfTwo(int n) {
        if (n <= 0) return false;
        int ctr = 0;
        for (int i = 0; i < Integer.SIZE; i++) {
            if (((n >> i) & 1) == 1) ctr++;
            if (ctr > 1) return false;
        }
        return ctr == 1;
    }

    // LC89
    public List<Integer> grayCode(int n) {
        int max = 1 << n;
        List<Integer> result = new ArrayList<>(max);
        result.add(0);
        int head = 1;
        for (int i = 0; i < n; i++) {
            for (int j = result.size() - 1; j >= 0; j--) {
                result.add(result.get(j) + head);
            }
            head = head << 1;
        }
        return result;
    }

    // LC89
    public List<Integer> grayCodeSet(int n) {
        if (n == 0) return new ArrayList<Integer>() {{
            add(0);
        }};
        int max = (1 << n);
        List<Integer> result = new ArrayList<>(max);
        Set<Integer> set = new HashSet<>(max);
        result.add(0);
        set.add(0);
        for (int i = 1; i < max; i++) {
            int former = result.get(result.size() - 1);
            for (int j = 0; j < n; j++) {
                int cur = former;
                if (((former >> j) & 1) == 1) {
                    cur = cur - (1 << j);
                } else {
                    cur = cur + (1 << j);
                }
                if (!set.contains(cur)) {
                    set.add(cur);
                    result.add(cur);
                    break;
                }
            }
        }
        return result;
    }

    // LC43
    public String multiply(String num1, String num2) {
        StringBuffer sb;
        // 假设num2更短, num1更长
        if (num1.length() < num2.length()) {
            String tmp = num2;
            num2 = num1;
            num1 = tmp;
        }
        List<String> each = new ArrayList<>(num2.length());
        for (int i = 0; i < num2.length(); i++) {
            int carry = 0;
            int twoDigit = num2.charAt(num2.length() - 1 - i) - '0';
            sb = new StringBuffer();
            for (int j = 0; j < num1.length(); j++) {
                int oneDigit = num1.charAt(num1.length() - 1 - j) - '0';
                int tmpResult = twoDigit * oneDigit + carry;
                carry = tmpResult / 10;
                sb.append(tmpResult % 10);
            }
            if (carry != 0) {
                sb.append(carry);
            }
            sb = sb.reverse();
            for (int k = 0; k < i; k++) {
                sb.append("0");
            }
            each.add(sb.toString());
        }

        String result = each.get(0);
        for (int i = 1; i < each.size(); i++) {
            result = add(result, each.get(i));
        }
        if (result.matches("^0+$")) {
            result = "0";
        }
        return result;
    }

    public String add(String num1, String num2) {
        StringBuffer sb = new StringBuffer();
        // 确保num1长 num2短
        if (num1.length() < num2.length()) {
            String tmp = num1;
            num1 = num2;
            num2 = tmp;
        }
        int carry = 0;
        int i = 0;
        for (; i < num2.length(); i++) {
            int twoDigit = num2.charAt(num2.length() - 1 - i) - '0';
            int oneDigit = num1.charAt(num1.length() - 1 - i) - '0';
            int tmpSum = twoDigit + oneDigit + carry;
            carry = tmpSum / 10;
            sb.append(tmpSum % 10);
        }
        for (; i < num1.length(); i++) {
            int oneDigit = num1.charAt(num1.length() - 1 - i) - '0';
            int tmpSum = oneDigit + carry;
            carry = tmpSum / 10;
            sb.append(tmpSum % 10);
        }
        if (carry != 0) sb.append(carry);
        sb = sb.reverse();
        return sb.toString();
    }

    // LC9
    public boolean isPalindrome(int x) {
        String s = String.valueOf(x);
        for (int i = 0; i < (s.length() + 1) / 2; i++) {
            if (s.charAt(i) != s.charAt(s.length() - 1 - i)) return false;
        }
        return true;
    }

    // LC15 3sum
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> result = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = n - 1;
            while (left < right) {
                int tmpSum = nums[i] + nums[left] + nums[right];
                if (tmpSum == 0) {
                    List<Integer> tmpResult = new ArrayList<>(3);
                    tmpResult.add(nums[i]);
                    tmpResult.add(nums[left]);
                    tmpResult.add(nums[right]);
                    result.add(tmpResult);
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (tmpSum > 0) {
                    right--;
                } else if (tmpSum < 0) {
                    left++;
                }
            }
        }
        return result;
    }

    // LC15 3sum TLE
    public List<List<Integer>> threeSumTLE(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        Map<Integer, List<Pair<Integer, Integer>>> twoSumIdxMap = new HashMap<>();
        Map<Integer, List<Integer>> arrReverseMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            arrReverseMap.putIfAbsent(nums[i], new ArrayList<>());
            arrReverseMap.get(nums[i]).add(i);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int twoSum = nums[i] + nums[j];
                twoSumIdxMap.putIfAbsent(twoSum, new ArrayList<>());
                twoSumIdxMap.get(twoSum).add(new Pair<>(i, j));
            }
        }
        Set<Pair<Integer, Pair<Integer, Integer>>> result = new HashSet<>();
        for (int twoSum : twoSumIdxMap.keySet()) {
            if (arrReverseMap.containsKey(0 - twoSum)) {

                List<Integer> oneSet = arrReverseMap.get(0 - twoSum);
                List<Pair<Integer, Integer>> twoSet = twoSumIdxMap.get(twoSum);

                for (Pair<Integer, Integer> tmpTwo : twoSet) {
                    for (int i : oneSet) {
                        int j = tmpTwo.getKey();
                        int k = tmpTwo.getValue();
                        if (i != j && i != k && j != k) {
                            int[] tmpArr = new int[]{nums[i], nums[j], nums[k]};
                            Arrays.sort(tmpArr);
                            result.add(new Pair<>(tmpArr[0], new Pair<>(tmpArr[1], tmpArr[2])));
                        }
                    }
                }
            }
        }
        List<List<Integer>> resultList = new ArrayList<>(result.size());
        for (Pair<Integer, Pair<Integer, Integer>> r : result) {
            resultList.add(new ArrayList<Integer>(3) {{
                add(r.getKey());
                add(r.getValue().getKey());
                add(r.getValue().getValue());
            }});
        }
        return resultList;
    }

    // LC1738
    public int kthLargestValue(int[][] matrix, int k) {
        // 分行亦或前缀
        int[][] matrixXorPrefix = new int[matrix.length][matrix[0].length + 1];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                matrixXorPrefix[i][j] = matrixXorPrefix[i][j - 1] ^ matrix[i][j - 1];
            }
        }
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k);
        for (int j = 0; j < matrix[0].length; j++) {
            int tmp = 0;
            for (int i = 0; i < matrix.length; i++) {
                tmp ^= matrixXorPrefix[i][j + 1];
                if (minHeap.size() < k) {
                    minHeap.offer(tmp);
                } else {
                    if (tmp > minHeap.peek()) {
                        minHeap.poll();
                        minHeap.offer(tmp);
                    }
                }
            }
        }
        return minHeap.peek();
    }

    // LC13
    public int romanToInt(String s) {
        Map<Character, Integer> m = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};
        int pre = m.get(s.charAt(0));
        int cur = 1;
        int sum = 0;
        while (cur < s.length()) {
            int tmp = m.get(s.charAt(cur++));
            if (pre < tmp) {
                sum -= pre;
            } else {
                sum += pre;
            }
            pre = tmp;
        }
        sum += pre;
        return sum;
    }

    // LC70
    public int climbStairs(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;
        // return climbStairs(n - 1) + climbStairs(n - 2);

        int preOne = 1, preTwo = 1, cur = 2; // 0阶算有一种方法
        for (int i = 3; i <= n; i++) {
            int tmp = cur + preTwo;
            preOne = preTwo;
            preTwo = cur;
            cur = tmp;
        }
        return cur;
    }

    // LC698
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % k != 0) return false;
        int subsetSum = sum / k;
        // 从空集的补集(全集)开始(11111) > 产生子集(10101) > 如果和为k > (选择)产生补集(01010) > 对补集使用backtrack函数 > 取消选择
        // 开一个数组存和值, 初始化为-1, 按需求和
        int[] sums = new int[1 << nums.length];
        Arrays.fill(sums, -1);

        return lc698Helper(nums, (1 << nums.length) - 1, new ArrayList<>(k), k, sums, subsetSum);
    }

    private boolean lc698Helper(int[] nums, int fullSetMask, List<Integer> selectedSet, int k, int[] sums, int subsetSum) {
        if (selectedSet.size() == k) {
            return true;
        }
        if (fullSetMask == 0) {
            return false;
        }
        // 产生二进制子集算法
        for (int subset = fullSetMask; subset != 0; subset = (subset - 1) & fullSetMask) {
            if (sums[subset] == -1) {
                sums[subset] = 0;
                for (int i = 0; i < nums.length; i++) {
                    if (((subset >> i) & 1) == 1) {
                        sums[subset] += nums[i];
                    }
                }
            }
            if (sums[subset] == subsetSum) {
                selectedSet.add(subset);
                // 求当前已经选的集合的并集
                int all = 0;
                for (int i : selectedSet) {
                    all |= i;
                }
                // 求当前并集的补集
                int sup = (1 << nums.length) - 1;
                sup ^= all;
                if (lc698Helper(nums, sup, selectedSet, k, sums, subsetSum)) {
                    return true;
                }
                selectedSet.remove(selectedSet.size() - 1);
            }
        }
        return false;
    }

    // LC698 TLE 注意剪枝技巧,剪枝后AC
    public boolean canPartitionKSubsetsTLE(int[] nums, int k) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % k != 0) return false;
        int subsetSum = sum / k;
        boolean[] option = new boolean[nums.length];
        return lc698helperTLE(nums, subsetSum, k, 0, option, 0);
    }

    private boolean lc698helperTLE(int[] nums, int subsetSum, int leftSetNum, int currentSum, boolean[] option, int start) {
        if (leftSetNum == 0) {
            return true;
        }
        if (currentSum == subsetSum) {
            return lc698helperTLE(nums, subsetSum, leftSetNum - 1, 0, option, 0); // 注意这里start改成了0, 从头开始
        }
        for (int i = start; i < nums.length; i++) { // 注意剪枝技巧, 用start作为开始下标, 不顾左边可能已经挑选的 (因为是证明存在性, 故若存在答案, 则必然存在一种从左到右依次划分的方法???)
            if (!option[i] && currentSum + nums[i] <= subsetSum) {
                option[i] = true;
                int tmpSum = currentSum + nums[i];
                if (lc698helperTLE(nums, subsetSum, leftSetNum, tmpSum, option, i + 1)) {
                    return true;
                }
                option[i] = false;
            }

        }
        return false;
    }

    // LC494 DP
    public int findTargetSumWays(int[] nums, int target) {
        // array[i] >=0
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (Math.abs(target) > sum) return 0;
        int t = sum * 2 + 1;
        int[][] dp = new int[nums.length + 1][t];
        // dp[i][j]: 加入前i个元素, 到达j的方案个数
        // dp[i][j] = dp[i-1][j-nums[i-1]] + dp[i-1][j+nums[i+1]], 注意使用sum做负下标修正偏移量
        dp[0][sum] = 1;

        for (int i = 1; i <= nums.length; i++) {
            for (int j = 0; j < t; j++) {
                int tmp = 0;
                if (j - nums[i - 1] >= 0) {
                    tmp += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] < t) {
                    tmp += dp[i - 1][j + nums[i - 1]];
                }
                dp[i][j] = tmp;
            }
        }

        return dp[nums.length][sum + target];
    }

    // LC416 DP 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        int halfSum = sum / 2;
        int[] dp = new int[halfSum + 1];
        // dp[i][j] 表示添加前i个元素 在背包大小限制为j的情况下能达到的最大值
        for (int i = 1; i <= nums.length; i++) {
            for (int j = halfSum; j >= 0; j--) {
//                dp[j] = dp[j];
                if (j - nums[i - 1] >= 0 && dp[j - nums[i - 1]] + nums[i - 1] <= halfSum) {
                    dp[j] = Math.max(dp[j], dp[j - nums[i - 1]] + nums[i - 1]);
                }
            }
            if (dp[halfSum] == halfSum) return true;
        }

        return false;
    }

    // LC494
    int lc494Result = 0;

    public int findTargetSumWaysDFS(int[] array, int target) {
        lc494Helper(array, target, 0, 0);
        return lc494Result;
    }

    private void lc494Helper(int[] array, int target, int currentIdx, int currentSum) {
        if (currentIdx == array.length) {
            if (currentSum == target) {
                lc494Result++;
            }
        } else {
            lc494Helper(array, target, currentIdx + 1, currentSum + array[currentIdx]);
            lc494Helper(array, target, currentIdx + 1, currentSum - array[currentIdx]);
        }
    }

    // LC503
    public int[] nextGreaterElements(int[] nums) {
        int[] snge = simpleNGE(nums);
        int[] doubleArray = new int[nums.length * 2];
        System.arraycopy(nums, 0, doubleArray, 0, nums.length);
        System.arraycopy(nums, 0, doubleArray, nums.length, nums.length);
        int[] dnge = simpleNGE(doubleArray);
        for (int i = 0; i < nums.length; i++) {
            if (snge[i] != -1) {
                continue;
            } else {
                snge[i] = dnge[i];
            }
        }
        return snge;
    }

    public int[] simpleNGE(int[] nums) {
        int n = nums.length;
        Deque<Integer> stack = new LinkedList<>();
        int[] result = new int[n];
        Arrays.fill(result, -1);
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                result[stack.pop()] = nums[i];
            }
            stack.push(i);
        }
        return result;
    }

    // LC993
    public boolean isCousins(TreeNode42 root, int x, int y) {
        if (root == null || root.left == null || root.right == null) return false;
        Deque<TreeNode42> q = new LinkedList<>();
        Map<TreeNode42, TreeNode42> father = new HashMap<>();
        Map<TreeNode42, Integer> layer = new HashMap<>();
        q.offer(root);
        int layerCtr = -1;
        boolean xFlag = false, yFlag = false;
        TreeNode42 xTN = null, yTN = null;
        while (!q.isEmpty()) {
            layerCtr++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                TreeNode42 tmp = q.poll();
                layer.put(tmp, layerCtr);
                if (tmp.left != null) {
                    father.put(tmp.left, tmp);
                    q.offer(tmp.left);
                }
                if (tmp.right != null) {
                    father.put(tmp.right, tmp);
                    q.offer(tmp.right);
                }
                if (tmp.val == x) {
                    xTN = tmp;
                    xFlag = true;
                }
                if (tmp.val == y) {
                    yTN = tmp;
                    yFlag = true;
                }
                if (xFlag && yFlag) break;
            }
        }
        return father.get(xTN) != father.get(yTN) && layer.get(xTN) == layer.get(yTN);
    }

    // LC451
    public String frequencySort(String s) {
        StringBuffer sb = new StringBuffer(s.length());

        Map<Character, Integer> freq = new HashMap<>();
        for (char c : s.toCharArray()) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }
        List<Pair<Character, Integer>> freqList = new ArrayList<>(freq.keySet().size());
        for (Map.Entry<Character, Integer> entry : freq.entrySet()) {
            freqList.add(new Pair<>(entry.getKey(), entry.getValue()));
        }
        Collections.sort(freqList, new Comparator<Pair<Character, Integer>>() {
            @Override
            public int compare(Pair<Character, Integer> o1, Pair<Character, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }
        });
        for (Pair<Character, Integer> pair : freqList) {
            for (int i = 0; i < pair.getValue(); i++) {
                sb.append(pair.getKey());
            }
        }

        return sb.toString();
    }

    // LC784
    public List<String> letterCasePermutation(String S) {
        List<String> result = new ArrayList<>();
        List<Integer> letterIdx = new ArrayList<>();
        for (int i = 0; i < S.length(); i++) {
            if (Character.isLetter(S.charAt(i))) {
                letterIdx.add(i);
            }
        }
        char[] lowerCase = new char[]{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
        char[] upperCase = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
        int maxMask = 1 << letterIdx.size();
        for (int mask = 0; mask < maxMask; mask++) {
            StringBuffer sb = new StringBuffer(S);
            for (int i = 0; i < letterIdx.size(); i++) {
                // 1 变 0 不变
                if (((mask >> i) & 1) == 1) {
                    int idx = letterIdx.get(i);
                    char c = S.charAt(idx);
                    sb.setCharAt(idx, Character.isLowerCase(c) ? upperCase[c - 'a'] : lowerCase[c - 'A']);
                }
            }
            result.add(sb.toString());
        }

        return result;
    }

    // LC480
    PriorityQueue<Long> maxHeap;
    PriorityQueue<Long> minHeap;

    public double[] medianSlidingWindow(int[] nums, int k) {
        minHeap = new PriorityQueue<>(Comparator.comparingLong(o -> o));
        maxHeap = new PriorityQueue<>(Comparator.comparingLong(o -> -o));

        int len = nums.length;
        double[] result = new double[len - k + 1];
        for (int i = 0; i < k; i++) {
            addNum(nums[i]);
        }
        for (int i = 0; i < len - k + 1; i++) {
            if (minHeap.size() == maxHeap.size()) {
                long sum = (minHeap.peek() + maxHeap.peek());
                result[i] = ((double) (sum)) / 2;
            } else {
                result[i] = (double) maxHeap.peek();
            }
            removeNum(nums[i]);
            if (i + k < len) {
                addNum(nums[i + k]);
            }
        }
        return result;

    }

    private void addNum(long i) {
        if (maxHeap.isEmpty() || i < maxHeap.peek()) {
            maxHeap.offer(i);
        } else {
            minHeap.offer(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    private void removeNum(long i) {
        if (maxHeap.contains(i)) {
            maxHeap.remove(i);
        } else {
            minHeap.remove(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

}

class TreeNode42 {
    int val;
    TreeNode42 left;
    TreeNode42 right;

    TreeNode42() {
    }

    TreeNode42(int val) {
        this.val = val;
    }

    TreeNode42(int val, TreeNode42 left, TreeNode42 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// LC155
class MinStack {
    Deque<Integer> stack;
    Deque<Integer> minStack;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        stack = new LinkedList<>();
        minStack = new LinkedList<>();
        minStack.push(Integer.MAX_VALUE);
    }

    public void push(int val) {
        stack.push(val);
        minStack.push(Math.min(minStack.peek(), val));
    }

    public void pop() {
        stack.pop();
        minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}