import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        int[] arr = new int[]{12, 12, 4, 56, 1, -100, 130};
//        char[] carr = new char[]{'F', 'J', 'J', 'A', 'J', 'F', 'C', 'H', 'J', 'B', 'E', 'G', 'G', 'F', 'A', 'C', 'I', 'F', 'J', 'C', 'J', 'C', 'H', 'C', 'A', 'D', 'G', 'H', 'B', 'F', 'G', 'C', 'C', 'A', 'E', 'B', 'H', 'J', 'E', 'I', 'F', 'D', 'E', 'A', 'C', 'D', 'B', 'D', 'J', 'J', 'C', 'F', 'D', 'D', 'J', 'H', 'A', 'E', 'C', 'D', 'J', 'D', 'G', 'G', 'B', 'C', 'E', 'G', 'H', 'I', 'D', 'H', 'F', 'E', 'I', 'B', 'D', 'E', 'I', 'E', 'C', 'J', 'G', 'I', 'D', 'E', 'D', 'J', 'C', 'A', 'C', 'C', 'D', 'I', 'J', 'B', 'D', 'H', 'H', 'J', 'G', 'B', 'G', 'A', 'H', 'E', 'H', 'E', 'D', 'E', 'J', 'E', 'J', 'C', 'F', 'C', 'J', 'G', 'B', 'C', 'I', 'I', 'H', 'F', 'A', 'D', 'G', 'F', 'C', 'C', 'F', 'G', 'C', 'J', 'B', 'B', 'I', 'C', 'J', 'J', 'E', 'G', 'H', 'C', 'I', 'G', 'J', 'I', 'G', 'G', 'J', 'G', 'G', 'E', 'G', 'B', 'I', 'J', 'B', 'H', 'D', 'H', 'G', 'F', 'C', 'H', 'C', 'D', 'A', 'G', 'B', 'H', 'H', 'B', 'J', 'C', 'A', 'F', 'J', 'G', 'F', 'E', 'B', 'F', 'E', 'B', 'B', 'A', 'E', 'F', 'E', 'H', 'I', 'I', 'C', 'G', 'J', 'D', 'H', 'E', 'F', 'G', 'G', 'D', 'E', 'B', 'F', 'J', 'J', 'J', 'D', 'H', 'E', 'B', 'D', 'J', 'I', 'F', 'C', 'I', 'E', 'H', 'F', 'E', 'G', 'D', 'E', 'C', 'F', 'E', 'D', 'E', 'A', 'I', 'E', 'A', 'D', 'H', 'G', 'C', 'I', 'E', 'G', 'A', 'H', 'I', 'G', 'G', 'A', 'G', 'F', 'H', 'J', 'D', 'F', 'A', 'G', 'H', 'B', 'J', 'A', 'H', 'B', 'H', 'C', 'G', 'F', 'A', 'C', 'C', 'B', 'I', 'G', 'G', 'B', 'C', 'J', 'J', 'I', 'E', 'G', 'D', 'I', 'J', 'I', 'C', 'G', 'A', 'J', 'G', 'F', 'J', 'F', 'C', 'F', 'G', 'J', 'I', 'E', 'B', 'G', 'F', 'A', 'D', 'A', 'I', 'A', 'E', 'H', 'F', 'D', 'D', 'C', 'B', 'J', 'I', 'J', 'H', 'I', 'C', 'D', 'A', 'G', 'F', 'I', 'B', 'E', 'D', 'C', 'J', 'G', 'I', 'H', 'E', 'C', 'E', 'I', 'I', 'B', 'B', 'H', 'J', 'C', 'F', 'I', 'D', 'B', 'F', 'H', 'F', 'A', 'C', 'A', 'A', 'B', 'D', 'C', 'A', 'G', 'B', 'G', 'F', 'E', 'G', 'A', 'A', 'A', 'C', 'J', 'H', 'H', 'G', 'C', 'C', 'B', 'C', 'E', 'B', 'E', 'F', 'I', 'E', 'E', 'D', 'I', 'H', 'G', 'F', 'A', 'H', 'B', 'J', 'B', 'G', 'H', 'C', 'C', 'B', 'G', 'C', 'B', 'A', 'E', 'G', 'A', 'J', 'G', 'D', 'C', 'I', 'G', 'F', 'G', 'G', 'A', 'J', 'E', 'I', 'D', 'E', 'A', 'F', 'A', 'H', 'C', 'E', 'D', 'D', 'D', 'H', 'I', 'F', 'F', 'A', 'F', 'A', 'A', 'C', 'J', 'D', 'J', 'H', 'I', 'F', 'A', 'C', 'B', 'C', 'A', 'C', 'C', 'H', 'A', 'J', 'I', 'B', 'A', 'I', 'F', 'J', 'C', 'I', 'B', 'C', 'E', 'E', 'E', 'J', 'G', 'F', 'E', 'I', 'A', 'A', 'E', 'B', 'J', 'H', 'H', 'H', 'A', 'H', 'J', 'E', 'F', 'E', 'F', 'G', 'J', 'D', 'I', 'D', 'I', 'F', 'B', 'J', 'D', 'A', 'A', 'D', 'F', 'G', 'B', 'J', 'H', 'F', 'A', 'D', 'H', 'C', 'B', 'A', 'J', 'H', 'I', 'F', 'H', 'E', 'G', 'B', 'A', 'F', 'F', 'A', 'C', 'D', 'G', 'I', 'I', 'J', 'H', 'H', 'C', 'J', 'G', 'B', 'A', 'D', 'B', 'F', 'J', 'D', 'I', 'A', 'F', 'F', 'F', 'F', 'A', 'E', 'B', 'C', 'G', 'H', 'E', 'B', 'B', 'A', 'G', 'D', 'C', 'C', 'E', 'A', 'C', 'F', 'G', 'A', 'I', 'F', 'B', 'H', 'J', 'G', 'C', 'B', 'H', 'D', 'A', 'H', 'B', 'H', 'H', 'C', 'A', 'F', 'I', 'C', 'F', 'A', 'C', 'J', 'I', 'H', 'H', 'F', 'B', 'B', 'D', 'E', 'C', 'J', 'F', 'C', 'E', 'A', 'J', 'E', 'C', 'A', 'E', 'B', 'A', 'J', 'F', 'J', 'J', 'J', 'H', 'H', 'C', 'I', 'E', 'G', 'G', 'H', 'J', 'J', 'H', 'H', 'H', 'J', 'H', 'A', 'G', 'I', 'C', 'E', 'C', 'D', 'G', 'G', 'F', 'H', 'D', 'G', 'H', 'A', 'E', 'I', 'D', 'A', 'H', 'G', 'E', 'A', 'B', 'F', 'I', 'C', 'A', 'F', 'B', 'A', 'I', 'F', 'G', 'I', 'F', 'D', 'A', 'B', 'J', 'B', 'D', 'F', 'G', 'J', 'J', 'A', 'A', 'C', 'H', 'G', 'F', 'B', 'I', 'I', 'J', 'A', 'H', 'D', 'F', 'E', 'F', 'J', 'B', 'F', 'C', 'G', 'E', 'A', 'G', 'H', 'E', 'H', 'H', 'F', 'I', 'G', 'C', 'C', 'G', 'J', 'B', 'H', 'F', 'H', 'D', 'I', 'B', 'D', 'I', 'F', 'H', 'I', 'D', 'F', 'G', 'G', 'E', 'A', 'C', 'A', 'G', 'H', 'G', 'H', 'J', 'F', 'D', 'F', 'G', 'D', 'D', 'C', 'J', 'C', 'J', 'G', 'G', 'G', 'G', 'H', 'H', 'G', 'D', 'E', 'H', 'G', 'C', 'B', 'F', 'I', 'F', 'C', 'H', 'J', 'I', 'A', 'F', 'D', 'C', 'F', 'C', 'E', 'E', 'D', 'D', 'C', 'G', 'B', 'F', 'E', 'J', 'C', 'I', 'E', 'D', 'B', 'B', 'I', 'I', 'I', 'H', 'C', 'E', 'C', 'J', 'F', 'G', 'A', 'I', 'J', 'D', 'I', 'C', 'G', 'F', 'I', 'E', 'I', 'E', 'F', 'A', 'G', 'E', 'J', 'A', 'I', 'A', 'D', 'A', 'G', 'J', 'F', 'E', 'D', 'I', 'A', 'E', 'J', 'I', 'C', 'J', 'B', 'F', 'B', 'E', 'C', 'E', 'F', 'G', 'E', 'J', 'J', 'I', 'E', 'D', 'F', 'C', 'H', 'H', 'B', 'G', 'D', 'I', 'I', 'F', 'B', 'G', 'C', 'F', 'J', 'B', 'G', 'J', 'H', 'D', 'G', 'C', 'C', 'I', 'I', 'E', 'I', 'B', 'H', 'B', 'I', 'G', 'F', 'H', 'G', 'C', 'J', 'D', 'C', 'E', 'G', 'F', 'C', 'H', 'D', 'A', 'C', 'D', 'H', 'B', 'C', 'H', 'I', 'B', 'A', 'J', 'C', 'B', 'D', 'J', 'D', 'H', 'F', 'B', 'A', 'G', 'G', 'J', 'I', 'E', 'F', 'A', 'D', 'H', 'D', 'B', 'C', 'A', 'H', 'F', 'G', 'B', 'F', 'H', 'B', 'H', 'I', 'J', 'D', 'H', 'I', 'B', 'C', 'D', 'G', 'A', 'E', 'A', 'A', 'I', 'F', 'I', 'F', 'B', 'B', 'I', 'F', 'A', 'E', 'I', 'A', 'B', 'G', 'C', 'F', 'I', 'A', 'F', 'I', 'D', 'H', 'B', 'I', 'I', 'B', 'J', 'F', 'E', 'B', 'B', 'B', 'D', 'C', 'J', 'E', 'J', 'J', 'G', 'D', 'F', 'F', 'F', 'G', 'I', 'H', 'J', 'J', 'G', 'D', 'G', 'F'};
        System.err.println(s.countSubstrings("aaa"));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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

    // LC1707 TBD
    public int[] maximizeXor(int[] nums, int[][] queries) {
        int[] result = new int[queries.length];
        return result;
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
        for (int k = i; k < j; k++) {
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


    // LC621 Simulation WA
    public int leastIntervalSimulation(char[] tasks, int n) {
        Map<Character, Integer> freq = new HashMap<>();
        for (char c : tasks) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }
        Map<Character, Integer> nextRunTime = new HashMap<>();
        for (char c : freq.keySet()) {
            nextRunTime.put(c, 1);
        }
        PriorityQueue<Character> pq = new PriorityQueue<>(new Comparator<Character>() {
            @Override
            public int compare(Character o1, Character o2) {
                if (nextRunTime.get(o1) == nextRunTime.get(o2)) {
                    return freq.get(o2) - freq.get(o1);
                } else {
                    return nextRunTime.get(o1) - nextRunTime.get(o2);
                }
            }
        });
        for (char c : freq.keySet()) {
            pq.offer(c);
        }
        int completedTaskCtr = 0;
        int time = 0;
        while (!pq.isEmpty()) {
            char thisTask = pq.poll();
            if (time < nextRunTime.get(thisTask)) {
                time = nextRunTime.get(thisTask);
            } else {
                time++;
            }
            nextRunTime.put(thisTask, time + n + 1);
            freq.put(thisTask, freq.get(thisTask) - 1);

            for (char c : freq.keySet()) {
                if (nextRunTime.get(c) < (time + 1)) {
                    nextRunTime.put(c, time + 1);
                }
            }

            Deque<Character> stack = new LinkedList<>();
            while (!pq.isEmpty()) {
                stack.push(pq.poll());
            }
            while (!stack.isEmpty()) {
                pq.offer(stack.pop());
            }

            if (freq.get(thisTask) != 0) {
                pq.offer(thisTask);
            }
            completedTaskCtr++;
            System.currentTimeMillis();
        }
        return time;
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
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        if (root1 == null && root2 != null) {
            return root2;
        }
        if (root1 != null && root2 == null) {
            return root1;
        }
        int result = root1.val + root2.val;
        root1.val = result;
        TreeNode r1L = mergeTrees(root1.left, root2.left);
        TreeNode r1R = mergeTrees(root1.right, root2.right);
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

    public int diameterOfBinaryTree(TreeNode root) {
        lc543ans = 1;
        lc543Recursive(root);
        return lc543ans - 1;
    }

    // 记录遍历左右子树经过节点数的最大值, 节点数-1 = 路径长度
    private int lc543Recursive(TreeNode root) {
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

    public TreeNode convertBST(TreeNode root) {
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
    public void flatten(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) return;
        Deque<TreeNode> stack = new LinkedList<>();
        List<TreeNode> inorder = new ArrayList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode tmp = stack.pop();
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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode anc = root;
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
    public boolean isCousins(TreeNode root, int x, int y) {
        if (root == null || root.left == null || root.right == null) return false;
        Deque<TreeNode> q = new LinkedList<>();
        Map<TreeNode, TreeNode> father = new HashMap<>();
        Map<TreeNode, Integer> layer = new HashMap<>();
        q.offer(root);
        int layerCtr = -1;
        boolean xFlag = false, yFlag = false;
        TreeNode xTN = null, yTN = null;
        while (!q.isEmpty()) {
            layerCtr++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                TreeNode tmp = q.poll();
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