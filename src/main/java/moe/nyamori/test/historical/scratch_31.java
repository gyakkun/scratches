package moe.nyamori.test.historical;

import java.util.*;
import java.util.stream.Collectors;

class scratch_31 {
    public static void main(String[] args) {
        scratch_31 s = new scratch_31();
        Long timing = System.currentTimeMillis();
        System.err.println(s.find132pattern(new int[]{1, 0, 1, -4, -3}));
        timing = System.currentTimeMillis() - timing;
        System.err.print("TIMING : " + timing + "ms");
    }

    // LC456 132模式, 单调栈, 题解
    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        Deque<Integer> possibleTop = new LinkedList<Integer>();
        possibleTop.push(nums[n - 1]);
        int maxK = Integer.MIN_VALUE;

        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] < maxK) {
                return true;
            }
            while (!possibleTop.isEmpty() && nums[i] > possibleTop.peek()) {
                maxK = possibleTop.pop();
            }
            // 如果等于, 则没必要push, 当然也可以不做这步判断
            if (nums[i] > maxK) {
                possibleTop.push(nums[i]);
            }
        }
        return false;
    }

    // LC71 简化Unix路径
    public String simplifyPath(String path) {
        String[] list = path.split("/");
        Set<String> set = new HashSet<>();
        set.add(".");
        set.add("..");
        set.add("");
        Deque<String> stack = new LinkedList<>();
        for (String s : list) {
            if (!set.contains(s)) {
                stack.push(s);
            } else if (s.equals("..")) {
                stack.pop();
            }
        }
        StringBuffer sb = new StringBuffer();
        sb.append('/');
        while (!stack.isEmpty()) {
            sb.append(stack.pollLast());
            sb.append('/');
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.equals("") ? "/" : sb.toString();
    }

    // LC72 编辑距离
    public int editDistance(String s1, String s2) {
        int s1l = s1.length(), s2l = s2.length();
        int[][] dp = new int[s1l + 1][s2l + 1];
        // 初始化, 从0个字符变为i个字符, 至少需要i步插入操作
        for (int i = 0; i <= s1l; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= s2l; i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= s1l; i++) {
            for (int j = 1; j <= s2l; j++) {
                int min = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1; // 从dp[i-1][j]删除 或者从dp[i][j-1]替换???
                if (s1.charAt(i - 1) != s2.charAt(j - 1)) {
                    dp[i][j] = Math.min(min, dp[i - 1][j - 1] + 1); // 从dp[i-1][j-1]插入一个
                } else {
                    dp[i][j] = Math.min(min, dp[i - 1][j - 1]); // 否则不用操作
                }
            }
        }
        return dp[s1l][s2l];
    }

    // LC97 交错字符串 递归 记忆数组
    public boolean isInterleave(String s1, String s2, String s3) {
        int s1l = s1.length(), s2l = s2.length(), s3l = s3.length();
        if (s1l + s2l != s3l) {
            return false;
        }
        Boolean[][] memo = new Boolean[s1l + 1][s2l + 1];
        return isInterleaveRecursive(s1, s2, s3, 0, 0, 0, memo);
    }

    private boolean isInterleaveRecursive(String a, String b, String c, int i, int j, int k, Boolean[][] memo) {
        if (k == c.length()) {
            memo[i][j] = true;
            return memo[i][j];
        }
        if (memo[i][j] != null) {
            return memo[i][j];
        }
        boolean result = false;
        if (i < a.length() && a.charAt(i) == c.charAt(k)) {
            result = isInterleaveRecursive(a, b, c, i + 1, j, k + 1, memo);
        }
        if (j < b.length() && b.charAt(j) == c.charAt(k)) {
            // 注意短路前面为真的结果, 不用进入这次匹配
            result |= isInterleaveRecursive(a, b, c, i, j + 1, k + 1, memo);
        }
        memo[i][j] = result;
        return memo[i][j];
    }


    // LC97 交错字符串 DP 滚动数组
    public boolean isInterleaveDP(String s1, String s2, String s3) {
        int s1l = s1.length(), s2l = s2.length(), s3l = s3.length();
        if (s1l + s2l != s3l) {
            return false;
        }
        boolean[] dp = new boolean[s2l + 1];
        dp[0] = true;
        // dp[i][j] 表示 s1的前i个字符和s2的前j个字符能否组成s3的前i+j个字符
        for (int i = 0; i <= s1l; i++) {
            for (int j = 0; j <= s2l; j++) {
                int s3p = i + j - 1;
                if (i > 0) {
                    // dp[i][j] = dp[i-1][j] && s3.charAt(s3p) == s1.charAt(i - 1);
                    dp[j] = dp[j] && s3.charAt(s3p) == s1.charAt(i - 1);
                }
                if (j > 0) {
                    // dp[i][j] |= dp[j-1][i] && && s3.charAt(s3p) == s2.charAt(j - 1);
                    dp[j] = dp[j] || dp[j - 1] && s3.charAt(s3p) == s2.charAt(j - 1);
                }
            }
        }

        return dp[s2l];
    }

    // LC792, 子序列匹配, 桶
    public int numMatchingSubseqBucket(String s, String[] words) {
        int result = 0;
        Map<Character, List<List<Character>>> bucket = new HashMap<>();
        for (String word : words) {
            bucket.putIfAbsent(word.charAt(0), new LinkedList<>());
            List<Character> tmpBucketItem = new LinkedList<>();
            for (char c : word.toCharArray()) {
                tmpBucketItem.add(c);
            }
            bucket.get(word.charAt(0)).add(tmpBucketItem);
        }
        for (char c : s.toCharArray()) {
            Set<Character> set = new HashSet<>(bucket.keySet());
            for (char key : set) {
                if (c == key) {
                    List<List<Character>> thisBucket = bucket.get(key);
                    ListIterator<List<Character>> it = thisBucket.listIterator();
                    while (it.hasNext()) {
                        List<Character> bucketItem = it.next();
                        it.remove();
                        bucketItem.remove(0);
                        if (bucketItem.size() == 0) {
                            result++;
                        } else {
                            bucket.putIfAbsent(bucketItem.get(0), new LinkedList<>());
                            if (bucketItem.get(0) != key) {
                                bucket.get(bucketItem.get(0)).add(bucketItem);
                            } else {
                                it.add(bucketItem);
                            }
                        }
                    }
                }
                if (bucket.get(key).size() == 0) {
                    bucket.remove(key);
                }
            }
        }
        return result;
    }

    // LC792, 子序列匹配, 暴力TLE
    public int numMatchingSubseqTLE(String s, String[] words) {
        int result = 0;
        for (String word : words) {
            int wPtr = 0;
            int sPtr = 0;
            while (wPtr < word.length() && sPtr < s.length()) {
                if (s.charAt(sPtr++) == word.charAt(wPtr)) {
                    wPtr++;
                    if (wPtr == word.length()) {
                        result++;
                        break;
                    }
                }
            }
        }
        return result;
    }

    // LC42 接雨水, 比较朴素的解法
    // 思路: 先得到从头/尾开始的递增/递减序列, 再处理中间最高柱子之间的部分
    public int trap(int[] height) {
        if (height.length == 0) return 0;

        int result = 0;

        Map<Integer, Integer> firstIndexOf = new HashMap<>();

        // firstIndexOf表
        for (int i = 0; i < height.length; i++) {
            firstIndexOf.putIfAbsent(height[i], i);
        }

        int n = height.length;
        List<Integer> lisFirst = simpleIncreasingSubsequence(height);
        int topEle = lisFirst.get(lisFirst.size() - 1);
//        List<Integer> lisFirst = lisFirstDup.stream().distinct().collect(Collectors.toList());

        int firstIdxOfTopEle = firstIndexOf.get(topEle);
        int lastIdxOfTopEle = -1;
        for (int i = height.length - 1; i >= 0; i--) {
            if (height[i] == topEle) {
                lastIdxOfTopEle = i;
                break;
            }
        }

        // 优先处理两个top之间的值
        if (firstIdxOfTopEle != lastIdxOfTopEle && lastIdxOfTopEle - firstIdxOfTopEle > 1) {
            for (int i = firstIdxOfTopEle + 1; i < lastIdxOfTopEle; i++) {
                result += topEle - height[i];
            }
        }

        if (lisFirst.size() > 1) {
            // 处理上升序列
            for (int i = 1; i < lisFirst.size(); i++) {
                int left = lisFirst.get(i - 1);
                int right = lisFirst.get(i);
                int firstIdxOfLeft = firstIndexOf.get(left);
                int firstIdxOfRight = firstIndexOf.get(right);
                if (firstIdxOfRight - firstIdxOfLeft > 1) {
                    for (int j = firstIdxOfLeft + 1; j < firstIdxOfRight; j++) {
                        result += left - height[j];
                    }
                }
            }
        }

        int[] secondHalf = new int[height.length - lastIdxOfTopEle];

        // 处理下降序列
        for (int i = height.length - 1; i >= lastIdxOfTopEle; i--) {
            secondHalf[height.length - 1 - i] = height[i];
        }

        firstIndexOf.clear();
        for (int i = 0; i < secondHalf.length; i++) {
            firstIndexOf.putIfAbsent(secondHalf[i], i);
        }

        List<Integer> lisSecond = simpleIncreasingSubsequence(secondHalf);

        if (lisSecond.size() > 1) {
            // 处理上升序列
            for (int i = 1; i < lisSecond.size(); i++) {
                int left = lisSecond.get(i - 1);
                int right = lisSecond.get(i);
                int firstIdxOfLeft = firstIndexOf.get(left);
                int firstIdxOfRight = firstIndexOf.get(right);
                if (firstIdxOfRight - firstIdxOfLeft > 1) {
                    for (int j = firstIdxOfLeft + 1; j < firstIdxOfRight; j++) {
                        result += left - secondHalf[j];
                    }
                }
            }
        }

        return result;
    }

    // Simple Increasing Subsequence from start
    private List<Integer> simpleIncreasingSubsequence(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if (nums.length == 0) return result;
        result.add(nums[0]);
        if (nums.length == 1) return result;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > result.get(result.size() - 1)) {
                result.add(nums[i]);
            }
        }
        return result;
    }

    // LC300 Greedy + Binary Search
    public List<Integer> LISGreedyBS(int[] nums) {
        int n = nums.length;
        List<Integer> tail = new ArrayList<>(n);
        tail.add(nums[0]);
        for (int i = 1; i < n; i++) {
            if (nums[i] >= tail.get(tail.size() - 1)) {
                tail.add(nums[i]);
            } else {
                int idx = binarySearchInList(tail, nums[i]);
//                if (idx != 0) {
                tail.set(idx, nums[i]);
//                }
            }
        }
        return tail;
    }

    private int binarySearchInListModForLC42(List<Integer> list, int target) {
        // 找出大于等于target的最小值的坐标
        int n = list.size();
        int l = 0, h = n - 1;
        while (l < h) {
            int mid = l + (h - l) / 2; // 取低位
            if (list.get(mid) < target) {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        if (list.get(h) >= target) {
            return h;
        } else {
            return -1;
        }
    }

    // LC1092
    public String shortestCommonSupersequence(String str1, String str2) {
        //   a c d b a c
        // c a e   b
        // 思路: 先找最长公共子序列, 然后逐个比较, 右移指针, 逐个填充
        String lcs = longestCommonSubsequenceString(str1, str2);
        StringBuffer answer = new StringBuffer();
        int i = 0, j = 0;
        for (char c : lcs.toCharArray()) {
            while (i < str1.length() && str1.charAt(i) != c) {
                answer.append(str1.charAt(i));
                i++;
            }
            while (j < str2.length() && str2.charAt(j) != c) {
                answer.append(str2.charAt(j));
                j++;
            }
            answer.append(c);
            i++;
            j++;
        }
        if (i < str1.length())
            answer.append(str1.substring(i));
        if (j < str2.length())
            answer.append(str2.substring(j));
        return answer.toString();
    }

    // 返回最终结果LCS
    public String longestCommonSubsequenceString(String text1, String text2) {
        String[][] dp = new String[text1.length() + 1][text2.length() + 1];
        for (String[] sa : dp) {
            Arrays.fill(sa, "");
        }
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + text1.charAt(i - 1);
                } else {
                    dp[i][j] = dp[i - 1][j].length() > dp[i][j - 1].length() ? dp[i - 1][j] : dp[i][j - 1];
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }


    // LC491

    List<Integer> temp;
    List<List<Integer>> findSubsequencesResult;

    public List<List<Integer>> findSubsequences(int[] nums) {
        findSubsequencesResult = new ArrayList<>();
        temp = new ArrayList<>();
        findSubsequencesRecursive(0, Integer.MIN_VALUE, nums);
        return findSubsequencesResult;
    }

    // 选择列表: 当前元素选或不选
    private void findSubsequencesRecursive(int cur, int last, int[] nums) {
        if (cur == nums.length) {
            if (temp.size() >= 2) {
                findSubsequencesResult.add(new ArrayList<>(temp));
            }
            return;
        }

        // 1. 做选择, 当前坐标元素大于等于上一个加入的元素, 则选择当前元素坐标的元素加入候选答案, idx指向下一个元素进行迭代
        if (nums[cur] >= last) {
            temp.add(nums[cur]);
            findSubsequencesRecursive(cur + 1, nums[cur], nums);
            // 2. 取消选择, 考虑不选择当前元素
            temp.remove(temp.size() - 1);
        }

        // 3. 什么情况不选择当前元素, 又不进入下一轮迭代?
        // 如果当前元素等于上一个加入的元素, 则不选当前元素入temp序列, 也不进入下一次迭代

        // 因为面对两个相同的元素(last==nums[cur]), 有4种情况:
        // （前者为last, 后者为nums[cur])
        // 1) 选择两者 (即1. 做选择)
        // 2) 选择前者, 不选择后者 (即保持last不变, 绕过cur, 进入下一次迭代(cur+1))
        // 3) 不选择前者, 选择后者 (即更新last, 选择cur, 进入下一次迭代, 但语义上和本轮迭代是完全一致的, 所以无法进行???)
        // 4) 不选择两者 (即既不选择当前元素, 也不进入下一次迭代, 直接return的情况)

        if (nums[cur] == last) {
            return;
        } else {
            findSubsequencesRecursive(cur + 1, last, nums);
        }
        // 1 2 2 2 3 4 5 6

    }

    // LC300 Greedy + Binary Search
    public int lengthOfLISGreedyBinarySearch(int[] nums) {
        int n = nums.length;
        List<Integer> tail = new ArrayList<>(n);
        tail.add(nums[0]);
        for (int i = 1; i < n; i++) {
            if (nums[i] > tail.get(tail.size() - 1)) {
                tail.add(nums[i]);
            } else {
                tail.set(binarySearchInList(tail, nums[i]), nums[i]);
            }
        }
        return tail.size();
    }

    private int binarySearchInList(List<Integer> list, int target) {
        // 找出大于等于target的最小值的坐标
        int n = list.size();
        int l = 0, h = n - 1;
        while (l < h) {
            int mid = l + (h - l) / 2; // 取低位
            if (list.get(mid) < target) {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        if (list.get(h) >= target) {
            return h;
        } else {
            return -1;
        }
    }

    // LC300 DP
    public int lengthOfLISDP(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        return Arrays.stream(dp).max().getAsInt();
    }

    // LC300 最长上升子序列, 递归+记忆数组
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        Integer[][] memo = new Integer[n + 1][n + 1];
        return lengthOfLISRecursive(-1, 0, nums, memo);
    }

    // 比较罕见的前向递归
    private int lengthOfLISRecursive(int pre, int cur, int[] nums, Integer[][] memo) {
        if (cur == nums.length) return 0;
        // 注意向右平移pre一位
        if (memo[pre + 1][cur] != null) return memo[pre + 1][cur];
        int stepOne = 0;
        // 初始状态, pre为-1
        if (pre < 0 || nums[pre] < nums[cur]) {
            stepOne = lengthOfLISRecursive(cur, cur + 1, nums, memo) + 1;
        }
        int stepTwo = lengthOfLISRecursive(pre, cur + 1, nums, memo);

        memo[pre + 1][cur] = Math.max(stepOne, stepTwo);
        return memo[pre + 1][cur];
    }

    // LC72 Edit Distance 编辑距离, 无法参考下面方法
    // 因为根据编辑距离的定义, 替换(==删除+插入各一次)也是一次操作
    // 以下方法求出的是最少的插入、删除次数
    public int minDelAndInsert(String a, String b) {
        String lcs = longestCommonSubsequenceString(a, b);
        int minDel = a.length() - lcs.length();
        int minInsert = b.length() - lcs.length();
        return minDel + minInsert;
    }

    // LC1143 最长相同子序列长度
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }

    // Longest Common Substring 最长相同子串
    public String longestCommonSubstring(String a, String b) {

        // dp[i][j] 表示a前i个字符与b前j个字符中最长相同子串的长度
        // 转移方程:
        //  1) 如果a[i] == b[j], dp[i][j] = dp[i-1][j-1] + 1
        //  2) 否则, dp[i][j] = 0
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        int maxLength = 0;
        int start = -1;

        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > maxLength) {
                        maxLength = dp[i][j];
                        start = i - maxLength; // 留意怎么取坐标
                    }
                } else {
                    dp[i][j] = 0;
                }
            }
        }

        return a.substring(start, start + maxLength);
    }

    // LC96: 推导卡特兰数
    public long numTrees(int n) {
        Long[] memo = new Long[n + 1];
        memo[0] = 1l;
        memo[1] = 1l;

        long result = numTreesRecursive(n, memo);
        return result;
    }

    private long numTreesRecursive(int n, Long[] memo) {
        if (memo[n] != null) {
            return memo[n];
        }
        long res = 0;
        for (int i = 1; i <= n; i++) {
            res += numTreesRecursive(i - 1, memo) * numTreesRecursive(n - i, memo);
        }
        memo[n] = res;
        return res;
    }

    // Minimum Deletions in a String to make it a Palindrome，怎么删掉最少字符构成回文
    // https://www.geeksforgeeks.org/minimum-number-deletions-make-string-palindrome/
    public int minDelToMakePalindrome(String s) {
        int n = s.length();
        // 搜索最长回文子序列的长度, 然后用总长度减去即可

        Integer[][] memo = new Integer[n + 1][n + 1];
        int longestPalindromeSubArrayLength = longestPalindromeSubArray(s, memo, 0, n - 1);

        return n - longestPalindromeSubArrayLength;
    }

    private int longestPalindromeSubArray(String s, Integer[][] memo, int l, int h) {
        if (l == h) {
            memo[l][h] = 1;
            return 1;
        }
        if (l > h) {
            return 0;
        }
        if (memo[l][h] != null) {
            return memo[l][h];
        }
        int res = 0;
        if (s.charAt(l) == s.charAt(h)) {
            res = longestPalindromeSubArray(s, memo, l + 1, h - 1) + 2;
        } else {
            res = Math.max(res, longestPalindromeSubArray(s, memo, l + 1, h));
            res = Math.max(res, longestPalindromeSubArray(s, memo, l, h - 1));
        }
        memo[l][h] = res;
        return res;
    }
}