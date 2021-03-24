import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        Long timing = System.currentTimeMillis();
        System.err.println(s.longestCommonSubsequence("horse", "ros"));
        timing = System.currentTimeMillis() - timing;
        System.err.print("TIMING : " + timing + "ms");

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
        String lcs = longestCommonSubsequence(a, b);
        int minDel = a.length() - lcs.length();
        int minInsert = b.length() - lcs.length();
        return minDel + minInsert;
    }

    // LC1143 最长相同子序列长度
    public String longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        List<Character> result = new LinkedList<>();
        int maxLen = 0;
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > maxLen) {
                        maxLen = dp[i][j];
                        result.add(text1.charAt(i - 1));
                    }
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        StringBuffer sb = new StringBuffer();
        for (char c : result) {
            sb.append(c);
        }
        return sb.toString();
//        return result.stream().map(String::valueOf).collect(Collectors.joining());
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