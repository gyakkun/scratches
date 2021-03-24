class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        Long timing = System.currentTimeMillis();
        System.err.println(s.longestCommonSubsequence("abcde", "ace"));
        timing = System.currentTimeMillis() - timing;
        System.err.print("TIMING : " + timing + "ms");

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