class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        Long timing = System.currentTimeMillis();
        System.err.println(s.numTrees(1000));
        timing = System.currentTimeMillis() - timing;
        System.err.print("TIMING : " + timing + "ms");

    }

    // LC96: 推导卡特兰数
    public long numTrees(int n) {
        Long[] memo = new Long[n + 1];
        memo[0] = 1l;
        memo[1] = 1l;

        return numTreesRecursive(n, memo);
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