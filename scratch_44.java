import java.util.Arrays;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.err.println(s.reversePairs(new int[]{2, 1, 4, 1, 3}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC961
    public int repeatedNTimes(int[] nums) {
        int len = nums.length;
        int n = len / 2;
        int[] freq = new int[10000];
        for (int i : nums) {
            freq[i]++;
            if (freq[i] == n) return i;
        }
        return -1;
    }

    // LC494
    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int sum = Arrays.stream(nums).sum();
        if (sum < target) return 0;
        int[][] dp = new int[n + 1][2 * sum + 1];
        // dp[i][j] 表示加入前i个数到达和j的方案数
        // 中点(0) 在 dp[sum]
        dp[0][sum] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= 2 * sum; j++) {
                int result = 0;
                if (j - nums[i - 1] >= 0) {
                    result += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] <= 2 * sum) {
                    result += dp[i - 1][j + nums[i - 1]];
                }
                dp[i][j] = result;
            }
        }
        return dp[n][sum + target];
    }

    // LC474
    public int findMaxForm(String[] strs, int m, int n) {
        int[] zeroCtr = new int[strs.length];
        int[] oneCtr = new int[strs.length];
        for (int i = 0; i < strs.length; i++) {
            for (char c : strs[i].toCharArray()) {
                if (c == '0') zeroCtr[i]++;
            }
            oneCtr[i] = strs[i].length() - zeroCtr[i];
        }
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= strs.length; i++) {
            int zeroNum = zeroCtr[i - 1];
            int oneNum = oneCtr[i - 1];
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    int result = dp[j][k];
                    if (j - zeroNum >= 0 && k - oneNum >= 0) {
                        result = Math.max(result, dp[j - zeroNum][k - oneNum] + 1);// 选择这一个字符串
                    }
                    dp[j][k] = result;
                }
            }
        }
        return dp[m][n];
    }

    // JZOF51 HARD
    public int reversePairs(int[] nums) {
        int n = nums.length;
        int[] sorted = new int[n];
        System.arraycopy(nums, 0, sorted, 0, n);
        Arrays.sort(sorted);
        for (int i = 0; i < n; i++) {
            nums[i] = Arrays.binarySearch(sorted, nums[i]) + 1;
        }
        int result = 0;
        BIT bit = new BIT(n);
        for (int i = n - 1; i >= 0; i--) {
            result += bit.query(nums[i] - 1);
            bit.update(nums[i], 1);
        }
        return result;

    }
}


class BIT {
    int len;
    int[] bit;

    public BIT(int n) {
        this.len = n;
        this.bit = new int[n + 1];
    }

    public int query(int idxFromOne) {
        int sum = 0;
        while (idxFromOne > 0) {
            sum += bit[idxFromOne];
            idxFromOne -= lowbit(idxFromOne);
        }
        return sum;
    }

    public void update(int idxFromOne, int delta) {
        while (idxFromOne <= len) {
            bit[idxFromOne] += delta;
            idxFromOne += lowbit(idxFromOne);
        }
    }


    private int lowbit(int x) {
        return x & (x ^ (x - 1));
    }
}