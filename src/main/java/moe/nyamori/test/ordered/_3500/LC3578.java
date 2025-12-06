package moe.nyamori.test.ordered._3500;

// Medium

/**
 * 3578. Count Partitions With Max-Min Difference at Most K
 * You are given an integer array nums and an integer k. Your task is to partition nums into one or more non-empty contiguous segments such that in each segment, the difference between its maximum and minimum elements is at most k.
 *
 * Return the total number of ways to partition nums under this condition.
 *
 * Since the answer may be too large, return it modulo 109 + 7.
 *
 * Constraints:
 *
 * 2 <= nums.length <= 5 * 104
 * 1 <= nums[i] <= 109
 * 0 <= k <= 109
 *
 * Tags:
 * Queue
 * Array
 * Dynamic Programming
 * Prefix Sum
 * Sliding Window
 * Monotonic Queue
 */
public class LC3578 {
    public static void main(String[] args) {
        var sol = new LC3578();
//        var nums = new int[]{9, 4, 1, 3, 7};
//        var k = 4;
        var nums = new int[]{3, 3, 4};
        var k = 0;
        System.err.println(sol.countPartitions(nums, k));
    }

    static final long MOD = 1_000_000_007L;


    public int countPartitions(int[] nums, int k) {
        var n = nums.length;
        var memo = new Long[n + 1]; // nulls
        /*var prefix = new Long[n + 1]; // nulls*/
        return (int) (helper(n - 1, nums, k, memo) % MOD);
    }

    private long helper(int endIdx, int[] nums, int k, Long[] memo/*, Long[] prefix*/) {
        if (endIdx <= 0) {
            // a number subtracts itself always equals 0, and 0<=k, hence 1 should be returned
            // ** counter-intuitive **
            // For endIdx less than zero, we still treat it as an existing number ???
            return 1;
        }
        if (memo[endIdx] != null) return memo[endIdx];
        var res = 0L;
        // find an L (lower bound), such that all sub ranges in [[L, endIdx], endIdx] satisfied the "k condition"
        // var minDeq = new LinkedList<Integer>();
        // var maxDeq = new LinkedList<Integer>();
        // iterate from endIdx backward, until the max-min > k
        var min = nums[endIdx];
        var max = nums[endIdx];
        var lowerBound = endIdx;
        for (var i = endIdx - 1; i >= 0; i--) {
            min = Math.min(min, nums[i]);
            max = Math.max(max, nums[i]);
            var diff = max - min;
            if (diff <= k) {
                lowerBound = i;
            } else {
                break;
            }
        }
        for (var i = lowerBound - 1; i < endIdx; i++) {
            res += (helper(i, nums, k, memo) + MOD) % MOD;
        }
        return memo[endIdx] = res;
    }

}
