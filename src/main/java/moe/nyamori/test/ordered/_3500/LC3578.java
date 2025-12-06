package moe.nyamori.test.ordered._3500;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedList;

// **Medium
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
    public static void main(String[] args) throws IOException {
        var sol = new LC3578();
//        var nums = new int[]{9, 4, 1, 3, 7};
//        var k = 4;
//        var nums = new int[]{3, 3, 4};
//        var k = 0;
        var k = 20_000_000;
        var nums = Arrays.stream(
                Files.readString(
                        Path.of(System.getProperty("user.home"), "LC3578.txt")
                ).trim().split(",")
        ).mapToInt(Integer::parseInt).toArray();
        var timing = System.currentTimeMillis();
        System.err.println(sol.countPartitions(nums, k));
        System.err.println("Timing: " + (System.currentTimeMillis() - timing) + "ms");
    }

    static final long MOD = 1_000_000_007L;


    public int countPartitions(int[] nums, int k) {
        var n = nums.length;
        var dp = new long[n + 1];
        dp[0] = 1L;
        var prefix = new long[n + 1]; // 左闭右开求和: SUM[i,j) = prefix[j] - prefix[i] -> 闭区间求和 SUM[i,j] = prefix[j+1] - prefix[i]
        prefix[0] = 1L; // dp[i+1] = prefix[i] - prefix[lower_bound-1]
        var minDeq = new LinkedList<Integer>(); // 单调队列 维护区间最值的**下标**
        var maxDeq = new LinkedList<Integer>();
        for (int i = 0, j = 0; i < n; i++) {
            while (!maxDeq.isEmpty() && nums[maxDeq.peekLast()] <= nums[i]) {
                maxDeq.pollLast();
            }
            maxDeq.offerLast(i);
            while (!minDeq.isEmpty() && nums[minDeq.peekLast()] >= nums[i]) {
                minDeq.pollLast();
            }
            minDeq.offerLast(i);
            // 滑动窗口
            while (!maxDeq.isEmpty() && !minDeq.isEmpty() && nums[maxDeq.peekFirst()] - nums[minDeq.peekFirst()] > k) {
                if (maxDeq.peekFirst() == j) maxDeq.pollFirst();
                if (minDeq.peekFirst() == j) minDeq.pollFirst();
                j++;
            }

            dp[i + 1] = (prefix[i] - (j > 0 ? prefix[j - 1] : 0) + MOD) % MOD;
            prefix[i + 1] = (prefix[i] + dp[i + 1]) % MOD;
        }

        return (int) (dp[n] % MOD);
    }

}
