package moe.nyamori.test.ordered._3400;

// Easy

import java.util.Arrays;

/**
 * 3432. Count Partitions with Even Sum Difference
 *
 * You are given an integer array nums of length n.
 *
 * A partition is defined as an index i where 0 <= i < n - 1, splitting the array into two non-empty subarrays such that:
 *
 * Left subarray contains indices [0, i].
 * Right subarray contains indices [i + 1, n - 1].
 * Return the number of partitions where the difference between the sum of the left and right subarrays is even.
 *
 */
public class LC3400 {
    public int countPartitions(int[] nums) {
        var sum = Arrays.stream(nums).sum();
        var leftSum = 0;
        var rightSum = -1;
        var res = 0;
        for(var i =0;i<nums.length-1;i++) {
            leftSum += nums[i];
            rightSum = sum-leftSum;
            if(Math.abs(leftSum-rightSum)%2==0) res++;
        }
        return res;
    }
}
