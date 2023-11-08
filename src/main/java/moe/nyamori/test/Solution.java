package moe.nyamori.test;

public class Solution {

    // Given an integer array nums, return the number of reverse pairs in the array.
    //
    //A reverse pair is a pair (i, j) where:
    //
    //0 <= i < j < nums.length and
    //nums[i] > 2 * nums[j].
    //
    //Example 1:
    //
    //Input: nums = [1,3,2,3,1]
    //Output: 2
    //Explanation: The reverse pairs are:
    //(1, 4) --> nums[1] = 3, nums[4] = 1, 3 > 2 * 1
    //(3, 4) --> nums[3] = 3, nums[4] = 1, 3 > 2 * 1
    //Example 2:
    //
    //Input: nums = [2,4,3,5,1]
    //Output: 3
    //Explanation: The reverse pairs are:
    //(1, 4) --> nums[1] = 4, nums[4] = 1, 4 > 2 * 1
    //(2, 4) --> nums[2] = 3, nums[4] = 1, 3 > 2 * 1
    //(3, 4) --> nums[3] = 5, nums[4] = 1, 5 > 2 * 1
    public int reversePairs(int[] nums) {
        int n = nums.length;
        int[] tmp = new int[n];
        return mergeSort(nums, tmp, 0, n - 1);
    }

    public int mergeSort(int[] nums, int[] tmp, int a, int b) {
        if (a >= b) return 0;
        int mid = (a + b) / 2;
        int res = mergeSort(nums, tmp, a, mid) + mergeSort(nums, tmp, mid + 1, b);
        int i = a, j = mid + 1;
        for (int k = a; k <= b; k++) {
            tmp[k] = nums[k];
        }
        for (int k = a; k <= b; k++) {
            if (i == mid + 1) {
                nums[k] = tmp[j++];
            } else if (j == b + 1 || tmp[i] <= tmp[j]) {
                nums[k] = tmp[i++];
            } else {
                nums[k] = tmp[j++];
                res += mid - i + 1;
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Solution s = new Solution();
        System.err.println(s.reversePairs(new int[]{1,3,2,3,1}));
    }

}