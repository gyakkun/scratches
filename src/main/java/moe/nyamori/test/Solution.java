package moe.nyamori.test;

public class Solution {

    public static void main(String[] args) {
        long timing = System.currentTimeMillis();
        Solution s = new Solution();
        System.err.println(s.twoSum(1, 2));
        timing = System.currentTimeMillis() - timing;
        System.err.println("Timing: " + timing + "ms.");
    }


    public int twoSum(int a, int b) {
        return a + b;
    }

}