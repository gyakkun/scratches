import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.countMaxOrSubsets(new int[]{3, 2, 1, 5}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC2044
    public int countMaxOrSubsets(int[] nums) {
        int n = nums.length;
        int max = Integer.MIN_VALUE, maxCount = 0;
        for (int mask = 1; mask < (1 << n); mask++) {
            int tmp = 0;
            for (int j = 0; j < n; j++) {
                if (((mask >> j) & 1) == 1) {
                    tmp |= nums[j];
                }
            }
            if (tmp > max) {
                max = tmp;
                maxCount = 1;
            } else if (tmp == max) {
                maxCount++;
            }
        }
        return maxCount;
    }
}