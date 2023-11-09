import java.util.*;

class Scratch {

    public static void main(String[] args) {
        int[] arr = new int[]{1, 10, 21, 22, 25};
        int k = 12;
        System.err.println(findKthPositive(arr, k));
    }

    public static int findKthPositive(int[] arr, int k) {
        if (k == 1 && arr[0] == 2) return 1;
        if (arr.length == 0) return k;
        int howManyPositiveIntegerAreLost = 0;
        howManyPositiveIntegerAreLost = arr[0] - 1;
        if (k <= howManyPositiveIntegerAreLost) {
            return k;
        }
        for (int i = 1; i < arr.length; i++) {
            howManyPositiveIntegerAreLost += arr[i] - arr[i - 1] - 1;
            if (k <= howManyPositiveIntegerAreLost) {
                return arr[i] - 1 - (howManyPositiveIntegerAreLost - k);
            }
        }
        return arr[arr.length - 1] + (k - howManyPositiveIntegerAreLost);
    }


    public static int minMatch(int[] nums, int n) {

        long nLong = (long) n;

        long result = 0;

        Arrays.sort(nums);

        long accumulateExpressRange = 0;
        long pivot = 0;

        for (int i = 0; i < nums.length; i++) {
            if (accumulateExpressRange >= nums[i]) {
                accumulateExpressRange += nums[i];
            }
            if (accumulateExpressRange >= nLong) {
                long maxTwoPowerLessThanNumsPivot = (long) Math.pow(2d, Math.floor(Math.log(pivot) / Math.log(2)));
                Set<Integer> tmpTwoPowerSet = new HashSet<>();
                for (int j : nums) {
                    if ((j & (j - 1)) == 0 && j <= maxTwoPowerLessThanNumsPivot) {
                        tmpTwoPowerSet.add(j);
                    }
                }

                long thisResult = 0;
                thisResult = (long) Math.floor(Math.log(pivot) / Math.log(2)) + 1 - tmpTwoPowerSet.size();
                return (int) thisResult;
            }
            long tmpMaxTwoPowerLessThanNumsI = (long) Math.pow(2d, Math.floor(Math.log(nums[i]) / Math.log(2)));
            long tmpMaxExpressRange = 0;
            if (tmpMaxTwoPowerLessThanNumsI == nums[i]) {
                tmpMaxExpressRange = 2 * tmpMaxTwoPowerLessThanNumsI - 1;
            } else {
                tmpMaxExpressRange = nums[i] + 2 * tmpMaxTwoPowerLessThanNumsI - 1;
            }

            if (accumulateExpressRange < tmpMaxExpressRange) {
                pivot = nums[i];
                accumulateExpressRange = tmpMaxExpressRange;
            }
            if (accumulateExpressRange >= nLong) {
                long maxTwoPowerLessThanNumsPivot = (long) Math.pow(2d, Math.floor(Math.log(pivot) / Math.log(2)));
                Set<Integer> tmpTwoPowerSet = new HashSet<>();
                for (int j : nums) {
                    if ((j & (j - 1)) == 0 && j <= maxTwoPowerLessThanNumsPivot) {
                        tmpTwoPowerSet.add(j);
                    }
                }

                long thisResult = 0;
                thisResult = (long) Math.floor(Math.log(pivot) / Math.log(2)) + 1 - tmpTwoPowerSet.size();
                return (int) thisResult;
            }
        }

        Set<Integer> twoPowerSet = new HashSet<>();

        long maxTwoPowerLessThanN = (long) Math.pow(2d, Math.floor(Math.log(nLong) / Math.log(2)));

        for (int i : nums) {
            if ((i & (i - 1)) == 0 && i <= maxTwoPowerLessThanN) twoPowerSet.add(i);
        }

        result = (long) Math.floor(Math.log(nLong) / Math.log(2)) + 1 - twoPowerSet.size();

        return (int) result;
    }


    public static int maxProfit(int k, int[] prices) {
        int result = 0;
        k = Math.min(k, prices.length / 2);
        int[][] sell = new int[prices.length][k + 1];
        int[][] buy = new int[prices.length][k + 1];

        for (int i = 0; i < k + 1; i++) {
            sell[0][i] = buy[0][i] = Integer.MIN_VALUE / 2;
        }
        sell[0][0] = 0;
        buy[0][0] = 0 - prices[0];

        for (int i = 1; i < prices.length; i++) {
            buy[i][0] = Math.max(buy[i - 1][0], sell[i - 1][0] - prices[i]);
            for (int j = 1; j <= k; j++) {
                buy[i][j] = Math.max(buy[i - 1][j], sell[i - 1][j] - prices[i]);
                sell[i][j] = Math.max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);
            }
        }

        for (int i = 0; i <= k; i++) {
            result = Math.max(result, sell[prices.length - 1][i]);
        }
        return result;
    }
}