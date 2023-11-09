import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        int[] arr = new int[]{7, 1, 5, 3, 6, 4};
        System.err.println(maxProfit(arr));
    }

    public static int maxProfit(int[] prices) {
        int[] dp = new int[prices.length];
        int max = 0;

        for (int i = 0; i < prices.length; i++) {
            for (int j = 0; j < prices.length; j++) {
                dp[i] = Math.max(dp[i], prices[j] - prices[i]);
                max = Math.max(max, dp[i]);
            }
        }
        return max;

    }

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {

        Set<String> set = new HashSet<>();

        double[] result = new double[queries.size()];

        for (List<String> q : queries) {
            for (String s : q) {
                set.add(s);
            }
        }

        for (List<String> q : equations) {
            for (String s : q) {
                set.add(s);
            }
        }

        int size = set.size();

        String[] sa = new String[size];
        set.toArray(sa);
        Map<String, Integer> sIdx = new HashMap<>();

        for (int i = 0; i < size; i++) {
            sIdx.put(sa[i], i);
        }

        double[][] mtx = new double[size][size];
        boolean[][] reachability = new boolean[size][size];
//        for (boolean[] ra : reachability) {
//            Arrays.fill(ra, false);
//        }

        for (int i = 0; i < equations.size(); i++) {
            int ntrIdx = sIdx.get(equations.get(i).get(0));
            int dtrIdx = sIdx.get(equations.get(i).get(1));
            mtx[ntrIdx][dtrIdx] = values[i];
            mtx[dtrIdx][ntrIdx] = 1d / values[i];
            reachability[ntrIdx][dtrIdx] = true;
            reachability[dtrIdx][ntrIdx] = true;
        }

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
//                if (i == j) continue;
                for (int k = 0; k < size; k++) {
//                    if (k == i) continue;
                    if (reachability[i][j] && reachability[j][k]) {
                        reachability[i][k] = true;
                        mtx[i][k] = mtx[i][j] * mtx[j][k];
                    }
                }
            }
        }

        for (int i = 0; i < queries.size(); i++) {
            int ntrIdx = sIdx.get(queries.get(i).get(0));
            int dtrIdx = sIdx.get(queries.get(i).get(1));
            if (reachability[ntrIdx][dtrIdx]) {
                result[i] = mtx[ntrIdx][dtrIdx];
            } else {
                result[i] = -1d;
            }
        }

        return result;
    }

    public static int lengthOfLISDP(int[] nums) {
        int n = nums.length;
        int[] arr = new int[n];
        Arrays.fill(arr, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    arr[i] = Math.max(arr[j] + 1, arr[i]);
                }
            }
        }

        return Arrays.stream(arr).max().getAsInt();
    }

    public static int lengthOfLISGreedyBS(int[] nums) {
        if (nums.length == 0) return 0;
        int n = nums.length;
        int len = 1;
        int[] d = new int[n + 1];
        d[1] = nums[0];

        for (int i = 1; i < n; i++) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0;
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return len;
    }

    public int eraseOverlapIntervalsBS(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int[] f = new int[intervals.length];
        int[] d = new int[intervals.length + 1];
        int len = 1;
        d[1] = intervals[0][0];


        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

        Arrays.fill(f, 1);

        for (int i = 1; i < intervals.length; i++) {
        }


        for (int i = 1; i < intervals.length; i++) {
            for (int j = 0; j < i; j++) {
                if (intervals[i][0] >= intervals[j][1]) {
                    f[i] = Math.max(f[i], f[j] + 1);
                }
            }
        }
        return intervals.length - Arrays.stream(f).max().getAsInt();
    }

    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int[] f = new int[intervals.length];
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

        // i,j
        // f[i] = 以第i个区间为结束, 可以选出的区间数量的最大值
        // f[i] = max(f[0]...f[i-1]) + 1,记为f[j]
        // 满足 j<i 且 r[j]<l[i]

        Arrays.fill(f, 1);

        for (int i = 1; i < intervals.length; i++) {
            for (int j = 0; j < i; j++) {
                if (intervals[j][1] <= intervals[i][0]) {
                    f[i] = Math.max(f[i], f[j] + 1);
                }
            }
        }
        return intervals.length - Arrays.stream(f).max().getAsInt();
    }

    public static int binarySearch(int[] sortedArray, int target) {
        int l = 0, r = sortedArray.length, mid;
        while (l <= r) {
            mid = (l + r) >>> 1;
            if (sortedArray[mid] == target) {
                return mid;
            } else if (sortedArray[mid] < target) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return -1;
    }


    public static List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> result = new ArrayList<>();

        char[] c = s.toCharArray();

        for (int i = 0; i < c.length; i++) {
            if (i + 2 < c.length && c[i] == c[i + 1] && c[i] == c[i + 2]) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(i);
                i += 2;
                while (c[i] == c[i - 1]) {
                    i++;
                    if (i > c.length) break;
                }
                i--;
                tmp.add(i);
                result.add(tmp);
            }
        }
        return result;
    }


    public static int fib(int n) {
        int nM2 = 0;
        int nM1 = 1;
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;
        int result = nM1;
        for (int i = 1; i < n; i++) {
            int temp = result;
            result = result + nM2;
            nM2 = temp;
        }
        return result;
    }

    public static int maxSubarraySumCircular(int[] A) {
        int aLength = A.length;
        int result = Integer.MIN_VALUE / 2;
        int[] localMaxArray = new int[aLength];
        Arrays.fill(localMaxArray, Integer.MIN_VALUE / 2);

        int localMax = Integer.MIN_VALUE / 2;
        int legacyKadaneResult = Integer.MIN_VALUE / 2;
        for (int i = 0; i < aLength; i++) {
            localMax = Math.max(A[i], A[i] + localMax);
            legacyKadaneResult = Math.max(localMax, legacyKadaneResult);
        }


        for (int i = 0; i < aLength; i++) {
            for (int j = 0; j < aLength; j++) {
                localMaxArray[i] = Math.max(A[(i + j) % aLength], localMaxArray[i] + A[(i + j) % aLength]);
            }
        }
        Arrays.sort(localMaxArray);
        return localMaxArray[aLength - 1];
    }

    public int lastStoneWeight(int[] stones) {
        List<Integer> l = Arrays.stream(stones).boxed().collect(Collectors.toList());

        while (l.size() >= 2) {
            l.sort((a, b) -> b - a);
            if (l.get(0) != l.get(1)) {
                l.set(0, l.get(0) - l.get(1));
                l.remove(1);
            } else {
                l.remove(1);
                l.remove(0);
            }
        }
        if (l.size() == 0) {
            return 0;
        }
        return l.get(0);
    }
}