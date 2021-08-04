import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.busRapidTransit(31,
                5,
                3,
                new int[]{6},
                new int[]{10}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LCP 20 ** bottom up dfs
    Map<Long, Long> lcp20Map;
    final long lcp20Mod = 1000000007L;
    int lcp20Inc, lcp20Dec;
    int[] lcp20Jump, lcp20Cost;

    public int busRapidTransit(int target, int inc, int dec, int[] jump, int[] cost) {
        lcp20Map = new HashMap<>();
        lcp20Map.put(0l, 0l);
        lcp20Map.put(1l, (long) inc);
        lcp20Inc = inc;
        lcp20Cost = cost;
        lcp20Jump = jump;
        lcp20Dec = dec;
        return (int) (lcp20Helper(target) % lcp20Mod);
    }

    private long lcp20Helper(long cur) {
        if (lcp20Map.containsKey(cur)) return lcp20Map.get(cur);
        long result = cur * lcp20Inc;
        for (int i = 0; i < lcp20Jump.length; i++) {
            long remainder = cur % lcp20Jump[i];
            if (remainder == 0l) {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i]);
            } else {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i] + remainder * lcp20Inc);
                result = Math.min(result, lcp20Helper((cur / lcp20Jump[i]) + 1) + lcp20Cost[i] + (lcp20Jump[i] - remainder) * lcp20Dec);
            }
        }
        lcp20Map.put(cur, result);
        return result;
    }

    // LC1940 Prime Locked
    public List<Integer> longestCommomSubsequence(int[][] arrays) {
        List<Integer> result = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            int count = 0;
            for (int[] arr : arrays) {
                int bsResult = Arrays.binarySearch(arr, i);
                if (bsResult >= 0) count++;
            }
            if (count == arrays.length) result.add(i);
        }
        return result;
    }

    // LC1781
    public int beautySum(String s) {
        char[] ca = s.toCharArray();
        int[] freq = new int[26];
        int left = 0;
        int result = 0;
        while (left < s.length()) {
            freq = new int[26];
            int right = left;
            while (right < s.length()) {
                freq[ca[right++] - 'a']++;
                int[] j = lc1781Judge(freq);
                if (j[0] != -1) {
                    result += freq[j[1]] - freq[j[0]];
                }
            }
            left++;
        }
        return result;
    }

    private int[] lc1781Judge(int[] freq) {
        int min = Integer.MAX_VALUE, minIdx = -1, max = 0, maxIdx = -1;
        int notZeroCount = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] != 0) notZeroCount++;
            if (freq[i] > max) {
                max = freq[i];
                maxIdx = i;
            }
            if (freq[i] != 0 && freq[i] < min) {
                min = freq[i];
                minIdx = i;
            }
        }
        if (notZeroCount <= 1 || max == min) return new int[]{-1, -1};
        return new int[]{minIdx, maxIdx};
    }

    // LC417 **
    boolean[][] lc417P, lc417A;
    int[][] lc417Direction = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        int n = heights.length, m = heights[0].length;
        lc417A = new boolean[n][m];
        lc417P = new boolean[n][m];
        for (int i = 0; i < n; i++) {
            lc417Helper(heights, i, 0, lc417P);
            lc417Helper(heights, i, m - 1, lc417A);
        }
        for (int i = 0; i < m; i++) {
            lc417Helper(heights, 0, i, lc417P);
            lc417Helper(heights, n - 1, i, lc417A);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (lc417P[i][j] && lc417A[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }
        return result;
    }

    private boolean lc417IdxLegalJudge(int row, int col) {
        return (row >= 0 && row < lc417P.length && col >= 0 && col < lc417P[0].length);
    }

    private void lc417Helper(int[][] heights, int row, int col, boolean[][] judge) {
        if (judge[row][col]) return;
        judge[row][col] = true;
        for (int[] dir : lc417Direction) {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if (lc417IdxLegalJudge(newRow, newCol) && heights[newRow][newCol] >= heights[row][col]) {
                lc417Helper(heights, newRow, newCol, judge);
            }
        }
    }


    // LC611 **
    public int triangleNumber(int[] nums) {
        // nums.length <=1000
        // A + B > C
        int n = nums.length, result = 0;
        if (n <= 2) return 0;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int k = i;
            for (int j = i + 1; j < n; j++) {
                while (k + 1 < n && nums[k + 1] < nums[i] + nums[j]) {
                    k++;
                }
                result += Math.max(k - j, 0);
            }
        }
        return result;
    }

    // LC167
    public int[] twoSum(int[] numbers, int target) {
        int n = numbers.length;
        for (int i = 0; i < n; i++) {
            int tmp = target - numbers[i];
            int bsResult = Arrays.binarySearch(numbers, i + 1, n, tmp);
            if (bsResult >= 0) return new int[]{i + 1, bsResult + 1};
        }
        return new int[]{-1, -1};
    }

    // LC1823
    public int findTheWinner(int n, int k) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int i = 1; i <= n; i++) s.add(i);
        int cur = 1;
        while (s.size() > 1) {
            int ctr = 1;
            while (ctr < k) {
                Integer higher = s.higher(cur);
                if (higher == null) higher = s.first();
                cur = higher;
                ctr++;
            }
            Integer next = s.higher(cur);
            if (next == null) next = s.first();
            s.remove(cur);
            cur = next;
        }
        return s.first();
    }

    // LC1567 Solution DP
    public int getMaxLen(int[] nums) {
        int n = nums.length;
        int[] pos = new int[2], neg = new int[2];
        if (nums[0] > 0) pos[0] = 1;
        if (nums[0] < 0) neg[0] = 1;
        int result = pos[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                pos[i % 2] = pos[(i - 1) % 2] + 1;
                neg[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
            } else if (nums[i] < 0) {
                pos[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
                neg[i % 2] = pos[(i - 1) % 2] + 1;
            } else {
                pos[i % 2] = neg[i % 2] = 0;
            }
            result = Math.max(result, pos[i % 2]);
        }
        return result;
    }

    // LC1567 慢
    public int getMaxLenSimple(int[] nums) {
        int n = nums.length;
        int[] nextZero = new int[n];
        int[] negCount = new int[n];
        Arrays.fill(nextZero, -1);
        int nextZeroIdx = -1;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] == 0) {
                nextZeroIdx = i;
            }
            nextZero[i] = nextZeroIdx;
        }
        negCount[0] = nums[0] < 0 ? 1 : 0;
        for (int i = 1; i < n; i++) {
            negCount[i] = negCount[i - 1] + (nums[i] < 0 ? 1 : 0);
        }
        int result = 0;
        // 在下一个0来临之前, 找到最大的偶数个负数所在IDX 求长度
        for (int i = 0; i < n; i++) {
            if (nums[i] != 0) {
                int curNegCount = negCount[i];
                if (nums[i] < 0) curNegCount--;
                int start = i, end = -1;
                if (nextZero[i] == -1) end = n - 1;
                else end = nextZero[i] - 1;
                int j;
                for (j = end; j >= start; j--) {
                    if (negCount[j] % 2 == curNegCount % 2) break;
                }
                result = Math.max(result, j - i + 1);
            }
        }
        return result;
    }

    // LC198
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[3]; // 滚数组
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[(i + 3) % 3] = Math.max(dp[(i - 1 + 3) % 3], dp[(i - 2 + 3) % 3] + nums[i]);
        }
        return dp[(n - 1 + 3) % 3];
    }

    // LC740 ** 打家劫舍
    public int deleteAndEarn(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int[] sum = new int[max + 1];
        for (int i : nums) sum[i] += i;
        if (max == 1) return sum[1];
        if (max == 2) return Math.max(sum[1], sum[2]);
        int[] dp = new int[max + 1];
        dp[1] = sum[1];
        dp[2] = Math.max(sum[1], sum[2]);
        for (int i = 3; i <= max; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + sum[i]);
        }
        return dp[max];
    }

    // LC673 **
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        int[] dp = new int[n], count = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(count, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[i] <= dp[j]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
            }
        }
        int max = Arrays.stream(dp).max().getAsInt();
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == max) {
                result += count[i];
            }
        }
        return result;
    }
}

// Interview 03.06
class AnimalShelf {
    int seq = 0;
    // type 0-cat 1-dog
    final int CAT = 0, DOG = 1;
    Map<Integer, Integer> idSeqMap = new HashMap<>();
    Deque<Integer> catQueue = new LinkedList<>();
    Deque<Integer> dogQueue = new LinkedList<>();

    public AnimalShelf() {

    }

    public void enqueue(int[] a) {
        // a[0] = id, a[1] = type
        int sequence = getSeq();
        idSeqMap.put(a[0], sequence);
        if (a[1] == CAT) {
            catQueue.offer(a[0]);
        } else {
            dogQueue.offer(a[0]);
        }
    }

    public int[] dequeueAny() {
        if (catQueue.isEmpty() && dogQueue.isEmpty()) {
            return new int[]{-1, -1};
        } else if (catQueue.isEmpty() && !dogQueue.isEmpty()) {
            return dequeueDog();
        } else if (!catQueue.isEmpty() && dogQueue.isEmpty()) {
            return dequeueCat();
        } else if (idSeqMap.get(catQueue.peek()) < idSeqMap.get(dogQueue.peek())) {
            return dequeueCat();
        } else {
            return dequeueDog();
        }

    }

    public int[] dequeueDog() {
        if (dogQueue.isEmpty()) return new int[]{-1, -1};
        int polledDogId = dogQueue.poll();
        idSeqMap.remove(polledDogId);
        return new int[]{polledDogId, DOG};
    }

    public int[] dequeueCat() {
        if (catQueue.isEmpty()) return new int[]{-1, -1};
        int polledCatId = catQueue.poll();
        idSeqMap.remove(polledCatId);
        return new int[]{polledCatId, CAT};
    }

    private int getSeq() {
        return seq++;
    }
}

// LC478
class Solution {
    double x_center;
    double y_center;
    double radius;

    public Solution(double radius, double x_center, double y_center) {
        this.x_center = x_center;
        this.y_center = y_center;
        this.radius = radius;
    }

    public double[] randPoint() {
        double len = Math.sqrt(Math.random()) * radius; // 注意开方 , 参考solution
        double theta = Math.random() * Math.PI * 2;

        double x = len * Math.sin(theta) + x_center;
        double y = len * Math.cos(theta) + y_center;
        return new double[]{x, y};
    }
}

// JZOF 59
class KthLargest {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        for (int i : nums) {
            add(i);
        }
    }

    public int add(int val) {
        if (pq.size() < k) {
            pq.offer(val);
        } else {
            if (val > pq.peek()) {
                pq.poll();
                pq.offer(val);
            }
        }
        return pq.peek();
    }
}

class quickSort {

    static Random r = new Random();

    public static void sort(int[] arr) {
        helper(arr, 0, arr.length - 1);
    }

    private static void helper(int[] arr, int start, int end) {
        if (start >= end) return;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }
        int pivotVal = arr[start];
        int left = start, right = end;
        while (left < right) {
            while (left < right && arr[right] > pivotVal) {
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivotVal) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivotVal;
        helper(arr, start, left - 1);
        helper(arr, right + 1, end);
    }

}

class quickSelect {
    static Random r = new Random();

    public static int topK(int[] arr, int topK) {
        return helper(arr, 0, arr.length - 1, topK);
    }

    private static Integer helper(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }
        int pivotVal = arr[start];
        int left = start, right = end;
        while (left < right) {
            while (left < right && arr[right] > pivotVal) {
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivotVal) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivotVal;
        if (left == arr.length - topK) return arr[left];
        Integer leftResult = helper(arr, start, left - 1, topK);
        if (leftResult != null) return leftResult;
        Integer rightResult = helper(arr, right + 1, end, topK);
        if (rightResult != null) return rightResult;
        return null;
    }
}