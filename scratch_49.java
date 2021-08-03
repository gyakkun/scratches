import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.getMaxLen(new int[]{-1, -2, -3, 0, 1}));
        System.out.println(s.getMaxLen(new int[]{-1, 2}));
        System.out.println(s.getMaxLen(new int[]{0, 1, -2, -3, -4}));
        System.out.println(s.getMaxLen(new int[]{1, -2, -3, 4}));
        System.out.println(s.getMaxLen(new int[]{1, 2, 3, 5, -6, 4, 0, 10}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1567 慢
    public int getMaxLen(int[] nums) {
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
