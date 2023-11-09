import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.TreeMap;

class Scratch {
    public static void main(String[] args) {
        minHeap = new PriorityQueue<>(Comparator.comparingLong(o -> o));
        maxHeap = new PriorityQueue<>(Comparator.comparingLong(o -> -o));
        int[] arr = new int[]{9, 7, 3, 5, 6, 2, 0, 8, 1, 9};
        System.err.println(findMaxAverage(arr, 6));

    }

//    给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。
//
//
//    示例：
//
//    输入：[1,12,-5,-6,50,3], k = 4
//    输出：12.75
//    解释：最大平均数 (12-5-6+50)/4 = 51/4 = 12.75

    public static double findMaxAverage(int[] nums, int k) {
        if (nums.length == 1) return nums[0];
        int len = nums.length;
        int n = len - k + 1; //滑动窗口个数
        long thisSum = 0;
        long prevSum = 0;
        for (int i = 0; i < k; i++) {
            prevSum += nums[i];
        }
        if (nums.length == k) return (double) prevSum / (double) k;
        long maxSum = Long.MIN_VALUE;
        maxSum = Math.max(maxSum, prevSum);
        for (int i = 1; i < n; i++) {
            thisSum = sumK(nums, prevSum, i, k);
            maxSum = Math.max(maxSum, thisSum);
            prevSum = thisSum;
        }
        return (double) maxSum / (double) k;
    }

    public static long sumK(int[] arr, long prevSum, int startIdx, int k) {
        return prevSum - arr[startIdx - 1] + arr[startIdx + k - 1];
    }

    public int characterReplacement(String s, int k) {
        int[] alphabet = new int[26];
        int left, right;
        int n = s.length();
        left = right = 0;
        int maxLetterCnt = 0;
        while (right < n) {
            alphabet[s.charAt(right) - 'A']++;
            maxLetterCnt = Math.max(maxLetterCnt, alphabet[s.charAt(right) - 'A']);
            if (right - left + 1 - maxLetterCnt > k) {
                alphabet[s.charAt(left) - 'A']--;
                left++;
            }
            right++;
        }
        return right - left;
    }

//The simplest of solutions comes from the basic idea of finding the median given a set of numbers. We know that by definition, a median is the center element (or an average of the two center elements). Given an unsorted list of numbers, how do we find the median element? If you know the answer to this question, can we extend this idea to every sliding window that we come across in the array?
//Is there a better way to do what we are doing in the above hint? Don't you think there is duplication of calculation being done there? Is there some sort of optimization that we can do to achieve the same result? This approach is merely a modification of the basic approach except that it simply reduces duplication of calculations once done.
//The third line of thought is also based on this same idea but achieving the result in a different way. We obviously need the window to be sorted for us to be able to find the median. Is there a data-structure out there that we can use (in one or more quantities) to obtain the median element extremely fast, say O(1) time while having the ability to perform the other operations fairly efficiently as well?

    public static PriorityQueue<Long> minHeap;
    public static PriorityQueue<Long> maxHeap;

    public Scratch() {
        minHeap = new PriorityQueue<>(Comparator.comparingLong(o -> o));
        maxHeap = new PriorityQueue<>(Comparator.comparingLong(o -> -o));
    }

    public static void addNum(long num) {
        if (maxHeap.isEmpty() || num < maxHeap.peek()) {
            maxHeap.offer(num);
        } else {
            minHeap.offer(num);
        }

        if (maxHeap.size() == minHeap.size() + 2) {
            minHeap.offer(maxHeap.poll());
        }
        if (minHeap.size() == maxHeap.size() + 1) {
            maxHeap.offer(minHeap.poll());
        }
    }

    public static void removeNum(long num) {
        // 先删除, 后平衡, 再添加, 再平衡
        if (maxHeap.contains(num)) {
            maxHeap.remove(num);
        } else {
            minHeap.remove(num);
        }

        while (minHeap.size() < maxHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }

        while (maxHeap.size() < minHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }

    }

    public static double[] medianSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        double[] result = new double[nums.length - k + 1];

        // 目标: 最大堆堆顶即是中位数

        // addNum(int i)

        int left = 0;
        int right = 0 + k - 1;

        // 先填充一个滑动窗口
        for (int i = 0; i < k; i++) {
            addNum(nums[i]);
        }

        for (int i = 0; right < len; i++) {
            if (maxHeap.size() == minHeap.size()) {
                long sum = maxHeap.peek() + minHeap.peek();
                result[i] = ((double) sum) / 2;
            } else {
                result[i] = (double) maxHeap.peek();
            }
            removeNum(nums[left]);
            left++;
            right++;
            if (right < len) {
                addNum(nums[right]);
            }
        }

        return result;
    }


    //100个房间，均有非负个糖果。从到尾取一遍，不能取相邻房间的，最多能取多少个#

    int maxCandyCount(int[] arr) {
        // 第二项, 0表示不取这个房间, 1表示取了这个房间
        int[][] dp = new int[arr.length][2];
        dp[0][0] = 0;
        dp[0][1] = arr[0];

        // dp[i][0] = Math.max( dp[i-1][0], dp[i-1][1])
        // dp[i][1] = dp[i-1][0]+arr[i]

    }


}

class Solution {
    class Obj {
        TreeMap<Integer, Integer> ml = new TreeMap<>(Comparator.reverseOrder());
        TreeMap<Integer, Integer> mr = new TreeMap<>();
        int lsz = 0, rsz = 0;

        public Integer get_first(TreeMap<Integer, Integer> map) {
            return map.keySet().iterator().next();//获取堆中第一个数
        }

        public void inc(TreeMap<Integer, Integer> map, int x) {
            map.put(x, map.getOrDefault(x, 0) + 1); //向堆中添加一个x
        }

        public void dec(TreeMap<Integer, Integer> map, int x) {
            if (map.get(x) == 1) map.remove(x);  //向堆中删除一个x
            else map.put(x, map.get(x) - 1);
        }

        Obj() {
        }

        public void insert(int x) { //插入，按照约定左侧一定比右侧小
            if (ml.size() == 0 || x < get_first(ml)) {
                inc(ml, x);
                lsz++;
            } else {
                inc(mr, x);
                rsz++;
            }
        }

        public void del(int x) {//删除
            if (ml.containsKey(x)) {
                dec(ml, x);
                lsz--;
            } else {
                dec(mr, x);
                rsz--;
            }
        }

        public double getMid() {//返回中位数
            int t = (lsz + rsz) % 2;
            while (lsz - rsz != t) {//调整两个堆的大小关系，使其满足l==r 或者l+1==r
                if (lsz - rsz > t) {//左边太多
                    int x = get_first(ml);
                    dec(ml, x);
                    inc(mr, x);
                    lsz--;
                    rsz++;
                } else {//右边太多
                    int x = get_first(mr);
                    dec(mr, x);
                    inc(ml, x);
                    rsz--;
                    lsz++;
                }
            }
            if (t == 1) return get_first(ml);
            else return (0.0 + get_first(ml) + get_first(mr)) / 2;
        }
    }

    public double[] medianSlidingWindow(int[] nums, int k) {
        double[] res = new double[nums.length - k + 1];
        int p = 0;
        Obj obj = new Obj();
        for (int i = 0; i < nums.length; i++) {
            obj.insert(nums[i]);
            if (i >= k) obj.del(nums[i - k]);
            if (i >= k - 1) res[p++] = obj.getMid();
        }
        return res;
    }

}
