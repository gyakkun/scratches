import java.util.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.err.println(s.lastStoneWeightII(new int[]{12, 35, 78, 10, 24, 37, 55, 66, 90, 10, 42, 44, 12, 35, 78, 10, 24, 37, 55, 66, 90, 10, 12, 35, 78, 10, 24, 37, 55, 66}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1049 DP
    public int lastStoneWeightII(int[] stones) {
        // 目标: 找到绝对值差最小的一个划分
        int n = stones.length;
        int sum = Arrays.stream(stones).sum();
        int bound = sum / 2;
        boolean[][] dp = new boolean[n + 1][bound + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= bound; j++) {
                if (stones[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - stones[i - 1]];
                }
            }
        }
        for (int i = bound; i >= 0; i--) {
            if (dp[n][i]) return sum - 2 * i;
        }
        return -1;
    }

    // INTERVIEW 16.15 **
    public int[] masterMind(String solution, String guess) {
        int[] freq = new int[26];
        int real = 0, fake = 0;

        for (int i = 0; i < 4; i++) {
            char sol = solution.charAt(i), gue = guess.charAt(i);
            if (sol == gue) real++;
            else {
                if (freq[sol - 'A'] < 0) fake++;
                freq[sol - 'A']++;
                if (freq[gue - 'A'] > 0) fake++;
                freq[gue - 'A']--;
            }
        }
        return new int[]{real, fake};
    }

    // LC1400 这都行???
    public boolean canConstruct(String s, int k) {
        if (s.length() < k) return false;
        if (s.length() == k) return true;
        int[] freq = new int[26];
        char[] cArr = s.toCharArray();
        for (char c : cArr) {
            freq[c - 'a']++;
        }

        // 贪心: 尽量保证freq[i]是偶数, 最多有一个是奇数
        int oddCtr = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] % 2 == 1) oddCtr++;
        }
        if (oddCtr > k) return false;

        return true;
    }

    public int fib(int n) {
        if (n == 0) return 0;
        long prev = 0, cur = 1;
        final long mod = 1000000007;
        for (int i = 0; i < n - 1; i++) {
            long tmp = cur;
            cur = (prev + cur) % mod;
            prev = tmp % mod;
        }
        return (int) cur;
    }

    // LC1109 区间修改 差分数组
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] diff = new int[n + 1];
        int[] result = new int[n];
        for (int[] book : bookings) {
            diff[book[0] - 1] += book[2];
            diff[book[1]] -= book[2];
        }
        result[0] = diff[0];
        for (int i = 1; i < n; i++) {
            result[i] = diff[i] + result[i - 1];
        }
        return result;
    }

    // LC1033
    public int[] numMovesStones(int a, int b, int c) {
        int[] abc = {a, b, c};
        Arrays.sort(abc);
        a = abc[0];
        b = abc[1];
        c = abc[2];
        int[] result = new int[2]; // [min,max]
        // 最多步数
        result[1] = (b - a - 1) + (c - b - 1);

        // 最少步数: 最多2步
        result[0] = 2;
        if (b == a + 1) result[0]--;
        if (c == b + 1) result[0]--;
        if (result[0] == 1 || result[0] == 0) return result;

        if (b == a + 2) {
            result[0]--;
        } else if (c == b + 2) {
            result[0]--;
        }

        return result;
    }

    // LC853 单调栈 + 排序 **
    public int carFleet(int target, int[] position, int[] speed) {
        int n = position.length;
        TreeMap<Integer, Integer> posSpeedMap = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            posSpeedMap.put(position[i], speed[i]);
        }
        Deque<Double> stack = new LinkedList<>(); // 栈顶时间小, 栈底时间大
        for (Map.Entry<Integer, Integer> entry : posSpeedMap.entrySet()) {
            int pos = entry.getKey();
            int spd = entry.getValue();
            double time = (target - pos + 0.0d) / (spd + 0.0d);
            while (!stack.isEmpty() && stack.peek() <= time) { // 如果栈顶的到达时间比当前小, 则说明前方有一辆车比较慢, 会合并
                stack.pop();
            }
            stack.push(time);
        }
        return stack.size();
    }

    // LC1776 Hard 单调栈 **
    public double[] getCollisionTimes(int[][] cars) {
        int n = cars.length;
        double[] result = new double[n];
        Arrays.fill(result, -1d);
        Deque<Integer> stack = new LinkedList<>(); // 逆序遍历, 存下标右侧的车(的下标), 栈顶的车快, 栈底的车慢
        for (int i = n - 1; i >= 0; i--) {
            // 找下一个速度比当前小的车
            while (!stack.isEmpty()) {
                if (cars[stack.peek()][1] >= cars[i][1]) {
                    stack.pop();
                } else {
                    // 如果栈顶的车辆没有追上下一辆车, 而当前车的速度比这辆车(右侧)大, 那必然能够追上
                    if (result[stack.peek()] < 0) {
                        break;
                    } else { // 否则栈顶辆车能追上下一辆车, 需要计算: 在栈顶车追上下一辆车前, 当前车能不能追上栈顶的车辆
                        double peekCollisionTime = result[stack.peek()];
                        double myCollisionTime = (double) (cars[stack.peek()][0] - cars[i][0]) / (double) (cars[i][1] - cars[stack.peek()][1]);
                        // 如果当前车追上栈顶车的时间小于等于栈顶车追上下一辆车, 则在栈顶车与下一辆车相撞前, 当前车能与栈顶车相撞
                        if (myCollisionTime <= peekCollisionTime) {
                            break;
                        } else {
                            stack.pop();
                        }
                    }
                }
            }
            if (!stack.isEmpty()) {
                result[i] = (double) (cars[stack.peek()][0] - cars[i][0]) / (double) (cars[i][1] - cars[stack.peek()][1]);
            }
            stack.push(i);
        }
        return result;
    }

    private int[] simplePge(int[] nums) {
        // 单调栈: 找到上一个更大的元素, 底大, 顶小
        int n = nums.length;
        int[] pge = new int[n];
        Arrays.fill(pge, -1);
        Deque<Integer> stack = new LinkedList<>();
        stack.push(nums[0]);
        for (int i = 1; i < n; i++) {
            while (!stack.isEmpty() && stack.peek() < nums[i]) {
                stack.pop();
            }
            if (!stack.isEmpty()) {
                pge[i] = stack.peek();
            }
            stack.push(nums[i]);
        }
        return pge;
    }

    // LC1014 利用数列的遍历顺序
    public int maxScoreSightseeingPair(int[] values) {
        // 两个数列 v[i] + i, v[j] - j
        // j>i, 考虑顺序遍历, 动态更新v[i]+i
        int result = Integer.MIN_VALUE;
        int first = values[0] + 0;
        for (int i = 1; i < values.length; i++) {
            int second = values[i] - i;
            result = Math.max(result, first + second);
            first = Math.max(first, values[i] + i);
        }
        return result;
    }

    // JZOF33 **
    public boolean verifyPostorder(int[] postorder) {
        return jzof33Helper(postorder, 0, postorder.length - 1);
    }

    private boolean jzof33Helper(int[] postorder, int start, int end) {
        // 考虑最开始的情形,end即为root
        if (start >= end) {
            return true;
        }
        int ptr = start;
        while (postorder[ptr] < postorder[end]) ptr++;
        int rightStart = ptr; // 右子树的start
        while (postorder[ptr] > postorder[end]) ptr++;
        return ptr == end && jzof33Helper(postorder, start, rightStart - 1) && jzof33Helper(postorder, rightStart, end - 1);
    }

    // LC669
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return null;
        TreeNode left = root.left, right = root.right; // 注意提前存下引用
        root.left = trimBST(left, low, high);
        root.right = trimBST(right, low, high);
        if (root.val < low) { // 该节点的值比下界还小, 说明左子树不能要
            if (root.right != null) {
                root.val = right.val;
                root.left = right.left;
                root.right = right.right;
            } else {
                root = null;
            }
        } else if (root.val > high) {
            if (root.left != null) {
                root.val = left.val;
                root.left = left.left;
                root.right = left.right;
            } else {
                root = null;
            }
        }
        return root;
    }

    // LC961
    public int repeatedNTimes(int[] nums) {
        int len = nums.length;
        int n = len / 2;
        int[] freq = new int[10000];
        for (int i : nums) {
            freq[i]++;
            if (freq[i] == n) return i;
        }
        return -1;
    }

    // LC494
    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int sum = Arrays.stream(nums).sum();
        if (sum < target) return 0;
        int[][] dp = new int[n + 1][2 * sum + 1];
        // dp[i][j] 表示加入前i个数到达和j的方案数
        // 中点(0) 在 dp[sum]
        dp[0][sum] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= 2 * sum; j++) {
                int result = 0;
                if (j - nums[i - 1] >= 0) {
                    result += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] <= 2 * sum) {
                    result += dp[i - 1][j + nums[i - 1]];
                }
                dp[i][j] = result;
            }
        }
        return dp[n][sum + target];
    }

    // LC474
    public int findMaxForm(String[] strs, int m, int n) {
        int[] zeroCtr = new int[strs.length];
        int[] oneCtr = new int[strs.length];
        for (int i = 0; i < strs.length; i++) {
            for (char c : strs[i].toCharArray()) {
                if (c == '0') zeroCtr[i]++;
            }
            oneCtr[i] = strs[i].length() - zeroCtr[i];
        }
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= strs.length; i++) {
            int zeroNum = zeroCtr[i - 1];
            int oneNum = oneCtr[i - 1];
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    int result = dp[j][k];
                    if (j - zeroNum >= 0 && k - oneNum >= 0) {
                        result = Math.max(result, dp[j - zeroNum][k - oneNum] + 1);// 选择这一个字符串
                    }
                    dp[j][k] = result;
                }
            }
        }
        return dp[m][n];
    }

    // JZOF51 HARD
    public int reversePairs(int[] nums) {
        int n = nums.length;
        int[] sorted = new int[n];
        System.arraycopy(nums, 0, sorted, 0, n);
        Arrays.sort(sorted);
        for (int i = 0; i < n; i++) {
            nums[i] = Arrays.binarySearch(sorted, nums[i]) + 1;
        }
        int result = 0;
        BIT bit = new BIT(n);
        for (int i = n - 1; i >= 0; i--) {
            result += bit.query(nums[i] - 1);
            bit.update(nums[i], 1);
        }
        return result;

    }
}


class BIT {
    int len;
    int[] bit;

    public BIT(int n) {
        this.len = n;
        this.bit = new int[n + 1];
    }

    public int query(int idxFromOne) {
        int sum = 0;
        while (idxFromOne > 0) {
            sum += bit[idxFromOne];
            idxFromOne -= lowbit(idxFromOne);
        }
        return sum;
    }

    public void update(int idxFromOne, int delta) {
        while (idxFromOne <= len) {
            bit[idxFromOne] += delta;
            idxFromOne += lowbit(idxFromOne);
        }
    }


    private int lowbit(int x) {
        return x & (x ^ (x - 1));
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// LC1116 多线程
class ZeroEvenOdd {
    private int n;
    ReentrantLock lock = new ReentrantLock();
    Condition zero = lock.newCondition();
    Condition num = lock.newCondition();
    volatile int index = 0;
    boolean zeroTurn = true;

    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
        while (index < n) {
            boolean flag = false;
            while (!flag) {
                if (lock.tryLock()) { // 试着使用trylock
                    flag = true;
                    try {
                        while (!zeroTurn) {
                            zero.await();
                        }
                        printNumber.accept(0);
                        zeroTurn = false;
                        num.signalAll();
                        index++;
                    } finally {
                        lock.unlock();
                    }
                }
            }
        }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
        for (int i = 2; i <= n; i += 2) {
            boolean flag = false;
            while (!flag) {
                if (lock.tryLock()) {
                    flag = true;
                    try {
                        while (zeroTurn || index % 2 == 1) {
                            num.await();
                        }
                        printNumber.accept(i);
                        zeroTurn = true;
                        zero.signalAll();
                    } finally {
                        lock.unlock();
                    }
                }
            }
        }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
        for (int i = 1; i <= n; i += 2) {
            boolean flag = false;
            while (!flag) {
                if (lock.tryLock()) {
                    flag = true;
                    try {
                        while (zeroTurn || index % 2 == 0) {
                            num.await();
                        }
                        printNumber.accept(i);
                        zeroTurn = true;
                        zero.signalAll();
                    } finally {
                        lock.unlock();
                    }
                }
            }
        }
    }
}

// LC1115 多线程
class FooBar {
    private int n;
    volatile boolean isFoo;

    public FooBar(int n) {
        this.n = n;
        isFoo = true;
    }

    public void foo(Runnable printFoo) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            synchronized (this) {
                while (!isFoo)
                    this.wait();
                // printFoo.run() outputs "foo". Do not change or remove this line.
                printFoo.run();
                this.isFoo = false;
                this.notifyAll();
            }
        }
    }

    public void bar(Runnable printBar) throws InterruptedException {
        for (int i = 0; i < n; i++) {
            synchronized (this) {
                while (isFoo)
                    this.wait();
                // printBar.run() outputs "bar". Do not change or remove this line.
                printBar.run();
                this.isFoo = true;
                this.notifyAll();
            }
        }
    }
}