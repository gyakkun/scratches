import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
//        int[][] zeroes = new int[][]{{0, 1, 2, 0}, {3, 4, 5, 2}, {1, 3, 1, 5}};
//        s.setZeroes(zeroes);
        System.err.println(s.longestPalindrome("abb"));
    }

    // 三壶问题,
    public int tripleBottle(int n, int a, int b) {
        if (n % 2 == 1 || (n / 2) % gcd(a, b) != 0) return -1;
        int[] three = new int[3];
        int half = n / 2;
        three[0] = n;
        int layer = 0;
        Deque<int[]> q = new LinkedList<>();
        q.offer(three);
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] tmp = q.poll();
                int nn = tmp[0], aa = tmp[1], bb = tmp[2];
                // 总共有多少种操作?
                // TODO
            }
        }

        return -1;
    }

    public int gcd(int a, int b) {
        return a % b != 0 ? gcd(b, a % b) : b;
    }

    // LC5 最长回文子串, 递归, 记忆, 慢

    private int longestLeft;
    private int longestRight;
    private int longestPalindromeLength;

    public String longestPalindrome(String s) {
        int n = s.length();
        longestPalindromeLength = 0;
        Boolean[][] judge = new Boolean[n + 1][n + 1];
        longestPalindromeJudgeRecursive(s, judge, 0, n - 1);
        return s.substring(longestLeft, longestRight + 1);
    }

    private boolean longestPalindromeJudgeRecursive(String s, Boolean[][] matrix, int l, int h) {
        if (l == h) {
            matrix[l][h] = true;
            if (longestPalindromeLength == 0) {
                longestPalindromeLength = 1;
                longestLeft = l;
                longestRight = h;
            }
            return true;
        }
        if (l > h) {
            return false;
        }
        if (matrix[l][h] != null) {
            return matrix[l][h];
        }
        if (l + 1 == h) {
            matrix[l][h] = s.charAt(l) == s.charAt(h);
            if (matrix[l][h] && longestPalindromeLength < 2) {
                longestPalindromeLength = 2;
                longestLeft = l;
                longestRight = h;
            }
            return matrix[l][h];
        } else {
            longestPalindromeJudgeRecursive(s, matrix, l, h - 1);
            longestPalindromeJudgeRecursive(s, matrix, l + 1, h);
            longestPalindromeJudgeRecursive(s, matrix, l + 1, h - 1);
            matrix[l][h] = s.charAt(l) == s.charAt(h) && matrix[l + 1][h - 1];
            if (matrix[l][h] && longestPalindromeLength < (h - l + 1)) {
                longestPalindromeLength = h - l + 1;
                longestLeft = l;
                longestRight = h;
            }
            return matrix[l][h];
        }
    }

    // LC516 最长回文子序列
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        // 用memo[l][h]记录[l,h]的最大回文子序列长度
        Integer[][] memo = new Integer[n + 1][n + 1];

        return longestPalindromeSubseqRecursive(s, memo, 0, n - 1);
    }

    private int longestPalindromeSubseqRecursive(String s, Integer[][] memo, int l, int h) {
        if (l == h) return 1;
        if (l > h) return 0;
        if (memo[l][h] != null) {
            return memo[l][h];
        }
        int res = 0; // 使用临时变量 避免赋值为null的问题
        if (s.charAt(l) == s.charAt(h)) {
            res = longestPalindromeSubseqRecursive(s, memo, l + 1, h - 1) + 2;
        } else {
            res = Math.max(res, longestPalindromeSubseqRecursive(s, memo, l + 1, h));
            res = Math.max(res, longestPalindromeSubseqRecursive(s, memo, l, h - 1));
        }
        memo[l][h] = res; // 避免赋值为null的问题
        return memo[l][h];
    }

    // LC341

    // This is the interface that allows for creating nested lists.
    // You should not implement it, or speculate about its implementation
    public interface NestedInteger {

        // @return true if this NestedInteger holds a single integer, rather than a nested list.
        public boolean isInteger();

        // @return the single integer that this NestedInteger holds, if it holds a single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger();

        // @return the nested list that this NestedInteger holds, if it holds a nested list
        // Return null if this NestedInteger holds a single integer
        public List<NestedInteger> getList();
    }

    public class NestedIterator implements Iterator<Integer> {

        Deque<Iterator<NestedInteger>> stack;

        public NestedIterator(List<NestedInteger> nestedList) {
            stack = new LinkedList<>();
            stack.push(nestedList.iterator());
        }

        @Override
        public Integer next() {
            return stack.peek().next().getInteger();
        }

        @Override
        public boolean hasNext() {
            while (!stack.isEmpty()) {
                Iterator<NestedInteger> it = stack.peek();
                if (!it.hasNext()) {
                    stack.pop();
                    continue;
                }
                NestedInteger ni = it.next();
                if (ni.isInteger()) {
                    List<NestedInteger> list = new ArrayList<>(1);
                    list.add(ni);
                    stack.push(list.iterator());
                    return true;
                }
                stack.push(ni.getList().iterator());
            }
            return false;
        }
    }


    // LC73, O(1)空间
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        if (m == 0 || n == 0) return;
        // matrix[0][0] == 0

        boolean zeroFirstRow = false;
        boolean zeroFirstColumn = false;

        // 处理第0列
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                zeroFirstColumn = true;
                break;
            }
        }

        // 处理第0行
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                zeroFirstRow = true;
                break;
            }
        }

        // 非首行、首列判断
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        // 非首列操作
        for (int i = 1; i < m; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        // 非首行操作
        for (int j = 1; j < n; j++) {
            if (matrix[0][j] == 0) {
                for (int i = 0; i < m; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
        // 首行操作
        if (zeroFirstRow) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
        if (zeroFirstColumn) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }


    // Find maximum possible stolen value from houses, House thief, 偷房子
    public int maxLoot(int[] houseVal) {
        int n = houseVal.length;

        if (n == 0) return 0;
        if (n == 1) return houseVal[1];
        if (n == 2) return Math.max(houseVal[0], houseVal[1]);

        int[][] dp = new int[n][2];
        // dp[i][0] 表示偷前i(i从0开始)个房子时候能得到的最大价值, 此时从最后一个房间没被偷
        // dp[i][1] 此时最后一个房间被偷
        dp[0][0] = 0;
        dp[0][1] = houseVal[0];
        dp[1][0] = houseVal[0];
        dp[1][1] = houseVal[1];
        for (int i = 2; i < n; i++) {
            dp[i][1] = Math.max(dp[i - 2][1] + houseVal[i], dp[i - 2][0] + houseVal[i]);
            dp[i][0] = Math.max(dp[i - 1][1], dp[i - 1][0]);
        }

        return Math.max(dp[n - 1][0], dp[n - 1][1]);
    }

    // Minimum jumps to reach the end，蛙跳最小步数问题, 动态规划
    public int minJumpsDP(int[] arr) {
        int n = arr.length;
        if (arr[0] == 0 || arr.length == 0) return -1;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        // dp[i] 表示到数组坐标i的最小步数
        // 转移方程: 对于每一个i, 遍历 0 < j < i, 如果可从j抵达i, 动态更新抵达i的最小步数
        // dp[i] = Math.min(dp[i], dp[j]+1)

        for (int i = 1; i < n; i++) {
            dp[i] = Integer.MAX_VALUE / 2;
            for (int j = 0; j < i; j++) {
                if (i <= j + arr[i] && dp[j] != Integer.MAX_VALUE / 2) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                    // 因为从数组的最左侧开始遍历, 一旦找到符合条件的左侧, 便一定是最小的步数, 此时剪枝即可
                    break;
                }
            }
        }

        return dp[n - 1];
    }

    // Minimum jumps to reach the end，蛙跳最小步数问题, 递归, 加记忆数组
    // https://www.geeksforgeeks.org/minimum-number-of-jumps-to-reach-end-of-a-given-array
    public int minJumps(int[] arr) {
        Integer[][] memo = new Integer[arr.length + 1][arr.length + 1];
        return minJumpsRecursive(0, arr.length - 1, arr, memo);
    }

    private int minJumpsRecursive(int l, int h, int[] arr, Integer[][] memo) throws IndexOutOfBoundsException {
        if (l < 0 || h > arr.length || l > h) {
            throw new IndexOutOfBoundsException("out of bound");
        }
        if (l == h) {
            return 0;
        }
        if (arr[l] == 0) {
            return Integer.MAX_VALUE / 2;
        }
        int min = Integer.MAX_VALUE / 2;
        for (int i = l + 1; i <= h && i <= arr[l] + l; i++) {
            int jump;
            if (memo[l][h] != null) {
                jump = memo[l][h];
            } else {
                jump = minJumpsRecursive(i, h, arr, memo);
                memo[i][h] = jump;
            }
            if (jump != Integer.MAX_VALUE / 2 && jump + 1 < min) {
                min = jump + 1;
            }
        }
        return min;
    }

    // Number factors，分解因子问题, 递归+记忆数组
    // https://blog.csdn.net/qq_42898536/article/details/109523993
    public int numberOfFactors(int n) {
        int[] memo = new int[n + 1];
        return numberOfFactorsRecursive(n, n, memo);
    }

    public int numberOfFactorsRecursive(int n, int target, int[] memo) {
        if (n > target) {
            return 0;
        }
        if (memo[n] != 0) {
            return memo[n];
        }
        int i = 2; // 从2开始
        memo[n] = 1; // 初始为1, 即1*n = n 这种情况, 否则1会重复计算
        for (; i * i < n; i++) {
            if (n % i == 0) {
                memo[n] += numberOfFactorsRecursive(i, target, memo) + numberOfFactorsRecursive(n / i, target, memo);
            }
        }
        if (i * i == n) memo[n] += numberOfFactorsRecursive(i, target, memo);
        return memo[n];
    }

    public int hammingWeight(int n) {
        return Integer.bitCount(n);
    }


    // Rod Cutting, 要存状态(切法), 思路和找硬币类似。暂未知道对错
    public List<Integer> rodCutting(int[] prices, int rodLength) {
        // 定义: price[i]表示长度为i+1的钢条的价格
        RCStatus[] dp = new RCStatus[rodLength + 1];
        // 初始化
        for (int i = 0; i <= rodLength; i++) {
            List<Integer> tmp = new ArrayList<>(i);
            for (int j = 0; j < i; j++) {
                tmp.add(1);
            }
            dp[i] = new RCStatus(tmp, prices[0] * i);
        }

        for (int i = 1; i <= rodLength; i++) {
            for (int j = 0; j < prices.length; j++) {
                if (i - (j + 1) > 0) {
                    // 如果dp单纯存利润
                    // dp[i] = Math.max(dp[i-j]+prices[j], dp[i])
                    if (dp[i].revenue < (dp[i - (j + 1)].revenue + prices[j])) {
                        List<Integer> list = new LinkedList<>(dp[i - (j + 1)].howToCut);
                        list.add(j + 1);
                        dp[i] = new RCStatus(list, dp[i - (j + 1)].revenue + prices[j]);
                    }
                }
            }
        }
        return dp[rodLength].howToCut;
    }

    class RCStatus {
        List<Integer> howToCut;
        int revenue;

        RCStatus(List<Integer> list, int i) {
            this.howToCut = list;
            this.revenue = i;
        }
    }

    // LC983 Top down
    public int minCostTicketsTopDown(int[] days, int[] costs) {
        int len = days[days.length - 1];
        int[] dp = new int[len + 1];
        int a, b, c;
        for (int i = 0; i < days.length; i++) {
            dp[days[i]] = -1;//标记当天要去旅行
        }
        for (int i = 1; i <= len; i++) {
            if (dp[i] == 0) {
                dp[i] = dp[i - 1];
            } else {
                a = dp[i - 1] + costs[0];//当天旅行
                //买7天的
                if (i - 7 >= 0) {
                    b = dp[i - 7] + costs[1];
                } else {
                    b = costs[1];
                }
                //买30天的
                if (i - 30 >= 0) {
                    c = dp[i - 30] + costs[2];
                } else {
                    c = costs[2];
                }
                dp[i] = Math.min(a, Math.min(b, c));
            }
        }
        return dp[len];
    }

    // LC983
    public int minCostTickets(int[] days, int[] costs) {
        Integer[] memo = new Integer[366];
        Set<Integer> set = new HashSet<>();
        for (int d : days) {
            set.add(d);
        }
        return minCostTicketsRecursive(1, memo, set, costs);
    }

    private int minCostTicketsRecursive(int day, Integer[] memo, Set<Integer> daySet, int[] costs) {
        if (day > 365) {
            return 0;
        }
        if (memo[day] != null) {
            return memo[day];
        }
        if (daySet.contains(day)) {
            memo[day] = Math.min(Math.min(minCostTicketsRecursive(day + 1, memo, daySet, costs) + costs[0],
                    minCostTicketsRecursive(day + 7, memo, daySet, costs) + costs[1]),
                    minCostTicketsRecursive(day + 30, memo, daySet, costs) + costs[2]);
        } else {
            memo[day] = minCostTicketsRecursive(day + 1, memo, daySet, costs);
        }
        return memo[day];
    }


    // LC322
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i : coins) {
            if (i <= amount)
                dp[i] = 1;
        }
        // dp[i] = Math.min(dp[i-coins[j]]+1, dp[i])
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i - coin > 0) {
                    dp[i] = Math.min(dp[i - coin] + 1, dp[i]);
                }
            }
        }

        return dp[amount] == (Integer.MAX_VALUE / 2) ? -1 : dp[amount];
    }

    // LC494, DP 题解
    public int findTargetSumWaysLC(int[] array, int target) {
        int[][] dp = new int[array.length][2001];
        dp[0][array[0] + 1000] = 1;
        dp[0][-array[0] + 1000] += 1;
        for (int i = 1; i < array.length; i++) {
            for (int sum = -1000; sum <= 1000; sum++) {
                if (sum + 1000 - array[i] > 0 && sum + 1000 + array[i] < 2001) {
                    dp[i][sum + 1000] = dp[i - 1][sum + 1000 - array[i]] + dp[i - 1][sum + 1000 + array[i]];
                }
            }
        }
        return target > 1000 ? 0 : dp[array.length - 1][target + 1000];
    }


    // Subset Sum, DP, Top Down
    public boolean existSubsetSumDP(int[] array, int target) {
        // dp[i][j]表示数组中 前i个数组成的全集 中是否有 和为j的子集
        // 状态转移:
        //  1) if(dp[i-1][j]) dp[i][j]=true
        //  2) else dp[i][j] = dp[i-1][j-array[i-1]]
        // 边界条件: 空集, 即dp[0][j] = false, dp[i][0]=true
        boolean[][] dp = new boolean[array.length + 1][target + 1];
        for (int i = 0; i <= array.length; i++) {
            dp[i][0] = true;
        }
        for (int i = 0; i <= array.length; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= array[i - 1]) { // 保证不越界
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - array[i - 1]];
                }
            }
            if (dp[i][target] == true) return true;
        }
        return false;
    }

    // LC494, 相当于遍历了决策树的暴力法
    public int findTargetSumWaysBruteForce(int[] array, int target) {
        int result = 0;
        int n = array.length;
        int fullSet = (1 << n) - 1;
        int[] sums = new int[1 << n];
        Arrays.fill(sums, Integer.MIN_VALUE);
        for (int subset = fullSet; subset >= 0; subset--) {
            // 补集
            int sup = subset ^ fullSet;
            if (sumsSubset(array, subset, sums) - sumsSubset(array, sup, sums) == target) {
                result++;
            }
        }
        return result;
    }

    // Subset sum

    public boolean existSubsetSum(int[] array, int target) {

        int n = array.length;
        int[] sums = new int[1 << n];
        Arrays.fill(sums, -1);
        int subset = (1 << n) - 1;
        for (int i = subset; i != 0; i--) {
            if (sumsSubset(array, i, sums) == target) {
                return true;
            }
        }

        return false;
    }

    // LC39, 同一个元素可以取不止一次, 待解决

    public List<List<Integer>> combinationSum(int[] array, int target) {
        int n = array.length;
        int[] sums = new int[1 << n];
        Arrays.fill(sums, -1);
        int fullSet = (1 << n) - 1;
        List<List<Integer>> result = new LinkedList<>();
        for (int subset = fullSet; subset != 0; subset--) {
            if (sumsSubset(array, subset, sums) == target) {
                List<Integer> tmp = new ArrayList<>();
                for (int j = 0; j < array.length; j++) {
                    if (((subset >> j) & 1) == 1) {
                        tmp.add(array[j]);
                    }
                }
                result.add(tmp);
            }
        }
        return result;
    }


    // 求所有子集和的递归方法, 带记忆, 较快
    private int sumsSubset(int[] array, int subsetBitmask, int[] sums) {
        if (sums[subsetBitmask] != Integer.MIN_VALUE) {
            return sums[subsetBitmask];
        }
        if (subsetBitmask == 0) {
            sums[0] = 0;
            return 0;
        }
        if (Integer.bitCount(subsetBitmask) == 1) {
            for (int i = 0; i < array.length; i++) {
                if (subsetBitmask >> i == 1) {
                    sums[subsetBitmask] = array[i];
                    return sums[subsetBitmask];
                }
            }
        }

        int A = subsetBitmask;
        // 求集合A所有元素的和
        // <=> 集合A的真子集a的元素的和 + a关于A的补集b的元素的和

        // 产生一个真子集a:
        int a = (A - 1) & A;
        // a关于A的补集b
        int b = a ^ A;
        sums[subsetBitmask] = sumsSubset(array, a, sums) + sumsSubset(array, b, sums);

        return sums[subsetBitmask];
    }

    // LC115 top down

    public int numDistinctTopDown(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        // dp[i][j] 表示 在s的前i个字符中, 出现的(t的前j个字符的子序列)的个数
        // dp[i][j] =
        //  1) s[i] == t[j], dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
        //  2) s[i] != t[j], dp[i][j] = dp[i-1][j]
        //
        // dp[i][0] = 1
        // dp[0][j] = 0
        for (int i = 0; i <= s.length(); i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[s.length()][t.length()];
    }

    // LC115 bottom up

    public int numDistinctBottomUp(String s, String t) {
        int[][] memo = new int[s.length()][t.length()];
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                memo[i][j] = -1;
            }
        }
        return numDistinctRecursive(s.length() - 1, t.length() - 1, memo, s, t);
    }

    private int numDistinctRecursive(int i, int j, int[][] memo, String s, String t) {
        // 目标: 返回s从0到i中, 出现(t从0到j的子序列)的次数
        if (j < 0) {
            return 1;
        }
        if (i < 0) {
            return 0;
        }
        if (memo[i][j] != -1) return memo[i][j];
        if (s.charAt(i) == t.charAt(j)) {
            memo[i][j] = numDistinctRecursive(i - 1, j - 1, memo, s, t) + numDistinctRecursive(i - 1, j, memo, s, t);
        } else {
            memo[i][j] = numDistinctRecursive(i - 1, j, memo, s, t);
        }
        return memo[i][j];
    }

    // LC560
    public int subarraySum(int[] nums, int k) {
        int[] prefix = new int[nums.length + 1];
        Map<Integer, Integer> m = new HashMap<>();
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }
        for (int i = 0; i <= nums.length; i++) {
            // 目标 : prefix[i] - prefix[j] = k , j<i
            // <=>   prefix[i] - k = prefix[j]
            // 考虑 i 的递增性, 每次判断完后往合集里加入i即可
            result += m.getOrDefault(prefix[i] - k, 0);
            m.put(prefix[i], m.getOrDefault(prefix[i], 0) + 1);
        }
        return result;
    }
}

class SolutionDFS {
    // 存储有向图
    List<List<Integer>> edges;
    // 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    int[] visited;
    // 用数组来模拟栈，下标 n-1 为栈底，0 为栈顶
    int[] result;
    // 判断有向图中是否有环
    boolean valid = true;
    // 栈下标
    int index;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<>());
        }
        visited = new int[numCourses];
        result = new int[numCourses];
        index = numCourses - 1;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        // 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for (int i = 0; i < numCourses && valid; ++i) {
            if (visited[i] == 0) {
                dfs(i);
            }
        }
        if (!valid) {
            return new int[0];
        }
        // 如果没有环，那么就有拓扑排序
        return result;
    }

    public void dfs(int u) {
        // 将节点标记为「搜索中」
        visited[u] = 1;
        // 搜索其相邻节点
        // 只要发现有环，立刻停止搜索
        for (int v : edges.get(u)) {
            // 如果「未搜索」那么搜索相邻节点
            if (visited[v] == 0) {
                dfs(v);
                if (!valid) {
                    return;
                }
            }
            // 如果「搜索中」说明找到了环
            else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        // 将节点标记为「已完成」
        visited[u] = 2;
        // 将节点入栈
        result[index--] = u;
    }
}

class SolutionBFS {
    // 存储有向图
    List<List<Integer>> edges;
    // 存储每个节点的入度
    int[] indeg;
    // 存储答案
    int[] result;
    // 答案下标
    int index;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<>());
        }
        indeg = new int[numCourses];
        result = new int[numCourses];
        index = 0;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            ++indeg[info[0]];
        }

        Queue<Integer> queue = new LinkedList<>();
        // 将所有入度为 0 的节点放入队列中
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        while (!queue.isEmpty()) {
            // 从队首取出一个节点
            int u = queue.poll();
            // 放入答案中
            result[index++] = u;
            for (int v : edges.get(u)) {
                --indeg[v];
                // 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
                if (indeg[v] == 0) {
                    queue.offer(v);
                }
            }
        }

        if (index != numCourses) {
            return new int[0];
        }
        return result;
    }
}
