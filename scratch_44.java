import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.IntConsumer;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.err.println(s.removeDuplicateLetters("abacb"));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC316 LC1081 Stack
    public String removeDuplicateLetters(String s) {
        char[] cArr = s.toCharArray();
        int[] freq = new int[26];
        boolean[] visited = new boolean[26];
        for (char c : cArr) {
            freq[c - 'a']++;
        }

        StringBuilder sb = new StringBuilder();
        for (char c : cArr) {
            while (sb.length() != 0 && c < sb.charAt(sb.length() - 1) && !visited[c - 'a']) {
                char tmp = sb.charAt(sb.length() - 1);
                if (freq[tmp - 'a'] > 0) {
                    sb.deleteCharAt(sb.length() - 1);
                    visited[tmp - 'a'] = false;
                } else {
                    break;
                }
            }
            freq[c - 'a']--;
            if (!visited[c - 'a']) {
                sb.append(c);
                visited[c - 'a'] = true;
            }
        }
        return sb.toString();
    }

    // LC857
    public double mincostToHireWorkers(int[] quality, int[] wage, int k) {
        List<Worker> wl = new ArrayList<>(quality.length);
        int minWage = Integer.MAX_VALUE;
        for (int i = 0; i < quality.length; i++) {
            wl.add(new Worker(i, quality[i], wage[i]));
            minWage = Math.min(minWage, wage[i]);
        }
        if (k == 1) return minWage;
        wl.sort(Comparator.comparingDouble(o -> o.getRatio()));
        PriorityQueue<Worker> pq = new PriorityQueue<>(Comparator.comparingDouble(o -> -o.quality));

        int sumQuality = 0;
        double sumWage = Integer.MAX_VALUE;

        for (Worker worker : wl) {
            if (pq.size() < k - 1) {
                pq.offer(worker);
                sumQuality += worker.quality;
            } else {
                sumWage = Math.min(sumWage, (sumQuality + worker.quality) * worker.getRatio());
                if (worker.quality < pq.peek().quality) {
                    Worker tmp = pq.poll();
                    sumQuality -= tmp.quality;
                    sumQuality += worker.quality;
                    pq.offer(worker);
                }
            }
        }
        return sumWage;
    }

    class Worker {
        int id;
        int quality;
        int wage;
        double ratio;

        public Worker(int id, int quality, int wage) {
            this.id = id;
            this.quality = quality;
            this.wage = wage;
            this.ratio = (wage + 0.0) / (quality + 0.0);
        }

        double getRatio() {
            return ratio;
        }

    }

    // LC458
    public int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
        if (buckets == 1) return 0;
        int shot = (int) Math.floor((minutesToTest + 0.0) / (minutesToDie + 0.0));
        int low = 1, high = buckets;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if ((Math.pow(shot + 1, mid)) >= (double) buckets) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    // LC279 DP
    public int numSquares(int n) {
        int bound = (int) Math.sqrt(n);
        int[][] dp = new int[n + 1][bound + 1];
        // dp[i][j] 表示 i 用前j个完全平方数表示 最少需要几个
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= bound; j++) {
                dp[i][j] = dp[i - 1][j] + 1; // 总能表示成上一个数 + 1*1, 也就是多一个平方数
                if (j != 1) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][j - 1]);
                }
                if (i - j * j >= 0) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - j * j][j] + 1);
                }
            }
        }
        return dp[n][bound];
    }

    // LC943 Hard WA
    public String shortestSuperstring(String[] words) {
        List<String> wordList = new ArrayList<>();
        for (String word : words) {
            wordList.add(word);
        }

        while (wordList.size() >= 2) {
            int maxReduce = -1;
            int maxReduceLeftIdx = -1;
            int maxReduceRightIdx = -1;
            String reduced = "";
            for (int i = 0; i < wordList.size(); i++) {
                for (int j = 0; j < wordList.size(); j++) {
                    if (i != j) {
                        for (int k = 1; k <= wordList.get(j).length(); k++) {
                            if (wordList.get(i).length() >= k
                                    && wordList.get(i).lastIndexOf(wordList.get(j).substring(0, k)) == wordList.get(i).length() - k) {

                                int reduceCount = k;
                                if (reduceCount > maxReduce) {
                                    maxReduceLeftIdx = i;
                                    maxReduceRightIdx = j;
                                    maxReduce = reduceCount;
                                    reduced = wordList.get(i) + wordList.get(j).substring(k);
                                }
                            }
                        }
                    }
                }
            }
            if (maxReduce == -1) {
                return String.join("", wordList);
            }
            List<String> newList = new ArrayList<>();
            for (int i = 0; i < wordList.size(); i++) {
                if (i != maxReduceLeftIdx && i != maxReduceRightIdx) {
                    newList.add(wordList.get(i));
                }
            }
            if (!reduced.equals(""))
                newList.add(reduced);
            wordList = newList;
        }
        return String.join("", wordList);
    }

    // LC728
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> ans = new ArrayList<>();
        for (int i = left; i <= right; i++) {
            if (lc728Judge(i)) {
                ans.add(i);
            }
        }
        return ans;
    }

    private boolean lc728Judge(int i) {
        int orig = i;
        while (i != 0) {
            if (i % 10 == 0) return false;
            if (orig % (i % 10) != 0) return false;
            i /= 10;
        }
        return true;
    }

    // LC507
    public boolean checkPerfectNumber(int num) {
        if (num == 1) return false;
        int sqrt = (int) Math.sqrt(num);
        int sum = 1;
        for (int i = 2; i <= sqrt; i++) {
            if (num % i == 0) {
                sum += i;
                if (num / i != i) {
                    sum += num / i;
                }
            }
            if (sum > num) return false;
        }
        return sum == num;
    }

    // LC1260
    public List<List<Integer>> shiftGrid(int[][] grid, int k) {
        int r = grid.length;
        int c = grid[0].length;
        k %= r * c;
        List<List<Integer>> result = new ArrayList<>(r);
        for (int i = 0; i < r; i++) {
            List<Integer> tmp = new ArrayList<>(c);
            for (int j = 0; j < c; j++) {
                int ith = i * c + j;
                ith = (ith - k + r * c) % (r * c);
                tmp.add(grid[ith / c][ith % c]);
            }
            result.add(tmp);
        }
        return result;
    }

    // LC1863
    public int subsetXORSum(int[] nums) {
        int mask = 1 << nums.length;
        int sum = 0;
        for (int i = 0; i < mask; i++) {
            int xor = 0;
            for (int j = 0; j < nums.length; j++) {
                if (((i >> j) & 1) == 1) {
                    xor ^= nums[j];
                }
            }
            sum += xor;
        }
        return sum;
    }

    // LC817
    public int numComponents(ListNode head, int[] nums) {
        Set<Integer> g = new HashSet<>();
        for (int i : nums) {
            g.add(i);
        }
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        boolean inGear = false;
        ListNode ptr = dummy;
        int result = 0;
        while (ptr != null) {
            if (g.contains(ptr.val)) {
                if (!inGear) {
                    inGear = true;
                    result++;
                }
            } else {
                inGear = false;
            }
            ptr = ptr.next;
        }
        return result;
    }

    // LC983
    Set<Integer> lc983DaySet;

    public int mincostTickets(int[] days, int[] costs) {
        Integer[] memo = new Integer[366];
        lc983DaySet = new HashSet<>();
        for (int d : days) {
            lc983DaySet.add(d);
        }
        return lc983Helper(1, costs, memo);
    }

    private int lc983Helper(int ithDay, int[] costs, Integer[] memo) {
        if (ithDay > 365) {
            return 0;
        }
        if (memo[ithDay] != null) return memo[ithDay];

        if (lc983DaySet.contains(ithDay)) {
            memo[ithDay] = Math.min(Math.min(
                    costs[0] + lc983Helper(ithDay + 1, costs, memo),
                    costs[1] + lc983Helper(ithDay + 7, costs, memo)),
                    costs[2] + lc983Helper(ithDay + 30, costs, memo));
        } else {
            memo[ithDay] = lc983Helper(ithDay + 1, costs, memo);
        }
        return memo[ithDay];
    }

    // LC322
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] == Integer.MAX_VALUE / 2 ? -1 : dp[amount];
    }

    // LC518 Learn from Solution
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = dp[i] + dp[i - coin];
            }
        }
        return dp[amount];
    }

    // LC1367
    // Map<Pair<ListNode, TreeNode>, Boolean> lc1367Memo;

    public boolean isSubPath(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;
        //  lc1367Memo = new HashMap<>();
        return lc1367Helper(head, root, head) || isSubPath(head, root.left) || isSubPath(head, root.right);
    }

    private boolean lc1367Helper(ListNode cur, TreeNode root, ListNode head) {
        if (cur == null) return true;
        if (root == null) return false;
        //  Pair<ListNode, TreeNode> status = new Pair<>(cur, root);
        //  if (lc1367Memo.containsKey(status)) return lc1367Memo.get(status);
        if (cur.val == root.val) {
            boolean result = lc1367Helper(cur.next, root.left, head) || lc1367Helper(cur.next, root.right, head);
            //  lc1367Memo.put(status, result);
            return result;
        }
        return false;
    }

    // LC10 Hard ** DP Solution
    public boolean isMatch(String s, String p) {
        int n = s.length(), m = p.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int i = 0; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (p.charAt(j - 1) != '*') {
                    if (lc10Helper(s, p, i, j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                } else {
                    dp[i][j] = dp[i][j - 2];
                    if (lc10Helper(s, p, i, j - 1)) {
                        dp[i][j] = dp[i][j - 2] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[n][m];
    }

    private boolean lc10Helper(String s, String p, int a, int b) { // a b 是从1开始算的, 指代s p 的下标
        if (a == 0) {
            return false;
        }
        if (p.charAt(b - 1) == '.') {
            return true;
        }
        return s.charAt(a - 1) == p.charAt(b - 1);
    }

    // JZOF14 LC343
    public int cuttingRope(int n) {
        // 1:  避免出现
        // 2: 2 : 0
        // 3: 3 : 0
        // 4: 4 : 0
        // 5: 2 3 : 1
        // 7: 4 3 : 5
        // 8: 3 3 2 : 10
        // 9： 3 3 3 : 18
        // 10： 3 3 4 : 26

        if (n == 2) return 1;
        if (n == 3) return 2;
        if (n == 4) return 4;
        int result = 1;
        while (n % 3 != 0) {
            n -= 2;
            result *= 2;
        }
        while (n != 0) {
            result *= 3;
            n -= 3;
        }
        return result;
    }

    // LC1001
    // BIT EXPERIMENT
    int[] bit;

    public void initBit(int n) {
        this.bit = new int[n + 1];
    }

    public void initBit(int[] nums) {
        this.bit = new int[nums.length + 1];
        for (int i = 1; i <= nums.length; i++) {
            bit[i] += nums[i - 1];
            int up = i + (i & -i);
            if (up <= nums.length) bit[up] += bit[i];
        }
    }

    public void updateBit(int idxFromOne, int delta) {
        while (idxFromOne < bit.length) {
            bit[idxFromOne] += delta;
            idxFromOne += (idxFromOne & -idxFromOne);
        }
    }

    public int sumBit(int idxFromOne) {
        int sum = 0;
        while (idxFromOne > 0) {
            sum += bit[idxFromOne];
            idxFromOne -= (idxFromOne & -idxFromOne);
        }
        return sum;
    }

    public int getBit(int idxFromOne) {
        return sumBit(idxFromOne) - sumBit(idxFromOne - 1);
    }

    // LC1001 TLE
    public int[] gridIllumination(int n, int[][] lamps, int[][] queries) {

        Map<Integer, Set<Integer>> row = new HashMap<>();
        Map<Integer, Set<Integer>> col = new HashMap<>();
        Map<Integer, Set<Integer>> leftCross = new HashMap<>();
        Map<Integer, Set<Integer>> rightCross = new HashMap<>();

        for (int[] l : lamps) {
            row.putIfAbsent(l[0], new HashSet<>());
            row.get(l[0]).add(l[1]);

            col.putIfAbsent(l[1], new HashSet<>());
            col.get(l[1]).add(l[0]);

            leftCross.putIfAbsent(l[1] - l[0], new HashSet<>());
            leftCross.get(l[1] - l[0]).add(l[0]);

            rightCross.putIfAbsent(l[0] + l[1], new HashSet<>());
            rightCross.get(l[0] + l[1]).add(l[0]);
        }

        int[] result = new int[queries.length];

        int[][] dir = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        for (int i = 0; i < queries.length; i++) {
            int[] q = queries[i];
            int r = q[0];
            int c = q[1];
            if ((row.containsKey(r) && !row.get(r).isEmpty())
                    || (col.containsKey(c) && !col.get(c).isEmpty())
                    || (leftCross.containsKey(c - r) && !leftCross.get(c - r).isEmpty())
                    || (rightCross.containsKey(c + r) && !rightCross.get(c + r).isEmpty())) {
                result[i] = 1;
            }

            for (int[] d : dir) {
                int tmpRow = r + d[0];
                int tmpCol = c + d[1];
                if (row.containsKey(tmpRow)) {
                    row.get(tmpRow).remove(tmpCol);
                }
                if (col.containsKey(tmpCol)) {
                    col.get(tmpCol).remove(tmpRow);
                }
                if (leftCross.containsKey(tmpCol - tmpRow)) {
                    leftCross.get(tmpCol - tmpRow).remove(tmpRow);
                }
                if (rightCross.containsKey(tmpCol + tmpRow)) {
                    rightCross.get(tmpCol + tmpRow).remove(tmpRow);
                }
            }
        }
        return result;
    }

    // LC879 DP
    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        int mod = 1000000007;
        int[] numPeoplePrefix = new int[group.length + 1];
        for (int i = 1; i <= group.length; i++) {
            numPeoplePrefix[i] = numPeoplePrefix[i - 1] + group[i - 1];
        }
        int gLen = group.length;
        int[][] dp = new int[n + 1][minProfit + 1];
        dp[0][0] = 1; // 空集 最小利润为0 有一种方案
        for (int i = 1; i <= gLen; i++) {
            int peo = group[i - 1], pro = profit[i - 1];
            for (int j = n; j >= 0; j--) {
                for (int k = minProfit; k >= 0; k--) {
                    if (j < peo) {
                        dp[j][k] = dp[j][k];
                    } else {
                        dp[j][k] = (dp[j][k] + dp[j - peo][Math.max(0, k - pro)]) % mod;
                    }
                }
            }
        }
        int sum = 0;
        for (int i = 0; i <= n; i++) {
            sum = (sum + dp[i][minProfit]) % mod;
        }
        return sum;
    }


    // LC879 多重背包 DFS 记忆化
    final long lc879Mod = 1000000007;
    Long[][][] lc879Memo;

    public int profitableSchemesMemo(int n, int minProfit, int[] group, int[] profit) {
        int[] numPeoplePrefix = new int[group.length + 1];
        for (int i = 1; i <= group.length; i++) {
            numPeoplePrefix[i] = numPeoplePrefix[i - 1] + group[i - 1];
        }
        lc879Memo = new Long[group.length + 1][n + 1][minProfit + 1];
        return (int) (lc879Helper(0, n, 0, group, profit, minProfit, numPeoplePrefix, n) % lc879Mod);
    }

    private long lc879Helper(int curIdx, int leftPeople, int curProfit, int[] group, int[] profit, int minProfit, int[] numPeoplePrefix, int totalPeople) {
        if (curIdx == group.length) {
            return curProfit >= minProfit ? 1 : 0;
        }
        if (lc879Memo[curIdx][leftPeople][curProfit] != null) {
            return lc879Memo[curIdx][leftPeople][curProfit];
        }
        if (leftPeople >= 0 && leftPeople < group[curIdx]) { // 如果有人剩但又不足够当前项目, 则跳过当前项目
            lc879Memo[curIdx][leftPeople][curProfit] = lc879Helper(curIdx + 1, leftPeople, curProfit, group, profit, minProfit, numPeoplePrefix, totalPeople) % lc879Mod;
            return lc879Memo[curIdx][leftPeople][curProfit];
        }

        // 如果剩下的所有项目的所需人数比当前人数还要小, 则说明剩下的项目的"状态"中的"剩余人数"项应该是等价的, 规约为总人数即可
        if (leftPeople >= (numPeoplePrefix[group.length] - numPeoplePrefix[curIdx])) {
            leftPeople = totalPeople;
        }
        lc879Memo[curIdx][leftPeople][curProfit] =
                (lc879Helper(curIdx + 1, leftPeople - group[curIdx], Math.min(minProfit, curProfit + profit[curIdx]), group, profit, minProfit, numPeoplePrefix, totalPeople) % lc879Mod
                        + lc879Helper(curIdx + 1, leftPeople, curProfit, group, profit, minProfit, numPeoplePrefix, totalPeople) % lc879Mod) % lc879Mod;
        return lc879Memo[curIdx][leftPeople][curProfit];
    }


    // LC982 使用二进制子集算法优化
    public int countTriplets(int[] nums) {
        Map<Integer, Integer> m = new HashMap<>();
        int max = 2;
        for (int i : nums) {
            while (max <= i) {
                max = max << 1;
            }
            for (int j : nums) {
                m.put(i & j, m.getOrDefault(i & j, 0) + 1);
            }
        }
        int result = 0;
        for (int i : nums) {
            int tmp = max - 1 - i;
            for (int j = tmp; j != 0; j = (j - 1) & tmp) {
                result += m.getOrDefault(j, 0);
            }
            result += m.getOrDefault(0, 0);
        }
        return result;
    }

    // LC357 DP
    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 1;
        if (n == 1) return 10;
        if (n == 2) return 91;
        if (n >= 10) n = 10;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 0;
        dp[2] = 9;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] * 10 + (9 * (int) Math.pow(10, i - 2) - dp[i - 1]) * (i - 1);
        }
        int result = 0;
        for (int i : dp) {
            result += i;
        }
        return (int) Math.pow(10, n) - result;
    }

    // JZOF16 快速幂
    public double myPow(double x, int n) {
        if (x == 0d) return 0d;
        boolean baseNegFlag = x < 0;
        boolean expNegFlag = n < 0;
        long longN = n;
        x = Math.abs(x);
        longN = Math.abs(longN);
        Map<Integer, Double> m = new HashMap<>();
        int logN = (int) (Math.log(longN) / Math.log(2));
        m.put(0, x);
        for (int i = 1; i <= logN + 1; i++, x *= x) {
            m.put(i, x * x);
        }
        double result = 1;
        for (int i = 0; i < 32; i++) {
            if (((longN >> i) & 1) == 1) {
                result *= m.get(i);
            }
        }
        if (expNegFlag) result = 1 / result;
        if (baseNegFlag && longN % 2 == 1) result = -result;

        return result;
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

// LC1656
class OrderedStream {
    String[] m;
    int ptr;
    int n;

    public OrderedStream(int n) {
        this.n = n;
        this.ptr = 1;
        m = new String[n + 1];
    }

    public List<String> insert(int idKey, String value) {
        m[idKey] = value;
        List<String> result = new ArrayList<>();
        for (int i = ptr; i <= n; i++, ptr++) {
            if (m[i] == null) break;
            result.add(m[i]);
        }
        return result;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}