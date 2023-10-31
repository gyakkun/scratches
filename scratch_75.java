import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Solution {
    public static void main(String[] args) {
        var s = new Solution();
        long timing = System.currentTimeMillis();
        System.err.println(s.maxSizeSlices(new int[]{1, 2, 3, 4, 5, 6}));
        timing = System.currentTimeMillis() - timing;
        System.err.println(timing + "ms");
    }

    // LC2544
    public int alternateDigitSum(int n) {
        int res = 0, numDigit = 0;
        int tmp = n;
        do {
            tmp /= 10;
            numDigit++;
        } while (tmp > 0);

        int sign = 0;
        if (numDigit % 2 == 0) {
            sign = -1;
        } else {
            sign = 1;
        }
        tmp = n;
        while(tmp!=0) {
            res += sign * (tmp % 10);
            sign *= -1;
            tmp /= 10;
        }
        return res;
    }

    // LC1388 Hard ** DP
    Integer[][] lc1388Memo;

    public int maxSizeSlices(int[] slices) {
        int len = slices.length;
        int res = Integer.MIN_VALUE;
        // 不选第一个
        lc1388Memo = new Integer[len + 1][len + 1];
        res = Math.max(res, lc1388Helper(len - 1, len / 3, slices, true, false));

        // 不选最后一个
        lc1388Memo = new Integer[len + 1][len + 1];
        res = Math.max(res, lc1388Helper(len - 1, len / 3, slices, false, true));

        return res;
    }

    private int lc1388Helper(int rangeInclusive, int selected, int[] slices, boolean shouldChooseFirst, boolean shouldChooseLast) {
        if (lc1388Memo[rangeInclusive][selected] != null) return lc1388Memo[rangeInclusive][selected];
        // while rangeIncl < 2 or selected = 0
        if (selected == 0) return 0;
        if (rangeInclusive == 0 && selected == 1) {
            if (!shouldChooseFirst) return Integer.MIN_VALUE;
            return slices[0];
        }
        if (rangeInclusive == 1 && selected == 1) {
            if (!shouldChooseFirst) return slices[1];
            return Math.max(slices[0], slices[1]);
        }
        if (rangeInclusive < 2 && selected >= 2) return Integer.MIN_VALUE;

        int res = Integer.MIN_VALUE;
        int choose = Integer.MIN_VALUE;
        // if choose idx=rangeIncl
        if (rangeInclusive == slices.length - 1 && !shouldChooseLast) {
            ;
        } else {
            choose = slices[rangeInclusive] + lc1388Helper(rangeInclusive - 2, selected - 1, slices, shouldChooseFirst, shouldChooseLast);
        }
        int notChoose = lc1388Helper(rangeInclusive - 1, selected, slices, shouldChooseFirst, shouldChooseLast);
        res = Math.max(choose, notChoose);
        return lc1388Memo[rangeInclusive][selected] = res;
    }

    // LC2003 Hard **c
    public int[] smallestMissingValueSubtree(int[] parents, int[] nums) {
        int[] res = new int[parents.length];
        Arrays.fill(res, 1);
        Map<Integer, Set<Integer>> children = new HashMap<>();
        // Build children map
        for (int i = 0; i < parents.length; i++) {
            children.computeIfAbsent(parents[i], j -> new HashSet<>())
                    .add(i);
        }
        // bottom up
        List<Integer> chain = new ArrayList<>();
        int oneLoc = -1;
        for (int i = 0; i < parents.length; i++) {
            if (nums[i] == 1) {
                oneLoc = i;
                break;
            }
        }
        int cur = oneLoc;
        Set<Integer> visited = new HashSet<>(), gene = new HashSet<>();
        int expectedLackNode = 1;
        while (cur != -1) {
            lc2003Helper(cur, nums, children, visited, gene);
            while (gene.contains(expectedLackNode)) expectedLackNode++;
            res[cur] = expectedLackNode;
            cur = parents[cur];
        }
        return res;
    }

    private void lc2003Helper(int cur, int[] nums,
                              Map<Integer, Set<Integer>> children,
                              Set<Integer> visited, Set<Integer> gene) {
        if (visited.contains(cur)) return;
        visited.add(cur);
        gene.add(nums[cur]);
        Set<Integer> ch = children.getOrDefault(cur, new HashSet<>());
        for (int c : ch) {
            lc2003Helper(c, nums, children, visited, gene);
        }
    }

    // 蒙特卡洛
    public double monteCarlo() {
        // range1 [-50,50]
        // range2 [-45,45]
        // min 0, max 45, 10 bars
        int[] freqMap = new int[10];
        Random r = new Random();
        double avgAlert = 0d;
        double avgDelay = 0d;
        double maxAlert = 0d;
        int zeroCount = 0;
        for (int i = 0; i < 10000000; i++) {
            double stir1 = r.nextDouble() * 100 - 50; // server check
            double stir2 = r.nextDouble() * 90 - 45; // client check
            while (stir1 < 120 * 60) stir1 += 50;
            while (stir2 < 120 * 60) stir2 += 45;
            double alertingTime = 0d;
            if (stir1 >= stir2) {
                alertingTime += stir1 - stir2;
            }
            if (alertingTime - 0d <= 1e-9) zeroCount++;
            avgAlert = (avgAlert * i + alertingTime) / (i + 1);
            avgDelay = (avgDelay * i + (stir1 - 120 * 60)) / (i + 1);
            maxAlert = Math.max(maxAlert, alertingTime);
            freqMap[(int) (alertingTime / 5d)]++;
        }
        System.err.println("Zero count " + zeroCount);
        return avgAlert; // ~9.25s, [0, 0s+,5s+...45s+] = [45%, 15%,9.4%,8.3%,7.2%,6.1%,5.0%,3.9%,2.8%,1.7%,0.56%]
    }

    // LC1003 **
    public boolean isValid(String s) {
        char[] ca = s.toCharArray();
        Deque<Character> stack = new LinkedList<>();
        for (char c : ca) {
            stack.push(c);
            if (stack.size() >= 3) {
                char p1 = stack.pop();
                char p2 = stack.pop();
                char p3 = stack.pop();
                if (p3 == 'a' && p2 == 'b' && p1 == 'c') continue;
                stack.push(p3);
                stack.push(p2);
                stack.push(p1);
            }
        }
        return stack.isEmpty();
    }

    // LC1499 Hard ** 单调队列
    public int findMaxValueOfEquation(int[][] points, int k) {
        // points按照[x,y] 中x的升序排列
        // 求0<=i<j<len, yi+yj+|xi-xj|的最大值, 其中|xi-xj|<=k

        // yi + yj + |xi-xj| = xj + yj + (yi - xi)
        Deque<int[]> dq = new LinkedList<>();
        int res = Integer.MIN_VALUE;
        for (int[] j : points) {
            int xj = j[0], yj = j[1];
            // 出队所有不满足 xj-xi 的元素
            while (!dq.isEmpty() && xj - dq.peekFirst()[1] > k) dq.pollFirst();
            if (!dq.isEmpty()) {
                res = Math.max(res, xj + yj + dq.peekFirst()[0]);
            }
            while (!dq.isEmpty() && yj - xj >= dq.peekLast()[0]) dq.pollLast();
            dq.offer(new int[]{yj - xj, xj});
        }
        return res;
    }

    // LC275
    public int hIndex2(int[] citations) {
        int len = citations.length;
        int[] m = new int[citations[len - 1] + 1];
        Arrays.fill(m, -1);
        for (int i = len - 1; i >= 0; i--) {
            m[citations[i]] = i;
        }
        int j = citations[len - 1];
        while (j >= 0) {
            int k = m[j];
            while (--j >= 0 && m[j] == -1) m[j] = k;
        }
        // H指数最高不超过最大引用次数, 最小为0, 考虑单调性, 可二分
        int lo = 0, hi = citations[len - 1];
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2; // 找最大值, 需要取上届
            // how to judge?
            // find the largest index that have mid citation
            int idx = m[mid]; // 找出大于等于mid的引用次数出现的最小下标
            int sum = len - idx;
            if (sum >= mid) { // valid candidate! can find in high half
                lo = mid;
            } else { // not enough, should find in the low half
                hi = mid - 1;
            }
        }
        return lo;
    }

    // LC274
    public int hIndex(int[] citations) {
        Map<Integer, Integer> collect = Arrays.stream(citations).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        TreeMap<Integer, Integer> tm = new TreeMap<>(collect); // [引用次数, 篇数]
        int res = 0, accu = 0;
        for (Map.Entry<Integer, Integer> e : tm.descendingMap().entrySet()) {
            accu += e.getValue();
            int h = Math.min(e.getKey(), accu);
            if (h > res) res = h;
        }
        return res;
    }

    // LC2558
    public long pickGifts(int[] gifts, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.reverseOrder());
        for (int i : gifts) pq.offer(i);
        for (int i = 0; i < k; i++) {
            int p = pq.poll();
            p = (int) Math.sqrt(p);
            pq.offer(p);
        }
        long res = 0;
        while (!pq.isEmpty()) {
            res += pq.poll();
        }
        return res;
    }

    // LC1465
    public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        List<Integer> hlist = new ArrayList<>(Arrays.stream(horizontalCuts).boxed().sorted().toList());
        List<Integer> vlist = new ArrayList<>(Arrays.stream(verticalCuts).boxed().sorted().toList());
        hlist.add(0, 0);
        hlist.add(h);
        vlist.add(0, 0);
        vlist.add(w);
        int maxHDiff = Integer.MIN_VALUE, maxVDiff = Integer.MIN_VALUE;
        for (int i = 0; i < hlist.size() - 1; i++) {
            int left = hlist.get(i), right = hlist.get(i + 1);
            int diff = right - left;
            if (diff > maxHDiff) {
                maxHDiff = diff;
            }
        }
        for (int i = 0; i < vlist.size() - 1; i++) {
            int left = vlist.get(i), right = vlist.get(i + 1);
            int diff = right - left;
            if (diff > maxVDiff) {
                maxVDiff = diff;
            }
        }
        long res = (long) maxHDiff * (long) maxVDiff;
        res %= 1000000007;
        return (int) res;
    }

    // LC2520
    public int countDigits(int num) {
        int victim = num;
        int res = 0;
        while (victim > 0) {
            int digit = victim % 10;
            if (num % digit == 0) res++;
            victim /= 10;
        }
        return res;
    }

    // LC2772 ** 差分数组 忘光了
    public boolean checkArray(int[] nums, int k) {
        int[] diff = new int[nums.length];
        diff[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            diff[i] = nums[i] - nums[i - 1];
        }
        int tracking = 0;
        for (int i = 0; i <= nums.length - k; i++) {
            tracking += diff[i];
            if (tracking < 0) {
                return false;
            }
            diff[i] -= tracking;
            if (i + k >= nums.length) continue;
            diff[i + k] += tracking;
            tracking = 0;
        }
        tracking = diff[0];
        for (int i = 1; i < nums.length; i++) {
            tracking += diff[i];
            if (tracking != 0) {
                return false;
            }
        }
        return true;
    }

    // LC2740
    public int findValueOfPartition(int[] nums) {
        Arrays.sort(nums);
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length - 1; i++) {
            int j = nums[i], k = nums[i + 1];
            if (k - j < res) {
                res = k - j;
            }
        }
        return res;
    }

    // LC2748
    public int countBeautifulPairs(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            int firstDigit = nums[i];
            while (firstDigit >= 10) firstDigit /= 10;
            for (int j = i + 1; j < nums.length; j++) {
                int lastDigit = nums[j] % 10;
                if (gcd(firstDigit, lastDigit) == 1) res++;
            }
        }
        return res;
    }

    // LC2697
    public String makeSmallestPalindrome(String s) {
        int len = s.length();
        int half = (len + 1) / 2;
        char[] carr = s.toCharArray();
        for (int i = 0; i < half; i++) {
            int oppose = len - i - 1;
            if (s.charAt(i) == s.charAt(oppose)) continue;
            char left = s.charAt(i);
            char right = s.charAt(oppose);
            if (left < right) {
                carr[oppose] = left;
            } else {
                carr[i] = right;
            }
        }
        return new String(carr);
    }

    // LC2698
    public int punishmentNumber(int n) {
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (lc2698Helper(i)) res += i * i;
        }
        return res;
    }

    private boolean lc2698Helper(int i) {
        int sqr = i * i;
        String str = Integer.toString(sqr);
        return lc2698TrySplit(str, i);
    }

    private boolean lc2698TrySplit(String str, int target) {
        if (str.length() == 1) return Integer.parseInt(str) == target;
        if (Integer.parseInt(str) == target) return true;
        if (str.chars().allMatch(i -> (char) i == '0')) return target == 0;
        int sum = str.chars().map(i -> (char) i).map(i -> i - '0').sum();
        if (sum == target) return true;
        if (sum > target) return false;
        for (int i = 1; i < str.length(); i++) {
            String sub = str.substring(0, i);
            int subInt = Integer.parseInt(sub);
            if (subInt > target) continue;
            int remainInt = target - subInt;
            String remainStr = str.substring(i);
            boolean hope = lc2698TrySplit(remainStr, remainInt);
            if (hope) return true;
        }
        return false;
    }

    // LC2682
    public int[] circularGameLosers(int n, int k) {
        Set<Integer> visited = new HashSet<>();
        int cur = 0, loop = 1;
        visited.add(0);
        while (true) {
            int next = cur + loop * k;
            next %= n;
            if (visited.contains(next)) break;
            visited.add(next);
            loop++;
            cur = next;
        }
        int[] res = new int[n - visited.size()];
        int counter = 0;
        for (int i = 0; i < n; i++) {
            if (visited.contains(i)) continue;
            res[counter++] = i + 1;
        }
        return res;
    }

    // LC2591 **
    public int distMoney(int money, int children) {
        if (money < children) return -1;
        if (money < 8) return 0;
        if (money - children < 7) return 0;
        int allocating = money - children;
        int count = Math.min(children, allocating / 7);
        int remain = children - count;
        allocating -= count * 7;
        if (remain == 0) {
            if (allocating > 0) {
                count--;
            }
        } else if (remain == 1) {
            if (allocating == 3) {
                count--;
            }
        }
        return count;
    }

    // LCP06
    public int minCount(int[] coins) {
        return Arrays.stream(coins).boxed()
                .map(i -> (int) Math.ceil((i / 2d)))
                .mapToInt(Integer::valueOf)
                .sum();
    }

    // LC1155
    Integer[][][] lc1155Memo;

    public int numRollsToTarget(int n, int k, int target) {
        lc1155Memo = new Integer[n + 1][k + 1][target + 1];
        return lc1155Helper(n, k, target);
    }

    private int lc1155Helper(int nDice, int kFace, int target) {
        if (nDice == 0 && target == 0) return 1;
        if (nDice <= 0) return 0;
        if (target <= 0) return 0;
        if (nDice * kFace < target) return 0; // 值最大情况下都凑不齐
        if (nDice * kFace == target) return 1;
        if (nDice > target) return 0; // 值最小的情况下都太大
        if (nDice == 1 && target <= kFace) return 1;
        if (lc1155Memo[nDice][kFace][target] != null) return lc1155Memo[nDice][kFace][target];
        long mod = 1000000007;
        long res = 0;
        // 只掷一次骰子
        for (int i = 1; i <= kFace; i++) {
            res += lc1155Helper(nDice - 1, kFace, target - i);
            res %= mod;
        }
        return lc1155Memo[nDice][kFace][target] = (int) res;
    }

    // LC2652
    public int sumOfMultiples(int n) {
        // 3 5 7
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (i % 3 == 0) res += i;
            else if (i % 5 == 0) res += i;
            else if (i % 7 == 0) res += i;
        }
        return res;
    }

    // LC2681 Hard **
    public int sumOfPower(int[] nums) {
        Arrays.sort(nums);
        long res = 0;
        long prefix = 0;
        long mae = 0;
        long mod = 1000000007;
        for (int num : nums) {
            mae = prefix + num;
            mae %= mod;
            prefix = prefix + mae;
            prefix %= mod;
            res += (((long) num * (long) num) % mod) * mae;
            res %= mod;
        }
        return (int) res;
    }

    // LC2592
    public int maximizeGreatness(int[] nums) {
        Map<Integer, Integer> m = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        TreeMap<Integer, Integer> tm = new TreeMap<>(m);
        int res = 0;
        for (int i : nums) {
            Integer higher = tm.higherKey(i);
            if (higher != null) {
                res++;
                int prevFreq = tm.get(higher);
                int newFreq = prevFreq - 1;
                tm.put(higher, newFreq);
                if (newFreq == 0) tm.remove(higher);
            } else {
                // 如果没有比他大的, 就直接找最小的凑数
                Integer minKey = tm.firstKey();
                assert minKey != null;
                int prevFreq = tm.get(minKey);
                int newFreq = prevFreq - 1;
                tm.put(minKey, newFreq);
                if (newFreq == 0) tm.remove(minKey);
            }
        }
        return res;
    }

    // LC2596
    public boolean checkValidGrid(int[][] grid) {
        int[][] directions = new int[][]{{-2, -1}, {-2, 1}, {-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}};
        int len = grid.length;
        int size = len * len;
        boolean[][] visited = new boolean[len][len];
        int[][] idxMap = new int[size][];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                idxMap[grid[i][j]] = new int[]{i, j};
            }
        }
        int r = idxMap[0][0], c = idxMap[0][1];
        if (r != 0 || c != 0) return false;
        visited[r][c] = true;
        for (int i = 1; i < size; i++) {
            int nr = idxMap[i][0], nc = idxMap[i][1];
            boolean pass = false;
            for (int[] d : directions) {
                int pr = r + d[0], pc = c + d[1];
                if (pr >= 0 && pr < len && pc >= 0 && pc < len && !visited[pr][pc] && pr == nr && pc == nc) {
                    pass = true;
                    break;
                }
            }
            if (!pass) {
                return false;
            }
            visited[nr][nc] = true;
            r = nr;
            c = nc;
        }
        return true;
    }

    // LC2660
    public int isWinner(int[] player1, int[] player2) {
        int v1 = 0, v2 = 0;
        for (int i = 0; i < player1.length; i++) {

            if ((i - 1 >= 0 && player1[i - 1] == 10) || (i - 2 >= 0 && player1[i - 2] == 10)) {
                v1 += 2 * player1[i];
            } else {
                v1 += player1[i];
            }
            if ((i - 1 >= 0 && player2[i - 1] == 10) || (i - 2 >= 0 && player2[i - 2] == 10)) {
                v2 += 2 * player2[i];
            } else {
                v2 += player2[i];
            }
        }
        if (v1 > v2) return 1;
        if (v2 > v1) return 2;
        return 0;
    }

    // LC2678
    public int countSeniors(String[] details) {
        return (int) Arrays.stream(details)
                .map(i -> i.substring(11, 13))
                .map(Integer::parseInt)
                .filter(i -> i > 60)
                .count();
    }

    // LC1362
    public int[] closestDivisors(int num) {
        int[] res = new int[]{Integer.MAX_VALUE, 0};
        int resDiff = Integer.MAX_VALUE;
        for (int i = 1; i < 3; i++) {
            int tmp = num + i;
            int sqrt = (int) Math.ceil(Math.sqrt(num));
            for (int j = 1; j <= sqrt; j++) {
                if (tmp % j != 0) continue;
                int another = tmp / j;
                if (Math.abs(j - another) < resDiff) {
                    res = new int[]{j, another};
                    resDiff = Math.abs(j - another);
                }
            }
        }
        return res;
    }

    // LC1432
    public int maxDiff(int num) {
        String victim = Integer.valueOf(num).toString();
        int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                String newNum = victim.replaceAll("" + i, "" + j);
                if (newNum.equals("0")) continue;
                else if (newNum.startsWith("0")) continue;
                int newInt = Integer.parseInt(newNum);
                max = Math.max(max, newInt);
                min = Math.min(min, newInt);
            }
        }
        return max - min;
    }

    // LC1478 Hard **
    interface TriFun<A, B, C, D> {
        D fun(A a, B b, C c);
    }

    public int minDistance(int[] houses, int k) {
        Arrays.sort(houses);
        int len = houses.length;
        Integer[][][] memo = new Integer[len + 1][len + 1][k + 1];
        TriFun<Integer, Integer, Integer, Integer> fun = new TriFun<>() {
            @Override
            public Integer fun(Integer l, Integer r, Integer j) {
                if (l >= r) return 0; // 等于说明是中位数下标, 小于说明已经交叉, 是无效位置
                if (j == 1) { // 只有一个邮箱, 则应该放置在下标中位数的房子的位置
                    return houses[r] - houses[l] // 两端点房子到邮箱的距离
                            + fun(l + 1, r - 1, 1); // 内测房子到邮箱的距离
                }
                if (memo[l][r][j] != null) return memo[l][r][j];
                int res = Integer.MAX_VALUE;
                // 总共还有 r-l+1 个房子
                // 最多可以放 j 个邮箱, j 的上界 = r-l+1
                // r - j + 2 <= r - r + l -1 +2 = l+1
                // 换而言之 在最理想的情况下 只需尝试一次(直接在l端点处放置一个) 然后进入下一层
                // 否则 要尝试在[l,r-j+1] 中每个位置都放一个 然后进入下一层

                for (int i = l; i < (r - j + 2); i++) {
                    res = Math.min(res, fun(l, i, 1) + fun(i + 1, r, j - 1));
                }
                return memo[l][r][j] = res;
            }
        };
        return fun.fun(0, len - 1, k);
    }

    // LC1466
    int lc1466Res = 0;
    boolean[] lc1466Visited;

    public int minReorder(int n, int[][] connections) {
        lc1466Visited = new boolean[n];
        Map<Integer, List<int[]>> forward = Arrays.stream(connections).collect(Collectors.groupingBy(i -> i[0]));
        Map<Integer, List<int[]>> reverse = Arrays.stream(connections).collect(Collectors.groupingBy(i -> i[1]));
        lc1466Helper(0, forward, reverse);
        return lc1466Res;
    }

    private void lc1466Helper(int startPoint, Map<Integer, List<int[]>> forward, Map<Integer, List<int[]>> reverse) {
        if (lc1466Visited[startPoint]) return;
        lc1466Visited[startPoint] = true;
        for (int[] i : forward.getOrDefault(startPoint, new ArrayList<>())) {
            if (lc1466Visited[i[1]]) continue;
            lc1466Res++;
            lc1466Helper(i[1], forward, reverse);
        }
        for (int[] j : reverse.getOrDefault(startPoint, new ArrayList<>())) {
            if (lc1466Visited[j[0]]) continue;
            lc1466Helper(j[0], forward, reverse);
        }
    }

    // LC1402 Hard ** 题解 
    // https://leetcode.cn/u/endlesscheng/
    public int maxSatisfaction(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int f = 0; // f(0) = 0
        int s = 0;
        for (int i = satisfaction.length - 1; i >= 0; i--) {
            s += satisfaction[i];
            if (s <= 0) { // 后面不可能找到更大的 f(k)
                break;
            }
            f += s; // f(k) = f(k-1) + s
        }
        return f;
    }

    // LC1665 Hard **
    public int minimumEffort(int[][] tasks) {
        Arrays.sort(tasks, (i, j) -> i[1] - i[0] - (j[1] - j[0]));
        int res = Integer.MIN_VALUE;
        for (int[] t : tasks) res = Math.max(res + t[0], t[1]);
        return res;
    }


    // LC2136 Hard **
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        int n = plantTime.length;
        Integer[] id = new Integer[n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
        }
        Arrays.sort(id, (i, j) -> growTime[j] - growTime[i]);
        int res = Integer.MIN_VALUE, plantingDays = 0;
        for (int i : id) {
            plantingDays += plantTime[i];
            res = Math.max(res, plantingDays + growTime[i]);
        }
        return res;
    }

    // LC2438
    public int[] productQueries(int n, int[][] queries) {
        List<Integer> binaryArr = toRadix(n, 2);
        List<Integer> twoPowers = new ArrayList<>();
        for (int i = 0; i < binaryArr.size(); i++) {
            if (binaryArr.get(i) == 0) continue;
            int rank = i;
            twoPowers.add(rank);
        }
        twoPowers.sort(Comparator.naturalOrder());
        int[] prefix = new int[twoPowers.size() + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + twoPowers.get(i - 1);
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            // inclusive
            int left = Math.max(0, queries[i][0]);
            int right = Math.min(twoPowers.size() - 1, queries[i][1]);
            int pow = prefix[right + 1] - prefix[left];
            res[i] = (int) quickPower(2, pow);
        }
        return res;
    }

    long mod = 1000000007L;

    private long quickPower(long num, long pow) {
        if (pow == 0L) return 1;
        long res = quickPower(num, pow / 2) % mod;
        res = (pow % 2 == 1 ? res * res * num : res * res) % mod;
        return res;
    }

    // LC2412 Hard **
    public long minimumMoney(int[][] transactions) {
        List<int[]> earnTrans = new ArrayList<>(), lossTrans = new ArrayList<>();
        for (int[] i : transactions) {
            if (i[1] >= i[0]) {
                earnTrans.add(i);
            } else {
                lossTrans.add(i);
            }
        }
        // 困难模式: 先做亏钱的, 从回报最少的开始
        // 然后做赚钱的, 从投入最大的开始
        earnTrans.sort(Comparator.comparingInt(i -> -i[0]));
        lossTrans.sort(Comparator.comparingInt(i -> i[1]));
        long mostLoss = Long.MAX_VALUE;
        long curMoney = 0;
        for (int[] i : lossTrans) {
            curMoney -= i[0];
            mostLoss = Math.min(mostLoss, curMoney);
            curMoney += i[1];
            mostLoss = Math.min(mostLoss, curMoney);
        }
        for (int[] i : earnTrans) {
            curMoney -= i[0];
            mostLoss = Math.min(mostLoss, curMoney);
            curMoney += i[1];
            mostLoss = Math.min(mostLoss, curMoney);
        }
        return -mostLoss;
    }

    // LC2396
    public boolean isStrictlyPalindromic(int n) {
        for (int rad = 2; rad < n - 1; rad++) {
            if (!isPalinDromicList(toRadix(n, rad))) return false;
        }
        return true;
    }

    private boolean isPalinDromicList(List<Integer> arr) {
        int half = arr.size() / 2;
        int len = arr.size();
        for (int i = 0; i < half; i++) {
            if (arr.get(i) != arr.get(len - 1 - i)) return false;
        }
        return true;
    }

    private List<Integer> toRadix(int num, int radix) {
        int left = num;
        List<Integer> res = new ArrayList<>();
        while (left > 0) {
            int remain = left % radix;
            left = left / radix;
            res.add(remain);
        }
        return res;
    }

    // LC2410
    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        int np = players.length, nt = trainers.length;
        int res = 0;
        if (np >= nt) {
            // 运动员更多, 教练员应该尽量匹配能力值高的
            Map<Integer, Integer> freq = Arrays.stream(players).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
            TreeMap<Integer, Integer> tm = new TreeMap<>(freq);
            Arrays.sort(trainers);
            for (int i : trainers) {
                Integer candidateVal = tm.floorKey(i); // 小于等于教练能力值的第一个运动员
                if (candidateVal == null) continue;
                int origCount = tm.get(candidateVal);
                if (origCount == 1) tm.remove(candidateVal);
                else tm.put(candidateVal, origCount - 1);
                res++;
            }
        } else {
            // 教练员更多, 运动员应该尽量匹配能力值高的
            Map<Integer, Integer> freq = Arrays.stream(trainers).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
            TreeMap<Integer, Integer> tm = new TreeMap<>(freq);
            Arrays.sort(players);
            for (int i : players) {
                Integer candidateVal = tm.ceilingKey(i); // 大于等于运动员能力值的第一个教练
                if (candidateVal == null) continue;
                int origCount = tm.get(candidateVal);
                if (origCount == 1) tm.remove(candidateVal);
                else tm.put(candidateVal, origCount - 1);
                res++;
            }
        }
        return res;
    }

    // LC2367
    public int arithmeticTriplets(int[] nums, int diff) {
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            idxMap.put(nums[i], i);
        }
        int res = 0;
        for (int i : nums) {
            int nextOne = i + diff;
            int nextTwo = nextOne + diff;
            if (idxMap.containsKey(nextOne) && idxMap.containsKey(nextTwo)) {
                res++;
            }
        }
        return res;
    }

    // LC2344 ** Hard
    public int minOperations(int[] nums, int[] numsDivide) {
        int gcd = gcd(numsDivide);
        Map<Integer, Integer> freqImMap = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        TreeMap<Integer, Integer> freqTm = new TreeMap<>(freqImMap);
        int counter = 0;
        for (var i : freqTm.entrySet()) {
            int k = i.getKey(), freq = i.getValue();
            if (k == gcd) return counter;
            if (k > gcd) return -1;
            // k < gcd
            if (gcd % k == 0) return counter;
            counter += freq;
        }
        return -1;
    }

    private int gcd(int[] numsDivide) {
        List<Integer> reverseSortedUniqNumsToDivide = Arrays.stream(numsDivide).boxed()
                .distinct()
                .sorted(Comparator.reverseOrder())
                .toList();
        Optional<Integer> reduce = reverseSortedUniqNumsToDivide.stream()
                .reduce(this::gcd);
        return reduce.get();
    }

    private int gcd(int a, int b) {
        return a % b == 0 ? b : gcd(b, a % b);
    }


    // LC2337
    public boolean canChange(String start, String target) {
        if (!start.replaceAll("_", "").equals(target.replaceAll("_", ""))) return false;
        // From left to right
        char[] startCarr = start.toCharArray(), targetCarr = target.toCharArray();
        int n = start.length();
        int startSpaceCounter = 0, startLetterCounter = 0;
        int targetSpaceCounter = 0, targetLetterCounter = 0;
        char wantedChar = 'L', unwantedChar = 'R';
        for (int i = 0; i < n; i++) {
            char s = startCarr[i], t = targetCarr[i];
            if (s == unwantedChar || t == unwantedChar) {
                if (Character.isLetter(s) && Character.isLetter(t) && s != t) {
                    return false;
                }
                startSpaceCounter = 0;
                startLetterCounter = 0;
                targetSpaceCounter = 0;
                targetLetterCounter = 0;
                continue;
            }
            if (s == '_') startSpaceCounter++;
            if (t == '_') targetSpaceCounter++;
            if (s == wantedChar) startLetterCounter++;
            if (t == wantedChar) targetLetterCounter++;
            if (!judge(startSpaceCounter, startLetterCounter, targetSpaceCounter, targetLetterCounter)) {
                return false;
            }
        }
        wantedChar = 'R';
        unwantedChar = 'L';
        for (int i = n - 1; i >= 0; i--) {
            char s = startCarr[i], t = targetCarr[i];
            if (s == unwantedChar || t == unwantedChar) {
                if (Character.isLetter(s) && Character.isLetter(t) && s != t) {
                    return false;
                }
                startSpaceCounter = 0;
                startLetterCounter = 0;
                targetSpaceCounter = 0;
                targetLetterCounter = 0;
                continue;
            }
            if (s == '_') startSpaceCounter++;
            if (t == '_') targetSpaceCounter++;
            if (s == wantedChar) startLetterCounter++;
            if (t == wantedChar) targetLetterCounter++;
            if (!judge(startSpaceCounter, startLetterCounter, targetSpaceCounter, targetLetterCounter)) {
                return false;
            }
        }
        return true;
    }

    private boolean judge(int startSpaceCounter, int startLetterCounter, int targetSpaceCounter, int targetLetterCounter) {
        return startSpaceCounter >= targetSpaceCounter && startLetterCounter <= targetLetterCounter;
    }

    private static ListNode prepareNode(int[] arr) {
        ListNode root = new ListNode(arr[0]);
        ListNode pivot = root;
        for (int i = 1; i < arr.length; i++) {
            pivot.next = new ListNode(arr[i]);
            pivot = pivot.next;
        }
        return root;
    }


    // LC2326
    int[][] directions = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] res = new int[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) res[i][j] = -1;
        BitSet bs = new BitSet(m * n);
        int dir = 0;
        ListNode node = head;
        int curX = 0, curY = 0;
        while (node != null) {
            res[curX][curY] = node.val;
            bs.set(curX * n + curY);
            if (node.next == null) break;
            int[] next = nextStep(curX, curY, dir, m, n, bs);
            curX = next[0];
            curY = next[1];
            dir = next[2];
            node = node.next;
        }
        return res;
    }

    private int[] nextStep(int curX, int curY, int dir, int m, int n, BitSet bs) {
        boolean shouldTurn = false;
        int nextDir = dir;
        int nextX = curX + directions[nextDir][0], nextY = curY + directions[nextDir][1];
        int nextRank = nextX * n + nextY;
        shouldTurn = isShouldTurn(m, n, bs, nextX, nextY, nextRank);
        int loopCount = 0;
        while (shouldTurn) {
            nextDir = (dir + 1) % 4;
            nextX = curX + directions[nextDir][0];
            nextY = curY + directions[nextDir][1];
            nextRank = nextX * n + nextY;
            shouldTurn = isShouldTurn(m, n, bs, nextX, nextY, nextRank);
            if (shouldTurn && ++loopCount >= 4) {
                throw new IllegalStateException("Next step is not available! next x: %d, next y: %d. m: %d, n: %d".formatted(nextX, nextY, m, n));
            }
        }
        return new int[]{nextX, nextY, nextDir};
    }

    private static boolean isShouldTurn(int m, int n, BitSet bs, int nextX, int nextY, int nextRank) {
        return (nextX < 0 || nextX >= m || nextY < 0 || nextY >= n || bs.get(nextRank));
    }


    // LC2316
    public long countPairs(int n, int[][] edges) {
        var dsu = new DSU();
        for (int i = 0; i < n; i++) dsu.add(i);
        for (int[] i : edges) {
            dsu.merge(i[0], i[1]);
        }
        List<Integer> groups = dsu.getAllGroups().values().stream().map(Set::size).toList();
        long sum = 0L;
        for (int i : groups) sum += i;
        long len = groups.size();
        long res = 0L;
        for (int i = 0; i < len; i++) {
            long the = groups.get(i);
            res += the * (sum - the);
            sum -= the;
        }
        return res;
    }
}

class DSU {
    Map<Integer, Integer> parent = new HashMap<>();
    Map<Integer, Integer> rank = new HashMap<>();

    public boolean add(int i) {
        if (parent.containsKey(i)) return false;
        parent.put(i, i);
        rank.put(i, 1);
        return true;
    }

    public int find(int i) { // find the root parent
        int root = i;
        int tmp;
        while ((tmp = parent.get(root)) != root) {
            root = tmp;
        }
        int ptr = i;
        while ((tmp = parent.get(ptr)) != root) { // Compress route
            parent.put(ptr, root);
            rank.put(root, rank.get(root) + 1); // merge by higher ranking
            ptr = tmp;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iParent = find(i);
        int jParent = find(j);
        if (iParent == jParent) return false;
        int iPRank = rank.get(iParent);
        int jPRank = rank.get(jParent);
        if (iPRank >= jPRank) {
            parent.put(jParent, iParent);
            rank.put(iParent, rank.get(iParent) + rank.get(jParent));
        } else {
            parent.put(iParent, jParent);
            rank.put(jParent, rank.get(iParent) + rank.get(jParent));
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        // Find all roots
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (int i : parent.keySet()) {
            int p = find(i);
            Set<Integer> s = res.computeIfAbsent(p, j -> new HashSet<>());
            s.add(i);
        }
        return res;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
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

// LC2671
class FrequencyTracker {

    Map<Integer, Integer> freq = new HashMap<>();
    Map<Integer, Set<Integer>> revFreq = new HashMap<>();

    public FrequencyTracker() {

    }

    public void add(int number) {
        int prevFreq = freq.getOrDefault(number, 0);
        int newFreq = prevFreq + 1;
        freq.put(number, newFreq);
        if (prevFreq != 0) {
            revFreq.get(prevFreq).remove(number);
            if (revFreq.get(prevFreq).isEmpty()) revFreq.remove(prevFreq);
        }
        revFreq.computeIfAbsent(newFreq, i -> new HashSet<>())
                .add(number);
    }

    public void deleteOne(int number) {
        int prevFreq = freq.getOrDefault(number, 0);
        int newFreq = prevFreq - 1;
        if (prevFreq == 0) return;
        if (prevFreq == 1) {
            freq.remove(number);
        } else {
            freq.put(number, newFreq);
        }
        if (prevFreq != 0) {
            revFreq.get(prevFreq).remove(number);
            if (revFreq.get(prevFreq).isEmpty()) revFreq.remove(prevFreq);
        }
        revFreq.computeIfAbsent(newFreq, i -> new HashSet<>())
                .add(number);
    }

    public boolean hasFrequency(int frequency) {
        return revFreq.containsKey(frequency);
    }
}


// LC1483 Hard ** 数上倍增
class TreeAncestor {
    int n;
    int[] parent;
    int[][] cache; // cache[node][power of two]

    public TreeAncestor(int n, int[] parent) {
        this.n = n;
        this.parent = parent;
        int log2 = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        cache = new int[n][log2];
        for (int i = 0; i < n; i++) {
            cache[i][0] = parent[i];
        }
        for (int i = 1; i < log2; i++) {
            for (int j = 0; j < n; j++) {
                int p = cache[j][i - 1];
                cache[j][i] = p < 0 ? -1 : cache[p][i - 1];
            }
        }
    }

    public int getKthAncestor(int node, int k) {
        while (k > 0 && node != -1) {
            node = cache[node][Integer.numberOfTrailingZeros(k)];
            k = k & (k - 1);
        }
        return node;
    }
}
