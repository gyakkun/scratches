import java.util.*;

class Scratch {

    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.isAlienSorted(new String[]{"apple", "app"}, "abcdefghijklmnopqrstuvwxyz"));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC953
    public boolean isAlienSorted(String[] words, String order) {
        if (words.length <= 1) return true;
        int[] letterOrderIdxMap = new int[128];
        for (int i = 0; i < order.length(); i++) letterOrderIdxMap[order.charAt(i)] = i;
        for (int i = 1; i < words.length; i++) {
            String prev = words[i - 1], cur = words[i];
            int minLen = Math.min(prev.length(), cur.length());
            int same = 0;
            for (int j = 0; j < minLen; j++) {
                if (prev.charAt(j) == cur.charAt(j)) {
                    same++;
                    continue;
                } else {
                    if (letterOrderIdxMap[prev.charAt(j)] > letterOrderIdxMap[cur.charAt(j)]) return false;
                    break;
                }
            }
            if (same == minLen && prev.length() > cur.length()) return false;
        }
        return true;
    }

    // LC858
    final double eps = 1e-6;

    public int mirrorReflection(int p, int q) {
        if (q == 0) return 0;
        if (q == p) return 1;
        // 向量表示: double[4] ： [x1,y1, x2, y2]
        double[] vec = new double[]{0d, 0d, p, q};
        int result = -1;
        while ((result = check(vec, p)) == -1) {
            vec = reflect(vec, p);
        }
        return result;
    }

    private double[] reflect(double[] vec, int sideLen) {
        double sx = vec[0], sy = vec[1], tx = vec[2], ty = vec[3];
        double k = -(ty - sy) / (tx - sx);
        double b = ty - k * tx;
        int whichSide = getWhichSide(sideLen, tx, ty);
        // 分别求解
        // 东 x = sideLen
        double[] p;
        if (whichSide != 0) {
            p = new double[]{sideLen, k * sideLen + b};
            if (p[1] >= 0d && p[1] <= (double) sideLen) return new double[]{tx, ty, p[0], p[1]};
        }
        // 南, y= 0
        if (whichSide != 1) {
            p = new double[]{-b / k, 0};
            if (p[0] >= 0d && p[0] <= (double) sideLen) return new double[]{tx, ty, p[0], p[1]};
        }
        // 西, x = 0
        if (whichSide != 2) {
            p = new double[]{0, b};
            if (p[1] >= 0d && p[1] <= (double) sideLen) return new double[]{tx, ty, p[0], p[1]};
        }
        // 北, y = sideLen
        if (whichSide != 3) {
            p = new double[]{(0d + sideLen - b) / k, sideLen};
            if (p[0] >= 0d && p[0] <= (double) sideLen) return new double[]{tx, ty, p[0], p[1]};
        }

        return new double[]{};

    }

    private int getWhichSide(int sideLen, double tx, double ty) {
        int which; // 0 1 2 3 东南西北
        if (Math.abs(tx - 0) < eps) {
            which = 2;
        } else if (Math.abs(ty - 0) < eps) {
            which = 1;
        } else if (Math.abs(ty - sideLen) < eps) {
            which = 3;
        } else if (Math.abs(tx - sideLen) < eps) {
            which = 0;
        } else {
            which = -1;
        }
        return which;
    }

    private int check(double[] vec, int sideLen) {
        // 右下 - 0
        if (Math.abs(vec[2] - sideLen) < eps && Math.abs(vec[3] - 0) < eps) return 0;
        // 右上 - 1
        if (Math.abs(vec[2] - sideLen) < eps && Math.abs(vec[3] - sideLen) < eps) return 1;
        // 左上 - 2
        if (Math.abs(vec[2] - 0) < eps && Math.abs(vec[3] - sideLen) < eps) return 2;
        return -1;
    }

    // LC1928 ** Try DP TAG: BF算法
    public int minCostDP(int limit, int[][] edges, int[] passingFees) {
        int n = passingFees.length, INF = Integer.MAX_VALUE / 2;
        int[][] dp = new int[n][limit + 1];
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], INF);
        dp[0][0] = passingFees[0];
        for (int i = 0; i <= limit; i++) {
            for (int[] e : edges) {
                int u = e[0], v = e[1], time = e[2];
                if (i - time >= 0) {
                    dp[u][i] = Math.min(dp[u][i], dp[v][i - time] + passingFees[u]);
                    dp[v][i] = Math.min(dp[v][i], dp[u][i - time] + passingFees[v]);
                }
            }
        }
        int result = INF;
        for (int i = 0; i <= limit; i++) result = Math.min(result, dp[n - 1][i]);
        if (result == INF) return -1;
        return result;
    }


    // LC1928 ** 联动 LC787 Dijkstra 变形
    public int minCost(int limit, int[][] edges, int[] passingFees) {
        int n = passingFees.length, INF = Integer.MAX_VALUE / 2;
        Map<Integer, Map<Integer, Integer>> mtx = new HashMap<>(); // 耗时矩阵
        for (int[] e : edges) {
            mtx.putIfAbsent(e[0], new HashMap<>());
            mtx.putIfAbsent(e[1], new HashMap<>());
            mtx.get(e[0]).put(e[1], Math.min(e[2], mtx.get(e[0]).getOrDefault(e[1], INF)));
            mtx.get(e[1]).put(e[0], Math.min(e[2], mtx.get(e[1]).getOrDefault(e[0], INF)));
        }
        Integer[][] minCost = new Integer[n][limit + 1]; // minCost[i][j] 表示在时间j到达车站i的最小费用, 同时当作visited使用(null即未访问)
        minCost[0][0] = passingFees[0];
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[2])); // [当前地点, 耗时, 费用]
        pq.offer(new int[]{0, 0, passingFees[0]});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int cur = p[0], timing = p[1], cost = p[2];
            if (cur == n - 1) return cost;
            if (minCost[cur][timing] != null && cost > minCost[cur][timing]) continue;
            if (minCost[cur][timing] == null) {
                minCost[cur][timing] = cost; // pq 保证了这个时间到cur的费用是最小的
            }
            for (int next : mtx.get(cur).keySet()) {
                int nTime = mtx.get(cur).get(next) + timing;
                int nCost = cost + passingFees[next];
                if (nTime <= limit && (minCost[next][nTime] == null || nCost < minCost[next][nTime])) {
                    pq.offer(new int[]{next, nTime, nCost});
                    minCost[next][nTime] = nCost;
                }
            }
        }
        return -1;
    }

    // LC1053 ** 极小化极大 minmax 类比nextPerm
    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        boolean hasResult = false;
        // 逆序寻找第一个降序
        int minMaxIdx = -1, minMaxVal = -1;
        for (int i = n - 2; i >= 0; i--) {
            if (arr[i] > arr[i + 1]) {
                for (int j = i + 1; j < n; j++) { // 然后再在i的右侧找比i小的最大值及其下标
                    if (arr[i] > arr[j]) {
                        hasResult = true;
                        if (arr[j] > minMaxVal) {
                            minMaxIdx = j;
                            minMaxVal = arr[j];
                        }
                    }
                }
                if (hasResult) { // 交换即为最小值
                    int tmp = arr[i];
                    arr[i] = arr[minMaxIdx];
                    arr[minMaxIdx] = tmp;
                    return arr;
                }
            }
        }
        return arr;
    }

    // LC1268 Trie + DFS
    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        int n = searchWord.length();
        List<List<String>> result = new ArrayList<>(n);
        Trie trie = new Trie();
        for (String w : products) {
            trie.addWord(w);
        }
        for (int i = 1; i <= n; i++) {
            List<String> tmp = new ArrayList<>();
            result.add(tmp);
            String keyword = searchWord.substring(0, i);
            TrieNode node = trie.getNode(keyword);
            if (node == null) continue;
            helper(node, tmp, new StringBuilder(keyword));
        }
        return result;
    }

    private void helper(TrieNode root, List<String> result, StringBuilder sb) {
        if (result.size() == 3) return;
        if (root.end > 0) {
            result.add(sb.toString());
        }
        for (char c = 'a'; c <= 'z'; c++) {
            if (root.children.containsKey(c)) {
                sb.append(c);
                helper(root.children.get(c), result, sb);
                sb.deleteCharAt(sb.length() - 1);
                if (result.size() == 3) return;
            }
        }
    }

    // LC1881 非常巧妙
    public String maxValue(String n, int x) {
        if (n.startsWith("-")) {
            for (int i = 1; i < n.length(); i++) {
                if (n.charAt(i) - '0' > x) {
                    return n.substring(0, i) + x + n.substring(i);
                }
            }
        } else {
            for (int i = 0; i < n.length(); i++) {
                if (n.charAt(i) - '0' < x) {
                    return n.substring(0, i) + x + n.substring(i);
                }
            }
        }
        return n + x;
    }

    // LC629 ** DP 注意递推式的推导
    Integer[][] lc629Memo = new Integer[1001][1001];
    final long lc629Mod = 1000000007;

    public int kInversePairs(int n, int k) {
        return lc629Helper(n, k);
    }

    // n 数字个数, k 逆序对个数
    private int lc629Helper(int n, int k) {
        if (k < 0 || n < 0) return 0;
        if (k == 0) return 1; // 如果没有逆序对, 则只有升序一种排列
        if (k == 1) return n - 1; // 如果只有一个逆序对, 则想象将 i 与 i+1 交换 (0<=i<n-1), 总共有n-1 种交换方法
        if (lc629Memo[n][k] != null) return lc629Memo[n][k];
        // 如何递推?
        // From Solution
        //  (n,k) 组成的逆序对可视作两部分, 记最右侧元素为p (p = 1...n)
        //  1) 则左侧元素为 1...p-1, p+1...n 的一个排列
        //     由 p 构成的逆序对个数为 n-p (左侧有 n - (p+1) +1 个元素比 p 大)
        //  2) 由 剩下这i-1个元素构成的逆序对个数, 我们预期它有 k-(n-p) 个逆序对, 因为逆序对只和相对大小有关, 剩下的n-1个数和连续的1...i-1并无区别
        //     故剩下的i-1个元素构成的逆序对的方案数 = f(n-1, k-(n-p))
        //  所以目标就是把 f(n-1,k-(n-p))  (1<=p<=n) 这么多方案个数求和起来即可
        // 递推式推导:
        // f(n,k) = SUM(p:0...n-1) f(n-1,k-p)
        // f(n,k-1) = SUM(p:0...n-1) f(n-1,k-1-p) = f(n,k) - f(n-1,k) + f(n-1,k-n)
        // f(n,k) = f(n,k-1) + f(n-1,k) - f(n-1,k-n)

        long result = lc629Helper(n, k - 1) + lc629Helper(n - 1, k) - lc629Helper(n - 1, k - n);
        result = ((result % lc629Mod) + lc629Mod) % lc629Mod; // result 有可能为负的模数处理方法
        return lc629Memo[n][k] = (int) result;
    }

    // JZOF II 091
    Integer[][] jzofii091Memo = new Integer[101][3];

    public int minCost(int[][] costs) {
        return jzofii091Helper(0, -1, costs);
    }

    private int jzofii091Helper(int idx, int preColor, int[][] cost) {
        if (idx == cost.length) return 0;
        if (preColor != -1 && jzofii091Memo[idx][preColor] != null) return jzofii091Memo[idx][preColor];
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < 3; i++) {
            if (i != preColor) {
                result = Math.min(result, cost[idx][i] + jzofii091Helper(idx + 1, i, cost));
            }
        }
        if (preColor == -1) return result;
        return jzofii091Memo[idx][preColor] = result;
    }

    // LCP 46
    class LCP46 {
        public int[] volunteerDeployment(int[] finalCnt, long totalNum, int[][] edges, int[][] plans) {
            int n = finalCnt.length + 1, m = plans.length;
            Node[][] mtx = new Node[m + 1][n];
            List<Set<Integer>> reach = new ArrayList<>();
            for (int i = 0; i < n; i++) reach.add(new HashSet<>());
            for (int[] e : edges) {
                reach.get(e[0]).add(e[1]);
                reach.get(e[1]).add(e[0]);
            }
            mtx[m][0] = new Node(0, 1l);
            for (int i = 1; i < n; i++) {
                mtx[m][i] = new Node(finalCnt[i - 1], 0);
            }
            for (int i = m - 1; i >= 0; i--) {
                int stadiumIdx = plans[i][1], which = plans[i][0];
                switch (which) {
                    case 1:
                        for (int j = 0; j < n; j++) {
                            if (j == stadiumIdx) {
                                mtx[i][j] = new Node(mtx[i + 1][j].val * 2l, mtx[i + 1][j].factor * 2l);
                            } else {
                                mtx[i][j] = new Node(mtx[i + 1][j].val, mtx[i + 1][j].factor);
                            }
                        }
                        break;
                    case 2:
                        Set<Integer> n1 = reach.get(stadiumIdx);
                        for (int j = 0; j < n; j++) {
                            if (!n1.contains(j)) mtx[i][j] = new Node(mtx[i + 1][j].val, mtx[i + 1][j].factor);
                            else {
                                mtx[i][j] = new Node(mtx[i + 1][j].val - mtx[i + 1][stadiumIdx].val, mtx[i + 1][j].factor - mtx[i + 1][stadiumIdx].factor);
                            }
                        }
                        break;
                    case 3:
                        Set<Integer> n2 = reach.get(stadiumIdx);
                        for (int j = 0; j < n; j++) {
                            if (!n2.contains(j)) mtx[i][j] = new Node(mtx[i + 1][j].val, mtx[i + 1][j].factor);
                            else {
                                mtx[i][j] = new Node(mtx[i + 1][j].val + mtx[i + 1][stadiumIdx].val, mtx[i + 1][j].factor + mtx[i + 1][stadiumIdx].factor);
                            }
                        }
                }
            }
            long k = 0;
            long b = 0;
            for (Node nn : mtx[0]) {
                k += nn.factor;
                b += nn.val;
            }
            double dx = (double) (totalNum - b) / (double) k;
            long x = (long) dx;
            int[] result = new int[n];
            for (int i = 0; i < n; i++) {
                result[i] = (int) (mtx[0][i].factor * x + mtx[0][i].val);
            }
            return result;
        }

        class Node {
            long val = 0;
            long factor = 0;

            public Node(long val, long factor) {
                this.val = val;
                this.factor = factor;
            }
        }
    }

    // JZOF II 044
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            long max = Long.MIN_VALUE;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                TreeNode p = q.poll();
                max = Math.max(max, p.val);
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
            result.add((int) max);
        }
        return result;
    }

    // LC2032
    public List<Integer> twoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
        boolean[] f1 = new boolean[101], f2 = new boolean[101], f3 = new boolean[101];
        for (int i : nums1) f1[i] = true;
        for (int i : nums2) f2[i] = true;
        for (int i : nums3) f3[i] = true;
        List<Integer> result = new ArrayList<>(101);
        for (int i = 0; i <= 100; i++) {
            if ((f1[i] && f2[i]) || (f1[i] && f3[i]) || (f2[i] && f3[i])) result.add(i);
        }
        return result;
    }


    // JZOF II 040 **
    public int maximalRectangle(String[] matrix) {
        if (matrix == null) return 0;
        if (matrix.length == 0) return 0;
        if (matrix[0].length() == 0) return 0;
        // 预处理 变成整型数组
        int m = matrix.length, n = matrix[0].length();
        int[][] mtx = new int[m][n];
        for (int i = 0; i < m; i++) {
            char[] ca = matrix[i].toCharArray();
            for (int j = 0; j < n; j++) {
                mtx[i][j] = ca[j] - '0';
            }
        }

        // 预处理 prefix[i][j] 存的是i,j左边有多少个1
        int[][] prefix = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (mtx[i - 1][j - 1] == 1) {
                    prefix[i][j] = prefix[i][j - 1] + 1;
                }
            }
        }

        int result = 0;
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                int width = prefix[i][j];
                // 从低行到高行, 实时更新最小宽度
                for (int k = i; k >= 0; k--) {
                    int height = i - k + 1;
                    width = Math.min(width, prefix[k][j]);
                    result = Math.max(result, height * width);
                }
            }
        }
        return result;

    }

    // LC1910
    public String removeOccurrences(String s, String part) {
        int idx = -1, pl = part.length();
        while ((idx = s.indexOf(part)) >= 0) {
            s = s.substring(0, idx) + s.substring(idx + pl);
        }
        return s;
    }

    // JZOF II 093
    Integer[][] jzofii093Memo = new Integer[1001][1001];

    public int lenLongestFibSubseq(int[] arr) {
        Map<Integer, TreeSet<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            m.putIfAbsent(arr[i], new TreeSet<>());
            m.get(arr[i]).add(i);
        }
        int result = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                int tmp = jzoffii093Helper(i, j, arr, m);
                if (tmp == 0) continue;
                result = Math.max(result, 2 + tmp);
            }
        }
        return result;
    }

    private int jzoffii093Helper(int prevIdx, int curIdx, int[] arr, Map<Integer, TreeSet<Integer>> m) {
        if (jzofii093Memo[prevIdx][curIdx] != null) return jzofii093Memo[prevIdx][curIdx];
        int sum = arr[prevIdx] + arr[curIdx];
        if (!m.containsKey(sum)) return jzofii093Memo[prevIdx][curIdx] = 0;
        Integer next = m.get(sum).higher(curIdx);
        if (next == null) return jzofii093Memo[prevIdx][curIdx] = 0;
        return jzofii093Memo[prevIdx][curIdx] = 1 + jzoffii093Helper(curIdx, next, arr, m);
    }

    // LC1690 **
    Integer[][] lc1690Memo = new Integer[1001][1001];
    int[] lc1690Prefix = new int[1001];

    public int stoneGameVII(int[] stones) {
        int n = stones.length;
        for (int i = 0; i < n; i++) {
            lc1690Prefix[i + 1] = lc1690Prefix[i] + stones[i];
        }
        return lc1690Helper(0, n - 1);
    }

    // 返回的是分差, 两者目标是一样的, 都是希望得分更高(alice得分更高有利于扩大分差, bob得分更高有利于缩小分差), 下面这个"分差"更大
    // "分差": 当前的得分 - 下次对手的最大得分(递归定义)
    private int lc1690Helper(int left, int right) {
        if (left >= right) {
            // ** left == right 只剩一个, 返回0
            return 0;
        }
        if (lc1690Memo[left][right] != null) return lc1690Memo[left][right];
        int removeLeftGain = lc1690Prefix[right + 1] - lc1690Prefix[left + 1], removeRightGain = lc1690Prefix[right] - lc1690Prefix[left];
        return lc1690Memo[left][right] = Math.max(removeLeftGain - lc1690Helper(left + 1, right),
                removeRightGain - lc1690Helper(left, right - 1));
    }

    // LC2025
    public int waysToPartition(int[] nums, int k) {
        // 结果最多为 n - 1
        int n = nums.length, result = 0;
        long sum = 0;
        Map<Long, TreeSet<Integer>> diffIdxSetMap = new HashMap<>();
        for (int i : nums) sum += i;

        long left = 0;
        for (int i = 1; i < n; i++) {
            left += nums[i - 1];
            long right = sum - left;
            long diff = left - right; // gap编号: i, diff: gap左侧减右侧
            diffIdxSetMap.putIfAbsent(diff, new TreeSet<>());
            diffIdxSetMap.get(diff).add(i);
        }
        if (diffIdxSetMap.containsKey(0l)) result = diffIdxSetMap.get(0l).size(); // 不修改任何数的情况下的结果

        for (int i = 0; i < n; i++) {
            long diff = k - nums[i];
            long revDiff = -diff;
            int tmpResult = 0;
            if (diffIdxSetMap.containsKey(revDiff)) {
                TreeSet<Integer> ts = diffIdxSetMap.get(revDiff);
                tmpResult += ts.subSet(i + 1, true, n, true).size();
            }
            if (diffIdxSetMap.containsKey(diff)) {
                TreeSet<Integer> ts = diffIdxSetMap.get(diff);
                tmpResult += ts.subSet(0, true, i, true).size();
            }
            result = Math.max(result, tmpResult);
        }
        return result;
    }

    // LC495
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        int result = 0, expire = 0;
        for (int i : timeSeries) {
            if (i < expire) {
                result += (duration - (expire - i));
            } else {
                result += duration;
            }
            expire = Math.max(expire, i + duration);
        }
        return result;
    }

    // LC2029 **
    // https://leetcode-cn.com/problems/stone-game-ix/solution/guan-jian-zai-yu-qiu-chu-hui-he-shu-by-e-mcgv/
    public boolean stoneGameIX(int[] stones) {
        int[] freq = new int[3];
        for (int i : stones) freq[i % 3]++;
        return check(new int[]{freq[0], freq[2], freq[1]}) || check(freq);
    }

    private boolean check(int[] freq) {
        // 1这个下标表示我们要抽的数, 为0时说明无牌可抽, 直接输掉
        if (freq[1] == 0) return false;
        freq[1]--;
        int turn = 1 + Math.min(freq[1], freq[2]) * 2 + freq[0]; // 这样下去可以玩多少个回合
        if (freq[1] > freq[2]) { // 最后剩下的都是我们想抽的牌, 对方想抽的牌就没有了
            turn++;
            freq[1]--;
        }
        return turn % 2 == 1 && freq[1] != freq[2]; // freq[1] == freq[2] 说明剩下没有牌了, 这时候BOB自动嬴
    }

    // LC974
    public int subarraysDivByK(int[] nums, int k) {
        int[] modMap = new int[k];
        modMap[0] = 1;
        int sum = 0, result = 0;
        for (int i : nums) {
            sum += i;
            int mod = (sum % k + k) % k; // sum%k有可能为负
            result += modMap[mod];
            modMap[mod]++;
        }
        return result;
    }

    // LC1727 **
    public int largestSubmatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] != 0 && matrix[i - 1][j] != 0) {
                    matrix[i][j] += matrix[i - 1][j];
                }
            }
        }
        int result = 0;
        for (int i = 0; i < m; i++) {
            Arrays.sort(matrix[i]);
            for (int j = n - 1; j >= 0; j--) {
                result = Math.max(result, matrix[i][j] * (n - j));
            }
        }
        return result;
    }

    // LC1359 Hard ** 组合数学
    public int countOrders(int n) {
        return lc1359Helper(n);
    }

    private int lc1359Helper(int n) {
        if (n <= 1) return 1;
        long result = 0;
        int prevLen = (n - 1) * 2;
        int numGaps = prevLen + 1;
        for (int i = 0; i < numGaps; i++) {
            result += numGaps - i; // 在占了第i个间隙之后, 自己右边也会多出一个间隙, 实际右侧可选间隙个数还是numGaps-i
            // result %= mod; // 这一步求模没有必要, 因为 n<=500, 求等差数列可知不可能超过1e9+7
        }
        result = result * lc1359Helper(n - 1);
        result %= 1000000007;
        return (int) result;
    }

    // LC505
    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        int m = maze.length, n = maze[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[2])); // [ r, c, dis ]
        pq.offer(new int[]{start[0], start[1], 0});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int r = p[0], c = p[1], dis = p[2];
            if (visited[r][c]) continue;
            visited[r][c] = true;
            if (r == destination[0] && c == destination[1]) return dis;

            for (int[] d : directions) {
                int nr = r, nc = c, ndis = dis;
                while (nr + d[0] >= 0 && nr + d[0] < m && nc + d[1] >= 0 && nc + d[1] < n && maze[nr + d[0]][nc + d[1]] != 1) {
                    nr += d[0];
                    nc += d[1];
                    ndis++;
                }
                if (!visited[nr][nc]) {
                    pq.offer(new int[]{nr, nc, ndis});
                }
            }
        }
        return -1;
    }

    // LC1016
    public boolean queryString(String s, int n) {
        BitSet bs = new BitSet(n + 1);
        int maxLen = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        char[] ca = s.toCharArray();
        for (int len = 1; len <= maxLen; len++) {
            int mask = 0, fullMask = (1 << len) - 1;
            for (int i = 0; i < len && i < ca.length; i++) {
                mask = (mask << 1) | (ca[i] - '0');
            }
            if (mask > 0 && mask <= n) bs.set(mask);
            for (int i = len; i < ca.length; i++) {
                mask = ((mask << 1) | (ca[i] - '0')) & fullMask;
                if (mask > 0 && mask <= n) bs.set(mask);
            }
            if (len < maxLen) {
                int total = bs.cardinality();
                if (total != (1 << len) - 1) return false;
            }
        }
        int total = bs.cardinality();
        if (total != n) return false;
        return true;
    }

    // LC546 ** 祖玛
    // Solution, From: https://leetcode-cn.com/problems/remove-boxes/solution/yi-chu-he-zi-by-leetcode-solution/546152
    Integer[][][] lc546Memo;

    public int removeBoxes(int[] boxes) {
        int n = boxes.length;
        lc546Memo = new Integer[n + 1][n + 1][n + 1];
        return lc546Helper(0, n - 1, 0, boxes);
    }

    private int lc546Helper(int left, int right, int k, int[] boxes) {
        if (left > right) return 0;
        if (lc546Memo[left][right][k] != null) return lc546Memo[left][right][k];

        int origRight = right, origK = k;
        while (left < right && boxes[right] == boxes[right - 1]) {
            right--;
            k++;
        }
        int result = lc546Helper(left, right - 1, 0, boxes) + (k + 1) * (k + 1);
        for (int i = left; i < right; i++) {
            if (boxes[i] == boxes[right]) {
                result = Math.max(result, lc546Helper(i + 1, right - 1, 0, boxes) + lc546Helper(left, i, k + 1, boxes));
            }
        }
        return lc546Memo[left][origRight][origK] = result;
    }

    // LC1437
    public boolean kLengthApart(int[] nums, int k) {
        if (k == 0) return true;
        int idx = 0;
        List<Integer> idxs = new ArrayList<>();
        while (idx < nums.length) {
            if (nums[idx] == 1) idxs.add(idx);
            idx++;
        }
        if (idxs.size() <= 1) return true;
        for (int i = 1; i < idxs.size(); i++) {
            int prev = idxs.get(i - 1), cur = idxs.get(i);
            if (cur - prev - 1 < k) return false;
        }
        return true;
    }

    // LC488 祖玛
    Map<String, Map<String, Integer>> lc488Memo;

    public int findMinStep(String board, String hand) {
        char[] ca = hand.toCharArray();
        Arrays.sort(ca);
        String sortedKey = new String(ca);
        lc488Memo = new HashMap<>();
        lc488Helper(board, hand);
        int result = lc488Helper(board, sortedKey);
        if (result == Integer.MAX_VALUE / 2) return -1;
        return result;
    }

    private int lc488Helper(String board, String hand) {
        if (lc488Memo.containsKey(board) && lc488Memo.get(board).containsKey(hand)) {
            return lc488Memo.get(board).get(hand);
        }
        lc488Memo.putIfAbsent(board, new HashMap<>());
        if (hand.equals("")) {
            board = zumaProcess(board);
            if (!board.equals("")) {
                lc488Memo.get(board).put(hand, Integer.MAX_VALUE / 2);
                return Integer.MAX_VALUE / 2;
            }
            lc488Memo.get(board).put(hand, 0);
            return 0;
        }
        if (board.equals("")) {
            lc488Memo.get(board).put(hand, 0);
            return 0;
        }
        int result = Integer.MAX_VALUE / 2;
        for (int i = 0; i <= board.length(); i++) {
            for (int j = 0; j < hand.length(); j++) {
                String tmpBoard = board.substring(0, i) + hand.charAt(j) + board.substring(i);
                tmpBoard = zumaProcess(tmpBoard);
                result = Math.min(result, 1 + lc488Helper(tmpBoard, hand.substring(0, j) + hand.substring(j + 1)));
            }
        }
        lc488Memo.get(board).put(hand, result);
        return result;
    }

    private String zumaProcess(String board) {
        int[] check;
        while ((check = zumaCheck(board)) != null) {
            board = board.substring(0, check[0]) + board.substring(check[1] + 1);
        }
        return board;
    }

    private int[] zumaCheck(String s) {
        int idx = 0;
        while (idx < s.length()) {
            int start = idx;
            while (idx + 1 < s.length() && s.charAt(idx) == s.charAt(idx + 1)) idx++;
            int end = idx;
            if (end - start + 1 >= 3) return new int[]{start, end};
            idx++;
        }
        return null;
    }


    // LC893 ** TAG: 卡特兰数
    Map<Integer, List<TreeNode>> lc893Memo = new HashMap<>();

    public List<TreeNode> allPossibleFBT(int n) {
        if (lc893Memo.containsKey(n)) return lc893Memo.get(n);
        List<TreeNode> result = new ArrayList<>();
        if (n == 1) {
            result.add(new TreeNode(0));
        } else if (n % 2 == 1) {
            for (int leftCount = 1; leftCount <= n - 1; leftCount++) {
                int rightCount = n - 1 - leftCount;
                for (TreeNode left : allPossibleFBT(leftCount)) {
                    for (TreeNode right : allPossibleFBT(rightCount)) {
                        TreeNode root = new TreeNode(0);
                        root.left = left;
                        root.right = right;
                        result.add(copyTree(root));
                    }
                }
            }
        }
        lc893Memo.put(n, result);
        return result;
    }


    private TreeNode copyTree(TreeNode root) {
        if (root == null) return null;
        TreeNode result = new TreeNode(root.val);
        result.left = copyTree(root.left);
        result.right = copyTree(root.right);
        return result;
    }

    // LC966
    public String[] spellchecker(String[] wordlist, String[] queries) {
        Map<String, Integer> lowerCaseIdxMap = new HashMap<>();
        Map<String, Integer> deVowelIdxMap = new HashMap<>();
        Set<String> wordSet = new HashSet<>();
        for (int i = 0; i < wordlist.length; i++) {
            lowerCaseIdxMap.putIfAbsent(wordlist[i].toLowerCase(), i);
            String deVowel = wordlist[i].toLowerCase().replaceAll("[aeiou]", "#");
            deVowelIdxMap.putIfAbsent(deVowel, i);
            wordSet.add(wordlist[i]);
        }
        String[] result = new String[queries.length];

        for (int i = 0; i < queries.length; i++) {
            if (wordSet.contains(queries[i])) {
                result[i] = queries[i];
                continue;
            }
            String lcKey = queries[i].toLowerCase();
            if (lowerCaseIdxMap.containsKey(lcKey)) {
                result[i] = wordlist[lowerCaseIdxMap.get(lcKey)];
                continue;
            }
            String dvKey = queries[i].toLowerCase().replaceAll("[aeiou]", "#");
            if (deVowelIdxMap.containsKey(dvKey)) {
                result[i] = wordlist[deVowelIdxMap.get(dvKey)];
                continue;
            }
            result[i] = "";
        }
        return result;
    }

    // LC1156
    public int maxRepOpt1(String text) {
        Map<Character, List<int[]>> letterAppearIdxPairMap = new HashMap<>(26);
        for (char c = 'a'; c <= 'z'; c++) {
            letterAppearIdxPairMap.put(c, new ArrayList<>());
        }
        char[] ca = text.toCharArray();
        int idx = 0;
        while (idx < ca.length) {
            int start = idx;
            while (idx + 1 < ca.length && ca[idx + 1] == ca[idx]) idx++;
            int end = idx;
            letterAppearIdxPairMap.get(ca[idx]).add(new int[]{start, end});
            idx++;
        }
        int maxLen = 0;
        for (char c = 'a'; c <= 'z'; c++) {
            List<int[]> appear = letterAppearIdxPairMap.get(c);
            for (int i = 0; i < appear.size(); i++) {
                int[] cur = appear.get(i);
                int curLen = cur[1] - cur[0] + 1;
                maxLen = Math.max(maxLen, curLen);
                if (i + 1 < appear.size()) {
                    int[] next = appear.get(i + 1);
                    int nextLen = next[1] - next[0] + 1;
                    if (next[0] - cur[1] == 2) {
                        // 中间只隔了一个字母 首先考虑当前c只有两段的情况
                        maxLen = Math.max(maxLen, curLen + nextLen);
                        // 如果c有三段或以上, 则从第三段调取一个字母过来填充
                        if (appear.size() >= 3) {
                            maxLen = Math.max(maxLen, curLen + nextLen + 1);
                        }
                    } else if (next[0] - cur[1] > 2) {
                        // 如果两段之间有超过一个字母, 则最多能抽掉一个过来填充
                        maxLen = Math.max(maxLen, curLen + 1);
                    }
                }
            }
            // 反过来再试一遍
            for (int i = appear.size() - 1; i >= 0; i--) {
                int[] cur = appear.get(i);
                int curLen = cur[1] - cur[0] + 1;
                maxLen = Math.max(maxLen, curLen);
                if (i - 1 >= 0) {
                    int[] next = appear.get(i - 1);
                    int nextLen = next[1] - next[0] + 1;
                    if (next[0] - cur[1] == -2) { // 注意符号, 或者用绝对值
                        // 中间只隔了一个字母 首先考虑当前c只有两段的情况
                        maxLen = Math.max(maxLen, curLen + nextLen);
                        // 如果c有三段或以上, 则从第三段调取一个字母过来填充
                        if (appear.size() >= 3) {
                            maxLen = Math.max(maxLen, curLen + nextLen + 1);
                        }
                    } else if (next[0] - cur[1] < -2) { // 注意符号, 或者用绝对值
                        // 如果两段之间有超过一个字母, 则最多能抽掉一个过来填充
                        maxLen = Math.max(maxLen, curLen + 1);
                    }
                }
            }
        }
        return maxLen;
    }

    // Interview 05.02
    public String printBin(double num) {
        double eps = 1e-11;
        StringBuilder bin = new StringBuilder("0.");
        while (true) {
            if (Math.abs(num - 0d) < eps) break;
            num = num * 2d;
            if (num >= 1d) {
                bin.append("1");
                num -= 1d;
            } else {
                bin.append("0");
            }
            if (bin.length() > 32) return "ERROR";
        }
        return bin.toString();
    }

    // LC1162 ** 多源最短路
    public int maxDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> q = new LinkedList<>();
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int landCount = 0, seaCount = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) seaCount++;
                else {
                    landCount++;
                    q.offer(new int[]{i, j});
                }
            }
        }
        if (landCount == m * n || seaCount == m * n) return -1;
        final int INF = Integer.MAX_VALUE / 2;
        int[][] minDistance = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) minDistance[i][j] = INF;
            }
        }
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                int r = p[0], c = p[1];
                if (grid[r][c] == 0 && minDistance[r][c] != INF) continue;
                if (grid[r][c] == 0) {
                    minDistance[r][c] = layer;
                }
                for (int[] d : directions) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] == 1 || minDistance[nr][nc] != INF) {
                        continue;
                    }
                    q.offer(new int[]{nr, nc});
                }
            }
        }
        int maxDistance = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    maxDistance = Math.max(maxDistance, minDistance[i][j]);
                }
            }
        }
        return maxDistance;
    }

    // LC1267
    public int countServers(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] rowSum = new int[m], colSum = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowSum[i] += grid[i][j];
                colSum[j] += grid[i][j];
            }
        }

        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && (rowSum[i] > 1 || colSum[j] > 1)) count++;
            }
        }

        return count;
    }

    // LC299
    public String getHint(String secret, String guess) {
        // secret.length = guess.length
        int n = secret.length();
        int[] sFreq = new int[10], gFreq = new int[10];
        char[] cs = secret.toCharArray(), cg = guess.toCharArray();
        int exactly = 0;
        for (int i = 0; i < n; i++) {
            if (cs[i] == cg[i]) {
                exactly++;
                continue;
            }
            sFreq[cs[i] - '0']++;
            gFreq[cg[i] - '0']++;
        }
        int blur = 0;
        for (int i = 0; i < 10; i++) {
            blur += Math.min(sFreq[i], gFreq[i]);
        }
        return exactly + "A" + blur + "B";
    }

    // LC1090
    public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
        int n = values.length;
        List<int[]> idxLabelSet = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            idxLabelSet.add(new int[]{labels[i], values[i]});
        }
        Collections.sort(idxLabelSet, Comparator.comparingInt(o -> -o[1]));
        int totalCount = 0, sum = 0;
        int[] labelFreq = new int[20001];
        for (int[] p : idxLabelSet) {
            int label = p[0], val = p[1];
            if (labelFreq[label] == useLimit) continue;
            if (totalCount == numWanted) break;
            totalCount++;
            sum += val;
            labelFreq[label]++;
        }
        return sum;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

// LC631
class Excel {

    Node[][] mtx;

    public Excel(int height, char width) {
        int h = height, w = width - 'A' + 1;
        mtx = new Node[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mtx[i][j] = new Node();
            }
        }
    }

    public void set(int row, char column, int val) {
        int r = row - 1, c = column - 'A';
        Node n = mtx[r][c];
        n.val = val;
        n.compose.clear();
    }

    public int get(int row, char column) {
        int r = row - 1, c = column - 'A';
        return mtx[r][c].eval();
    }

    public int sum(int row, char column, String[] numbers) {
        int r = row - 1, c = column - 'A';
        List<Node> compose = new ArrayList<>();
        for (String q : numbers) {
            if (q.indexOf(":") < 0) {
                char icc = q.charAt(0);
                int ic = icc - 'A';
                int ir = Integer.valueOf(q.substring(1)) - 1;
                compose.add(mtx[ir][ic]);
            } else {
                // 左上 : 右下
                String[] arr = q.split(":");
                int leftCol = arr[0].charAt(0) - 'A';
                int leftRow = Integer.valueOf(arr[0].substring(1)) - 1;
                int rightCol = arr[1].charAt(0) - 'A';
                int rightRow = Integer.valueOf(arr[1].substring(1)) - 1;
                for (int i = leftRow; i <= rightRow; i++) {
                    for (int j = leftCol; j <= rightCol; j++) {
                        compose.add(mtx[i][j]);
                    }
                }
            }
        }
        mtx[r][c].compose = compose;
        return mtx[r][c].eval();
    }

    private Node getNode(String q) {
        char icc = q.charAt(0);
        int ic = icc - 'A';
        int ir = Integer.valueOf(q.substring(1)) - 1;
        return mtx[ir][ic];
    }


    class Node {
        int val = 0;
        List<Node> compose = new ArrayList<>();

        public int eval() {
            if (compose.size() == 0) {
                return val;
            }
            int result = 0;
            for (Node n : compose) {
                result += n.eval();
            }
            return result;
        }
    }
}

// JZOF 09
class CQueue {
    Deque<Integer> stack1 = new LinkedList<>();
    Deque<Integer> stack2 = new LinkedList<>();

    public CQueue() {

    }

    public void appendTail(int value) {
        stack1.push(value);
    }

    public int deleteHead() {
        if (stack1.size() == 0) return -1;
        while (stack1.size() > 1) {
            stack2.push(stack1.pop());
        }
        int result = stack1.pop();
        while (!stack2.isEmpty()) stack1.push(stack2.pop());
        return result;
    }
}

class Trie {
    TrieNode root = new TrieNode();

    public void addWord(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) cur.children.put(c, new TrieNode());
            cur = cur.children.get(c);
            cur.path++;
        }
        cur.end++;
    }

    public boolean removeWord(String word) {
        TrieNode target = getNode(word);
        if (target == null) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children.get(c).path-- == 1) {
                cur.children.remove(c);
                return true;
            }
            cur = cur.children.get(c);
        }
        cur.end--;
        return true;
    }

    public boolean search(String word) {
        TrieNode target = getNode(word);
        return target.end > 0;
    }

    public boolean startsWith(String word) {
        return getNode(word) != null;
    }

    public void insert(String word) {
        addWord(word);
    }

    public int countWordsStartingWith(String prefix) {
        TrieNode target = getNode(prefix);
        if (target == null) return 0;
        return target.path;
    }

    public int countWordsEqualTo(String word) {
        TrieNode target = getNode(word);
        if (target == null) return 0;
        return target.end;
    }

    public void erase(String word) {
        removeWord(word);
    }

    public TrieNode getNode(String prefix) {
        TrieNode cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return null;
            cur = cur.children.get(c);
        }
        return cur;
    }
}

class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    int end = 0;
    int path = 0;
}