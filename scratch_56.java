import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        Codec codec = new Codec();
        long timing = System.currentTimeMillis();


        // [2,3,4]
        //[[1,1,0,4],[2,2,1,9]]
        //[1,2,1]
        System.out.println(s.getBiggestThree(new int[][]{{3, 4, 5, 1, 3}, {3, 3, 4, 2, 3}, {20, 30, 200, 40, 10}, {1, 5, 5, 4, 1}, {4, 3, 2, 2, 5}}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC272
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingDouble(o -> -Math.abs(target - (double) o)));
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            if (pq.size() < k) pq.offer(p.val);
            else {
                if (Math.abs(target - (double) p.val) < Math.abs(target - (double) pq.peek())) {
                    pq.poll();
                    pq.offer(p.val);
                }
            }
            if (p.left != null) q.offer(p.left);
            if (p.right != null) q.offer(p.right);
        }
        return new ArrayList<>(pq);
    }

    // LC1878
    public int[] getBiggestThree(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        TreeSet<Integer> ts = new TreeSet<>(Comparator.reverseOrder());
        for (int i = 0; i < m; i++) {
            loop:
            for (int j = 0; j < n; j++) {
                // "边长"为0
                ts.add(grid[i][j]);
                for (int k = 1; ; k++) { // "边长"
                    int sum = grid[i][j];
                    int row = i;
                    int leftCol = -1, rightCol = -1;
                    // 前半
                    for (int p = 0; p < k; p++) {
                        leftCol = j - 1 - p;
                        rightCol = j + 1 + p;
                        int curRow = row + 1 + p;
                        if (leftCol < 0 || rightCol >= n || curRow >= m) continue loop;
                        sum += grid[curRow][leftCol] + grid[curRow][rightCol];
                    }

                    // 后半
                    row += k;
                    for (int p = 0; p < k; p++) {
                        leftCol += 1;
                        rightCol -= 1;
                        int curRow = row + 1 + p;
                        if (leftCol < 0 || rightCol >= n || curRow >= m) continue loop;
                        sum += grid[curRow][leftCol] + grid[curRow][rightCol];
                    }
                    sum -= grid[i + 2 * k][j];
                    ts.add(sum);

                }
            }
        }
        int[] result = new int[ts.size() > 3 ? 3 : ts.size()];
        Iterator<Integer> it = ts.iterator();
        for (int i = 0; i < result.length; i++) {
            result[i] = it.next();
        }
        return result;
    }

    // LC638
    Map<Integer, Map<Integer, Integer>> lc638Memo;

    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        lc638Memo = new HashMap<>();
        List<List<Integer>> validSpecial = new ArrayList<>();
        for (List<Integer> sp : special) if (isValidSpecial(sp, needs, price)) validSpecial.add(sp);
        // 背包问题
        int[] needArr = new int[needs.size()];
        for (int i = 0; i < needs.size(); i++) needArr[i] = needs.get(i);
        // 购买前i个礼包, 不超过need的情况下的最小价格
        return helper(validSpecial.size(), needArr, price, validSpecial);
    }

    // 只使用前idx个大礼包, 顶格满足needs需求时候的最低价格
    private int helper(int idx, int[] needs, List<Integer> price, List<List<Integer>> special) {
        int status = calStatus(needs), n = needs.length;
        if (lc638Memo.containsKey(idx) && lc638Memo.get(idx).containsKey(status)) return lc638Memo.get(idx).get(status);
        if (idx == 0) {
            int result = 0;
            for (int i = 0; i < needs.length; i++) {
                result += needs[i] * price.get(i);
            }
            lc638Memo.putIfAbsent(idx, new HashMap<>());
            lc638Memo.get(idx).put(status, result);
            return result;
        }
        int[] origNeeds = Arrays.copyOf(needs, needs.length);
        int result = Integer.MAX_VALUE;
        loop:
        for (int i = 0; ; i++) {// 看选择多少个礼包(idx-1)
            for (int j = 0; j < n; j++) {
                if (needs[j] - i * special.get(idx - 1).get(j) < 0) break loop;
            }
            int tempPrice = i * special.get(idx - 1).get(n);
            // 减少相应礼包数量
            for (int j = 0; j < n; j++) {
                needs[j] -= i * special.get(idx - 1).get(j);
            }
            tempPrice += helper(idx - 1, needs, price, special);
            result = Math.min(result, tempPrice);
            // 还原
            for (int j = 0; j < n; j++) {
                needs[j] = origNeeds[j];
            }
        }
        lc638Memo.putIfAbsent(idx, new HashMap<>());
        lc638Memo.get(idx).put(status, result);
        return result;
    }

    private int calStatus(int[] needs) {
        int base = 11;
        int accu = 1;
        int result = 0;
        for (int i = 0; i < needs.length; i++) {
            result += accu * needs[i];
            accu *= base;
        }
        return result;
    }

    private boolean isValidSpecial(List<Integer> sp, List<Integer> needs, List<Integer> price) {
        int sum = 0;
        for (int i = 0; i < sp.size() - 1; i++) {
            if (sp.get(i) > needs.get(i)) {
                return false;
            }
            sum += price.get(i) * sp.get(i); // 单价*数量
        }
        if (sum < sp.get(sp.size() - 1)) {
            return false; // 如果总价比单买还贵也没必要买这个大礼包了
        }
        return true;
    }

    // LC1522 N叉树的直径
    class Lc1522 {
        int d = 0;

        public int diameter(Node root) {
            helper(root);
            return d;
        }

        private int helper(Node root) {
            if (root == null) return 0;
            if (root.children.size() == 0) return 1;
            int max1 = 0, max2 = 0;
            for (Node c : root.children) {
                int pathLen = helper(c);
                if (pathLen > max1) {
                    max2 = max1;
                    max1 = pathLen;
                } else if (pathLen > max2) {
                    max2 = pathLen;
                }
            }
            d = Math.max(d, max1 + max2);
            return Math.max(max1, max2) + 1;
        }

        class Node {
            public int val;
            public List<Node> children;


            public Node() {
                children = new ArrayList<Node>();
            }

            public Node(int _val) {
                val = _val;
                children = new ArrayList<Node>();
            }

            public Node(int _val, ArrayList<Node> _children) {
                val = _val;
                children = _children;
            }
        }
    }

    // LC1245 无向树的直径
    int lc1245Diameter = 0;

    public int treeDiameter(int[][] edges) {
        int n = edges.length + 1; // k条边, k+1个节点
        List<List<Integer>> mtx = new ArrayList<>(n);
        int[] indegree = new int[n];
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int[] e : edges) {
            mtx.get(e[0]).add(e[1]);
            mtx.get(e[1]).add(e[0]);
        }
        boolean[] visited = new boolean[n];
        lc1245Helper(0, visited, mtx);
        return lc1245Diameter;
    }

    private int lc1245Helper(int idx, boolean[] visited, List<List<Integer>> mtx) {
        visited[idx] = true;
        int max1 = 0, max2 = 0;
        for (int next : mtx.get(idx)) {
            if (!visited[next]) {
                int pathLen = lc1245Helper(next, visited, mtx);
                if (pathLen > max1) {
                    max2 = max1;
                    max1 = pathLen;
                } else if (pathLen > max2) {
                    max2 = pathLen;
                }
            }
        }
        lc1245Diameter = Math.max(lc1245Diameter, max1 + max2);
        return Math.max(max1, max2) + 1;
    }


    // LC492
    public int[] constructRectangle(int area) {
        int sqrt = (int) Math.sqrt(area);
        if (sqrt * sqrt == area) return new int[]{sqrt, sqrt};
        int minDelta = area - sqrt * sqrt;
        int[] result = new int[]{sqrt, sqrt};
        for (int w = sqrt; w >= 1; w--) {
            int l = area / w;
            int actualArea = l * w;
            int delta = area - actualArea;
            if (delta < minDelta) {
                minDelta = delta;
                result = new int[]{l, w};
            }
        }
        return result;
    }

    // LC1457
    int lc1457Result = 0, lc1457Mask = 0;

    public int pseudoPalindromicPaths(TreeNode root) {
        lc1457Helper(root);
        return lc1457Result;
    }

    private void lc1457Helper(TreeNode root) {
        if (root == null) return;
        lc1457Mask ^= (1 << root.val);
        if (root.left == null && root.right == null) {
            if (Integer.bitCount(lc1457Mask) <= 1) lc1457Result++;
        }
        lc1457Helper(root.left);
        lc1457Helper(root.right);
        lc1457Mask ^= (1 << root.val);
    }


    // LC487
    public int findMaxConsecutiveOnes(int[] nums) {
        if (nums.length == 1) return 1;
        if (nums.length == 2) {
            int sum = nums[0] + nums[1];
            if (sum > 0) return 2;
            return 1;
        }
        // 只能翻转1个0, 问最长连续1
        int n = nums.length;
        int[] rightOnes = new int[n]; // 右侧连续1的个数
        int oneCount = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            rightOnes[i] = oneCount;
            if (nums[i] == 0) oneCount = 0;
            else oneCount++;
        }
        int max = nums[0], leftOneCount = nums[0];
        for (int i = 1; i < n; i++) {
            max = Math.max(max, leftOneCount + rightOnes[i] + 1);
            if (nums[i] == 0) leftOneCount = 0;
            else leftOneCount++;
        }
        return max;
    }

    // LC1041 **
    public boolean isRobotBounded(String instructions) {
        int[][] direction = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        int d = 0, x = 0, y = 0;
        for (char c : instructions.toCharArray()) {
            switch (c) {
                case 'G':
                    x += direction[d][0];
                    y += direction[d][1];
                    break;
                case 'L':
                    d = (d - 1 + 4) % 4;
                    break;
                case 'R':
                    d = (d + 1) % 4;
                    break;
            }
        }
        return d != 0 || (x == 0 && y == 0);
    }

    // LC251
    class Vector2D {
        int x = 0, y = 0;
        int[][] vector;

        public Vector2D(int[][] vec) {
            vector = vec;
        }

        public int next() {
            int result = vector[x][y];
            if (y < vector[x].length - 1) y++;
            else {
                x++;
                y = 0;
            }
            return result;
        }

        public boolean hasNext() {
            if (x < vector.length && y < vector[x].length) return true;
            if (x >= vector.length) return false;
            if (y >= vector[x].length) {
                x++;
                y = 0;
                return hasNext();
            }
            return false;
        }
    }

    // JZOF 31 LC946 **
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new LinkedList<>();
        int popIdx = 0;
        for (int i = 0; i < pushed.length; i++) {
            stack.push(pushed[i]);
            while (!stack.isEmpty() && stack.peek() == popped[popIdx]) {
                stack.pop();
                popIdx++;
            }
        }
        return popIdx == popped.length;
    }


    // LC1969 **
    public int minNonZeroProduct(int p) {
        // a+b 和不变, |a-b|越大, a*b 越小
        // 如果算上0, 总共有 2^p 个数 -> 2^(p-1) 个数的正反码 -> 必然可以凑出 2^(p-1) 个1和(2^p -2)
        // 因为现在0不在, 所以总共凑出了(2^(p-1)-1)个1和(2^p -2), 以及一个(2^p -1)
        // 快速幂即可
        final long mod = 1000000007;
        long result = quickPowMod((1l << p) - 2l, (1l << (p - 1)) - 1l, mod);
        long remain = ((1l << p) - 1l) % mod;
        return (int) ((remain * result) % mod);
    }

    private long quickPowMod(long base, long exp, long mod) {
        long result = 1l;
        long accu = base % mod;
        while (exp != 0l) {
            if ((exp & 1l) == 1l) {
                result *= accu;
                result %= mod;
            }
            accu *= accu;
            accu %= mod;
            exp >>= 1l;
        }
        return result;
    }

    // LC767 **
    public String reorganizeString(String s) {
        PriorityQueue<Pair<Character, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o.getValue()));
        Deque<Pair<Character, Integer>> q = new LinkedList<>();
        Map<Character, Integer> freq = new HashMap<>(26);
        for (char c : s.toCharArray()) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }
        for (Map.Entry<Character, Integer> e : freq.entrySet()) {
            pq.offer(new Pair<>(e.getKey(), e.getValue()));
        }
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            Pair<Character, Integer> p = pq.poll();
            sb.append(p.getKey());
            q.offer(new Pair<>(p.getKey(), p.getValue() - 1));
            if (q.size() == 2) {
                if (q.peek().getValue() > 0) {
                    pq.offer(q.peek());
                }
                q.poll();
            }
        }
        return sb.length() < s.length() ? "" : sb.toString();
    }

    // LC294 博弈
    public boolean canWin(String currentState) {
        List<String> next = generatePossibleNextMoves(currentState);
        if (next.size() == 0) return false;
        for (String n : next) {
            if (!canWin(n)) return true;
        }
        return false;
    }

    // LC293
    public List<String> generatePossibleNextMoves(String currentState) {
        List<String> result = new ArrayList<>();
        for (int i = 1; i < currentState.length(); i++) {
            if (currentState.charAt(i) == '+' && currentState.charAt(i - 1) == '+') {
                result.add(currentState.substring(0, i - 1) + "--" + currentState.substring(i + 1));
            }
        }
        return result;
    }

    // LC1697 离线算法
    public boolean[] distanceLimitedPathsExist(int n, int[][] edgeList, int[][] queries) {
        boolean[] result = new boolean[queries.length];
        Arrays.sort(edgeList, Comparator.comparingInt(o -> o[2]));
        DSUArray dsu = new DSUArray(n);
        for (int i = 0; i < n; i++) dsu.add(i);
        int lastEdge = 0;
        Map<int[], Integer> queryIdxMap = new HashMap<>();
        for (int i = 0; i < queries.length; i++) {
            queryIdxMap.put(queries[i], i);
        }
        Arrays.sort(queries, Comparator.comparingInt(o -> o[2]));
        for (int i = 0; i < queries.length; i++) {
            int[] q = queries[i];
            int start = q[0], end = q[1], limit = q[2];
            int edgeIdx = lastEdge;
            while (edgeIdx < edgeList.length && edgeList[edgeIdx][2] < limit) {
                dsu.merge(edgeList[edgeIdx][0], edgeList[edgeIdx][1]);
                edgeIdx++;
            }
            if (dsu.isConnected(start, end)) result[queryIdxMap.get(q)] = true;
            lastEdge = edgeIdx;
        }
        return result;
    }

    // LC1769
    public int[] minOperations(String boxes) {
        int n = boxes.length();
        int leftOper = 0, leftOneCount = 0, rightOper = 0, rightOneCount = 0;
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            if (boxes.charAt(i) == '1') {
                rightOneCount++;
                rightOper += i;
            }
        }
        result[0] = rightOper;
        for (int i = 1; i < n; i++) {
            if (boxes.charAt(i - 1) == '1') {
                leftOneCount++;
                rightOneCount--;
            }
            leftOper += leftOneCount; // 每往左移一步, 左边的"1"到该点的路程各自加1, 总共加左边1的个数
            rightOper -= rightOneCount; // 如果这个位置本身是1, 通过这里的减, 其实自身到自身的路程已经为0了
            result[i] = leftOper + rightOper;
        }
        return result;
    }

    // LCP33 **
    public int storeWater(int[] bucket, int[] vat) {
        int n = bucket.length, maxVat = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) maxVat = Math.max(maxVat, vat[i]);
        if (maxVat == 0) return 0;
        int minOper = Integer.MAX_VALUE;
        for (int i = 1; i <= maxVat; i++) {
            minOper = Math.min(minOper, getOper(bucket, vat, i));
        }
        return minOper;
    }

    private int getOper(int[] bucket, int[] vat, int stores) { // stores: 倒水次数
        int n = bucket.length, oper = 0;
        for (int i = 0; i < n; i++) {
            int buck = vat[i] / stores + (vat[i] % stores == 0 ? 0 : 1); // 上取整, 即倒水之前要加到多少水
            oper += (buck > bucket[i] ? buck - bucket[i] : 0); // 如果已经超过最小值就不用再加水了
        }
        return stores + oper;
    }

    private boolean check(long[] cur, int[] vat) {
        for (int i = 0; i < cur.length; i++) {
            if (cur[i] < vat[i]) return false;
        }
        return true;
    }

    // LC358 ** Heap
    public String rearrangeString(String s, int k) {
        if (k == 0) return s;
        PriorityQueue<Pair<Character, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o.getValue()));
        Map<Character, Integer> freq = new HashMap<>(26);
        Deque<Pair<Character, Integer>> q = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
        for (Map.Entry<Character, Integer> e : freq.entrySet()) pq.offer(new Pair<>(e.getKey(), e.getValue()));
        while (!pq.isEmpty()) {
            Pair<Character, Integer> p = pq.poll();
            sb.append(p.getKey());
            q.offer(new Pair<>(p.getKey(), p.getValue() - 1));
            if (q.size() == k) {
                if (q.peek().getValue() > 0) {
                    pq.offer(q.peek());
                }
                q.poll();
            }
        }
        return sb.length() < s.length() ? "" : sb.toString();
    }


    // LC453
    public int minMoves(int[] nums) {
        // 每次n-1个数增加1, 问增加多少次相等
        // 等同于每次挑一个数减少1, 问总共减少多少次相等
        int min = Integer.MAX_VALUE;
        long sum = 0;
        for (int i : nums) {
            min = Math.min(min, i);
            sum += i;
        }
        return (int) (sum - (min + 0l) * (nums.length + 0l));
    }

    // LC1093
    public double[] sampleStats(int[] count) {
        // min max avg middle popular
        double[] result = new double[5];
        int total = 0;
        for (int i : count) total += i;
        if (total % 2 == 0) {
            // 找 half, half+1
            int first = -1, second = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half) {
                    first = i;
                }
                if (first != -1 && accu >= half + 1) {
                    second = i;
                    break;
                }
            }
            result[3] = (first + second + 0d) / 2d;
        } else {
            // 找half+1
            int first = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half + 1) {
                    first = i;
                    break;
                }
            }
            result[3] = first;
        }
        for (int i = 0; i < 256; i++) {
            if (count[i] != 0) {
                result[0] = i;
                break;
            }
        }
        for (int i = 255; i >= 0; i--) {
            if (count[i] != 0) {
                result[1] = i;
                break;
            }
        }
        long sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += (count[i] + 0l) * (i + 0l);
        }
        result[2] = (sum + 0d) / (total + 0d);
        int popCount = 0, popNum = -1;
        for (int i = 0; i < 256; i++) {
            if (count[i] > popCount) {
                popNum = i;
                popCount = count[i];
            }
        }
        result[4] = popNum;
        return result;
    }

    // LC1058
    public String minimizeError(String[] prices, int target) {
        double max = 0d, min = 0d, sum = 0d;
        List<Double> priceList = new ArrayList<>(prices.length);
        for (String p : prices) {
            double d = Double.parseDouble(p);
            if (p.endsWith("000")) { // 整数会对shouldFloorCount造成影响, 导致shouldFloorCount少1, 最终的和会变大
                target -= (int) d;
                continue;
            }
            priceList.add(d);
            max += Math.ceil(d);
            min += Math.floor(d);
            sum += d;
        }
        if (max < (target + 0d)) return "-1";
        if (min > (target + 0d)) return "-1";
        int shouldFloorCount = (int) (max - target);
        if (shouldFloorCount == 0) {
            return String.format("%.3f", max - sum);
        }
        if (shouldFloorCount == prices.length) {
            return String.format("%.3f", sum - min);
        }
        PriorityQueue<Double> floorPq = new PriorityQueue<>(Comparator.comparingDouble(d -> -(d - Math.floor(d))));
        List<Double> shouldCeilList = new ArrayList<>(priceList.size() - shouldFloorCount);
        for (double d : priceList) {
            if (floorPq.size() < shouldFloorCount) {
                floorPq.offer(d);
            } else {
                double deltaOut = d - Math.floor(d);
                double peek = floorPq.peek();
                double deltaIn = peek - Math.floor(peek);
                if (deltaOut < deltaIn) {
                    floorPq.poll();
                    floorPq.offer(d);
                    shouldCeilList.add(peek);
                } else {
                    shouldCeilList.add(d);
                }
            }
        }
        double totalDelta = 0d;
        while (!floorPq.isEmpty()) {
            double p = floorPq.poll();
            totalDelta += p - Math.floor(p);
        }
        for (double d : shouldCeilList) {
            totalDelta += Math.ceil(d) - d;
        }
        return String.format("%.3f", totalDelta);
    }


    // LC1508 前缀和 + 暴力
    public int rangeSum(int[] nums, int n, int left, int right) {
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + nums[i];
        List<Integer> accu = new ArrayList<>((n + 1) * n / 2);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                accu.add(prefix[j + 1] - prefix[i]);
            }
        }
        Collections.sort(accu);
        long sum = 0, mod = 1000000007;
        for (int i = left - 1; i < right; i++) {
            sum += accu.get(i);
            sum %= mod;
        }
        return (int) (sum % mod);
    }

    // LC1078
    public String[] findOcurrences(String text, String first, String second) {
        List<String> result = new ArrayList<>();
        String[] words = text.split(" ");
        String prev = '\0' + "";
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            if (prev.equals(first) && cur.equals(second)) {
                if (i + 1 < words.length) {
                    result.add(words[i + 1]);
                }
            }
            prev = cur;
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1815 Try Simulate Anneal 模拟退火
    class Lc1815 {
        List<Integer> toSatisfy = new ArrayList<>();
        int max = 0;
        int n;
        int batchSize;

        public int maxHappyGroups(int batchSize, int[] groups) {
            if (batchSize == 1) return groups.length;
            this.batchSize = batchSize;
            int modZeroGroupCount = 0;
            for (int i : groups) {
                if (i % batchSize == 0) modZeroGroupCount++;
                else toSatisfy.add(i);
            }
            if (toSatisfy.size() < 2) return modZeroGroupCount + toSatisfy.size();
            n = toSatisfy.size();
            // 多次使用模拟退火
            for (int i = 0; i < 32; i++) {
                simulateAnneal();
            }
            return modZeroGroupCount + max;
        }

        private void simulateAnneal() {
            Collections.shuffle(toSatisfy); // 随机化
            for (double t = 1e6; t >= 1e-5; t *= 0.97d) {
                int i = (int) (n * Math.random()), j = (int) (n * Math.random());
                int prevScore = evaluate();
                // 交换
                int tmp = toSatisfy.get(i);
                toSatisfy.set(i, toSatisfy.get(j));
                toSatisfy.set(j, tmp);

                int nextScore = evaluate();
                int delta = nextScore - prevScore;
                if (delta < 0 && Math.pow(Math.E, (delta / t)) <= (double) (Math.random())) {
                    tmp = toSatisfy.get(i);
                    toSatisfy.set(i, toSatisfy.get(j));
                    toSatisfy.set(j, tmp);
                }
            }
        }

        private int evaluate() {
            // 评价函数 直接返回在当前排列下有多少组可以好评
            int result = 0, sum = 0;
            for (int i : toSatisfy) {
                sum += i;
                if (sum % batchSize == 0) {
                    result++; // 能收到这一组的好评
                    sum = 0;
                }
            }
            if (sum > 0) result++; // 总是能收到第一组的好评
            max = Math.max(max, result);
            return result;
        }

    }


    // LC286
    public void wallsAndGates(int[][] rooms) {
        int m = rooms.length, n = rooms[0].length;
        final int INF = Integer.MAX_VALUE;
        int[][] direction = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int layer = -1;
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == 0) {
                    q.offer(new int[]{i, j});
                }
            }
        }
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                if (rooms[p[0]][p[1]] == INF) rooms[p[0]][p[1]] = layer;
                for (int[] d : direction) {
                    int nr = p[0] + d[0], nc = p[1] + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc] && rooms[nr][nc] != -1) {
                        q.offer(new int[]{nr, nc});
                    }
                }
            }
        }
        return;
    }

    // LC1976 **
    final long lc1976Inf = Long.MAX_VALUE / 2;
    Integer[] lc1976Memo;

    public int countPaths(int n, int[][] roads) {
        // 最短路的数量
        lc1976Memo = new Integer[n];
        long[][] matrix = new long[n][n];
        long[][] minDist = new long[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(matrix[i], lc1976Inf);
            Arrays.fill(minDist[i], lc1976Inf);
        }
        for (int[] r : roads) {
            matrix[r[0]][r[1]] = r[2];
            matrix[r[1]][r[0]] = r[2];
            minDist[r[0]][r[1]] = r[2];
            minDist[r[1]][r[0]] = r[2];
        }
        for (int i = 0; i < n; i++) {
            minDist[i][i] = 0;
        }
        // Floyd 求出任意两点间的最短距离
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (j != k) {
                        minDist[j][k] = minDist[k][j] = Math.min(minDist[j][k], minDist[j][i] + minDist[i][k]);
                    }
                }
            }
        }
        return lc1976Helper(0, minDist, matrix);
    }

    private int lc1976Helper(int cur, long[][] minDist, long[][] matrix) {
        if (cur == minDist.length - 1) return 1; // 已经到达最后一个节点
        if (lc1976Memo[cur] != null) return lc1976Memo[cur];
        int n = minDist.length;
        long result = 0;
        final long mod = 1000000007;
        for (int next = 0; next < n; next++) {
            if (matrix[cur][next] != lc1976Inf && minDist[0][cur] + minDist[next][n - 1] + matrix[cur][next] == minDist[0][n - 1]) {
                result += lc1976Helper(next, minDist, matrix);
                result %= mod;
            }
        }
        return lc1976Memo[cur] = (int) (result % mod);
    }

    // LC1017 **
    public String baseNeg2(int n) {
        StringBuilder sb = new StringBuilder();
        List<Integer> result = toBase(n, -2);
        for (int i : result) sb.append(i);
        return sb.toString();
    }

    public List<Integer> toBase(int num, int base) {
        if (num == 0) return Arrays.asList(0);
        List<Integer> result = new ArrayList<>();
        while (num != 0) {
            int r = ((num % base) + Math.abs(base)) % Math.abs(base);
            result.add(r);
            num -= r;
            num /= base;
        }
        Collections.reverse(result);
        return result;
    }

    // LC1256
    public String encode(int num) {
        // 0 -> ""
        // 2 -> 0
        // 3 -> 1
        // 4 -> 00
        // 5 -> 01
        // 6 -> 10
        // 7 -> 11
        // 8 -> 000
        if (num == 0) return "";
        num++;
        boolean flag = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            if ((num >> (32 - i - 1) & 1) == 1) {
                if (!flag) {
                    flag = true;
                    continue;
                }
            }
            if (flag) {
                sb.append(num >> (32 - i - 1) & 1);
            }
        }
        return sb.toString();
    }

    // LC1218
    Map<Integer, TreeSet<Integer>> lc1218IdxMap;
    Integer[] lc1218Memo;

    public int longestSubsequence(int[] arr, int difference) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        lc1218IdxMap = new HashMap<>();
        lc1218Memo = new Integer[n + 1];
        int max = 1;
        for (int i = 0; i < n; i++) {
            lc1218IdxMap.putIfAbsent(arr[i], new TreeSet<>());
            lc1218IdxMap.get(arr[i]).add(i);
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int result = 1 + lc1218Helper(i, difference, arr, visited);
                max = Math.max(max, result);
            }
        }
        return max;
    }

    private int lc1218Helper(int idx, int difference, int[] arr, boolean[] visited) {
        visited[idx] = true;
        if (lc1218Memo[idx] != null) return lc1218Memo[idx];
        int expected = arr[idx] + difference;
        if (lc1218IdxMap.get(expected) != null) {
            Integer nextIdx = lc1218IdxMap.get(expected).higher(idx);
            if (nextIdx != null) {
                return lc1218Memo[idx] = 1 + lc1218Helper(nextIdx, difference, arr, visited);
            }
        }
        return lc1218Memo[idx] = 0;
    }

    // JZOF II 102 LC494
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i : nums) sum += i;
        if (Math.abs(target) > sum) return 0;
        int OFFSET = sum, n = nums.length;
        int[][] dp = new int[2][OFFSET * 2 + 1];
        dp[0][OFFSET] = 1; // 加入0个数, 和为 (0+OFFSET) 的个数为0
        for (int i = 1; i <= n; i++) {
            for (int total = 0; total <= 2 * sum; total++) {
                int result = 0;
                // 背包问题
                if (total - nums[i - 1] >= 0) {
                    result += dp[(i - 1) % 2][total - nums[i - 1]];
                }
                if (total + nums[i - 1] <= 2 * OFFSET) {
                    result += dp[(i - 1) % 2][total + nums[i - 1]];
                }
                dp[i % 2][total] = result;
            }
        }
        return dp[n % 2][OFFSET + target];
    }

    // LC1087
    public String[] expand(String s) {
        List<String> l = braceExpansionII(s);
        return l.toArray(new String[l.size()]);
    }

    // LC1096 ** DFS
    public List<String> braceExpansionII(String expression) {
        Set<String> result = helper(expression);
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    private Set<String> helper(String chunk) {
        if (chunk.length() == 0) return new HashSet<>();
        Set<String> result = new HashSet<>();
        Set<String> peek = new HashSet<>();
        char[] ca = chunk.toCharArray();
        int i = 0, n = ca.length;
        while (i < n) {
            char c = ca[i];
            if (c == '{') {
                int numParenthesis = 1;
                int start = ++i; // 括号对内的起始下标(不包括括号)
                while (numParenthesis != 0) {
                    if (ca[i] == '{') numParenthesis++;
                    if (ca[i] == '}') numParenthesis--;
                    i++;
                }
                Set<String> next = helper(chunk.substring(start, i - 1));
                peek = merge(peek, next);
                continue;
            } else if (c == ',') {
                result.addAll(peek);
                peek.clear();
                i++;
                continue;
            } else { // 不会遍历到 '{'
                StringBuilder word = new StringBuilder();
                while (i < n && Character.isLetter(ca[i])) {
                    word.append(ca[i]);
                    i++;
                }
                Set<String> tmp = new HashSet<>();
                tmp.add(word.toString());
                peek = merge(peek, tmp);
            }
        }
        if (i == n) result.addAll(peek);
        return result;
    }

    private Set<String> merge(Set<String> prefix, Set<String> suffix) {
        if (suffix.size() == 0) return prefix;
        if (prefix.size() == 0) return suffix;
        Set<String> result = new HashSet<>();
        for (String p : prefix) {
            for (String s : suffix) {
                result.add(p + s);
            }
        }
        return result;
    }

    // ** BFS
    public List<String> braceExpansionIiBfs(String expression) {
        expression = "{" + expression + "}"; // 预防 "a,{b}c"这种情况
        Deque<String> q = new LinkedList<>();
        q.offer(expression);
        Set<String> result = new HashSet<>();
        while (!q.isEmpty()) {
            String p = q.poll();
            if (p.indexOf("{") < 0) {
                result.add(p);
                continue;
            }
            // ** 找最深的括号对
            int idx = 0, left = -1, right = -1;
            while (p.charAt(idx) != '}') {
                if (p.charAt(idx) == '{') left = idx;
                idx++;
            }
            right = idx;
            String prefix = p.substring(0, left);
            String suffix = p.substring(right + 1);
            String[] middle = p.substring(left + 1, right).split(",");

            for (String m : middle) {
                q.offer(prefix + m + suffix);
            }
        }
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    // LC1807
    public String evaluate(String s, List<List<String>> knowledge) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        Map<String, String> map = new HashMap<>(knowledge.size());
        for (List<String> k : knowledge) {
            map.put(k.get(0), k.get(1));
        }
        int left = -1, right = -1;
        boolean inParenthesis = false;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '(') {
                left = i;
                inParenthesis = true;
            } else if (c == ')') {
                right = i;
                inParenthesis = false;
                String key = s.substring(left + 1, right);
                sb.append(map.getOrDefault(key, "?"));
            } else {
                if (inParenthesis) continue;
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // LC814
    public TreeNode pruneTree(TreeNode root) {
        if (!subtreeHasOne(root)) return null;
        lc814Helper(root);
        return root;
    }

    private void lc814Helper(TreeNode root) {
        if (root == null) return;
        if (!subtreeHasOne(root.left)) {
            root.left = null;
        } else {
            lc814Helper(root.left);
        }
        if (!subtreeHasOne(root.right)) {
            root.right = null;
        } else {
            lc814Helper(root.right);
        }
    }

    private boolean subtreeHasOne(TreeNode root) {
        if (root == null) return false;
        if (root.val == 1) return true;
        return subtreeHasOne(root.left) || subtreeHasOne(root.right);
    }

    // LC306
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        // 选前两个数
        for (int i = 1; i <= n / 2; i++) {
            long first = Long.parseLong(num.substring(0, i));
            if (String.valueOf(first).length() != i) continue;
            for (int j = i + 1; j < n; j++) {
                long second = Long.parseLong(num.substring(i, j));
                if (String.valueOf(second).length() != j - i) continue;
                if (judge(first, second, j, num)) return true;
            }
        }
        return false;
    }

    private boolean judge(long first, long second, int idx, String num) {
        if (idx == num.length()) return true;
        long sum = first + second;
        if (num.indexOf(String.valueOf(sum), idx) != idx) return false;
        return judge(second, sum, idx + String.valueOf(sum).length(), num);
    }


    // LC311 矩阵乘法
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        // axb mult bxc = axc
        int a = mat1.length, b = mat1[0].length, c = mat2[0].length;
        int[][] result = new int[a][c];
        for (int i = 0; i < a; i++) {
            for (int k = 0; k < b; k++) {
                if (mat1[i][k] == 0) continue;
                for (int j = 0; j < c; j++) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return result;
    }

    // LC259 ** Solution O(n^2)
    public int threeSumSmaller(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int result = 0;
        for (int i = 0; i < n - 2; i++) {
            result += twoSumSmaller(nums, i + 1, target - nums[i]);
        }
        return result;
    }

    private int twoSumSmaller(int[] nums, int startIdx, int target) {
        int result = 0;
        int left = startIdx, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                result += right - left;
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    // LC1243
    public List<Integer> transformArray(int[] arr) {
        int n = arr.length;
        List<Integer> prev = Arrays.stream(arr).boxed().collect(Collectors.toList());
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < n; i++) cur.add(-1);
        while (true) {
            cur = helper(prev);
            if (cur.equals(prev)) return cur;
            prev = cur;
        }
    }

    private List<Integer> helper(List<Integer> prev) {
        int n = prev.size();
        List<Integer> cur = new ArrayList<>();
        cur.add(prev.get(0));
        for (int i = 1; i < n - 1; i++) {
            // 假如一个元素小于它的左右邻居，那么该元素自增 1。
            // 假如一个元素大于它的左右邻居，那么该元素自减 1。
            if (prev.get(i) < prev.get(i - 1) && prev.get(i) < prev.get(i + 1)) {
                cur.add(prev.get(i) + 1);
            } else if (prev.get(i) > prev.get(i - 1) && prev.get(i) > prev.get(i + 1)) {
                cur.add(prev.get(i) - 1);
            } else {
                cur.add(prev.get(i));
            }
        }
        cur.add(prev.get(n - 1));
        return cur;
    }


    // Interview 17.09 LC264 UglyNumber 丑数
    public int getKthMagicNumber(int k) {
        // Prime Factor 3,5,7
        long[] factor = {3, 5, 7};
        PriorityQueue<Long> pq = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        pq.offer(1l);
        set.add(1l);
        long result = -1;
        for (int i = 0; i < k; i++) {
            long p = pq.poll();
            result = p;
            for (long f : factor) {
                if (set.add(f * p)) {
                    pq.offer(f * p);
                }
            }
        }
        return (int) result;
    }

    // LC365
    public boolean canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        Deque<int[]> q = new LinkedList<>();
        Set<Pair<Integer, Integer>> visited = new HashSet<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{jug1Capacity, jug2Capacity});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            Pair<Integer, Integer> pair = new Pair<>(p[0], p[1]);
            if (visited.contains(pair)) continue;
            visited.add(pair);
            if (p[0] == targetCapacity || p[1] == targetCapacity) return true;
            if (p[0] + p[1] == targetCapacity) return true;
            // 倒满一侧
            pair = new Pair<>(jug1Capacity, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{jug1Capacity, p[1]});
            }
            pair = new Pair<>(p[0], jug2Capacity);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], jug2Capacity});
            }
            // 倒掉一侧
            pair = new Pair<>(0, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{0, p[1]});
            }
            pair = new Pair<>(p[0], 0);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], 0});
            }
            // 一侧倒向另一侧
            if (p[0] < jug1Capacity) {
                int jug1Empty = jug1Capacity - p[0];
                int jug2ToJug1 = Math.min(p[1], jug1Empty);
                pair = new Pair<>(p[0] + jug2ToJug1, p[1] - jug2ToJug1);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] + jug2ToJug1, p[1] - jug2ToJug1});
                }
            }
            if (p[1] < jug2Capacity) {
                int jug2Empty = jug2Capacity - p[1];
                int jug1ToJug2 = Math.min(p[0], jug2Empty);
                pair = new Pair<>(p[0] - jug1ToJug2, p[1] + jug1ToJug2);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] - jug1ToJug2, p[1] + jug1ToJug2});
                }
            }
        }
        return false;
    }

    // LC439 ** Great Solution
    public String parseTernary(String expression) {
        int len = expression.length();
        int level = 0;
        for (int i = 1; i < len; i++) {
            if (expression.charAt(i) == '?') level++;
            if (expression.charAt(i) == ':') level--;
            if (level == 0) {
                return expression.charAt(0) == 'T' ?
                        parseTernary(expression.substring(2, i)) : parseTernary(expression.substring(i + 1));
            }
        }
        return expression;
    }

    // LC385
    public NestedInteger deserialize(String s) {
        NestedInteger root = new NestedInteger();
        if (s.charAt(0) != '[') {
            root.setInteger(Integer.parseInt(s));
            return root;
        }
        Deque<NestedInteger> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '[') {
                NestedInteger next = new NestedInteger();
                stack.push(next);
            } else if (c == ']') {
                NestedInteger pop = stack.pop();
                if (sb.length() != 0) {
                    pop.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                if (!stack.isEmpty()) {
                    stack.peek().add(pop);
                    continue;
                } else {
                    return pop;
                }
            } else if (c == ',') {
                NestedInteger peek = stack.peek();
                if (sb.length() != 0) {
                    peek.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                continue;
            } else {
                sb.append(c);
            }
        }
        return null;
    }
}

// LC385
class NestedInteger {
    // Constructor initializes an empty nested list.
    public NestedInteger() {

    }

    // Constructor initializes a single integer.
    public NestedInteger(int value) {

    }

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger() {
        return false;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger() {
        return -1;
    }

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value) {
        ;
    }

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni) {
        ;
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return empty list if this NestedInteger holds a single integer
    public List<NestedInteger> getList() {
        return null;
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


class Trie {
    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }


    public boolean addWord(String word) {
        if (search(word)) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) {
                cur.children.put(c, new TrieNode());
            }
            cur = cur.children.get(c);
            cur.path++;
        }
        cur.end++;
        return true;
    }

    public boolean remove(String word) {
        if (!search(word)) return false;
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
        return target != null && target.end > 0;
    }

    public boolean beginWith(String prefix) {
        return getNode(prefix) != null;
    }

    private TrieNode getNode(String prefix) {
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


// LC211
class WordDictionary {
    Trie trie;

    public WordDictionary() {
        trie = new Trie();
    }

    public void addWord(String word) {
        trie.addWord(word);
    }

    public boolean search(String word) {
        return searchHelper("", word);
    }

    private boolean searchHelper(String prefix, String suffix) {
        if (suffix.equals("")) return trie.search(prefix);
        StringBuilder sb = new StringBuilder(prefix);
        for (int i = 0; i < suffix.length(); i++) {
            if (suffix.charAt(i) != '.') {
                sb.append(suffix.charAt(i));
            } else {
                for (int j = 0; j < 26; j++) {
                    sb.append((char) ('a' + j));
                    if (!trie.beginWith(sb.toString())) {
                        sb.deleteCharAt(sb.length() - 1);
                        continue;
                    }
                    if (searchHelper(sb.toString(), suffix.substring(i + 1))) return true;
                    sb.deleteCharAt(sb.length() - 1);
                }
                // 一旦所有'.'的可能性都被尝试, 且无一匹配, 即可返回false
                return false;
            }
        }
        // 如果全程没有'.', 返回Trie中是否有这个word
        return trie.search(sb.toString());
    }
}

// LC126
class Lc126 {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = new ArrayList<>();
        Node start = new Node(null, beginWord);
        Set<String> visited = new HashSet<>(wordList.size());
        Map<String, Set<String>> edge = new HashMap<>();
        for (String w : wordList) edge.put(w, new HashSet<>());
        edge.put(beginWord, new HashSet<>());
        for (int i = 0; i < wordList.size(); i++) {
            for (int j = i + 1; j < wordList.size(); j++) {
                String iw = wordList.get(i), jw = wordList.get(j);
                if (oneLetterDiff(iw.toCharArray(), jw.toCharArray())) {
                    edge.get(iw).add(jw);
                    edge.get(jw).add(iw);
                }
            }
            if (oneLetterDiff(wordList.get(i).toCharArray(), beginWord.toCharArray())) {
                edge.get(beginWord).add(wordList.get(i));
            }
        }

        Deque<Node> q = new LinkedList<>();
        List<Node> resultNodeList = new ArrayList<>();
        int layer = -1, finalLayer = -1;
        q.offer(start);
        while (!q.isEmpty()) {
            int qs = q.size();
            Set<String> thisLayerVisited = new HashSet<>();
            layer++;
            if (finalLayer != -1 && layer > finalLayer) break;
            for (int i = 0; i < qs; i++) {
                Node p = q.poll();
                if (p.word.equals(endWord)) {
                    finalLayer = layer;
                    resultNodeList.add(p);
                    continue;
                }
                if (visited.contains(p.word)) continue;
                thisLayerVisited.add(p.word);
                if (finalLayer == -1) {
                    for (String next : edge.get(p.word)) {
                        if (!visited.contains(next)) {
                            q.offer(new Node(p, next));
                        }
                    }
                }
            }
            // 注意求所有情况的时候不能根据在同一层已经访问过的边剪枝, 应该等该层全部访问后, 再将已访问节点标记
            visited.addAll(thisLayerVisited);
        }
        for (Node resultNode : resultNodeList) {
            List<String> r = new LinkedList<>();
            while (resultNode != null) {
                r.add(0, resultNode.word);
                resultNode = resultNode.prev;
            }
            result.add(r);
        }
        return result;
    }

    private boolean oneLetterDiff(char[] a, char[] b) {
        int ctr = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) ctr++;
            if (ctr == 2) return false;
        }
        return true;
    }

    class Node {
        Node prev;
        String word;

        public Node(Node prev, String word) {
            this.prev = prev;
            this.word = word;
        }
    }
}

class DisjointSetUnion<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(T i, T j) {
        if (!father.containsKey(i) || !father.containsKey(j)) return false;
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(T i) {
        return father.containsKey(i);
    }

}

class DSUArray {
    int[] father;
    int[] rank;
    int size;

    public DSUArray(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray() {
        this.size = Integer.MAX_VALUE >> 16;
        father = new int[Integer.MAX_VALUE >> 16];
        rank = new int[Integer.MAX_VALUE >> 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            rank[root]++;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }


}

// LC428
class Codec {
    // Encodes a tree to a single string.
    public String serialize(Node root) {
        if (root == null) return "()";
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        if (root.children == null || root.children.size() == 0) {
            sb.append("()");
            return sb.toString();
        }
        sb.append("(");
        for (Node c : root.children) {
            sb.append(serialize(c));
            sb.append(",");
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append(")");
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public Node deserialize(String data) {
        if (data.equals("()")) return null;
        int left = data.indexOf("("), right = data.lastIndexOf(")");
        Node root = new Node(Integer.parseInt(data.substring(0, left)));
        List<Node> children = new ArrayList<>();
        if (right - left == 1) {
            root.children = children;
            return root;
        }
        int idx = left + 1;
        boolean digitStart = false;
        int digitStartIdx = -1;
        while (idx < right) {
            if (Character.isDigit(data.charAt(idx))) {
                if (!digitStart) {
                    digitStart = true;
                    digitStartIdx = idx;
                }
                idx++;
                continue;
            } else if (data.charAt(idx) == '(') {
                // 开始括号匹配
                int parenthesisCount = 1, ppIdx = idx + 1;
                while (parenthesisCount != 0) {
                    if (data.charAt(ppIdx) == '(') parenthesisCount++;
                    if (data.charAt(ppIdx) == ')') parenthesisCount--;
                    ppIdx++;
                }
                children.add(deserialize(data.substring(digitStartIdx, ppIdx)));
                idx = ppIdx;
                digitStart = false;
                digitStartIdx = -1;
            } else {
                idx++;
            }
        }
        root.children = children;
        return root;
    }

    class Node {
        public int val;
        public List<Node> children;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    }
}


// LC431
class CodecLc431 {
    // left siblings right children
    // Encodes an n-ary tree to a binary tree.
    public TreeNode encode(Node root) {
        if (root == null) return null;
        TreeNode tn = new TreeNode(root.val);
        if (root.children.size() == 0) return tn;
        TreeNode rightForChildren = tn;
        TreeNode siblings = null, firstSiblings = null;
        for (Node c : root.children) {
            TreeNode cn = encode(c);
            if (siblings == null) {
                siblings = cn;
                firstSiblings = cn;
                continue;
            } else {
                siblings.left = cn;
                siblings = cn;
            }
        }
        tn.right = firstSiblings;
        return tn;
    }

    // Decodes your binary tree to an n-ary tree.
    public Node decode(TreeNode root) {
        if (root == null) return null;
        Node n = new Node(root.val);
        List<Node> children = new ArrayList<>();
        if (root.right == null) {
            n.children = children;
            return n;
        }
        TreeNode siblings = root.right;
        while (siblings != null) {
            children.add(decode(siblings));
            siblings = siblings.left;
        }
        n.children = children;
        return n;
    }

    class Node {
        public int val;
        public List<Node> children;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    }
}
