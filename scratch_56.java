import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

//        System.out.println(s.rearrangeString("cihefgcceegaaebajgcfchhegfcehjbchfbjdjgdcedd", 6));
//        System.out.println(s.rearrangeString("abcdabcdabdeac", 4));
//        System.out.println(s.rearrangeString("aabbcc", 3));
//        System.out.println(s.rearrangeString("abb", 2));
//        System.out.println(s.rearrangeString("aabbccdd", 3));
        System.out.println(s.rearrangeString("cijfgcdadbhffgjdjccbdgeihbdcbjdijajehgihihdijghcbffiedjahbdjbbjcfggaj", 6));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC453
    public int minMoves(int[] nums) {
        // 每次n-1个数增加1, 问增加多少次相等
        // 等同于每次挑一个数减少1, 问总共减少多少次相等
        int min = Arrays.stream(nums).min().getAsInt();
        int result = 0;
        for (int i : nums) result += i - min;
        return result;
    }

    // LC358 发牌算法 WA TBD
    char[] lc358Result;

    public String rearrangeString(String s, int k) {
        if (k == 0) return s;
        int n = s.length();
        // 使相同字母的间隔至少为k, 不行则返回空串
        int[] freq = new int[256];
        for (char c : s.toCharArray()) {
            freq[c]++;
        }
        int maxFreq = Arrays.stream(freq).max().getAsInt();
        int gapNum = n / k;
        int maxGap = n / k;
        int lastGapSize = 0;
        if (n % k != 0) {
            maxGap++;
            lastGapSize = n % k;
        }
        if (maxGap < maxFreq) return "";
        List<Pair<Character, Integer>> pairList = new ArrayList<>(26);
        for (int i = 0; i < 256; i++) {
            if (freq[i] != 0) {
                pairList.add(new Pair<>((char) i, freq[i]));
            }
        }
        pairList.sort(Comparator.comparingInt(o -> -o.getValue()));

        this.lc358Result = new char[n];

        // 数下有多少个freq 为 max的个数c, 然后先将最后的gap 的 len - c个用频率最少的几个来填充

        int maxFreqCount = 0;
        for (Pair<Character, Integer> p : pairList) {
            if (p.getValue() == maxFreq) {
                maxFreqCount++;
            } else {
                break;
            }
        }

        int alreadyFilled = 0;
        if (lastGapSize != 0) {
            int backIdx = n - 1;
            int origPairSize = pairList.size();
            int remainTurn = lastGapSize - maxFreqCount;
            if (remainTurn > 0) {
                alreadyFilled = remainTurn;
                for (int i = origPairSize - 1; i >= 0 && remainTurn > 0; i--) {
                    Pair<Character, Integer> lastPair = pairList.get(i);
                    pairList.remove(i);
                    lc358Result[backIdx--] = lastPair.getKey();
                    if (lastPair.getValue() != 1) {
                        pairList.add(new Pair<>(lastPair.getKey(), lastPair.getValue() - 1));
                    }
                    remainTurn--;
                }
            }
        }

        int idx = 0;
        char[] ca = new char[n - alreadyFilled];
        for (Pair<Character, Integer> p : pairList) {
            for (int i = 0; i < p.getValue(); i++) {
                ca[idx++] = p.getKey();
            }
        }

        for (int i = 0; i < n - alreadyFilled; i++) {
            idx = getIdx(i, n, maxGap, k);
            lc358Result[idx] = ca[i];
        }


        // 除了max freq 的之外 其他占据最后一个gap的都应该是频率为
        int[] lastOccur = new int[256];
        Arrays.fill(lastOccur, -1);
        for (int i = 0; i < n; i++) {
            if (lastOccur[lc358Result[i]] != -1) {
                if (i - lastOccur[lc358Result[i]] < k) return "";
            }
            lastOccur[lc358Result[i]] = i;
        }
        return new String(lc358Result);
    }

    private int getIdx(int current, int length, int gapNum, int gapLength) {
        int round = current / gapNum;
        int whichGap = current % gapNum;
        int result = round + whichGap * gapLength;

        if (result >= length) {
            return getIdx(current + 1, length, gapNum, gapLength);
        }
        if (this.lc358Result[result] != 0) {
            return getIdx(current + 1, length, gapNum, gapLength);
        }
        return result;
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