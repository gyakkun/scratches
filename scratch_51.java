import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.numSubarrayBoundedMax(new int[]{2, 1, 4, 3}, 2, 3));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1679
    public int maxOperations(int[] nums, int k) {
        Map<Integer, Integer> m = new HashMap<>();
        int result = 0;
        for (int i : nums) {
            if (m.containsKey(k - i)) {
                result += 1;
                m.put(k - i, m.get(k - i) - 1);
                if (m.get(k - i) == 0) {
                    m.remove(k - i);
                }
            } else {
                m.put(i, m.getOrDefault(i, 0) + 1);
            }
        }
        return result;
    }

    // LC795 ** from Solution
    public int numSubarrayBoundedMax(int[] nums, int lowBound, int highBound) {
        return lc795Helper(nums, highBound) - lc795Helper(nums, lowBound - 1);
    }

    private int lc795Helper(int[] arr, int bound) { // 最大值小于等于bound的子数组的数量
        int cur = 0, result = 0;
        for (int i : arr) {
            cur = i <= bound ? cur + 1 : 0;
            result += cur;
        }
        return result;
    }

    // LC258
    public int addDigits(int num) {
        if (num < 10) return num;
        while (num >= 10) {
            int sum = 0;
            while (num != 0) {
                sum += num % 10;
                num /= 10;
            }
            num = sum;
        }
        return num;
    }

    // Interview 01.07
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) { // 先上下 再斜对角线(左上, 右下)
            for (int j = 0; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = tmp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }

    // LC797
    List<List<Integer>> lc797Result;

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        lc797Result = new ArrayList<>();
        List<Integer> path = new ArrayList<>(graph.length);
        path.add(0);
        lc797Dfs(0, graph, path);
        return lc797Result;
    }

    private void lc797Dfs(int cur, int[][] graph, List<Integer> path) {
        if (cur == graph.length - 1) {
            lc797Result.add(new ArrayList<>(path));
            return;
        }
        for (int next : graph[cur]) {
            path.add(next);
            lc797Dfs(next, graph, path);
            path.remove(path.size() - 1);
        }
    }

    // LC1671
    public int minimumMountainRemovals(int[] nums) {
        int n = nums.length, result = nums.length;
        int[] left = new int[n], right = new int[n];
        int[] leftTop = new int[n], rightTop = new int[n];
        left[0] = 1;
        leftTop[0] = nums[0];
        right[n - 1] = 1;
        rightTop[n - 1] = nums[n - 1];
        TreeSet<Integer> ts = new TreeSet<>();
        ts.add(nums[0]);
        for (int i = 1; i < n; i++) {
            Integer ceil = ts.ceiling(nums[i]);
            if (ceil != null) {
                ts.remove(ceil);
            }
            ts.add(nums[i]);
            left[i] = ts.size();
            leftTop[i] = ts.last();
        }
        ts.clear();
        ts.add(nums[n - 1]);
        for (int i = n - 2; i >= 0; i--) {
            Integer ceil = ts.ceiling(nums[i]);
            if (ceil != null) {
                ts.remove(ceil);
            }
            ts.add(nums[i]);
            right[i] = ts.size();
            rightTop[i] = ts.last();
        }
        for (int i = 1; i <= n - 2; i++) {
            // 判定条件: 同为两边的峰, 且长度必须超过1
            if (leftTop[i] == rightTop[i] && left[i] != 1 && right[i] != 1) {
                result = Math.min(result, n - (left[i] + right[i] - 1));
            }
        }
        return result;
    }

    // LC1099
    public int twoSumLessThanK(int[] nums, int k) {
        if (nums.length == 0) return -1;
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1, result = Integer.MIN_VALUE;
        while (left < right) {
            if (nums[left] + nums[right] < k) {
                result = Math.max(result, nums[left] + nums[right]);
                left++;
            } else if (nums[left] + nums[right] >= k) right--;
        }
        return result == Integer.MIN_VALUE ? -1 : result;
    }

    // Interview 16.05 ** LC172
    public int trailingZeroes(int n) {
        int result = 0;
        while (n != 0) {
            n /= 5;
            result += n;
        }
        return result;
    }

    // LC67 JZOF II 002
    public String addBinary(String a, String b) {
        int carry = 0;
        StringBuilder sb = new StringBuilder();
        int i = 0;
        for (; i < Math.min(a.length(), b.length()); i++) {
            int abit = a.charAt(a.length() - 1 - i) - '0', bbit = b.charAt(b.length() - 1 - i) - '0';
            int thisBit = (carry + abit + bbit) % 2;
            carry = (carry + abit + bbit) / 2;
            sb.append(thisBit);
        }
        while (i < a.length()) {
            int bit = a.charAt(a.length() - 1 - i) - '0';
            int thisBit = (carry + bit) % 2;
            carry = (carry + bit) / 2;
            sb.append(thisBit);
            i++;
        }
        while (i < b.length()) {
            int bit = b.charAt(b.length() - 1 - i) - '0';
            int thisBit = (carry + bit) % 2;
            carry = (carry + bit) / 2;
            sb.append(thisBit);
            i++;
        }
        if (carry == 1) {
            sb.append(1);
        }
        return sb.reverse().toString();
    }

    // JZOF II 098
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][1] = 1;
        }
        for (int i = 1; i <= n; i++) {
            dp[1][i] = 1;
        }
        for (int i = 2; i <= m; i++) {
            for (int j = 2; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m][n];
    }

    // LC1749
    public int maxAbsoluteSum(int[] nums) {
        int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE, curSeg = 0;
        for (int i : nums) {
            curSeg = Math.max(i, curSeg + i);
            max = Math.max(max, curSeg);
        }

        curSeg = 0;
        for (int i : nums) {
            curSeg = Math.min(i, curSeg + i);
            min = Math.min(min, curSeg);
        }
        int absMax = Math.max(0, Math.max(Math.abs(max), Math.abs(min)));
        return absMax;
    }

    // LC1519
    Integer[][] lc1519Memo;

    public int[] countSubTrees(int n, int[][] edges, String labels) {
        // 0 是根
        lc1519Memo = new Integer[n + 1][];
        boolean[] visited = new boolean[n];
        char[] labelCa = labels.toCharArray();
        int[] result = new int[n];
        List<List<Integer>> edgeList = new ArrayList<>(n);
        List<List<Integer>> outList = new ArrayList<>(n);
        for (int i = 0; i < n; i++) edgeList.add(new ArrayList<>());
        for (int i = 0; i < n; i++) outList.add(new ArrayList<>());
        for (int[] e : edges) {
            edgeList.get(e[0]).add(e[1]);
            edgeList.get(e[1]).add(e[0]);
        }
        Deque<Integer> q = new LinkedList<>();
        q.offer(0);
        while (!q.isEmpty()) {
            int p = q.poll();
            if (visited[p]) continue;
            visited[p] = true;
            for (int next : edgeList.get(p)) {
                if (!visited[next]) {
                    outList.get(p).add(next);
                    q.offer(next);
                }
            }
        }
        lc1519Dfs(0, labelCa, result, outList);
        for (int i = 0; i < n; i++) {
            result[i] = lc1519Memo[i][labelCa[i] - 'a'];
        }
        return result;
    }

    private Integer[] lc1519Dfs(int cur, char[] labelCa, int[] result, List<List<Integer>> outList) {
        if (lc1519Memo[cur] != null) return lc1519Memo[cur];
        Integer[] count = new Integer[26];
        for (int i = 0; i < 26; i++) count[i] = 0;
        count[labelCa[cur] - 'a']++;
        for (int next : outList.get(cur)) {
            Integer[] childColorTable = lc1519Dfs(next, labelCa, result, outList);
            for (int i = 0; i < 26; i++) {
                count[i] += childColorTable[i];
            }
        }
        return lc1519Memo[cur] = count;
    }

    // LC205
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> m = new HashMap<>();
        Map<Character, Character> reverseM = new HashMap<>();
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        for (int i = 0; i < cs.length; i++) {
            if (!m.containsKey(cs[i])) {
                m.put(cs[i], ct[i]);
                if (!reverseM.containsKey(ct[i])) {
                    reverseM.put(ct[i], cs[i]);
                } else {
                    return false;
                }
            } else {
                if (m.get(cs[i]) != ct[i]) return false;
                if (reverseM.get(ct[i]) != cs[i]) return false;
            }
        }
        return true;
    }

    // LC787
    int lc787Result = Integer.MAX_VALUE;

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        // k站中转, 即最多可以坐k+1次航班
        // flights[i] = [fromi, toi, pricei]
        int[][] reach = new int[n][n];
        int[] maxStop = new int[n]; // 经停i 的最大跳数
        int[] minCost = new int[n];
        Arrays.fill(minCost, Integer.MAX_VALUE / 2);
        for (int[] r : reach) Arrays.fill(r, -1);
        for (int[] f : flights) reach[f[0]][f[1]] = f[2];
        lc787Dfs(src, dst, reach, maxStop, minCost, k + 1, 0);
        return lc787Result == Integer.MAX_VALUE ? -1 : lc787Result;
    }

    private void lc787Dfs(int cur, int dst, int[][] reach, int[] maxStop, int[] minCost, int limit, int price) {
        if (cur == dst) {
            lc787Result = Math.min(lc787Result, price);
            return;
        }
        if (limit > 0) {
            for (int i = 0; i < reach.length; i++) {
                if (reach[cur][i] != -1) {
                    int minCostToI = minCost[i], costFromCurToI = price + reach[cur][i];
                    if (costFromCurToI > lc787Result) continue; // 剪枝, 如果在这一站的中转都要比最小结果大, 就没必要DFS下去了
                    if (minCostToI > costFromCurToI) {
                        lc787Dfs(i, dst, reach, maxStop, minCost, limit - 1, costFromCurToI);
                        minCost[i] = costFromCurToI;
                        maxStop[i] = limit - 1;
                    } else if (maxStop[i] < limit - 1) {
                        lc787Dfs(i, dst, reach, maxStop, minCost, limit - 1, costFromCurToI);
                    }
                }
            }
        }
    }

    // LC282 ** Hard
    List<String> lc282Result = new ArrayList<>();

    public List<String> addOperators(String num, int target) {
        lc282Dfs(0, num, target, new StringBuilder(), 0, 0, 0);
        return lc282Result;
    }

    private void lc282Dfs(int idx, String num, int target, StringBuilder sb, int cur, int pre, int accumulate) {
        if (idx == num.length()) {
            if (accumulate == target && cur == 0) {
                lc282Result.add(sb.substring(1));
            }
            return;
        }
        // 溢出判断
        if ((cur * 10 + num.charAt(idx) - '0') / 10 != cur) return;

        cur = cur * 10 + num.charAt(idx) - '0';
        String curStr = String.valueOf(cur);

        // 空操作
        if (cur > 0) {
            lc282Dfs(idx + 1, num, target, sb, cur, pre, accumulate);
        }

        // +
        sb.append("+");
        sb.append(cur);
        lc282Dfs(idx + 1, num, target, sb, 0, cur, accumulate + cur);
        sb.delete(sb.length() - 1 - curStr.length(), sb.length());


        if (sb.length() != 0) {

            // -
            sb.append("-");
            sb.append(cur);
            lc282Dfs(idx + 1, num, target, sb, 0, -cur, accumulate - cur);
            sb.delete(sb.length() - 1 - curStr.length(), sb.length());

            // *
            sb.append("*");
            sb.append(cur);
            lc282Dfs(idx + 1, num, target, sb, 0, cur * pre, accumulate - pre + cur * pre);
            sb.delete(sb.length() - 1 - curStr.length(), sb.length());

        }
    }

    // LC1896 ** Hard
    public int minOperationsToFlip(String expression) {
        Deque<int[]> stack = new LinkedList<>(); // [p,q] 表示变为0要p步, 变为1要q步
        Deque<Character> ops = new LinkedList<>();
        Set<Character> rightSideOp = new HashSet<Character>() {{
            add('0');
            add('1');
            add(')');
        }};
        for (char c : expression.toCharArray()) {
            if (rightSideOp.contains(c)) {
                if (c == '0') stack.push(new int[]{0, 1});
                else if (c == '1') stack.push(new int[]{1, 0});
                else if (c == ')') ops.pop();

                if (ops.size() != 0 && ops.peek() != '(') {
                    char op = ops.pop();
                    int[] pair2 = stack.pop();
                    int[] pair1 = stack.pop();
                    int[] newEntry;
                    if (op == '&') {
                        newEntry = new int[]{
                                Math.min(pair1[0], pair2[0]),
                                Math.min(pair1[1] + pair2[1], Math.min(pair1[1], pair2[1]) + 1)
                        };
                    } else {
                        newEntry = new int[]{
                                Math.min(pair1[0] + pair2[0], Math.min(pair1[0], pair2[0]) + 1),
                                Math.min(pair1[1], pair2[1])
                        };
                    }
                    stack.push(newEntry);
                }
            } else {
                ops.push(c);
            }
        }
        int[] last = stack.pop();
        return Math.max(last[0], last[1]);
    }

    // LC1223
    public List<String> removeSubfolders(String[] folder) {
        Set<String> prefix = new HashSet<>();
        Arrays.sort(folder, Comparator.comparingInt(o -> o.length()));
        for (String f : folder) {
            int ptr = 0;
            while (ptr < f.length()) {
                int last = ptr + 1;
                while (last != f.length() && f.charAt(last) != '/') last++;
                if (prefix.contains(f.substring(0, last))) break;
                ptr = last;
            }
            if (ptr == f.length()) prefix.add(f);
        }
        return new ArrayList<>(prefix);
    }
}

// LC 1226 哲学家进餐
class DiningPhilosophers {

    ReentrantLock[] locks = new ReentrantLock[]{
            new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock()
    };

    ReentrantLock pickBoth = new ReentrantLock();


    public DiningPhilosophers() {

    }

    // call the run() method of any runnable to execute its code
    public void wantsToEat(int philosopher,
                           Runnable pickLeftFork,
                           Runnable pickRightFork,
                           Runnable eat,
                           Runnable putLeftFork,
                           Runnable putRightFork) throws InterruptedException {
        int left = (philosopher + 1) % 5, right = philosopher;

        pickBoth.lock();

        locks[left].lock();
        locks[right].lock();
        pickLeftFork.run();
        pickRightFork.run();

        pickBoth.unlock();

        eat.run();

        putLeftFork.run();
        putRightFork.run();

        locks[left].unlock();
        locks[right].unlock();
    }
}

// Interview 16.02
class WordsFrequency {
    Map<Pair<Integer, Integer>, Integer> m = new HashMap<>();

    public WordsFrequency(String[] book) {
        for (String word : book) {
            Pair<Integer, Integer> tmp = new Pair<>(word.hashCode(), word.length());
            m.put(tmp, m.getOrDefault(tmp, 0) + 1);
        }
    }

    public int get(String word) {
        return m.getOrDefault(new Pair<>(word.hashCode(), word.length()), 0);
    }
}

// JZOF II 062
class Trie {

    TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new TrieNode();
            }
            cur = cur.children[c - 'a'];
        }
        cur.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return cur.isEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
        for (char c : prefix.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return true;
    }

    class TrieNode {
        TrieNode[] children = new TrieNode[26];
        boolean isEnd = false;
    }
}