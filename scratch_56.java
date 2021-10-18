import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.longestSubsequence(new int[]{1, 3, 5, 7}, 1));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1218 TLE
    Map<Integer, TreeSet<Integer>> idxMap = new HashMap<>();

    public int longestSubsequence(int[] arr, int difference) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        int max = 1;
        for (int i = 0; i < n; i++) {
            idxMap.putIfAbsent(arr[i], new TreeSet<>());
            idxMap.get(arr[i]).add(i);
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int result = helper(i, difference, arr, visited);
                max = Math.max(max, result);
            }
        }
        return max;
    }

    private int helper(int idx, int difference, int[] arr, boolean[] visited) {
        visited[idx] = true;
        int expected = arr[idx] + difference;
        if(idxMap.get(expected)!=null){
            Integer nextIdx = idxMap.get(expected).higher(idx);
            if(nextIdx!=null){
                return 1 + helper(nextIdx, difference, arr, visited);
            }
        }
        return 0;
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
