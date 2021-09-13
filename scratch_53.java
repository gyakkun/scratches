
import javax.swing.plaf.TreeUI;
import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.hasPath(
                new int[][]{{0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 1, 0}, {1, 1, 0, 1, 1}, {0, 0, 0, 0, 0}},
                new int[]{0, 4},
                new int[]{4, 4}
        ));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC490
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        int m = maze.length, n = maze[0].length;
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int[] p = q.poll();
            if (visited[p[0]][p[1]]) continue;
            visited[p[0]][p[1]] = true;
            if (p[0] == destination[0] && p[1] == destination[1]) return true;

            // 上下左右 边缘都是墙壁

            // 上
            int upWallIdx = 0;
            for (int i = p[0]; i >= 0; i--) {
                if (maze[i][p[1]] == 1) {
                    upWallIdx = i + 1;
                    break;
                }
            }
            q.offer(new int[]{upWallIdx, p[1]});

            // 下
            int downWallIdx = m - 1;
            for (int i = p[0]; i < m; i++) {
                if (maze[i][p[1]] == 1) {
                    downWallIdx = i - 1;
                    break;
                }
            }
            q.offer(new int[]{downWallIdx, p[1]});

            // 左
            int leftWallIdx = 0;
            for (int i = p[1]; i >= 0; i--) {
                if (maze[p[0]][i] == 1) {
                    leftWallIdx = i + 1;
                    break;
                }
            }
            q.offer(new int[]{p[0], leftWallIdx});


            int rightWallIdx = n - 1;
            for (int i = p[1]; i < n; i++) {
                if (maze[p[0]][i] == 1) {
                    rightWallIdx = i - 1;
                    break;
                }
            }
            q.offer(new int[]{p[0], rightWallIdx});
        }
        return false;
    }

    // LC1273
    Map<Integer, Integer> lc1273ParentMap;
    Map<Integer, Set<Integer>> lc1273ChildrenMap;
    Integer[] lc1273SubTreeSum;
    int[] lc1273Value;

    public int deleteTreeNodes(int nodes, int[] parent, int[] value) {
        lc1273ParentMap = new HashMap<>();
        lc1273ChildrenMap = new HashMap<>();
        this.lc1273Value = value;
        lc1273SubTreeSum = new Integer[nodes];
        for (int i = 0; i < nodes; i++) {
            lc1273ParentMap.put(i, parent[i]);
            lc1273ChildrenMap.putIfAbsent(i, new HashSet<>());
            lc1273ChildrenMap.putIfAbsent(parent[i], new HashSet<>());
            lc1273ChildrenMap.get(parent[i]).add(i);
        }
        lc1273Helper(lc1273ChildrenMap.get(-1).iterator().next());
        Deque<Integer> toRemove = new LinkedList<>();
        for (int i = 0; i < nodes; i++) {
            if (lc1273SubTreeSum[i] == 0) {
                toRemove.offer(i);
            }
        }
        while (!toRemove.isEmpty()) {
            int n = toRemove.poll();
            if (lc1273ChildrenMap.containsKey(n)) {
                for (int child : lc1273ChildrenMap.get(n)) {
                    toRemove.offer(child);
                }
            }
            lc1273ChildrenMap.remove(n);
            if (lc1273ChildrenMap.containsKey(lc1273ParentMap.get(n)))
                lc1273ChildrenMap.get(lc1273ParentMap.get(n)).remove(n);
        }
        int result = 0;
        for (Set<Integer> s : lc1273ChildrenMap.values()) {
            result += s.size();
        }
        return result;
    }

    private int lc1273Helper(int idx) {
        if (lc1273SubTreeSum[idx] != null) return lc1273SubTreeSum[idx];
        lc1273SubTreeSum[idx] = lc1273Value[idx];
        for (int child : lc1273ChildrenMap.get(idx)) {
            lc1273SubTreeSum[idx] += lc1273Helper(child);
        }
        return lc1273SubTreeSum[idx];
    }

    // LC1779
    public int nearestValidPoint(int x, int y, int[][] points) {
        int minIdx = -1, minDistance = Integer.MAX_VALUE;
        for (int i = 0; i < points.length; i++) {
            int[] p = points[i];
            if (p[0] == x || p[1] == y) {
                int dis = Math.abs(p[0] - x) + Math.abs(p[1] - y);
                if (dis < minDistance) {
                    minIdx = i;
                    minDistance = dis;
                }
            }
        }
        return minIdx;
    }

    // LC1564
    public int maxBoxesInWarehouse(int[] boxes, int[] warehouse) {
        Arrays.sort(boxes);
        // Next Smaller Element
        int n = warehouse.length;
        int min = warehouse[0];
        for (int i = 0; i < n; i++) {
            if (warehouse[i] >= min) {
                warehouse[i] = min;
            } else {
                min = warehouse[i];
            }
        }
        int ptr = n - 1, result = 0;
        for (int i = 0; i < boxes.length; i++) {
            if (ptr < 0) break;
            while (ptr >= 0 && warehouse[ptr] < boxes[i]) ptr--;
            if (ptr-- >= 0) result++;
        }
        return result;
    }

    // LC1580 O(nlog(n))
    public int maxBoxesInWarehouseII(int[] boxes, int[] warehouse) {
        Arrays.sort(boxes);
        int n = warehouse.length;
        int min = Arrays.stream(warehouse).min().getAsInt();
        int leftMost = -1, rightMost = -1;
        int curMin = warehouse[0];
        for (int i = 0; i < n; i++) {
            if (warehouse[i] != min) {
                if (warehouse[i] >= curMin) {
                    warehouse[i] = curMin;
                } else {
                    curMin = warehouse[i];
                }
            } else {
                leftMost = i;
                break;
            }
        }

        curMin = warehouse[n - 1];
        for (int i = n - 1; i >= 0; i--) {
            if (warehouse[i] != min) {
                if (warehouse[i] >= curMin) {
                    warehouse[i] = curMin;
                } else {
                    curMin = warehouse[i];
                }
            } else {
                rightMost = i;
                break;
            }
        }

        for (int i = leftMost; i < rightMost; i++) {
            warehouse[i] = min;
        }

        int leftPtr = rightMost, rightPtr = rightMost + 1;
        int result = 0;

        for (int i = 0; i < boxes.length; i++) {
            if (leftPtr < 0 && rightPtr >= n) break;

            // 比较左右两边哪一侧的格子的绝对值差最小

            int shadowLeftPtr = leftPtr, shadowRightPtr = rightPtr;

            while (shadowLeftPtr >= 0 && warehouse[shadowLeftPtr] < boxes[i]) shadowLeftPtr--;
            while (shadowRightPtr < n && warehouse[shadowRightPtr] < boxes[i]) shadowRightPtr++;

            int leftDistance = -1;
            if (shadowLeftPtr >= 0) leftDistance = warehouse[shadowLeftPtr] - boxes[i];
            int rightDistance = -1;
            if (shadowRightPtr < n) rightDistance = warehouse[shadowRightPtr] - boxes[i];

            if (leftDistance == -1 && rightDistance == -1) break;
            if ((leftDistance == -1 && rightDistance != -1) || (leftDistance != -1 && rightDistance != -1 && leftDistance > rightDistance)) {
                rightPtr = ++shadowRightPtr;
                result++;
            } else if ((leftDistance != -1 && rightDistance == -1) || (leftDistance != -1 && rightDistance != -1 && leftDistance <= rightDistance)) {
                leftPtr = --shadowLeftPtr;
                result++;
            }

        }
        return result;

    }

    // LC1283 二分
    public int smallestDivisor(int[] nums, int threshold) {
        int lo = 1, hi = Integer.MAX_VALUE;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sumDivide(nums, mid) <= threshold) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int sumDivide(int[] nums, int divider) {
        int result = 0;
        for (int i : nums) result += divideUpper(i, divider);
        return result;
    }

    private int divideUpper(int a, int b) {
        if (b == 0) throw new ArithmeticException("Zero divider!");
        if (a == 0) return 0;
        if (a % b == 0) return a / b;
        return a / b + 1;
    }


    // LC1415
    class Lc1415 {
        int kth = 0;
        int targetTh, len;
        String result;
        char[] valid = {'a', 'b', 'c'};

        public String getHappyString(int n, int k) {
            targetTh = k;
            len = n;
            backtrack(new StringBuilder());
            if (result == null) return "";
            return result;
        }

        private void backtrack(StringBuilder cur) {
            if (cur.length() == len) {
                if (++kth == targetTh) {
                    result = cur.toString();
                }
                return;
            }
            for (char c : valid) {
                if ((cur.length() > 0 && cur.charAt(cur.length() - 1) != c) || cur.length() == 0) {
                    cur.append(c);
                    backtrack(cur);
                    cur.deleteCharAt(cur.length() - 1);
                }
            }
        }
    }

    // LC536
    public TreeNode str2tree(String s) {
        if (s.equals("")) return null;
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        boolean number = true, left = false, right = false;
        TreeNode root = new TreeNode();
        int val = -1, pair = 0;
        int startOfLeft = -1, endOfLeft = -1, startOfRight = -1, endOfRight = -1;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (number && (c == '(' || c == ')')) {
                number = false;
                val = Integer.valueOf(sb.toString());
                root.val = val;
                left = true;
                startOfLeft = i;
            }
            if (number) {
                sb.append(c);
            } else if (left) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfLeft = i;
                        TreeNode leftNode = str2tree(s.substring(startOfLeft + 1, endOfLeft));
                        root.left = leftNode;
                        startOfRight = i + 1;
                        right = true;
                        left = false;
                    }
                }
            } else if (right) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfRight = i;
                        TreeNode rightNode = str2tree(s.substring(startOfRight + 1, endOfRight));
                        root.right = rightNode;
                        right = false;
                    }
                }
            }
        }
        if (number) {
            val = Integer.valueOf(sb.toString());
            root.val = val;
        }
        return root;
    }

    // LC1608
    public int specialArray(int[] nums) {
        int[] count = new int[1001];
        for (int i : nums) count[i]++;
        int ctr = 0;
        for (int i = 1000; i >= 0; i--) {
            ctr += count[i];
            if (ctr == i) return i;
        }
        return -1;
    }

    // LC447
    public int numberOfBoomerangs(int[][] points) {
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int[] pi = points[i];
            Map<Integer, Integer> m = new HashMap<>();
            for (int j = 0; j < points.length; j++) {
                if (i != j) {
                    int[] pj = points[j];
                    int distance = (pi[0] - pj[0]) * (pi[0] - pj[0]) + (pi[1] - pj[1]) * (pi[1] - pj[1]);
                    m.put(distance, m.getOrDefault(distance, 0) + 1);
                }
            }
            for (int e : m.keySet()) {
                result += m.get(e) * (m.get(e) - 1);
            }
        }
        return result;
    }

    // LC1955 ** DP
    public int countSpecialSubsequences(int[] nums) {
        int i0 = 0, i1 = 0, i2 = 0;
        final int mod = 1000000007;
        for (int i : nums) {
            switch (i) {
                case 0:
                    i0 = ((i0 * 2) + 1) % mod;
                    break;
                case 1:
                    i1 = (((i1 * 2) % mod) + i0) % mod;
                    break;
                case 2:
                    i2 = (((i2 * 2) % mod) + i1) % mod;
                    break;
                default:
                    continue;
            }
        }
        return i2;
    }

    // LC600 ** 数位DP
    public int findIntegers(int n) {
        if (n == 0) return 1;
        int[] dp = new int[32];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < 32; i++) { // fib???
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        int prev = 0, result = 0;
        int len = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        for (int i = len; i >= 1; i--) {
            int cur = (n >> (i - 1)) & 1;
            if (cur == 1) {
                result += dp[i];
            }
            if (cur == 1 && prev == 1) break;
            prev = cur;
            if (i == 1) result++;
        }
        return result;
    }


    // LC898 ** 看题解
    // https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/solution/zi-shu-zu-an-wei-huo-cao-zuo-by-leetcode/
    public int subarrayBitwiseORs(int[] arr) {
        Set<Integer> result = new HashSet<>();
        Set<Integer> cur = new HashSet<>();
        for (int i : arr) {
            Set<Integer> tmp = new HashSet<>();
            for (int j : cur) { // 最多有32个数 (1的个数是递增的) ???
                tmp.add(i | j);
            }
            tmp.add(i); // 记得加上自身(长度为1)
            cur = tmp;
            result.addAll(cur);
        }
        return result.size();
    }

    // LC248
    public int strobogrammaticInRange(String low, String high) {
        int count = 0;
        for (int i = low.length(); i <= high.length(); i++) {
            List<String> result = findStrobogrammatic(i);
            if (i > low.length() && i < high.length()) {
                count += result.size();
                continue;
            }
            for (String s : result) {
                if (bigIntCompare(s, low) >= 0 && bigIntCompare(s, high) <= 0) {
                    count++;
                }
            }
        }
        return count;
    }

    private int bigIntCompare(String a, String b) {
        if (a.equals(b)) return 0;
        if (a.length() < b.length()) return -1;
        if (a.length() > b.length()) return 1;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) > b.charAt(i)) return 1;
            if (a.charAt(i) < b.charAt(i)) return -1;
        }
        return 0;
    }

    // LC247
    int[] validDigit = {0, 1, 6, 8, 9};
    int[] symmetryDigit = {0, 1, 8};
    List<String> lc247Result;

    public List<String> findStrobogrammatic(int n) {
        lc247Result = new ArrayList<>();
        if (n == 1) return Arrays.asList("0", "1", "8");
        lc247Helper(new StringBuilder(), n);
        return lc247Result;
    }

    private void lc247Helper(StringBuilder sb, int total) {
        if (sb.length() == total / 2) {
            if (sb.charAt(0) == '0') return;
            if (total % 2 == 1) {
                for (int i : symmetryDigit) {
                    String r = sb.toString() + i + getReverse(sb);
                    lc247Result.add(r);
                }
            } else {
                String r = sb + getReverse(sb);
                lc247Result.add(r);
            }
            return;
        }
        for (int i : validDigit) {
            if (i == 0 && sb.length() == 0) continue;
            sb.append(i);
            lc247Helper(sb, total);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    private String getReverse(StringBuilder input) {
        StringBuilder sb = new StringBuilder();
        for (int i = input.length() - 1; i >= 0; i--) {
            if (input.charAt(i) == '6') {
                sb.append('9');
            } else if (input.charAt(i) == '9') {
                sb.append('6');
            } else {
                sb.append(input.charAt(i));
            }
        }
        return sb.toString();
    }

    // LC246
    public boolean isStrobogrammatic(String num) {
        int[] notValid = {2, 3, 4, 5, 7};
        char[] ca = num.toCharArray();
        for (int i = 0; i <= ca.length / 2; i++) {
            char c = ca[i];
            for (int j : notValid) if (c - '0' == j) return false;
            if (c == '6') {
                if (ca[ca.length - 1 - i] != '9') return false;
            } else if (c == '9') {
                if (ca[ca.length - 1 - i] != '6') return false;
            } else {
                if (ca[ca.length - 1 - i] != c) return false;
            }
        }
        return true;
    }

    // LC1953 Hint: 只和最大时间有关
    public long numberOfWeeks(int[] milestones) {
        long sum = 0;
        long max = Long.MIN_VALUE;
        for (int i : milestones) {
            sum += i;
            max = Math.max(max, i);
        }
        long remain = sum - max;
        max = Math.min(remain + 1, max);
        return remain + max;
    }

    // LC249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<>();
        Map<Integer, Map<String, Integer>> m = new HashMap<>();
        for (String s : strings) {
            m.putIfAbsent(s.length(), new HashMap<>());
            Map<String, Integer> inner = m.get(s.length());
            inner.put(s, inner.getOrDefault(s, 0) + 1);
        }

        for (Map<String, Integer> s : m.values()) {
            while (!s.isEmpty()) {
                String w = s.keySet().iterator().next();
                // 构造
                List<String> list = new ArrayList<>();
                char[] ca = w.toCharArray();
                for (int i = 0; i < 26; i++) {
                    for (int j = 0; j < ca.length; j++) {
                        ca[j] = (char) (((ca[j] - 'a' + 1) % 26) + 'a');
                    }
                    String built = new String(ca);
                    if (s.containsKey(built)) {
                        int count = s.get(built);
                        s.remove(built);
                        for (int j = 0; j < count; j++)
                            list.add(built);
                    }
                }
                result.add(list);
            }
        }
        return result;
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

// JZOF II 30 LC380
class RandomizedSet {

    Map<Integer, Integer> idxMap = new HashMap<>();
    List<Integer> entities = new ArrayList<>();

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {

    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified element.
     */
    public boolean insert(int val) {
        if (idxMap.containsKey(val)) return false;
        idxMap.put(val, entities.size());
        entities.add(val);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!idxMap.containsKey(val)) return false;
        int lastEntity = entities.get(entities.size() - 1);
        int targetIdx = idxMap.get(val);
        entities.set(targetIdx, lastEntity);
        idxMap.put(lastEntity, targetIdx);
        idxMap.remove(val);
        entities.remove(entities.size() - 1);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        int idx = (int) (Math.random() * entities.size());
        return entities.get(idx);
    }
}