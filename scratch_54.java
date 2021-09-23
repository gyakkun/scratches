import java.math.BigInteger;
import java.util.*;
import java.util.function.Function;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.maximumBeauty(new int[]{-7, -8, 8, 0, -7}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1788 Hard
    public int maximumBeauty(int[] flowers) {
        int n = flowers.length;
        int[] prefix = new int[n + 1];
        int[] negPrefix = new int[n + 1];
        int[] negCount = new int[n + 1];
        Map<Integer, Integer> firstAppear = new HashMap<>();
        for (int i = 0; i < n; i++) {
            firstAppear.putIfAbsent(flowers[i], i);
            prefix[i + 1] = prefix[i] + flowers[i];
            negPrefix[i + 1] = negPrefix[i] + (flowers[i] > 0 ? 0 : flowers[i]);
            negCount[i + 1] = negCount[i] + (flowers[i] < 0 ? 1 : 0);
        }
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int firstAppearIdx = firstAppear.get(flowers[i]);
            if (firstAppearIdx == i) continue;
            int totalNumCount = i - firstAppearIdx + 1;
            int negNumCount = negCount[i + 1] - negCount[firstAppearIdx];
            int rest = totalNumCount - negNumCount;
            if (flowers[i] < 0) rest += 2;
            if (rest < 2) continue;
            int judge = prefix[i + 1] - prefix[firstAppearIdx] - (negPrefix[i + 1] - negPrefix[firstAppearIdx]);
            if (flowers[i] < 0) judge += 2 * flowers[i];
            result = Math.max(result, judge);
        }
        return result;
    }

    // JZOF II 070 二分 LC540
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid % 2 == 1) {
                mid--;
            }
            if (nums[mid] == nums[mid + 1]) {
                lo = mid + 2;
            } else {
                hi = mid;
            }
        }
        return nums[lo];
    }

    // LC1546 **
    public int maxNonOverlapping(int[] nums, int target) {
        int n = nums.length;
        int prefix = 0;
        int idx = 0;
        int result = 0;
        while (idx < n) {
            Set<Integer> prefixSet = new HashSet<>();
            prefixSet.add(0);
            prefix = 0;
            while (idx < n) {
                prefix = prefix + nums[idx];
                if (prefixSet.contains(prefix - target)) {
                    result++;
                    break;
                } else {
                    prefixSet.add(prefix);
                    idx++;
                }
            }
            idx++;
        }
        return result;
    }

    // LC889 ** 前序 后续 重建二叉树 不唯一
    public TreeNode constructFromPrePost(int[] preorder, int[] postorder) {
        if (preorder.length == 0) return null;
        TreeNode root = new TreeNode(preorder[0]);
        if (preorder.length == 1) return root;
        int len = 0;
        for (int i = 0; i < postorder.length; i++) {
            if (postorder[i] == preorder[1]) {
                len = i + 1; // + 1 方便算len
                break;
            }
        }

        root.left = constructFromPrePost(Arrays.copyOfRange(preorder, 1, 1 + len), Arrays.copyOfRange(postorder, 0, len));
        root.right = constructFromPrePost(Arrays.copyOfRange(preorder, 1 + len, preorder.length), Arrays.copyOfRange(postorder, len, postorder.length - 1));
        return root;
    }

    // LC640
    public String solveEquation(String equation) {
        final String INF_RES = "Infinite solutions";
        final String NO_SOL = "No solution";
        final String SOL_IS_ZERO = "x=0";

        String[] parts = equation.split("=");
        int[] left = lc640Helper(parts[0]), right = lc640Helper(parts[1]);

        int leftFactor = left[0], leftConst = left[1];
        int rightFactor = right[0], rightConst = right[1];

        // 无解 / 无限解
        if (leftFactor == rightFactor) {
            if (leftConst == rightConst) return INF_RES;
            else return NO_SOL;
        }
        // 解只有一个且为0
        if (leftConst == rightConst) {
            if (leftFactor != rightFactor) return SOL_IS_ZERO;
            else return INF_RES;
        }
        return "x=" + (rightConst - leftConst) / (leftFactor - rightFactor);
    }

    private int[] lc640Helper(String halfEq) {
        // 左侧
        int factor = 0, constVal = 0;
        boolean comesDigit = false;
        int cur = 0, sign = 1;
        for (char c : halfEq.toCharArray()) {
            if (c == 'x') {
                if (cur == 0 && !comesDigit) factor += 1 * sign;
                else factor += cur * sign;
                sign = 1;
                cur = 0;
                comesDigit = false;
            } else if (c == '+') {
                constVal += cur * sign;
                cur = 0;
                sign = 1;
                comesDigit = false;
            } else if (c == '-') {
                constVal += cur * sign;
                cur = 0;
                sign = -1;
                comesDigit = false;
            } else if (Character.isDigit(c)) {
                cur = cur * 10 + (c - '0');
                comesDigit = true;
            }
        }
        constVal += cur * sign;
        return new int[]{factor, constVal};
    }

    // LC1110
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> result = new ArrayList<>();
        Set<Integer> toDelete = new HashSet<>();
        for (int i : to_delete) toDelete.add(i);
        Function<TreeNode, TreeNode> dfs = new Function<TreeNode, TreeNode>() {
            @Override
            public TreeNode apply(TreeNode root) {
                if (root == null) return null;
                root.left = this.apply(root.left);
                root.right = this.apply(root.right);
                if (toDelete.contains(root.val)) {
                    if (root.left != null) {
                        result.add(root.left);
                    }
                    if (root.right != null) {
                        result.add(root.right);
                    }
                    root = null;
                }
                return root;
            }
        };
        root = dfs.apply(root);
        if (root != null) result.add(root);
        return result;
    }

    // LC1410
    public String entityParser(String text) {
        StringBuilder sb = new StringBuilder();
        int ptr = 0;
        while (ptr < text.length()) {
            if (text.charAt(ptr) != '&') {
                sb.append(text.charAt(ptr));
                ptr++;
                continue;
            }
            if (ptr + 6 <= text.length() && text.startsWith("&quot;", ptr)) {
                sb.append("\"");
                ptr += 6;
            } else if (ptr + 6 <= text.length() && text.startsWith("&apos;", ptr)) {
                sb.append("\'");
                ptr += 6;
            } else if (ptr + 5 <= text.length() && text.startsWith("&amp;", ptr)) {
                sb.append("&");
                ptr += 5;
            } else if (ptr + 4 <= text.length() && text.startsWith("&gt;", ptr)) {
                sb.append(">");
                ptr += 4;
            } else if (ptr + 4 <= text.length() && text.startsWith("&lt;", ptr)) {
                sb.append("<");
                ptr += 4;
            } else if (ptr + 7 <= text.length() && text.startsWith("&frasl;", ptr)) {
                sb.append("/");
                ptr += 7;
            } else {
                sb.append(text.charAt(ptr));
                ptr++;
            }
        }
        return sb.toString();
    }

    // LC1105 ** DP DFS
    public int minHeightShelves(int[][] books, int shelfWidth) {
        int n = books.length;
        Integer[][] memo = new Integer[n + 1][n + 1];
        return lc1105Helper(0, 0, books, shelfWidth, memo);
    }

    private int lc1105Helper(int cur, int floor, int[][] books, int shelfWidth, Integer[][] memo) {
        if (cur == books.length) {
            return 0;
        }
        if (memo[cur][floor] != null) return memo[cur][floor];
        int width = 0, height = 0;
        int result = Integer.MAX_VALUE / 2;
        int i;
        for (i = cur; i < books.length; i++) {
            if (width + books[i][0] > shelfWidth) break;
            width += books[i][0];
            height = Math.max(height, books[i][1]);
            int next = height + lc1105Helper(i + 1, floor + 1, books, shelfWidth, memo);
            result = Math.min(result, next);
        }
        return memo[cur][floor] = result;
    }

    // LC2007
    public int[] findOriginalArray(int[] changed) {
        List<Integer> result = new ArrayList<>();
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i : changed) freq.put(i, freq.getOrDefault(i, 0) + 1);
        // 如果有0, 则一定有2的倍数个0
        if (freq.containsKey(0)) {
            if (freq.get(0) % 2 == 1) return new int[0];
            for (int i = 0; i < freq.get(0) / 2; i++) result.add(0);
            freq.remove(0);
        }
        Arrays.sort(changed);  // 从小往大找, 规避如[16,32,8,64]的测例
        for (int i : changed) {
            if (!freq.containsKey(i) || freq.get(i) == 0) continue;
            if (i % 2 == 0) {
                int lack = 0;
                if (!freq.containsKey(i * 2)) lack++;
                if (!freq.containsKey(i / 2)) lack++;
                if (lack == 2) return new int[0];

                if (freq.containsKey(i * 2)) {
                    result.add(i);
                    freq.put(i, freq.get(i) - 1);
                    if (freq.get(i) == 0) freq.remove(i);
                    freq.put(i * 2, freq.get(i * 2) - 1);
                    if (freq.get(i * 2) == 0) freq.remove(i * 2);
                }

                if (freq.containsKey(i / 2)) {
                    result.add(i / 2);
                    freq.put(i / 2, freq.get(i / 2) - 1);
                    if (freq.get(i / 2) == 0) freq.remove(i / 2);
                    if (!freq.containsKey(i)) return new int[0];
                    freq.put(i, freq.get(i) - 1);
                    if (freq.get(i) == 0) freq.remove(i);
                }
            } else {
                if (!freq.containsKey(i * 2)) return new int[0];
                result.add(i);
                freq.put(i, freq.get(i) - 1);
                if (freq.get(i) == 0) freq.remove(i);
                freq.put(i * 2, freq.get(i * 2) - 1);
                if (freq.get(i * 2) == 0) freq.remove(i * 2);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC725
    public ListNode[] splitListToParts(ListNode head, int k) {
        ListNode[] result = new ListNode[k];
        int len = 0;
        ListNode cur = head;
        while (cur != null) {
            len++;
            cur = cur.next;
        }
        int partLen = len / k;
        int remain = 0;
        if (partLen == 0) {
            partLen = 1;
        } else {
            remain = len - k * partLen;
        }
        cur = head;
        int ptr = 0;
        int resultCtr = 0;
        ListNode partHead = head;
        while (ptr < len) {
            ptr++;
            if (ptr % partLen == 0) {
                if ((remain - 1) >= 0) {
                    cur = cur.next;
                    remain--;
                }
                if (cur == null) break;
                ListNode origNext = cur.next;
                cur.next = null;
                result[resultCtr++] = partHead;
                partHead = origNext;
                cur = origNext;
            } else {
                if (cur == null) break;
                cur = cur.next;
            }
        }
        return result;
    }

    // LC8
    public int myAtoi(String s) {
        char[] ca = s.toCharArray();
        boolean skipSpace = false;
        boolean checkSign = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (!skipSpace && c == ' ') continue;
            skipSpace = true;
            if (skipSpace && !checkSign) {
                checkSign = true;
                if (c != '+' && c != '-' && !Character.isDigit(c)) return 0;
                if (c == '-') {
                    sb.append('-');
                    continue;
                }
                if (c == '+') {
                    continue;
                }
            }
            if (!Character.isDigit(c)) break;
            sb.append(c);
        }
        if (sb.length() == 0) return 0;
        if (sb.toString().equals("-")) return 0;
        // 这里可以考虑:1)这里可以考虑和Integer.MAX/MIN .toString() 比较字符串字典序
        BigInteger b = new BigInteger(sb.toString());
        if (b.compareTo(new BigInteger("" + Integer.MAX_VALUE)) > 0) return Integer.MAX_VALUE;
        if (b.compareTo(new BigInteger("" + Integer.MIN_VALUE)) < 0) return Integer.MIN_VALUE;
        return Integer.parseInt(sb.toString());
    }

    // LC650 BFS
    public int minSteps(int n) {
        boolean[][] visited = new boolean[n * 2][n * 2];
        Deque<int[]> q = new LinkedList<>();
        // 当前个数 剪贴板个数
        q.offer(new int[]{1, 0});
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (p[0] == n) return layer;
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                // 复制
                if (p[0] <= n && !visited[p[0]][p[0]]) {
                    q.offer(new int[]{p[0], p[0]});
                }
                // 粘贴
                if (p[0] + p[1] <= n && !visited[p[0] + p[1]][p[0]]) {
                    // if (p[0] + p[1] == n) return layer + 1;  //少这步特判会快1ms
                    q.offer(new int[]{p[0] + p[1], p[1]});
                }
            }
        }
        return -1;
    }

    // LC758 LC616 就硬匹配 问你气不气
    public String boldWords(String[] words, String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        boolean[] mask = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (String w : words) {
                int idx = i;
                while ((idx = s.indexOf(w, idx)) != -1) {
                    for (int j = 0; j < w.length(); j++) {
                        mask[idx + j] = true;
                    }
                    idx += w.length();
                }
            }
        }
        StringBuilder result = new StringBuilder();
        int ptr = 0;
        int boldLen = 0;
        while (ptr < ca.length) {
            if (!mask[ptr]) {
                if (boldLen > 0) {
                    result.append("<b>");
                    result.append(s, ptr - boldLen, ptr);
                    result.append("</b>");
                    result.append(ca[ptr]);
                    boldLen = 0;
                } else {
                    result.append(ca[ptr]);
                }
            } else {
                boldLen++;
            }
            ptr++;
        }
        if (boldLen > 0) {
            result.append("<b>");
            result.append(s, ptr - boldLen, ptr);
            result.append("</b>");
        }
        return result.toString();
    }

    // LC1309
    public String freqAlphabets(String s) {
        StringBuilder sb = new StringBuilder();
        int ptr = 0;
        char[] ca = s.toCharArray();
        while (ptr < ca.length) {
            if (ptr + 2 < ca.length && ca[ptr + 2] == '#') {
                int idx = (ca[ptr] - '0') * 10 + (ca[ptr + 1] - '0') - 1;
                sb.append((char) ('a' + idx));
                ptr += 3;
            } else {
                sb.append((char) ('a' + ca[ptr] - '1'));
                ptr++;
            }
        }
        return sb.toString();
    }

    // LC1602
    public TreeNode findNearestRightNode(TreeNode root, TreeNode u) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                TreeNode p = q.poll();
                if (p == u) {
                    if (i == qs - 1) return null;
                    return q.poll();
                }
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
        }
        return null;
    }

    // LC1199 Hard 可以二分, 也可以直接优先队列解决, 非常优雅
    public int minBuildTime(int[] blocks, int split) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i : blocks) {
            pq.offer(i);
        }
        while (pq.size() > 1) {
            // 队列里最小的两个任务耗时中的大者+分裂时间, 可视为大者分裂出另一个工人完成小两者中较小的任务后, 再完成自己任务所耗的总时间
            // 这样较小的任务的时间就被"掩盖"了
            pq.offer(Math.max(pq.poll(), pq.poll()) + split);
        }
        return pq.poll();
    }


    // LCP 04 匈牙利算法 二分图的最大匹配 Hard **
    public int domino(int n, int m, int[][] broken) {
        // 统计
        Set<Integer> brokenSet = new HashSet<>();
        int[][] direction = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] b : broken) {
            brokenSet.add(b[0] * m + b[1]);
        }
        // 建图
        List<List<Integer>> mtx = new ArrayList<>(m * n); // 邻接矩阵
        for (int i = 0; i < m * n; i++) {
            mtx.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int idx = i * m + j;
                if (!brokenSet.contains(idx)) {
                    for (int[] d : direction) {
                        if (lcp04Check(i + d[0], j + d[1], n, m, brokenSet)) {
                            int nextIdx = (i + d[0]) * m + j + d[1];
                            mtx.get(idx).add(nextIdx);
                        }
                    }
                }
            }
        }
        boolean[] visited;
        int[] p = new int[m * n];
        Arrays.fill(p, -1);
        int result = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if ((i + j) % 2 == 0 && !brokenSet.contains(i * m + j)) {
                    visited = new boolean[m * n];
                    if (lcp04(i * m + j, visited, mtx, p, brokenSet)) {
                        result++;
                    }
                }
            }
        }
        return result;
    }

    private boolean lcp04(int i, boolean[] visited, List<List<Integer>> mtx, int[] p, Set<Integer> brokenSet) {
        if (brokenSet.contains(i)) return false;
        for (int next : mtx.get(i)) {
            if (!visited[next]) {
                visited[next] = true;
                if (p[next] == -1 || lcp04(p[next], visited, mtx, p, brokenSet)) {
                    p[next] = i;
                    return true;
                }
            }
        }
        return false;
    }

    private boolean lcp04Check(int row, int col, int n, int m, Set<Integer> brokenSet) {
        return row >= 0 && row < n && col >= 0 && col < m && !brokenSet.contains(row * m + col);
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
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return cur.end > 0;
    }

    public boolean startsWith(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return true;
    }
}

class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    int end = 0;
    int path = 0;
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