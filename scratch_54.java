import javafx.util.Pair;

import java.math.BigInteger;
import java.util.*;
import java.util.List;
import java.util.function.Function;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.stoneGameII(new int[]{8270, 7145, 575, 5156, 5126, 2905, 8793, 7817, 5532, 5726, 7071, 7730, 5200, 5369, 5763, 7148, 8287, 9449, 7567, 4850, 1385, 2135, 1737, 9511, 8065, 7063, 8023, 7729, 7084, 8407}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1140 ** 注意状态定义
    int[] prefix;
    int[][] memo;
    BitSet visit;

    public int stoneGameII(int[] piles) {
        int n = piles.length;
        prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + piles[i];
        memo = new int[n + 1][n + 1];
        visit = new BitSet((n + 1) * (n + 1));
        return helper(n, 1, 0);
    }

    private int helper(int n, int m, int ptr) { // 返回的是先手能从剩下的石堆中能取到的最多的石子数量
        if (ptr == n) return 0;
        if (visit.get(m * (n + 1) + ptr)) return memo[m][ptr];
        visit.set(m * (n + 1) + ptr);
        // 如果取值范围覆盖到了n, 则全部拿走
        if (ptr + 2 * m >= n) return memo[m][ptr] = prefix[n] - prefix[ptr];
        int maxGain = Integer.MIN_VALUE;
        for (int len = 1; len <= 2 * m; len++) {
            if (ptr + len > n) break;
            int remain = prefix[n] - prefix[ptr];
            int myGainThisTime = prefix[len + ptr] - prefix[ptr];
            int advGain = helper(n, Math.max(len, m), ptr + len);
            // 现在剩余的所有石子- 己方本轮的所有石子 -  对方将来得到得到的所有石子 = 己方将来得到的所有石子
            int myGainFuture = remain - myGainThisTime - advGain;
            maxGain = Math.max(maxGain, myGainFuture + myGainThisTime);
        }
        return memo[m][ptr] = maxGain;
    }

    // LC1025
    Boolean[] lc1025Memo = new Boolean[1001];

    public boolean divisorGame(int n) {
        if (n <= 1) return false;
        if (lc1025Memo[n] != null) return lc1025Memo[n];
        int sqrt = (int) Math.sqrt(n);
        for (int i = 2; i <= sqrt; i++) {
            if (n % i == 0) {
                if (!divisorGame(n - i) || !divisorGame(n - n / i)) {
                    return lc1025Memo[n] = true;
                }
            }
        }
        if (!divisorGame(n - 1)) return lc1025Memo[n] = true;
        return lc1025Memo[n] = false;
    }

    // LC1686 **
    public int stoneGameVI(int[] aliceValues, int[] bobValues) {
        int n = aliceValues.length;
        List<int[]> totalValuesAndIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int totalValues = aliceValues[i] + bobValues[i];
            totalValuesAndIdx.add(new int[]{totalValues, i});
        }
        Collections.sort(totalValuesAndIdx, Comparator.comparingInt(o -> -o[0]));
        int alice = 0, bob = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) alice += aliceValues[totalValuesAndIdx.get(i)[1]];
            else bob += bobValues[totalValuesAndIdx.get(i)[1]];
        }
        if (alice == bob) return 0;
        if (alice > bob) return 1;
        return -1;
    }

    // LC877
    Boolean[] lc877Memo;

    public boolean stoneGame(int[] piles) {
        LinkedList<Integer> pileList = new LinkedList<>();
        for (int i : piles) pileList.add(i);
        lc877Memo = new Boolean[piles.length];
        return lc877Helper(pileList, 0, 0, 0);
    }

    private boolean lc877Helper(LinkedList<Integer> pileList, int myGain, int advGain, int status) {
        if (pileList.size() == 0) {
            return myGain > advGain;
        }
        if (lc877Memo[status] != null) return lc877Memo[status];
        // 左侧
        int left = pileList.getFirst(), right = pileList.getLast();
        pileList.removeFirst();
        boolean result = lc877Helper(pileList, advGain, myGain + left, status + 1);
        pileList.addFirst(left);
        if (!result) return lc877Memo[status] = true;

        pileList.removeLast();
        result = lc877Helper(pileList, advGain, myGain + right, status);
        pileList.addLast(right);
        if (!result) return lc877Memo[status] = true;

        return lc877Memo[status] = false;
    }


    // LC1908
    BitSet lc1908Visited;
    BitSet lc1908Memo;
    int lc1908Base;

    public boolean nimGame(int[] piles) {
        lc1908Base = Arrays.stream(piles).max().getAsInt() + 1;
        int status = calStatus(piles);
        lc1908Memo = new BitSet(status + 1);
        lc1908Visited = new BitSet(status + 1);
        return lc1908Helper(piles);
    }

    private boolean lc1908Helper(int[] piles) {
        int status = calStatus(piles);
        if (status == 0) return false;
        if (lc1908Visited.get(status)) return lc1908Memo.get(status);
        lc1908Visited.set(status);
        for (int i = 0; i < piles.length; i++) {
            if (piles[i] > 0) {
                for (int j = 1; j <= piles[i]; j++) {
                    piles[i] -= j;
                    boolean lose = lc1908Helper(piles);
                    piles[i] += j;
                    if (!lose) {
                        lc1908Memo.set(status);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private int calStatus(int[] piles) {
        int status = 0;
        for (int i : piles) {
            status *= lc1908Base;
            status += i;
        }
        return status;
    }


    // LC1973
    int lc1973Result = 0;

    public int equalToDescendants(TreeNode root) {
        if (root == null) return 0;
        lc1973Dfs(root);
        return lc1973Result;
    }

    private int lc1973Dfs(TreeNode root) {
        if (root == null) return 0;
        int sum = lc1973Dfs(root.left) + lc1973Dfs(root.right);
        if (root.val == sum) lc1973Result++;
        return sum + root.val;
    }

    // LC430
    class Lc430 {
        public Node flatten(Node head) {
            if (head == null) return null;
            Deque<Node> dfs = new LinkedList<>(); // for dfs
            dfs.push(head);
            Deque<Node> stack = new LinkedList<>(); // for linking
            while (!dfs.isEmpty()) {
                Node p = dfs.pop();
                stack.push(p);
                if (p.next != null) dfs.push(p.next);
                if (p.child != null)
                    dfs.push(p.child); // rule: if child is not null, make child next, so last push child (for first pop)
            }
            Node next = null;
            while (!stack.isEmpty()) {
                Node p = stack.pop();
                p.child = null; // make the child null or error occur
                p.next = next;
                next = p;
                if (!stack.isEmpty()) {
                    p.prev = stack.peek();
                } else {
                    p.prev = null;
                }
            }
            return head;
        }

        class Node {
            public int val;
            public Node prev;
            public Node next;
            public Node child;
        }
    }

    // Interview 04.01
    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        Map<Integer, Set<Integer>> edge = new HashMap<>();
        for (int[] e : graph) {
            edge.putIfAbsent(e[0], new HashSet<>());
            edge.get(e[0]).add(e[1]);
        }
        Deque<Integer> q = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        q.offer(start);
        while (!q.isEmpty()) {
            int p = q.poll();
            if (visited.contains(p)) continue;
            if (p == target) return true;
            visited.add(p);
            for (int next : edge.getOrDefault(p, new HashSet<>())) {
                if (!visited.contains(next)) {
                    q.offer(next);
                }
            }
        }
        return false;
    }

    // LC440 ** 第k大字典序数
    public int findKthNumber(int upperBound, int k) {
        int cur = 1;
        k--; // 从1开始。 如果从0开始不需要减
        while (k != 0) {
            int num = lc440Helper(cur, upperBound);
            if (num <= k) {
                cur++;
                k -= num;
            } else {
                cur *= 10;
                k--;
            }
        }
        return cur;
    }

    private int lc440Helper(int prefix, int upperBound) {// inclusive
        int count = 0;
        for (long cur = prefix, next = prefix + 1; cur <= upperBound; cur *= 10, next *= 10) {
            count += Math.min(upperBound + 1, next) - cur;
        }
        // 如: 数字位数1位的时候 有多少个以1开头的个数?
        // a=1, b=2, count += 2-1
        // 上界为12
        // a=10 b=20, 10 11 12, count += 12+1-10
        return count;
    }

    // LC708 ** JZOF II 029
    class Lc708 {
        public Node insert(Node head, int insertVal) {
            if (head == null) {
                Node n = new Node(insertVal);
                n.next = n;
                return n;
            }
            Function<Node, Node> deal = cur -> {
                Node n = new Node(insertVal);
                Node origNext = cur.next;
                cur.next = n;
                n.next = origNext;
                return head;
            };
            Node cur = head;
            if (cur.next == head) {
                return deal.apply(cur);
            }
            while (true) {
                if (cur.val > cur.next.val) { // cur就是分界点, 就是末尾
                    if (insertVal >= cur.val || insertVal <= cur.next.val) {
                        return deal.apply(cur);
                    }
                } else if (insertVal >= cur.val && insertVal <= cur.next.val) {
                    return deal.apply(cur);
                }
                cur = cur.next;
                // 说明已经转了一圈, 还没有收获, 即圈上所有数都是同一个值
                if (cur.next == head) {
                    return deal.apply(cur);
                }
            }
        }

        class Node {
            public int val;
            public Node next;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, Node _next) {
                val = _val;
                next = _next;
            }
        }

    }

    // LC548 Hard **
    public boolean splitArray(int[] nums) {
        // 要找到三个点, 去除这三个点的数分成四段, 使得四段的和相等
        // 0 1 2 3 4 5 6 i=1 j=3 k = 5, 至少要有7个数
        // 1 <= 1 <= n-6
        // 3 <= i+2 <= j <= n-4
        // 5 <= j+2 <= k <= n-2
        if (nums.length < 7) return false;
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + nums[i];
        // 要点: 枚举中间的点, i,j,k 的j
        for (int j = 3; j <= n - 4; j++) {
            Set<Integer> set = new HashSet<>();
            for (int i = 1; i <= j - 2; i++) {
                if (prefix[i] == prefix[j] - prefix[i + 1]) {
                    set.add(prefix[i]);
                }
            }
            for (int k = j + 2; k <= n - 2; k++) {
                int sum = prefix[n] - prefix[k + 1];
                if (sum == prefix[k] - prefix[j + 1] && set.contains(sum))
                    return true;
            }
        }
        return false;
    }


    // LC325
    public int maxSubArrayLen(int[] nums, int k) {
        int result = 0, n = nums.length, prefix = 0;
        Map<Integer, Integer> prefixIdxMap = new HashMap<>();
        prefixIdxMap.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix += nums[i];
            if (prefixIdxMap.containsKey(prefix - k)) {
                result = Math.max(result, i - prefixIdxMap.get(prefix - k));
            }
            prefixIdxMap.putIfAbsent(prefix, i);
        }
        return result;
    }

    // LC768 Hard **  LC769 Medium 单调栈 非常精妙
    // https://leetcode-cn.com/problems/max-chunks-to-make-sorted-ii/solution/zui-duo-neng-wan-cheng-pai-xu-de-kuai-ii-deng-jie-/
    public int maxChunksToSorted(int[] arr) {
        Deque<Integer> stack = new LinkedList<>(); // 单调递增栈
        for (int i : arr) {
            if (!stack.isEmpty() && i < stack.peek()) {
                int head = stack.pop();
                while (!stack.isEmpty() && i < stack.peek()) stack.pop();
                stack.push(head);
            } else {
                stack.push(i);
            }
        }
        return stack.size();
    }

    // LC1788 Hard
    public int maximumBeauty(int[] flowers) {
        final int OFFSET = 10000;
        int[] firstAppear = new int[20001];
        Arrays.fill(firstAppear, -1);
        int prefix = 0, result = Integer.MIN_VALUE;
        for (int i : flowers) {
            if (firstAppear[i + OFFSET] == -1) {
                firstAppear[i + OFFSET] = prefix;
                prefix += i > 0 ? i : 0;
            } else {
                result = Math.max(result, (prefix += (i > 0 ? i : 0)) - firstAppear[i + OFFSET] + (i < 0 ? i << 1 : 0));
            }
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

// Interview 16.25 LRU
class LRUCache {
    int cap;
    Map<Integer, Node> m = new HashMap<>();
    Map<Node, Integer> reverse = new HashMap<>();
    Node head;
    Node tail;

    public LRUCache(int capacity) {
        this.cap = capacity;
        head = new Node(-1);
        tail = new Node(-1);
        head.next = head.prev = tail;
        tail.next = tail.prev = head;
    }

    public int get(int key) {
        if (!m.containsKey(key)) return -1;
        int result = m.get(key).val;
        // 将Node 移到头部
        Node victim = m.get(key);
        unlink(victim);
        Node origHeadNext = head.next;
        head.next = victim;
        origHeadNext.prev = victim;
        victim.next = origHeadNext;
        victim.prev = head;
        return result;
    }

    public void put(int key, int value) {
        if (m.containsKey(key)) {
            m.get(key).val = value;
        } else {
            if (m.size() == cap) {
                Node victim = tail.prev;
                int vKey = reverse.get(victim);
                unlink(victim);
                m.remove(vKey);
                reverse.remove(victim);
            }
            Node n = new Node(value);
            m.put(key, n);
            reverse.put(n, key);
        }
        get(key);
    }

    // Node 双向链表, 然后用Map来存指针
    class Node {
        int val;
        Node prev;
        Node next;

        public Node(int val) {
            this.val = val;
        }
    }

    private void unlink(Node n) {
        Node origPrev = n.prev, origNext = n.next;
        if (origNext != null) origNext.prev = origPrev;
        if (origPrev != null) origPrev.next = origNext;
        n.next = null;
        n.prev = null;
    }
}