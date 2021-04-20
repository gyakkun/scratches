import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        TreeNode a0 = new TreeNode(1);
        TreeNode a1 = new TreeNode(2);
        TreeNode a2 = new TreeNode(2);
        TreeNode a3 = new TreeNode(3);
        TreeNode a4 = new TreeNode(3);
        TreeNode a5 = new TreeNode(3);
        TreeNode a6 = new TreeNode(3);

        a0.left = a1;
        a0.right = a2;
        a1.left = a3;
        a1.right = a4;
        a2.left = a5;
        a2.right = a6;

        long timing = System.currentTimeMillis();
        System.err.println(s.isPalindrome("ab_a"));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");

    }

    // LC125
    public boolean isPalindrome(String s) {
        s = s.toLowerCase().replaceAll("/\\s+/","").replaceAll("_","").replaceAll("[^\\w\\d]+","");

        int half = s.length() / 2;
        for (int i = 0; i < half; i++) {
            if (s.charAt(i) != s.charAt(s.length() - i - 1)) {
                return false;
            }
        }
        return true;
    }

    // LC122
    public int maxProfit(int[] prices) {
        int n = prices.length;

        int[][] dp = new int[n][2]; // dp[i][j] 表示第i天持有j(0/1)份股票时候的最大利润
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][1] + prices[i], dp[i - 1][0]);
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
        }
        return dp[n][0];
    }

    // LC118
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>(numRows);
        List<Integer> firstRow = new ArrayList<>(1);
        firstRow.add(1);
        result.add(firstRow);
        if (numRows == 1) return result;
        List<Integer> secondRow = new ArrayList<>(2);
        secondRow.add(1);
        secondRow.add(1);
        result.add(secondRow);
        if (numRows == 2) return result;

        for (int i = 2; i < numRows; i++) {
            List<Integer> thisRow = new LinkedList<>();
            thisRow.add(1);
            List<Integer> lastRow = result.get(result.size() - 1);
            for (int j = 0; j < lastRow.size() - 1; j++) {
                thisRow.add(lastRow.get(j) + lastRow.get(j + 1));
            }
            thisRow.add(1);
            result.add(thisRow);
        }
        return result;
    }

    // LC116 Solution O(1) space
    public Node connectSolution(Node root) {
        if (root == null) {
            return root;
        }
        // 从根节点开始
        Node leftmost = root;
        while (leftmost.left != null) {
            // 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            Node head = leftmost;
            while (head != null) {
                // CONNECTION 1
                head.left.next = head.right;
                // CONNECTION 2
                if (head.next != null) {
                    head.right.next = head.next.left;
                }
                // 指针向后移动
                head = head.next;
            }
            // 去下一层的最左的节点
            leftmost = leftmost.left;
        }
        return root;
    }

    // LC116, O(n) space
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Deque<Node> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                Node n = q.poll();
                if (n.left != null) q.offer(n.left);
                if (n.right != null) q.offer(n.right);
                if (i == len - 1) continue;
                n.next = q.peek();
            }
        }
        return root;
    }

    // LC108
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int left, int right) {
        if (left == right) return new TreeNode(nums[left]);
        if (left > right) return null;

        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelper(nums, left, mid - 1);
        root.right = sortedArrayToBSTHelper(nums, mid + 1, right);
        return root;
    }

    // LC105 Solution
    private Map<Integer, Integer> indexMap;

    public TreeNode buildTreeHelper(int[] preorder, int[] inorder, int preorderLeft, int preorderRight, int inorderLeft, int inorderRight) {
        if (preorderLeft > preorderRight) {
            return null;
        }

        // 前序遍历中的第一个节点就是根节点
        int preorderRoot = preorderLeft;
        // 在中序遍历中定位根节点
        int inorderRoot = indexMap.get(preorder[preorderRoot]);

        // 先把根节点建立出来
        TreeNode root = new TreeNode(preorder[preorderRoot]);
        // 得到左子树中的节点数目
        int sizeLeftSubtree = inorderRoot - inorderLeft;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = buildTreeHelper(preorder, inorder, preorderLeft + 1, preorderLeft + sizeLeftSubtree, inorderLeft, inorderRoot - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = buildTreeHelper(preorder, inorder, preorderLeft + sizeLeftSubtree + 1, preorderRight, inorderRoot + 1, inorderRight);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        // 构造哈希映射，帮助我们快速定位根节点
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return buildTreeHelper(preorder, inorder, 0, n - 1, 0, n - 1);
    }


    Set<String> lc87Memo = new HashSet<>();

    // LC87
    public boolean isScramble(String s1, String s2) {
        assert s1.length() == s2.length();
        if (s1 == s2) return true;
        if (lc87Memo.contains(s1 + "#" + s2)) return false;

        int[] alphabet = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            alphabet[s1.charAt(i) - 'a']++;
            alphabet[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (alphabet[i] != 0) {
                lc87Memo.add(s1 + "#" + s2);
                return false;
            }
        }
        if (s1.length() <= 3) return true;

        for (int i = 1; i <= s1.length() - 1; i++) {
            // 如果 x+y -> x+y, 则 x与x幂等, y亦然
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            // 如果 x+y -> y+x, 则x与y幂等, 反之亦然
            if (isScramble(s1.substring(0, i), s2.substring(s2.length() - i)) && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) {
                return true;
            }
        }
        lc87Memo.add(s1 + "#" + s2);
        return false;
    }

    // LC27 移除数组中指定值的元素并返回新长度
    public int removeElement(int[] nums, int val) {
        // 1 2 3 3 4 3 3
        int k = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] == val) {
                if (i != k) {
                    nums[i] ^= nums[k];
                    nums[k] ^= nums[i];
                    nums[i] ^= nums[k];
                }
                k--;
            }
        }
        return k + 1;
    }

    // LC220 桶
    public boolean containsNearbyAlmostDuplicateBucket(int[] nums, int k, int t) {
        int n = nums.length;
        Map<Long, Long> bucket = new HashMap<>();
        long step = (long) t + 1;
        for (int i = 0; i < n; i++) {
            long id = getBucketId(nums[i], step);
            if (bucket.containsKey(id)) {
                return true;
            }
            if (bucket.containsKey(id - 1) && Math.abs((long) nums[i] - bucket.get(id - 1)) < step) {
                return true;
            }
            if (bucket.containsKey(id + 1) && Math.abs((long) nums[i] - bucket.get(id + 1)) < step) {
                return true;
            }
            bucket.put(id, (long) nums[i]);
            if (i >= k) {
                bucket.remove(getBucketId(nums[i - k], step));
            }
        }
        return false;
    }

    private long getBucketId(long num, long step) {
        if (num >= 0) {
            return num / step;
        } else {
            return ((num + 1) / step) - 1; // 注意保证[-k/2,0]这个范围的num和[0,k/2]范围的num拥有相同的桶ID?
        }
    }

    // LC220 滑动窗口 + 有序集合, Java TreeSet / C++ set
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        // TreeMap<Integer, Integer> tm = new TreeMap<>();
        int n = nums.length;
        TreeSet<Long> ts = new TreeSet<>(Comparator.comparingLong(o -> o));
        for (int i = 0; i < n; i++) {
            Long ceiling = ts.ceiling((long) nums[i] - t); // 返回set中大于等于 x - t 的最小值
            if (ceiling != null && ceiling <= (long) nums[i] + (long) t) {
                return true;
            }
            ts.add((long) nums[i]);
            if (i >= k) {
                ts.remove((long) nums[i - k]);
            }
        }
        return false;
    }

    // LC104
    public int maxDepth(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int layer = 0;

        // 取层数
        while (!q.isEmpty()) {
            layer++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                if (q.peek().left != null) {
                    q.offer(q.peek().left);
                }
                if (q.peek().left != null) {
                    q.offer(q.peek().right);
                }
                q.poll();
            }
        }
        return layer;
    }

    // LC101
    public boolean isSymmetric(TreeNode root) {
        return isSymmerticHelper(root, root);
    }

    private boolean isSymmerticHelper(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val && isSymmerticHelper(a.left, b.right) && isSymmerticHelper(a.right, b.left);
    }

    // LC101 solution
    public boolean check(TreeNode u, TreeNode v) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(u);
        q.offer(v);
        while (!q.isEmpty()) {
            u = q.poll();
            v = q.poll();
            if (u == null && v == null) {
                continue;
            }
            if ((u == null || v == null) || (u.val != v.val)) {
                return false;
            }

            q.offer(u.left);
            q.offer(v.right);

            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
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

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}