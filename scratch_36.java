
import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();

        long timing = System.currentTimeMillis();
        // 10,9,2,5,3,7,101,18
        System.err.println(s.lengthOfLIS(new int[]{5, 9, 18, 54, 108, 540, 90, 180, 360, 720}));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");

    }

    // LC137 躺平 选择哈希
    public int singleNumberLc137(int[] nums) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : nums) {
            m.put(i, m.getOrDefault(i, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : m.entrySet()) {
            if (entry.getValue() == 1) return entry.getKey();
        }
        return -1;
    }

    // LC136 位运算
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
            result ^= nums[i];
        }
        return result;
    }

    // LC300 最长递增子序列LIS 使用TreeSet API
    public int lengthOfLIS(int[] nums) {
        TreeSet<Integer> tail = new TreeSet<>();
        for (int i : nums) {
            Integer ceiling = tail.ceiling(i);
            if (ceiling != null) {
                tail.remove(ceiling);
            }
            tail.add(i);
        }
        return tail.size();
    }

    // LC368 Solution
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int[] dp = new int[n]; // dp[i] 表示以i为约数的最大整除子集大小
        Arrays.fill(dp, 1);
        int maxSize = 1;
        int maxVal = nums[0];


        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            if (dp[i] > maxVal) {
                maxVal = nums[i];
                maxSize = dp[i];
            }
        }

        List<Integer> result = new LinkedList<>();
        if (maxSize == 1) {
            result.add(nums[0]);
            return result;
        }

        for (int i = n - 1; i >= 0 && maxVal > 0; i--) {
            if (dp[i] == maxSize && maxVal % nums[i] == 0) {
                result.add(nums[i]);
                maxVal = nums[i];
                maxSize--;
            }
        }
        return result;

    }

    // LC134
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int result = -1;

        List<double[]> unitCost = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            unitCost.add(new double[]{i, (double) gas[i] / (double) cost[i]});
        }
        unitCost.sort(new Comparator<double[]>() {
            @Override
            public int compare(double[] o1, double[] o2) {
                return (int) (o2[1] - o1[1]);
            }
        });

        for (int i = 0; i < n; i++) {
            double[] firstStation = unitCost.get(i);
            int idx = (int) firstStation[0];
            int currentGas = 0;
            boolean flag = false;
            for (int j = 0; j < n; j++) {
                currentGas = currentGas + gas[(idx + j) % n] - cost[(idx + j) % n];
                if (currentGas < 0) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                continue;
            } else {
                return idx;
            }
        }

        return result;
    }

    // LC130, My Solution: 变通使用并查集
    public void solve(char[][] board) {
        int rowNum = board.length;
        int colNum = board[0].length;
        DisjointSetUnionString dsus = new DisjointSetUnionString();

        // 用一个长度为2的整形组表示'O'点的行列坐标, 用visited二维数组表示该'O'点是否已经访问过
        List<int[]> oPoints = new LinkedList<>();
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                if (board[i][j] == 'O') {
                    oPoints.add(new int[]{i, j});
                }
            }
        }

        Set<String> invalid = new HashSet<>();

        for (int[] point : oPoints) {
            int row = point[0];
            int col = point[1];
            dsus.add("" + row + ',' + col);

            // CHECKVALID
            if (row == 0 || row == rowNum - 1 || col == 0 || col == colNum - 1) {
                invalid.add("" + row + ',' + col);
            }

            if (row >= 1) {
                if (board[row - 1][col] == 'O') {
                    dsus.add("" + (row - 1) + ',' + col);
                    dsus.merge("" + (row - 1) + ',' + col, "" + row + ',' + col);
                }
            }
            if (row <= rowNum - 2) {
                if (board[row + 1][col] == 'O') {
                    dsus.add("" + (row + 1) + ',' + col);
                    dsus.merge("" + (row + 1) + ',' + col, "" + row + ',' + col);
                }
            }
            if (col >= 1) {
                if (board[row][col - 1] == 'O') {
                    dsus.add("" + row + ',' + (col - 1));
                    dsus.merge("" + row + ',' + (col - 1), "" + row + ',' + col);
                }
            }
            if (col <= colNum - 2) {
                if (board[row][col + 1] == 'O') {
                    dsus.add("" + row + ',' + (col + 1));
                    dsus.merge("" + row + ',' + (col + 1), "" + row + ',' + col);
                }
            }
        }

        Map<String, Set<String>> allGroup = dsus.getAllGroups();
        for (String invalidGroupLeader : invalid) {
            String leader = dsus.find(invalidGroupLeader);
            allGroup.remove(leader);
        }
        for (Map.Entry<String, Set<String>> entry : allGroup.entrySet()) {
            for (String s : entry.getValue()) {
                List<Integer> coor = Arrays.stream(s.split(",")).map(Integer::valueOf).collect(Collectors.toList());
                board[coor.get(0)][coor.get(1)] = 'X';
            }
        }

    }

    // LC363
    public int maxSumSubmatrix(int[][] matrix, int target) {
        int ans = Integer.MIN_VALUE;
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; ++i) { // 枚举上边界
            int[] sum = new int[n];
            for (int j = i; j < m; ++j) { // 枚举下边界
                for (int c = 0; c < n; ++c) {
                    sum[c] += matrix[j][c]; // 更新每列的元素和
                }
                TreeSet<Integer> sumSet = new TreeSet<Integer>();
                sumSet.add(0);
                int s = 0;
                for (int v : sum) {
                    s += v;
                    Integer ceil = sumSet.ceiling(s - target);
                    if (ceil != null) {
                        ans = Math.max(ans, s - ceil);
                    }
                    sumSet.add(s);
                }
            }
            // prefix[r] - prefix[l] <= target
            //
            // prefix[l] >= prefix[r] - target
            //
            // ceil >= prefix[r] - target
            // <-> ceil = prefix[l] (若存在)
            //
            // prefix[r] - prefix[l] = 区间和
            // ans  = Math.max(ans, prefix[r] - prefix[l])
        }
        return ans;
    }

    // LC363前置: 最大子段和 (未优化空间)
    public int maxSumSubArray(int[] arr) {
        int n = arr.length;
        int[] dp = new int[n + 1]; // 以i结尾的最大子段和, 注意子段长度不能为0
        dp[0] = arr[0];
        int result = dp[0];
        for (int i = 1; i < n; i++) {
            dp[i] = Math.max(0, dp[i - 1]) + arr[i];
            result = Math.max(result, dp[i]);
        }
        return result; // 若子段长度可为0, 则取Math.max(result,0)
    }

    // LC128
    public int longestConsecutive(int[] nums) {
        DisjointSetUnion dsu = new DisjointSetUnion();
        for (int i = 0; i < nums.length; i++) {
            dsu.add(nums[i]);
            if (dsu.contains(nums[i] - 1)) {
                dsu.merge(nums[i], nums[i] - 1);
            }
            if (dsu.contains(nums[i] + 1)) {
                dsu.merge(nums[i], nums[i] + 1);
            }
        }
        Map<Integer, Set<Integer>> map = dsu.getAllGroups();
        int result = 0;
        for (Map.Entry<Integer, Set<Integer>> entry : map.entrySet()) {
            result = Math.max(entry.getValue().size(), result);
        }
        return result;
    }

    Integer[] lc91Memo;

    // LC91
    public int numDecodings(String s) {
        lc91Memo = new Integer[s.length() + 1];
        return numDecodingsHelper(s, 0);
    }

    private int numDecodingsHelper(String s, int idx) {

        if (idx >= s.length()) return 1;

        if (s.charAt(idx) == '0') return 0;

//        if (idx + 1 == s.length()) {
//            return numDecodingsHelper(s, idx + 1);
//        }

        if (lc91Memo[idx] != null) {
            return lc91Memo[idx];
        }
        int one = numDecodingsHelper(s, idx + 1);
        int two = 0;
        if (idx + 2 <= s.length() && s.substring(idx, idx + 2).compareTo("26") <= 0) {
            two = numDecodingsHelper(s, idx + 2);
        }

        lc91Memo[idx] = one + two;
        return lc91Memo[idx];

    }

    // LC127
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        int layer = 0;
        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) return 0;
        Set<String> visited = new HashSet<>();

        Deque<String> q = new LinkedList<>();
        q.offer(beginWord);
        while (!q.isEmpty()) {
            int len = q.size();
            layer++;
            for (int i = 0; i < len; i++) {
                String tmp = q.poll();
                if (visited.contains(tmp)) continue;
                visited.add(tmp);
                if (tmp.equals(endWord)) return layer;
                Iterator<String> it = wordSet.iterator();
                while (it.hasNext()) {
                    String next = it.next();
                    if (!visited.contains(next) && oneLetterDiff(next, tmp)) {
                        q.offer(next);
                    }
                }
            }
        }

        return 0;
    }

    private boolean oneLetterDiff(String a, String b) {
        // if(a.length()!=b.length()) return false;
        int ctr = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) != b.charAt(i)) ctr++;
        }
        return ctr == 1;
    }

    // LC125
    public boolean isPalindrome(String s) {
        s = s.toLowerCase().replaceAll("/\\s+/", "").replaceAll("_", "").replaceAll("[^\\w\\d]+", "");

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


class DisjointSetUnion {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(int i) {
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
    public int find(int i) {
        //先找到根 再压缩
        int root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            int origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iFather = find(i);
        int jFather = find(j);
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

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (Integer i : father.keySet()) {
            int f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<Integer> s = new HashSet<Integer>();
        for (Integer i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}

class DisjointSetUnionString {

    Map<String, String> father;
    Map<String, Integer> rank;

    public DisjointSetUnionString() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(String i) {
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
    public String find(String i) {
        //先找到根 再压缩
        String root = i;
        while (!father.get(root).equals(root)) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (!father.get(i).equals(root)) {
            String origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(String i, String j) {
        String iFather = find(i);
        String jFather = find(j);
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

    public boolean isConnected(String i, String j) {
        return find(i) == find(j);
    }

    public Map<String, Set<String>> getAllGroups() {
        Map<String, Set<String>> result = new HashMap<>();
        // 找出所有根
        for (String i : father.keySet()) {
            String f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<String> s = new HashSet<>();
        for (String i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}