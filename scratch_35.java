import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.maxCoins(new int[]{3, 1, 5, 8}));
    }

    // LC337
    public int rob(TreeNode root) {
        // 构造两个map<node,int>, 分别存储rob该节点和不rob该节点能取到的最大值
        Map<TreeNode, Integer> choose = new HashMap<>();
        Map<TreeNode, Integer> notChoose = new HashMap<>();
        choose.put(null, 0);
        notChoose.put(null, 0);
        robLc337Helper(root, choose, notChoose);
        return Math.max(choose.getOrDefault(root, 0), notChoose.getOrDefault(root, 0));
    }

    private void robLc337Helper(TreeNode root, Map<TreeNode, Integer> choose, Map<TreeNode, Integer> notChoose) {
        if (root.left == null && root.right == null) {
            choose.put(root, root.val);
            notChoose.put(root, 0);
            return;
        }
        if (root.left != null) robLc337Helper(root.left, choose, notChoose);
        if (root.right != null) robLc337Helper(root.right, choose, notChoose);
        choose.put(root, root.val
                + notChoose.getOrDefault(root.left, 0)
                + notChoose.getOrDefault(root.right, 0)
        );
        // 若不选root, 则其左/右 选/不选共4中, 挑出最大值
        int max = Math.max(choose.getOrDefault(root.left, 0), notChoose.getOrDefault(root.left, 0))
                + Math.max(choose.getOrDefault(root.right, 0), notChoose.getOrDefault(root.right, 0));
        notChoose.put(root, max);
    }

    // LC312
    public int maxCoins(int[] nums) {
        int n = nums.length;
        Integer[][] memo = new Integer[n + 2][n + 2];
        int[] balloon = new int[n + 2];
        balloon[0] = balloon[n + 1] = 1;
        for (int i = 1; i <= n; i++) {
            balloon[i] = nums[i - 1];
        }
        return maxCoinsHelper(balloon, 0, n + 1, memo);
    }

    private int maxCoinsHelper(int[] balloon, int l, int r, Integer[][] memo) {
        if (l + 1 >= r) return 0;
        if (memo[l][r] != null) return memo[l][r];
        int result = 0;
        for (int i = l + 1; i < r; i++) {
            int sum = balloon[l] * balloon[i] * balloon[r];
            sum += maxCoinsHelper(balloon, l, i, memo) + maxCoinsHelper(balloon, i, r, memo);
            result = Math.max(result, sum);
        }
        memo[l][r] = result;
        return result;
    }

    // LC213
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[][] dp = new int[n][2];
        // dp[i][0] 表示不rob 第i家能取到的最大值
        // dp[i][1] 表示  rob 第i家能取到的最大值
        // dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1])
        // dp[i][1] = Math.max(dp[i-2][0]+nums[i], dp[i-2][1]+nums[i])

        // 假设第0家必须不被rob, 则令 dp[0][1] = Integer.MIN_VALUE;
        // 此时的最后一家可被rob, 最大值在dp[n-1][1],dp[n-1][0]中
        dp[0][0] = 0;
        dp[0][1] = Integer.MIN_VALUE;
        dp[1][0] = 0;
        dp[1][1] = nums[1];
        for (int i = 2; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = Math.max(dp[i - 2][0] + nums[i], dp[i - 2][1] + nums[i]);
        }
        int onePossibleResult = Math.max(dp[n - 1][1], dp[n - 1][0]);

        // 假设第0家必须被rob, 此时最后一家必须不被rob, 最大值在dp[n-2][0],dp[n-2][1]中
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        dp[1][0] = nums[0];
        dp[1][1] = nums[1];
        for (int i = 2; i < n - 1; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = Math.max(dp[i - 2][0] + nums[i], dp[i - 2][1] + nums[i]);
        }
        int anotherPossibleResult = Math.max(dp[n - 2][1], dp[n - 2][0]);
        int result = Math.max(onePossibleResult, anotherPossibleResult);
        return result;

    }

}


// LC208 二叉树实现
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
        TrieNode former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {

            if (former.val == '#') former.val = word.charAt(i);

            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) {
                former.sibling = new TrieNode(word.charAt(i));
                former = former.sibling;
            }
            if (former.child == null) former.child = new TrieNode();
            former = former.child;
        }

        if (former.val == '#') former.val = word.charAt(i);

        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) {
            former.sibling = new TrieNode(word.charAt(i));
            former = former.sibling;
        }
        if (former.child == null) former.child = new TrieNode();
        former.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {
            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) return false;
        return former.isEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode former = root;
        int i;
        for (i = 0; i < prefix.length() - 1; i++) {
            former = former.searchSibling(prefix.charAt(i));
            if (former.val != prefix.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(prefix.charAt(i));
        if (former.val != prefix.charAt(i)) return false;
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */

class TrieNode {
    Character val;
    Boolean isEnd;
    TrieNode child;
    TrieNode sibling;

    public TrieNode() {
        this.val = '#';
        this.isEnd = false;
    }

    public TrieNode(Character c) {
        this.val = c;
        this.isEnd = false;
    }

    public TrieNode searchSibling(Character c) {
        TrieNode former = this;
        while (former.sibling != null) {
            if (former.val == c) return former;
            former = former.sibling;
        }
        return former;
    }

    public TrieNode searchChildren(Character c) {
        TrieNode former = this;
        while (former.child != null) {
            if (former.val == c) return former;
            former = former.child;
        }
        return former;
    }
}

class TrieHM {
    Map<String, Boolean> m;

    /**
     * Initialize your data structure here.
     */
    public TrieHM() {
        m = new HashMap<>();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        for (int i = 0; i < word.length(); i++) {
            m.putIfAbsent(word.substring(0, i + 1), false);
        }
        m.put(word, true);
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        return m.getOrDefault(word, false);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        return m.containsKey(prefix);
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