import java.util.Arrays;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        TreeNode t1 = new TreeNode(1);
        TreeNode t0 = new TreeNode(0);
        TreeNode t2 = new TreeNode(2);
        t1.left = t0;
        t1.right = t2;

        System.err.println(s.trimBST(t1, 1, 2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC669
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return null;
        TreeNode left = root.left, right = root.right; // 注意提前存下引用
        root.left = trimBST(left, low, high);
        root.right = trimBST(right, low, high);
        if (root.val < low) { // 该节点的值比下界还小, 说明左子树不能要
            if (root.right != null) {
                root.val = right.val;
                root.left = right.left;
                root.right = right.right;
            } else {
                root = null;
            }
        } else if (root.val > high) {
            if (root.left != null) {
                root.val = left.val;
                root.left = left.left;
                root.right = left.right;
            } else {
                root = null;
            }
        }
        return root;
    }

    // LC961
    public int repeatedNTimes(int[] nums) {
        int len = nums.length;
        int n = len / 2;
        int[] freq = new int[10000];
        for (int i : nums) {
            freq[i]++;
            if (freq[i] == n) return i;
        }
        return -1;
    }

    // LC494
    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int sum = Arrays.stream(nums).sum();
        if (sum < target) return 0;
        int[][] dp = new int[n + 1][2 * sum + 1];
        // dp[i][j] 表示加入前i个数到达和j的方案数
        // 中点(0) 在 dp[sum]
        dp[0][sum] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= 2 * sum; j++) {
                int result = 0;
                if (j - nums[i - 1] >= 0) {
                    result += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] <= 2 * sum) {
                    result += dp[i - 1][j + nums[i - 1]];
                }
                dp[i][j] = result;
            }
        }
        return dp[n][sum + target];
    }

    // LC474
    public int findMaxForm(String[] strs, int m, int n) {
        int[] zeroCtr = new int[strs.length];
        int[] oneCtr = new int[strs.length];
        for (int i = 0; i < strs.length; i++) {
            for (char c : strs[i].toCharArray()) {
                if (c == '0') zeroCtr[i]++;
            }
            oneCtr[i] = strs[i].length() - zeroCtr[i];
        }
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= strs.length; i++) {
            int zeroNum = zeroCtr[i - 1];
            int oneNum = oneCtr[i - 1];
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    int result = dp[j][k];
                    if (j - zeroNum >= 0 && k - oneNum >= 0) {
                        result = Math.max(result, dp[j - zeroNum][k - oneNum] + 1);// 选择这一个字符串
                    }
                    dp[j][k] = result;
                }
            }
        }
        return dp[m][n];
    }

    // JZOF51 HARD
    public int reversePairs(int[] nums) {
        int n = nums.length;
        int[] sorted = new int[n];
        System.arraycopy(nums, 0, sorted, 0, n);
        Arrays.sort(sorted);
        for (int i = 0; i < n; i++) {
            nums[i] = Arrays.binarySearch(sorted, nums[i]) + 1;
        }
        int result = 0;
        BIT bit = new BIT(n);
        for (int i = n - 1; i >= 0; i--) {
            result += bit.query(nums[i] - 1);
            bit.update(nums[i], 1);
        }
        return result;

    }
}


class BIT {
    int len;
    int[] bit;

    public BIT(int n) {
        this.len = n;
        this.bit = new int[n + 1];
    }

    public int query(int idxFromOne) {
        int sum = 0;
        while (idxFromOne > 0) {
            sum += bit[idxFromOne];
            idxFromOne -= lowbit(idxFromOne);
        }
        return sum;
    }

    public void update(int idxFromOne, int delta) {
        while (idxFromOne <= len) {
            bit[idxFromOne] += delta;
            idxFromOne += lowbit(idxFromOne);
        }
    }


    private int lowbit(int x) {
        return x & (x ^ (x - 1));
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