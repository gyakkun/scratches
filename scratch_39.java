import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        System.err.println(s.containsDuplicate(new int[]{1, 1, 1, 3, 3, 4, 3, 2, 4, 2}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC230
    int lc230Ctr = 0;
    int lc230Result = -1;

    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return lc230Result;
    }

    private void inorder(TreeNode root, int k) {
        if (root.left != null) inorder(root.left, k);
        lc230Ctr++;
        if (lc230Ctr == k) {
            lc230Result = root.val;
            return;
        }
        if (root.right != null) inorder(root.right, k);
    }

    // LC633
    public boolean judgeSquareSum(int c) {
        for (long a = 0; a * a <= c; a++) {
            double b = Math.sqrt(c - a * a);
            if (b == (int) b) {
                return true;
            }
        }
        return false;
    }

    // LC217
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i : nums) {
            if (!s.add(i)) return true;
        }
        return false;
    }

    // LC218 TBD
    public List<List<Integer>> getSkyline(int[][] buildings) {
        return null;
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
