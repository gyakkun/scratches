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

    // LC234 O(1) Space O(n) Time
    public boolean isPalindrome(ListNode head) {
        // get mid
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode middleDummy = new ListNode();
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
            if (fast != null) {
                fast = fast.next;
            }
        }
        middleDummy.next = slow;

        // reverse the right part
        ListNode prev = null;
        ListNode cur = slow;
        while (cur != null) {
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }

        // judge
        ListNode end = prev;
        ListNode first = dummy.next;
        boolean result = true;
        while (end != null) {
            if (end.val != first.val) {
                result = false;
                break;
            }
            end = end.next;
            first = first.next;
        }

        // recover
        cur = prev; // end
        prev = null;
        while (cur != null) {
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }

        return result;
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
