class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.countSquares(new int[][]{{0, 1, 1, 1}, {1, 1, 1, 1}, {0, 1, 1, 1}}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1277
    public int countSquares(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1]; // dp[i][j] 表示以matrix[i][j]为右下角的最大正方形边长
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    dp[i + 1][j + 1] = 0;
                } else {
                    dp[i + 1][j + 1] = 1 + Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]);
                }
                result += dp[i + 1][j + 1];
            }
        }
        return result;
    }

    // LC147
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return head;
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode lastSorted = head, cur = head.next;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode prev = dummy;
                while (prev.next.val <= cur.val) {
                    prev = prev.next;
                }
                lastSorted.next = cur.next;
                cur.next = prev.next;
                prev.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    // JSOF 52 LC160
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode aPtr = headA, bPtr = headB;
        int aLen = 0, bLen = 0;
        while (aPtr != null && aPtr.next != null) {
            aLen++;
            aPtr = aPtr.next;
        }
        while (bPtr != null && bPtr.next != null) {
            bLen++;
            bPtr = bPtr.next;
        }
        if (aPtr != bPtr) return null;
        ListNode fast = aLen > bLen ? headA : headB;
        ListNode slow = fast == headA ? headB : headA;
        int aheadStep = Math.abs(aLen - bLen);
        while (aheadStep != 0) {
            fast = fast.next;
            aheadStep--;
        }
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
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
