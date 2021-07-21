class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

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
