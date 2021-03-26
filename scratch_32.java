import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        System.err.println(s.findMedianSortedArrays(new int[]{0, 0}, new int[]{0, 0}));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int totalLen = nums1.length + nums2.length;
        boolean oddFlag = totalLen % 2 == 1;
        int finalLen = totalLen / 2 + 1;
        List<Integer> l = new ArrayList<>(finalLen);
        int[] longer = nums1.length > nums2.length ? nums1 : nums2;
        int[] shorter = longer == nums1 ? nums2 : nums1;
        int shorterPtr = 0, longerPtr = 0;
        while (l.size() < finalLen) {
            if (shorterPtr != shorter.length && longerPtr != longer.length) {
                if (shorter[shorterPtr] < longer[longerPtr]) {
                    l.add(shorter[shorterPtr++]);
                } else {
                    l.add(longer[longerPtr++]);
                }
            } else if (shorterPtr == shorter.length) {
                l.add(longer[longerPtr++]);
            } else {
                l.add(shorter[shorterPtr++]);
            }
        }
        if (oddFlag) return l.get(l.size() - 1);
        return ((double) l.get(l.size() - 1) + (double) l.get(l.size() - 2)) / 2d;
    }

    // LC83
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                cur = cur.next;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }


    // LC82
    public ListNode deleteDuplicatesLC82(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public class ListNode {
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

}