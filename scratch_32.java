import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        System.err.println(s.longestCommonPrefix(new String[]{"ab", "a"}));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC14
    public String longestCommonPrefix(String[] strs) {
        if(strs.length==0) return "";
        StringBuffer sb = new StringBuffer();
        sb.append(strs[0]);
        for (int i = 1; i < strs.length; i++) {
            if (sb.length() == 0) return "";
            if (sb.length() > strs[i].length()) sb.delete(strs[i].length(), sb.length());
            for (int j = 0; j < strs[i].length(); j++) {
                if (j + 1 > sb.length()) break;
                if (strs[i].charAt(j) != sb.charAt(j)) {
                    sb.delete(j, sb.length());
                    break;
                }
            }
        }
        return sb.toString();
    }

    // LC7, 不能使用long, 注意溢出判断
    public int reverse(int x) {
        if (x == 0) return 0;
        boolean negFlag = x < 0;
        if (x < 0) x = -x;
        int result = 0;
        while (x != 0) {
            // 溢出判断
            if (result > Integer.MAX_VALUE / 10) {
                return 0;
            }
            if (result * 10 > Integer.MAX_VALUE - x % 10) {
                return 0;
            }

            result = result * 10 + x % 10;
            x /= 10;
        }
        return negFlag ? -result : result;
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