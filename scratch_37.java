import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();


        long timing = System.currentTimeMillis();

//        ["LRUCache","put","put","get","put","get","put","get","get","get"]
//[[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]

        LRUCache lru = new LRUCache(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.get(1);
        lru.put(3, 3);
        lru.get(2);
        lru.put(4, 4);
        lru.get(1);
        lru.get(3);
        lru.get(4);

        System.err.println(s.shipWithinDays(new int[]{361, 321, 186, 186, 67, 283, 36, 471, 304, 218, 60, 78, 149, 166, 282, 384, 61, 242, 426, 275, 236, 221, 27, 261, 487, 90, 468, 19, 453, 241}, 15));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1011 1000ms+
    public int shipWithinDays(int[] weights, int D) {
        int total = 0;
        int maxSingle = Integer.MIN_VALUE;
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : weights) {
            total += i;
            maxSingle = Math.max(maxSingle, i);
            ts.add(total);
        }
        int average = (int) Math.ceil((double) total / (double) D);
        int dCtr = 0;
        int totalCtr = 0;
        for (int i = Math.max(maxSingle, average); i <= average + maxSingle; i++) {
            totalCtr = 0;
            for (dCtr = 0; dCtr < D; dCtr++) {
                Integer floor = ts.floor(totalCtr + i);
                if (floor != null && totalCtr < total) {
                    totalCtr = floor;
                } else {
                    break;
                }
            }
            if (totalCtr == total) {
                return i;
            }
        }
        return average + maxSingle;
    }

    // LC149 Hard
//    public int maxPoints(int[][] points) {
//        int pointCount = points.length;
//        int maxCount = 1;
//        for (int i = 0; i < pointCount - 1; i++) {
//            // find the line with max point on it including point[i]
//            int count = 1;
//            for (int j = i + 1; i + 1 < pointCount; j++) {
//                if (points[i][1] != points[j][1]) {
//                    // y = kx + b
//                    double k = (double) (points[i][0] - points[j][0]) / (double) (points[i][1] - points[j][1]);
//                }
//            }
//        }
//
//    }

    // LC148
    public ListNode sortList(ListNode head) {
        return sortListHelper(head, null);
    }

    private ListNode sortListHelper(ListNode head, ListNode tail) {
        if (head == null) return null;
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode fast = head, slow = head;
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }

        ListNode left = sortListHelper(head, slow);
        ListNode right = sortListHelper(slow, tail);
        ListNode newHead = sortMerge(left, right);
        return newHead;
    }

    private ListNode sortMerge(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode tmp = dummy, p1 = l1, p2 = l2;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                tmp.next = p1;
                p1 = p1.next;
            } else {
                tmp.next = p2;
                p2 = p2.next;
            }
            tmp = tmp.next;
        }
        if (p1 != null) {
            tmp.next = p1;
        } else if (p2 != null) {
            tmp.next = p2;
        }
        return dummy.next;
    }

    // LC140
    private List<String> lc140Result = new LinkedList<>();
    private int longestWordLen = 0;

    // LC140
    public List<String> wordBreak140(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        for (String word : wordSet) {
            longestWordLen = Math.max(longestWordLen, word.length());
        }
        wordBreak140Backtrack(s, wordSet, 0, new LinkedList<>());
        return lc140Result;
    }

    private void wordBreak140Backtrack(String s, Set<String> wordSet, int curIdx, List<String> curList) {
        if (curIdx == s.length()) {
            lc140Result.add(String.join(" ", curList));
            return;
        }
        for (int i = 1; i <= longestWordLen; i++) {
            if (curIdx + i <= s.length() && wordSet.contains(s.substring(curIdx, curIdx + i))) {
                curList.add(s.substring(curIdx, curIdx + i));
                wordBreak140Backtrack(s, wordSet, curIdx + i, curList);
                curList.remove(curList.size() - 1);
            }
        }
    }

    // LC139
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] reachable = new boolean[n + 1];
        reachable[0] = true;
        Set<String> wordSet = new HashSet<>(wordDict);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (reachable[i] && wordSet.contains(s.substring(i, j))) {
                    reachable[j] = true;
                }
            }
        }
        return reachable[n];
    }

    // LC138
    public Node copyRandomList(Node head) {
        if (head == null) {
            return head;
        }
        Map<Node, Node> map = new HashMap<>();
        Node node = head;
        while (node != null) {
            Node temp = new Node(node.val);
            map.put(node, temp);
            node = node.next;
        }
        node = head;
        while (node != null) {
            map.get(node).next = map.get(node.next);
            map.get(node).random = map.get(node.random);
            node = node.next;
        }
        return map.get(head);
    }

}

// LC146
class LRUCache extends LinkedHashMap<Integer, Integer> {
    private int capacity;

    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    // 这个可不写
    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */


// LC138
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
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
