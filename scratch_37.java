import javafx.util.Pair;

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

        System.err.println(s.fractionToDecimal(1, 333));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC169
    public int majorityElement(int[] nums) {
        int half = nums.length / 2;
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : nums) {
            m.put(i, m.getOrDefault(i, 0) + 1);
            if (m.get(i) > half) {
                return i;
            }
        }
        return -1;
    }

    // LC166 almost Solution
    public String fractionToDecimal(int numerator, int denominator) {
        long num = numerator;
        long den = denominator;
        num = Math.abs(num);
        den = Math.abs(den);
        String left = String.valueOf(num / den);
        if ((numerator < 0 && denominator > 0) || (numerator > 0 && denominator < 0)) left = "-" + left;
        long remainder = num % den;
        StringBuffer sb = new StringBuffer(left);
        sb.append(".");
        if (remainder == 0) {
            return left;
        }
        Map<Long, Integer> map = new HashMap<>();
        while (remainder != 0) {
            if (map.containsKey(remainder)) {
                sb.insert(map.get(remainder), "(");
                sb.append(")");
                break;
            }
            map.put(remainder, sb.length());
            remainder *= 10;
            sb.append(remainder / den);
            remainder %= den;
        }
        return sb.toString();
    }

    // LC162
    public int findPeakElement(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1]) return i;
        }
        return nums.length - 1;
    }

    // LC04.04
    Map<TreeNode, Integer> treeHeight = new HashMap<>();
    boolean lc0404Flag = false;

    public boolean isBalanced(TreeNode root) {
        checkHeight(root);
        return !lc0404Flag;
    }

    private int checkHeight(TreeNode root) {
        if (root == null) return 0;
        if (treeHeight.containsKey(root)) return treeHeight.get(root);
        int height = Math.max(checkHeight(root.left), checkHeight(root.right)) + 1;
        if (Math.abs(checkHeight(root.left) - checkHeight(root.right)) > 1) lc0404Flag = true;
        treeHeight.put(root, height);
        return height;
    }


    // LC152 乘积最大子数组
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] dpMax = Arrays.copyOf(nums, n); // dpMax[i] 表示以nums[i] 结尾的最大子数组的积
        int[] dpMin = Arrays.copyOf(nums, n); // dpMin 表示以nums[i]结尾的最小乘积
        for (int i = 1; i < n; i++) {
            dpMax[i] = Math.max(Math.max(dpMax[i - 1] * nums[i], dpMin[i - 1] * nums[i]), nums[i]);
            dpMin[i] = Math.min(Math.min(dpMin[i - 1] * nums[i], dpMax[i - 1] * nums[i]), nums[i]);
        }
        return Arrays.stream(dpMax).max().getAsInt();
    }

    // LC149 Hard
    public int maxPoints(int[][] points) {
        Map<Double, Integer> slash = new HashMap<>();
        int result = 1;
        for (int i = 0; i < points.length - 1; i++) {
            slash.clear();
            int horizon = 1;
            int dup = 0;
            int tmpMax = 1;
            for (int j = i + 1; j < points.length; j++) {
                if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
                    dup++;
                } else if (points[i][1] == points[j][1]) {
                    horizon++;
                    tmpMax = Math.max(tmpMax, horizon);
                } else {
                    double k;
                    if (points[i][0] == points[j][0]) {
                        k = 0d;
                    } else {
                        k = ((double) (points[i][0] - points[j][0])) / ((double) (points[i][1] - points[j][1]));
                    }
                    slash.put(k, slash.getOrDefault(k, 1) + 1);
                    tmpMax = Math.max(tmpMax, slash.get(k));
                }
            }
            result = Math.max(result, tmpMax + dup);
        }
        return result;
    }

    // LC1011 Solution
    public int shipWithinDaysSolution(int[] weights, int D) {
        // 确定二分查找左右边界
        int left = Arrays.stream(weights).max().getAsInt(), right = Arrays.stream(weights).sum();
        while (left < right) {
            int mid = (left + right) / 2;
            // need 为需要运送的天数
            // cur 为当前这一天已经运送的包裹重量之和
            int need = 1, cur = 0;
            for (int weight : weights) {
                if (cur + weight > mid) {
                    ++need;
                    cur = 0;
                }
                cur += weight;
            }
            if (need <= D) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // LC1011 141ms
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
        int left = Math.max(maxSingle, average), right = average + maxSingle;
        while (left < right) {
            int midCap = left + (right - left) / 2;
            int days = howManyDays(D, total, ts, midCap);
            if (days <= D) {
                right = midCap;
            } else {
                left = midCap + 1;
            }
        }
        return left;
    }

    private int howManyDays(int D, int total, TreeSet<Integer> ts, int cap) {
        int totalCtr;
        int dCtr;
        totalCtr = 0;
        for (dCtr = 0; dCtr < D; dCtr++) {
            Integer floor = ts.floor(totalCtr + cap);
            if (floor != null && totalCtr < total) {
                totalCtr = floor;
            } else {
                break;
            }
        }
        if (totalCtr == total) {
            return dCtr;
        }
        return Integer.MAX_VALUE;
    }

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