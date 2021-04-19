import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        TreeNode a0 = new TreeNode(1);
        TreeNode a1 = new TreeNode(2);
        TreeNode a2 = new TreeNode(2);
        TreeNode a3 = new TreeNode(3);
        TreeNode a4 = new TreeNode(3);
        TreeNode a5 = new TreeNode(3);
        TreeNode a6 = new TreeNode(3);

        a0.left = a1;
        a0.right = a2;
        a1.left = a3;
        a1.right = a4;
        a2.left = a5;
        a2.right = a6;

        long timing = System.currentTimeMillis();
        System.err.println(s.isScramble("eebaacbcbcadaaedceaaacadccdabcdefg",
                "eadcaacabaddaceacbceaabeccdabcdefg"));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");

    }

    Set<String> lc87Memo = new HashSet<>();

    // LC87
    public boolean isScramble(String s1, String s2) {
        assert s1.length() == s2.length();
        if (s1 == s2) return true;
        if (lc87Memo.contains(s1 + "#" + s2)) return false;

        int[] alphabet = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            alphabet[s1.charAt(i) - 'a']++;
            alphabet[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (alphabet[i] != 0) {
                lc87Memo.add(s1 + "#" + s2);
                return false;
            }
        }
        if (s1.length() <= 3) return true;

        for (int i = 1; i <= s1.length() - 1; i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(0, i), s2.substring(s2.length() - i)) && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) {
                return true;
            }
        }
        lc87Memo.add(s1 + "#" + s2);
        return false;
    }

    // LC27 移除数组中指定值的元素并返回新长度
    public int removeElement(int[] nums, int val) {
        // 1 2 3 3 4 3 3
        int k = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] == val) {
                if (i != k) {
                    nums[i] ^= nums[k];
                    nums[k] ^= nums[i];
                    nums[i] ^= nums[k];
                }
                k--;
            }
        }
        return k + 1;
    }

    // LC220 桶
    public boolean containsNearbyAlmostDuplicateBucket(int[] nums, int k, int t) {
        int n = nums.length;
        Map<Long, Long> bucket = new HashMap<>();
        long step = (long) t + 1;
        for (int i = 0; i < n; i++) {
            long id = getBucketId(nums[i], step);
            if (bucket.containsKey(id)) {
                return true;
            }
            if (bucket.containsKey(id - 1) && Math.abs((long) nums[i] - bucket.get(id - 1)) < step) {
                return true;
            }
            if (bucket.containsKey(id + 1) && Math.abs((long) nums[i] - bucket.get(id + 1)) < step) {
                return true;
            }
            bucket.put(id, (long) nums[i]);
            if (i >= k) {
                bucket.remove(getBucketId(nums[i - k], step));
            }
        }
        return false;
    }

    private long getBucketId(long num, long step) {
        if (num >= 0) {
            return num / step;
        } else {
            return ((num + 1) / step) - 1; // 注意保证[-k/2,0]这个范围的num和[0,k/2]范围的num拥有相同的桶ID?
        }
    }

    // LC220 滑动窗口 + 有序集合, Java TreeSet / C++ set
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        // TreeMap<Integer, Integer> tm = new TreeMap<>();
        int n = nums.length;
        TreeSet<Long> ts = new TreeSet<>(Comparator.comparingLong(o -> o));
        for (int i = 0; i < n; i++) {
            Long ceiling = ts.ceiling((long) nums[i] - t); // 返回set中大于等于 x - t 的最小值
            if (ceiling != null && ceiling <= (long) nums[i] + (long) t) {
                return true;
            }
            ts.add((long) nums[i]);
            if (i >= k) {
                ts.remove((long) nums[i - k]);
            }
        }
        return false;
    }

    // LC104
    public int maxDepth(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int layer = 0;

        // 取层数
        while (!q.isEmpty()) {
            layer++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                if (q.peek().left != null) {
                    q.offer(q.peek().left);
                }
                if (q.peek().left != null) {
                    q.offer(q.peek().right);
                }
                q.poll();
            }
        }
        return layer;
    }

    // LC101
    public boolean isSymmetric(TreeNode root) {
        return isSymmerticHelper(root, root);
    }

    private boolean isSymmerticHelper(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val && isSymmerticHelper(a.left, b.right) && isSymmerticHelper(a.right, b.left);
    }

    // LC101 solution
    public boolean check(TreeNode u, TreeNode v) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(u);
        q.offer(v);
        while (!q.isEmpty()) {
            u = q.poll();
            v = q.poll();
            if (u == null && v == null) {
                continue;
            }
            if ((u == null || v == null) || (u.val != v.val)) {
                return false;
            }

            q.offer(u.left);
            q.offer(v.right);

            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
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