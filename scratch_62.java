import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.atMostNGivenDigitSet(
                new String[]{"1", "3", "5", "7"},
                100
        ));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC902 **
    public int atMostNGivenDigitSet(String[] digits, int n) {
        String nStr = String.valueOf(n);
        char[] ca = nStr.toCharArray();
        int k = ca.length;
        int[] dp = new int[k + 1];
        dp[k] = 1;

        for (int i = k - 1; i >= 0; i--) {
            int digit = ca[i] - '0';
            for (String dStr : digits) {
                int d = Integer.valueOf(dStr);
                if (d < digit) {
                    dp[i] += Math.pow(digits.length, /*剩下的位数可以随便选*/ k - i - 1);
                } else if (d == digit) {
                    dp[i] += dp[i + 1];
                }
            }
        }

        for (int i = 1; i < k; i++) {
            dp[0] += Math.pow(digits.length, i);
        }
        return dp[0];
    }

    // LC825
    public int numFriendRequests(int[] ages) {
        // 以下情况 X 不会向 Y 发送好友请求
        // age[y] <= 0.5 * age[x] + 7
        // age[y] > age[x]
        // age[y] > 100 && age[x] < 100
        int result = 0;
        Arrays.sort(ages);
        int idx = 0, n = ages.length;
        while (idx < n) {
            int age = ages[idx], same = 1;
            while (idx + 1 < n && ages[idx + 1] == ages[idx]) {
                same++;
                idx++;
            }

            // 找到 大于 0.5 * age + 7 的最小下标
            int lo = 0, hi = idx - 1, target = (int) (0.5 * age + 7);
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (ages[mid] > target) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            int count = 0;
            if (ages[lo] > target) {
                count = idx - lo;
            }
            result += same * count;
            idx++;
        }
        return result;
    }

    // LC1609
    public boolean isEvenOddTree(TreeNode root) {
        int layer = -1;
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            if (layer % 2 == 0) { // 偶数层
                int prev = -1;
                for (int i = 0; i < qs; i++) {
                    TreeNode p = q.poll();
                    int v = p.val;
                    if (v % 2 == 0) return false;
                    if (i > 0) {
                        if (v <= prev) return false;
                    }
                    prev = v;
                    if (p.left != null) q.offer(p.left);
                    if (p.right != null) q.offer(p.right);
                }
            } else { // 奇数层
                int prev = -1;
                for (int i = 0; i < qs; i++) {
                    TreeNode p = q.poll();
                    int v = p.val;
                    if (v % 2 == 1) return false;
                    if (i > 0) {
                        if (v >= prev) return false;
                    }
                    prev = v;
                    if (p.left != null) q.offer(p.left);
                    if (p.right != null) q.offer(p.right);
                }
            }
        }
        return true;
    }


    // LC1044
    final long mod = 1000000007l;
    final long base1 = 29;
    final long base2 = 31;
    String s;

    public String longestDupSubstring(String s) {
        this.s = s;
        char[] ca = s.toCharArray();
        int lo = 0, hi = s.length() - 1;
        String result = "";
        String tmp = null;
        while (lo < hi) { // 找最大值
            int mid = lo + (hi - lo + 1) / 2;
            if ((tmp = helper(ca, mid)) != null) {
                result = tmp;
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }

    private String helper(char[] ca, int len) {
        Set<Integer> m1 = new HashSet<>();
        Set<Integer> m2 = new HashSet<>();
        long hash1 = 0, hash2 = 0, accu1 = 1, accu2 = 1;
        for (int i = 0; i < len; i++) {
            hash1 *= base1;
            hash1 %= mod;
            hash2 *= base2;
            hash2 %= mod;
            hash1 += ca[i] - 'a';
            hash1 %= mod;
            hash2 += ca[i] - 'a';
            hash2 %= mod;
            accu1 *= base1;
            accu1 %= mod;
            accu2 *= base2;
            accu2 %= mod;
        }
        m1.add((int) hash1);
        m2.add((int) hash2);
        for (int i = len; i < ca.length; i++) {
            String victim = s.substring(i - len + 1, i + 1);
            hash1 = (((hash1 * base1 - accu1 * (ca[i - len] - 'a')) % mod) + mod + ca[i] - 'a') % mod;
            hash2 = (((hash2 * base2 - accu2 * (ca[i - len] - 'a')) % mod) + mod + ca[i] - 'a') % mod;
            if (m1.contains((int) hash1)
                    && m2.contains((int) hash2)
            ) {
                return victim;
            }
            m1.add((int) hash1);
            m2.add((int) hash2);
        }
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