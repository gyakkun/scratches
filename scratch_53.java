import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println();

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1283 二分
    public int smallestDivisor(int[] nums, int threshold) {
        int lo = 1, hi = Integer.MAX_VALUE;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sumDivide(nums, mid) <= threshold) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int sumDivide(int[] nums, int divider) {
        int result = 0;
        for (int i : nums) result += divideUpper(i, divider);
        return result;
    }

    private int divideUpper(int a, int b) {
        if (b == 0) throw new ArithmeticException("Zero divider!");
        if (a == 0) return 0;
        if (a % b == 0) return a / b;
        return a / b + 1;
    }


    // LC1415
    class Lc1415 {
        int kth = 0;
        int targetTh, len;
        String result;
        char[] valid = {'a', 'b', 'c'};

        public String getHappyString(int n, int k) {
            targetTh = k;
            len = n;
            backtrack(new StringBuilder());
            if (result == null) return "";
            return result;
        }

        private void backtrack(StringBuilder cur) {
            if (cur.length() == len) {
                if (++kth == targetTh) {
                    result = cur.toString();
                }
                return;
            }
            for (char c : valid) {
                if ((cur.length() > 0 && cur.charAt(cur.length() - 1) != c) || cur.length() == 0) {
                    cur.append(c);
                    backtrack(cur);
                    cur.deleteCharAt(cur.length() - 1);
                }
            }
        }
    }

    // LC536
    public TreeNode str2tree(String s) {
        if (s.equals("")) return null;
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        boolean number = true, left = false, right = false;
        TreeNode root = new TreeNode();
        int val = -1, pair = 0;
        int startOfLeft = -1, endOfLeft = -1, startOfRight = -1, endOfRight = -1;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (number && (c == '(' || c == ')')) {
                number = false;
                val = Integer.valueOf(sb.toString());
                root.val = val;
                left = true;
                startOfLeft = i;
            }
            if (number) {
                sb.append(c);
            } else if (left) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfLeft = i;
                        TreeNode leftNode = str2tree(s.substring(startOfLeft + 1, endOfLeft));
                        root.left = leftNode;
                        startOfRight = i + 1;
                        right = true;
                        left = false;
                    }
                }
            } else if (right) {
                if (c == '(') pair++;
                else if (c == ')') {
                    pair--;
                    if (pair == 0) {
                        endOfRight = i;
                        TreeNode rightNode = str2tree(s.substring(startOfRight + 1, endOfRight));
                        root.right = rightNode;
                        right = false;
                    }
                }
            }
        }
        if (number) {
            val = Integer.valueOf(sb.toString());
            root.val = val;
        }
        return root;
    }

    // LC1608
    public int specialArray(int[] nums) {
        int[] count = new int[1001];
        for (int i : nums) count[i]++;
        int ctr = 0;
        for (int i = 1000; i >= 0; i--) {
            ctr += count[i];
            if (ctr == i) return i;
        }
        return -1;
    }

    // LC447
    public int numberOfBoomerangs(int[][] points) {
        int result = 0;
        for (int i = 0; i < points.length; i++) {
            int[] pi = points[i];
            Map<Integer, Integer> m = new HashMap<>();
            for (int j = 0; j < points.length; j++) {
                if (i != j) {
                    int[] pj = points[j];
                    int distance = (pi[0] - pj[0]) * (pi[0] - pj[0]) + (pi[1] - pj[1]) * (pi[1] - pj[1]);
                    m.put(distance, m.getOrDefault(distance, 0) + 1);
                }
            }
            for (int e : m.keySet()) {
                result += m.get(e) * (m.get(e) - 1);
            }
        }
        return result;
    }

    // LC1955 ** DP
    public int countSpecialSubsequences(int[] nums) {
        int i0 = 0, i1 = 0, i2 = 0;
        final int mod = 1000000007;
        for (int i : nums) {
            switch (i) {
                case 0:
                    i0 = ((i0 * 2) + 1) % mod;
                    break;
                case 1:
                    i1 = (((i1 * 2) % mod) + i0) % mod;
                    break;
                case 2:
                    i2 = (((i2 * 2) % mod) + i1) % mod;
                    break;
                default:
                    continue;
            }
        }
        return i2;
    }

    // LC600 ** 数位DP
    public int findIntegers(int n) {
        if (n == 0) return 1;
        int[] dp = new int[32];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < 32; i++) { // fib???
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        int prev = 0, result = 0;
        int len = Integer.SIZE - Integer.numberOfLeadingZeros(n);
        for (int i = len; i >= 1; i--) {
            int cur = (n >> (i - 1)) & 1;
            if (cur == 1) {
                result += dp[i];
            }
            if (cur == 1 && prev == 1) break;
            prev = cur;
            if (i == 1) result++;
        }
        return result;
    }


    // LC898 ** 看题解
    // https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/solution/zi-shu-zu-an-wei-huo-cao-zuo-by-leetcode/
    public int subarrayBitwiseORs(int[] arr) {
        Set<Integer> result = new HashSet<>();
        Set<Integer> cur = new HashSet<>();
        for (int i : arr) {
            Set<Integer> tmp = new HashSet<>();
            for (int j : cur) { // 最多有32个数 (1的个数是递增的) ???
                tmp.add(i | j);
            }
            tmp.add(i); // 记得加上自身(长度为1)
            cur = tmp;
            result.addAll(cur);
        }
        return result.size();
    }

    // LC248
    public int strobogrammaticInRange(String low, String high) {
        int count = 0;
        for (int i = low.length(); i <= high.length(); i++) {
            List<String> result = findStrobogrammatic(i);
            if (i > low.length() && i < high.length()) {
                count += result.size();
                continue;
            }
            for (String s : result) {
                if (bigIntCompare(s, low) >= 0 && bigIntCompare(s, high) <= 0) {
                    count++;
                }
            }
        }
        return count;
    }

    private int bigIntCompare(String a, String b) {
        if (a.equals(b)) return 0;
        if (a.length() < b.length()) return -1;
        if (a.length() > b.length()) return 1;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) > b.charAt(i)) return 1;
            if (a.charAt(i) < b.charAt(i)) return -1;
        }
        return 0;
    }

    // LC247
    int[] validDigit = {0, 1, 6, 8, 9};
    int[] symmetryDigit = {0, 1, 8};
    List<String> lc247Result;

    public List<String> findStrobogrammatic(int n) {
        lc247Result = new ArrayList<>();
        if (n == 1) return Arrays.asList("0", "1", "8");
        lc247Helper(new StringBuilder(), n);
        return lc247Result;
    }

    private void lc247Helper(StringBuilder sb, int total) {
        if (sb.length() == total / 2) {
            if (sb.charAt(0) == '0') return;
            if (total % 2 == 1) {
                for (int i : symmetryDigit) {
                    String r = sb.toString() + i + getReverse(sb);
                    lc247Result.add(r);
                }
            } else {
                String r = sb + getReverse(sb);
                lc247Result.add(r);
            }
            return;
        }
        for (int i : validDigit) {
            if (i == 0 && sb.length() == 0) continue;
            sb.append(i);
            lc247Helper(sb, total);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    private String getReverse(StringBuilder input) {
        StringBuilder sb = new StringBuilder();
        for (int i = input.length() - 1; i >= 0; i--) {
            if (input.charAt(i) == '6') {
                sb.append('9');
            } else if (input.charAt(i) == '9') {
                sb.append('6');
            } else {
                sb.append(input.charAt(i));
            }
        }
        return sb.toString();
    }

    // LC246
    public boolean isStrobogrammatic(String num) {
        int[] notValid = {2, 3, 4, 5, 7};
        char[] ca = num.toCharArray();
        for (int i = 0; i <= ca.length / 2; i++) {
            char c = ca[i];
            for (int j : notValid) if (c - '0' == j) return false;
            if (c == '6') {
                if (ca[ca.length - 1 - i] != '9') return false;
            } else if (c == '9') {
                if (ca[ca.length - 1 - i] != '6') return false;
            } else {
                if (ca[ca.length - 1 - i] != c) return false;
            }
        }
        return true;
    }

    // LC1953 Hint: 只和最大时间有关
    public long numberOfWeeks(int[] milestones) {
        long sum = 0;
        long max = Long.MIN_VALUE;
        for (int i : milestones) {
            sum += i;
            max = Math.max(max, i);
        }
        long remain = sum - max;
        max = Math.min(remain + 1, max);
        return remain + max;
    }

    // LC249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<>();
        Map<Integer, Map<String, Integer>> m = new HashMap<>();
        for (String s : strings) {
            m.putIfAbsent(s.length(), new HashMap<>());
            Map<String, Integer> inner = m.get(s.length());
            inner.put(s, inner.getOrDefault(s, 0) + 1);
        }

        for (Map<String, Integer> s : m.values()) {
            while (!s.isEmpty()) {
                String w = s.keySet().iterator().next();
                // 构造
                List<String> list = new ArrayList<>();
                char[] ca = w.toCharArray();
                for (int i = 0; i < 26; i++) {
                    for (int j = 0; j < ca.length; j++) {
                        ca[j] = (char) (((ca[j] - 'a' + 1) % 26) + 'a');
                    }
                    String built = new String(ca);
                    if (s.containsKey(built)) {
                        int count = s.get(built);
                        s.remove(built);
                        for (int j = 0; j < count; j++)
                            list.add(built);
                    }
                }
                result.add(list);
            }
        }
        return result;
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

// JZOF II 30 LC380
class RandomizedSet {

    Map<Integer, Integer> idxMap = new HashMap<>();
    List<Integer> entities = new ArrayList<>();

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {

    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified element.
     */
    public boolean insert(int val) {
        if (idxMap.containsKey(val)) return false;
        idxMap.put(val, entities.size());
        entities.add(val);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!idxMap.containsKey(val)) return false;
        int lastEntity = entities.get(entities.size() - 1);
        int targetIdx = idxMap.get(val);
        entities.set(targetIdx, lastEntity);
        idxMap.put(lastEntity, targetIdx);
        idxMap.remove(val);
        entities.remove(entities.size() - 1);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        int idx = (int) (Math.random() * entities.size());
        return entities.get(idx);
    }
}