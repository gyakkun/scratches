import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.containsNearbyDuplicate(new int[]{1, 2, 3, 1, 2, 3}, 2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1345
    public int minJumps(int[] arr) {
        Map<Integer, Set<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            m.putIfAbsent(arr[i], new HashSet<>());
            Set<Integer> ts = m.get(arr[i]);
            ts.add(i);
        }
        Deque<Integer> q = new LinkedList<>();
        q.offer(0);
        BitSet visited = new BitSet(arr.length);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int p = q.poll();
                int val = arr[p], idx = p;
                if (visited.get(idx)) continue;
                visited.set(idx);

                if (idx == arr.length - 1) return layer;

                // i + 1
                if (idx + 1 < arr.length && !visited.get(idx + 1)) {
                    q.add(idx + 1);
                }

                // i-1
                if (idx - 1 >= 0 && !visited.get(idx - 1)) {
                    q.add(idx - 1);
                }

                // same value
                for (int smi : m.getOrDefault(val, new HashSet<>())) {
                    if (!visited.get(smi)) {
                        q.add(smi);
                    }
                }
                m.remove(val);
            }
        }
        return -1;
    }

    // LC219
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, TreeSet<Integer>> m = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            m.putIfAbsent(nums[i], new TreeSet<>());
            TreeSet<Integer> ts = m.get(nums[i]);
            int lb = i - k;
            int hb = i + k;
            Integer hr = ts.higher(lb);
            Integer lw = ts.lower(hb);
            if (hr != null && Math.abs(i - hr) <= k) {
                System.err.println(ts.ceiling(hr));
                return true;
            }
            if (lw != null && Math.abs(i - lw) <= k) {
                System.err.println(ts.floor(lw));
                return true;
            }
            ts.add(i);
        }
        return false;
    }

    // LC1220
    Long[][] memo;
    final long mod = 1000000007;

    public int countVowelPermutation(int n) {
        memo = new Long[n + 1][6];
        long result = 0;
        for (int i = 0; i < 5; i++) {
            result = (result + helper(n - 1, i)) % mod;
        }
        return (int) result;
    }

    private long helper(int remainLetters, int currentLetterIdx) {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        if (remainLetters == 0) return 1;
        if (memo[remainLetters][currentLetterIdx] != null) return memo[remainLetters][currentLetterIdx] % mod;
        switch (currentLetterIdx) {
            case 0: // a
                return memo[remainLetters][currentLetterIdx]
                        = helper(remainLetters - 1, 1);
            case 1: // e
                return memo[remainLetters][currentLetterIdx]
                        = (helper(remainLetters - 1, 0)
                        + helper(remainLetters - 1, 2)) % mod;
            case 2:
                return memo[remainLetters][currentLetterIdx]
                        = (helper(remainLetters - 1, 0)
                        + helper(remainLetters - 1, 1)
                        + helper(remainLetters - 1, 3)
                        + helper(remainLetters - 1, 4)) % mod;
            case 3:
                return memo[remainLetters][currentLetterIdx]
                        = (helper(remainLetters - 1, 2)
                        + helper(remainLetters - 1, 4));
            case 4:
                return memo[remainLetters][currentLetterIdx]
                        = helper(remainLetters - 1, 0) % mod;
        }
        return 0;
    }

    // LC1036
    public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
        if (blocked.length < 2) return true;
        final int bound = 1000000;
        final int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        TreeSet<Integer> rowSet = new TreeSet<>(), colSet = new TreeSet<>();
        List<int[]> allPoints = new ArrayList<>(Arrays.stream(blocked).collect(Collectors.toList())) {{
            add(source);
            add(target);
        }};
        for (int[] b : allPoints) {
            rowSet.add(b[0]);
            colSet.add(b[1]);
        }

        int rid = rowSet.first() == 0 ? 0 : 1, cid = colSet.first() == 0 ? 0 : 1; // bound it from 0 to 999999
        Iterator<Integer> rit = rowSet.iterator(), cit = colSet.iterator();
        Map<Integer, Integer> rowMap = new HashMap<>(), colMap = new HashMap<>();
        int pr = -1, pc = -1;
        if (rit.hasNext()) rowMap.put((pr = rit.next()), rid);
        if (cit.hasNext()) colMap.put((pc = cit.next()), cid);
        while (rit.hasNext()) {
            int nr = rit.next();
            rid += (nr == pr + 1) ? 1 : 2;
            rowMap.put(nr, rid);
            pr = nr;
        }
        while (cit.hasNext()) {
            int nc = cit.next();
            cid += (nc == pc + 1) ? 1 : 2;
            colMap.put(nc, cid);
            pc = nc;
        }
        int rBound = (pr == bound - 1) ? rid : rid + 1;
        int cBound = (pc == bound - 1) ? cid : cid + 1;
        boolean[][] mtx = new boolean[rBound + 1][cBound + 1]; // use as visited[][] too
        for (int[] b : blocked) {
            mtx[rowMap.get(b[0])][colMap.get(b[1])] = true;
        }
        int sr = rowMap.get(source[0]), sc = colMap.get(source[1]), tr = rowMap.get(target[0]), tc = colMap.get(target[1]);
        Deque<int[]> q = new LinkedList<>();
        q.offer(new int[]{sr, sc});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            if (p[0] == tr && p[1] == tc) return true;
            if (mtx[p[0]][p[1]]) continue;
            mtx[p[0]][p[1]] = true;
            for (int[] d : dir) {
                int inr = p[0] + d[0], inc = p[1] + d[1];
                if (inr >= 0 && inr <= rBound && inc >= 0 && inc <= cBound && !mtx[inr][inc]) {
                    q.offer(new int[]{inr, inc});
                }
            }
        }
        return false;
    }

    // LC2022
    public int[][] construct2DArray(int[] original, int m, int n) {
        int len = original.length;
        if (len != m * n) return new int[][]{};
        int[][] result = new int[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(original, i * n, result[i], 0, n);
        }
        return result;
    }

    // LC472
    Trie lc472Trie = new Trie();

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Arrays.sort(words, Comparator.comparingInt(o -> o.length()));
        List<String> result = new ArrayList<>();
        for (String w : words) {
            if (w.length() == 0) continue;
            if (lc472Helper(w, 0)) {
                result.add(w);
            } else {
                lc472Trie.addWord(w);
            }
        }
        return result;
    }

    private boolean lc472Helper(String word, int startIdx) {
        if (word.length() == startIdx) return true;
        Trie.TrieNode cur = lc472Trie.root;
        for (int i = startIdx; i < word.length(); i++) {
            char c = word.charAt(i);
            if (cur.children[c] == null) return false;
            cur = cur.children[c];
            if (cur.end > 0) {
                if (lc472Helper(word, i + 1)) return true;
            }
        }
        return false;
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
    final long lc1044Mod = 1000000007l;
    final long lc1044Base1 = 29;
    final long lc1044Base2 = 31;
    String lc1044S;

    public String longestDupSubstring(String s) {
        this.lc1044S = s;
        char[] ca = s.toCharArray();
        int lo = 0, hi = s.length() - 1;
        String result = "";
        String tmp = null;
        while (lo < hi) { // 找最大值
            int mid = lo + (hi - lo + 1) / 2;
            if ((tmp = lc1044Helper(ca, mid)) != null) {
                result = tmp;
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }

    private String lc1044Helper(char[] ca, int len) {
        Set<Integer> m1 = new HashSet<>();
        Set<Integer> m2 = new HashSet<>();
        long hash1 = 0, hash2 = 0, accu1 = 1, accu2 = 1;
        for (int i = 0; i < len; i++) {
            hash1 *= lc1044Base1;
            hash1 %= lc1044Mod;
            hash2 *= lc1044Base2;
            hash2 %= lc1044Mod;
            hash1 += ca[i];
            hash1 %= lc1044Mod;
            hash2 += ca[i];
            hash2 %= lc1044Mod;
            accu1 *= lc1044Base1;
            accu1 %= lc1044Mod;
            accu2 *= lc1044Base2;
            accu2 %= lc1044Mod;
        }
        m1.add((int) hash1);
        m2.add((int) hash2);
        for (int i = len; i < ca.length; i++) {
            String victim = lc1044S.substring(i - len + 1, i + 1);
            hash1 = (((hash1 * lc1044Base1 - accu1 * (ca[i - len])) % lc1044Mod) + lc1044Mod + ca[i]) % lc1044Mod;
            hash2 = (((hash2 * lc1044Base2 - accu2 * (ca[i - len])) % lc1044Mod) + lc1044Mod + ca[i]) % lc1044Mod;
            if (m1.contains((int) hash1) && m2.contains((int) hash2)) {
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

class Trie {
    TrieNode root = new TrieNode();

    public boolean search(String query) {
        TrieNode result = getNode(query);
        return result != null && result.end > 0;
    }

    public boolean beginWith(String query) {
        return getNode(query) != null;
    }

    public void addWord(String word) {
        // if (getNode(word) != null) return;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c] == null) {
                cur.children[c] = new TrieNode();
            }
            cur = cur.children[c];
            cur.path++;
        }
        cur.end++;
    }

    public boolean removeWord(String word) {
        if (getNode(word) == null) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c].path-- == 1) {
                cur.children[c] = null;
                return true;
            }
            cur = cur.children[c];
        }
        cur.end--;
        return true;
    }


    private TrieNode getNode(String query) {
        TrieNode cur = root;
        for (char c : query.toCharArray()) {
            if (cur.children[c] == null) return null;
            cur = cur.children[c];
        }
        return cur;
    }


    class TrieNode {
        TrieNode[] children = new TrieNode[128];
        int end = 0;
        int path = 0;
    }
}