import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


//        System.out.println(s.minTapsGreedy(7, new int[]{1, 2, 1, 0, 2, 1, 0, 1}));
        System.out.println(s.combinationSum2(new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                30));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC40 给的数字有重复 选择不可重复 可选个数100 无法位运算枚举 哈希超时
    List<List<Integer>> result;
    Set<Integer> hash53, hash59;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        hash53 = new HashSet<>();
        hash59 = new HashSet<>();
        Arrays.sort(candidates);
        result = new ArrayList<>();
        helper(candidates, 0, new ArrayList<>(), target);
        return result;
    }

    private void helper(int[] candidates, int curIdx, List<Integer> selected, int remain) {
        if (remain == 0) {
            int h53 = hash(selected, 53), h59 = hash(selected, 59);
            if (!hash53.contains(h53) || !hash59.contains(h59)) {
                hash53.add(h53);
                hash59.add(h59);
                result.add(new ArrayList<>(selected));
            }
            return;
        }
        for (int i = curIdx; i < candidates.length; i++) {
            int c = candidates[i];
            if (c > remain) break;
            if (c <= remain) {
                selected.add(c);
                helper(candidates, i + 1, selected, remain - c);
                selected.remove(selected.size() - 1);
            }
        }
    }

    private int hash(List<Integer> selected, int prime) {
        long result = 0, accu = 1, mod = 1000000007;
        for (int i = 0; i < selected.size(); i++) {
            result = (result + accu * selected.get(i)) % mod;
            accu = (accu * prime) % mod;
        }
        return (int) (result);
    }

    // LC39 可重复
    List<List<Integer>> lc39Result;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        lc39Result = new ArrayList<>();
        lc39Helper(candidates, 0, new ArrayList<>(), target);
        return lc39Result;
    }

    private void lc39Helper(int[] candidates, int curIdx, List<Integer> selected, int remain) {
        if (remain == 0) {
            lc39Result.add(new ArrayList<>(selected));
            return;
        }
        for (int i = curIdx; i < candidates.length; i++) {
            int c = candidates[i];
            if (c <= remain) {
                selected.add(c);
                lc39Helper(candidates, i, selected, remain - c);
                selected.remove(selected.size() - 1);
            }
        }
    }

    // LC166
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        long num = Math.abs(numerator), den = Math.abs(0l + denominator);
        String left = String.valueOf(num / den);
        if ((0l + numerator) * (0l + denominator) < 0l) left = "-" + left;
        long remainder = num % den;
        if (remainder == 0l) return left;
        left += ".";
        Map<Long, Integer> map = new HashMap<>();
        StringBuilder sb = new StringBuilder(left);
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

    // LC405
    public String toHex(int num) {
        if (num == 0) return "0";
        char[] hex = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
        StringBuilder result = new StringBuilder();
        // int -> 4byte ,1byte = 8bit = 2*4bit
        for (int i = 1; i <= 8; i++) {
            int offset = i * 4;
            int this4bit = (num >> (32 - offset)) & 0x0f;
            if (result.length() == 0 && this4bit == 0) continue;
            result.append(hex[this4bit]);
        }
        return result.toString();
    }

    // LC1057
    public int[] assignBikesI(int[][] workers, int[][] bikes) {
        int nw = workers.length, nb = bikes.length;
        int[][] distance = new int[nw][nb];
        int[] result = new int[nw];
        boolean[] visitedBike = new boolean[nb];
        boolean[] visitedWorker = new boolean[nb];
        TreeSet<int[]> ts = new TreeSet<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (distance[o1[0]][o1[1]] == distance[o2[0]][o2[1]]) {
                    if (o1[0] == o2[0]) {
                        return o1[1] - o2[1];
                    }
                    return o1[0] - o2[0];
                }
                return distance[o1[0]][o1[1]] - distance[o2[0]][o2[1]];
            }
        });
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < nb; j++) {
                distance[i][j] = Math.abs(workers[i][0] - bikes[j][0]) + Math.abs(workers[i][1] - bikes[j][1]);
            }
        }
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < nb; j++) {
                ts.add(new int[]{i, j});
            }
        }
        Iterator<int[]> it = ts.iterator();
        while (it.hasNext()) {
            int[] next = it.next();
            if (visitedBike[next[1]]) {
                it.remove();
                continue;
            }
            if (visitedWorker[next[0]]) {
                it.remove();
                continue;
            }
            result[next[0]] = next[1];
            visitedWorker[next[0]] = true;
            visitedBike[next[1]] = true;
        }
        return result;
    }

    // LC1066
    public int assignBikes(int[][] workers, int[][] bikes) {
        int nw = workers.length, nb = bikes.length;
        // dp[mask][mask]
        int[][] dp = new int[1 << nw][1 << nb];

        for (int mw = 0; mw < 1 << nw; mw++) {
            Arrays.fill(dp[mw], Integer.MAX_VALUE / 2);
        }
        dp[0][0] = 0;

        for (int mw = 1; mw < 1 << nw; mw++) {
            for (int mb = 1; mb < 1 << nb; mb++) {
                if (Integer.bitCount(mw) > Integer.bitCount(mb)) continue;
                for (int w = 0; w < nw; w++) {
                    if (((mw >> w) & 1) == 1) {
                        int parentWorkerMask = mw ^ (1 << w);
                        for (int b = 0; b < nb; b++) {
                            if (((mb >> b) & 1) == 1) {
                                int parentBikeMask = mb ^ (1 << b);
                                int distance = Math.abs(workers[w][0] - bikes[b][0]) + Math.abs(workers[w][1] - bikes[b][1]);
                                dp[mw][mb] = Math.min(dp[mw][mb], dp[parentWorkerMask][parentBikeMask] + distance);
                            }
                        }
                    }
                }
            }
        }
        int min = Integer.MAX_VALUE / 2;
        for (int i = 0; i < 1 << nb; i++) {
            min = Math.min(min, dp[(1 << nw) - 1][i]);
        }
        return min;
    }

    // JZOF II 055
    class BSTIterator {
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode cur;

        public BSTIterator(TreeNode root) {
            cur = root;
        }

        public int next() { // 先序遍历
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            int result = cur.val;
            cur = cur.right;
            return result;
        }

        public boolean hasNext() {
            return cur != null || !stack.isEmpty();
        }
    }

    // JZOF 26
    public boolean isSubStructure(TreeNode a, TreeNode b) {
        // 空树不是任何树的子结构
        if (a == null || b == null) return false;
        return helper(a, b) || isSubStructure(a.left, b) || isSubStructure(a.right, b);
    }

    private boolean helper(TreeNode a, TreeNode b) {
        if (b == null) return true;
        if (a == null || a.val != b.val) return false;
        return helper(a.left, b.left) && helper(a.right, b.right);
    }

    // LC1024 DP
    public int videoStitching(int[][] clips, int time) {
        int[] dp = new int[time + 1]; // 表示当前下标能覆盖到的最远距离
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i = 1; i <= time; i++) {
            for (int[] c : clips) {
                if (c[0] < i && i <= c[1]) { // 如果i在该片段的覆盖范围内 (注意点还是线)
                    dp[i] = Math.min(dp[i], 1 + dp[c[0]]);
                }
            }
        }
        return dp[time] == Integer.MAX_VALUE / 2 ? -1 : dp[time];
    }


    // LC45
    Integer[] lc45Memo;

    public int jump(int[] nums) {
        lc45Memo = new Integer[nums.length + 1];
        return lc45Helper(0, nums);
    }

    private int lc45Helper(int curIdx, int[] nums) {
        if (curIdx >= nums.length - 1) return 0;
        if (lc45Memo[curIdx] != null) return lc45Memo[curIdx];
        int min = Integer.MAX_VALUE / 2; // 防溢出
        for (int i = 1; i <= nums[curIdx]; i++) {
            min = Math.min(min, 1 + lc45Helper(curIdx + i, nums));
        }
        return lc45Memo[curIdx] = min;
    }

    // LC1326
    public int minTapsGreedy(int n, int[] ranges) {
        int[] land = new int[n]; // 表示覆盖范围内最远覆盖到的土地下标
        for (int i = 0; i < n; i++) {
            int l = Math.max(i - ranges[i], 0);
            int r = Math.min(i + ranges[i], n);
            for (int j = l; j < r; j++) { // 最多两百次, 视作常数
                land[j] = Math.max(land[j], r); // 更新范围内最远覆盖到的土地下标
            }
        }
        int ctr = 0, cur = 0;
        while (cur < n) {
            if (land[cur] == 0) return -1; // 如果有土地未被覆盖到
            cur = land[cur];
            ctr++;
        }
        return ctr;
    }

    public int minTaps(int n, int[] ranges) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i <= n; i++) {
            if (ranges[i] == 0) continue;
            tm.put(Math.max(i - ranges[i], 0), Math.min(Math.max(tm.getOrDefault(i - ranges[i], Integer.MIN_VALUE), i + ranges[i]), n));
        }
        int result = Integer.MAX_VALUE;
        loop:
        for (Map.Entry<Integer, Integer> i : tm.entrySet()) { // 从i开始
            if (i.getKey() > 0) break;
            LinkedList<Map.Entry<Integer, Integer>> candidateQueue = new LinkedList<>();
            candidateQueue.add(i);
            while (candidateQueue.getLast().getValue() < n) {
                Map.Entry<Integer, Integer> last = candidateQueue.getLast();
                NavigableMap<Integer, Integer> intersect = tm.subMap(last.getKey(), false, last.getValue(), true);
                if (intersect.isEmpty()) break loop;
                Map.Entry<Integer, Integer> candidate = null;
                int rightMost = last.getValue();
                for (Map.Entry<Integer, Integer> j : intersect.entrySet()) {
                    if (j.getValue() > rightMost) {
                        candidate = j;
                        rightMost = j.getValue();
                    }
                }
                if (candidate == null) break;
                candidateQueue.add(candidate);
            }
            if (candidateQueue.getLast().getValue() < n) break;
            result = Math.min(result, candidateQueue.size());
            if (result == 1) return 1;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}