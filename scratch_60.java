import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.minimumSwap(
                "xxyyxyxyxx",
                "xyyxyxxxyx"));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1433
    public boolean checkIfCanBreak(String s1, String s2) {
        int[] freq1 = new int[26], freq2 = new int[26];
        char[] ca1 = s1.toCharArray(), ca2 = s2.toCharArray();
        int n = ca1.length;
        for (int i = 0; i < n; i++) {
            freq1[ca1[i] - 'a']++;
            freq2[ca2[i] - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            int min = Math.min(freq1[i], freq2[i]);
            freq1[i] -= min;
            freq2[i] -= min;
        }
        int[] origFreq1 = Arrays.copyOf(freq1, 26);
        int[] origFreq2 = Arrays.copyOf(freq2, 26);

        // 预处理完之后, freq 的同一个位置只能是一正 一零, 或者两个0
        // 对于正数的位置, 如freq2[b] 的位置为正, 现在假设s1压制s2, 则从freq1[c....z] 的位置借数, 贪心地从小开始借, 直到能够把freq2[b]的正对冲掉
        // 如果对冲不掉, 则s1压制s2失败

        // 如果s1压制s2
        boolean s1BreakS2 = true;
        for (int i = 0; i < 26; i++) {
            if (freq1[i] == 0 && freq2[i] == 0) continue;

            if (freq2[i] > 0) {
                int count = freq2[i];
                for (int j = i + 1; j < 26; j++) {
                    if (freq1[j] > 0) {
                        int min = Math.min(count, freq1[j]);
                        count -= min;
                        freq1[j] -= min;
                        if (count == 0) break;
                    }
                }
                if (count > 0) {
                    s1BreakS2 = false;
                    break;
                }
            }
        }
        if (s1BreakS2) return true;

        freq1 = Arrays.copyOf(origFreq1, 26);
        freq2 = Arrays.copyOf(origFreq2, 26);

        boolean s2BreakS1 = true;
        for (int i = 0; i < 26; i++) {
            if (freq1[i] == 0 && freq2[i] == 0) continue;

            if (freq1[i] > 0) {
                int count = freq1[i];
                for (int j = i + 1; j < 26; j++) {
                    if (freq2[j] > 0) {
                        int min = Math.min(count, freq2[j]);
                        count -= min;
                        freq2[j] -= min;
                        if (count == 0) break;
                    }
                }
                if (count > 0) {
                    s2BreakS1 = false;
                    break;
                }
            }
        }
        if (s2BreakS1) return true;

        return false;
    }

    // LC910 ** 学习贪心思路
    public int smallestRangeII(int[] nums, int k) {
        int n = nums.length;
        Arrays.sort(nums);
        int result = nums[n - 1] - nums[0];
        for (int i = 0; i < n - 1; i++) {
            int a = nums[i], b = nums[i + 1];
            int hi = Math.max(a + k, nums[n - 1] - k);
            int lo = Math.min(nums[0] + k, b - k);
            result = Math.min(result, hi - lo);
        }
        return result;
    }

    // LC1826
    public int badSensor(int[] sensor1, int[] sensor2) {
        int idx = 0, n = sensor1.length;
        while (idx < n && sensor1[idx] == sensor2[idx]) idx++;
        if (idx == n || idx == n - 1) return -1;
        // 考虑是sensor1 异常还是sensor2异常

        // 如果sensor1 异常
        int tmpIdx1 = idx;
        while (tmpIdx1 + 1 < n && sensor1[tmpIdx1] == sensor2[tmpIdx1 + 1]) tmpIdx1++;


        // 如果sensor2 异常
        int tmpIdx2 = idx;
        while (tmpIdx2 + 1 < n && sensor1[tmpIdx2 + 1] == sensor2[tmpIdx2]) tmpIdx2++;

        if (tmpIdx1 == n - 1 && tmpIdx2 == n - 1) return -1;
        if (tmpIdx1 == n - 1) return 1;
        return 2;
    }

    // LC1542 ** 非常巧妙
    public int longestAwesome(String s) {
        Integer[] memo = new Integer[1 << 10];
        memo[0] = -1;
        char[] ca = s.toCharArray();
        int result = 1;
        int n = ca.length;
        int mask = 0;
        for (int i = 0; i < n; i++) {
            mask ^= (1 << (ca[i] - '0'));
            // 上一次出现的同频状态 (同频: 指的是各数字频率奇偶性相同)
            if (memo[mask] != null) {
                result = Math.max(result, i - memo[mask]);
            } else {
                memo[mask] = i;
            }

            // 允许有其中一个数字的频率奇偶性与上次不一样
            for (int j = 0; j < 10; j++) {
                int oddMask = mask ^ (1 << j);
                if (memo[oddMask] != null) {
                    result = Math.max(result, i - memo[oddMask]);
                }
            }
        }
        return result;
    }

    // LC1986
    Integer[][] lc1986Memo = new Integer[1 << 15][16];

    public int minSessions(int[] tasks, int sessionTime) {
        int n = tasks.length;
        int fullMask = (1 << n) - 1;

        return lc1986Helper(0, tasks, sessionTime, sessionTime, fullMask);
    }

    // 返回最小任务格数
    private int lc1986Helper(int mask, int[] tasks, int remainTime, int sessionTime, int fullMask) {
        if (mask == fullMask) return 1;
        if (lc1986Memo[mask][remainTime] != null) return lc1986Memo[mask][remainTime];
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < tasks.length; i++) {
            if (((mask >> i) & 1) == 1) continue; // 当前任务已经完成
            if (remainTime - tasks[i] >= 0) {
                result = Math.min(result, lc1986Helper(mask | (1 << i), tasks, remainTime - tasks[i], sessionTime, fullMask));
            } else {
                result = Math.min(result, 1 + lc1986Helper(mask, tasks, sessionTime, sessionTime, fullMask));
            }
        }
        return lc1986Memo[mask][remainTime] = result;
    }

    // LC563
    int lc563Result = 0;

    public int findTilt(TreeNode root) {
        lc563Helper(root);
        return lc563Result;
    }

    public int lc563Helper(TreeNode root) {
        if (root == null) return 0;
        int left = lc563Helper(root.left);
        int right = lc563Helper(root.right);
        int sum = root.val + left + right;
        lc563Result += Math.abs(left - right);
        return sum;
    }

    // LC1247 **
    public int minimumSwap(String s1, String s2) {
        char[] ca1 = s1.toCharArray(), ca2 = s2.toCharArray();
        int x = 0, y = 0;
        for (int i = 0; i < ca1.length; i++) {
            if (ca1[i] == ca2[i]) continue;
            if (ca1[i] == 'x') x++;
            else y++;
        }
        if ((x + y) % 2 == 1) return -1;
        return x / 2 + y / 2 + x % 2 + y % 2;
    }

    // LC1409 ** 树状数组解法
    public int[] processQueries(int[] queries, int m) {
        int n = queries.length;
        BIT bit = new BIT(m + n + 1);
        int[] pos = new int[m + 1];
        for (int i = 1; i <= m; i++) {
            pos[i] = n + i;
            bit.set(n + i, 1);
        }
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            int cur = pos[queries[i]];
            bit.set(cur, 0);
            result[i] = bit.sumRange(0, cur - 1);
            cur = n - i;
            pos[queries[i]] = cur;
            bit.set(cur, 1);
        }
        return result;
    }

    // LC1666 ** 理解翻转规则
    class Lc1666 {
        // 你可以按照下列步骤修改从 leaf到 root的路径中除 root 外的每个节点 cur：
        // 如果cur有左子节点，则该子节点变为cur的右子节点。
        // cur的原父节点变为cur的左子节点。
        //
        public Node flipBinaryTree(Node root, Node leaf) {
            return helper(leaf);
        }

        private Node helper(Node cur) {
            if (cur == null) return null;
            Node parent = cur.parent;
            //断开当前节点和父节点的联系
            if (parent != null) {
                if (cur == parent.left) {
                    parent.left = null;
                } else {
                    parent.right = null;
                }
                cur.parent = null;
            }
            helper(parent);
            if (parent != null) {
                if (cur.left != null) {
                    cur.right = cur.left;
                }
                cur.left = parent;
                parent.parent = cur;
            }
            return cur;
        }

        class Node {
            public int val;
            public Node left;
            public Node right;
            public Node parent;
        }
    }

    // LC437 **
    Map<Integer, Integer> lc437Prefix;

    public int pathSumIii(TreeNode root, int targetSum) {
        lc437Prefix = new HashMap<>();
        lc437Prefix.put(0, 1); // root, sum=0
        return lc437Helper(root, 0, targetSum);
    }

    private int lc437Helper(TreeNode root, int cur, int target) {
        if (root == null) return 0;
        cur += root.val;
        int result = lc437Prefix.getOrDefault(cur - target, 0);
        lc437Prefix.put(cur, lc437Prefix.getOrDefault(cur, 0) + 1);
        result += lc437Helper(root.left, cur, target);
        result += lc437Helper(root.right, cur, target);
        lc437Prefix.put(cur, lc437Prefix.get(cur) - 1);
        return result;
    }

    // LC124
    int lc124Result = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        lc124Helper(root);
        return lc124Result;
    }

    private int lc124Helper(TreeNode root) {
        if (root == null) return 0;
        int left = lc124Helper(root.left);
        int right = lc124Helper(root.right);
        int val = root.val;
        int thisNode = val + Math.max(0, left) + Math.max(0, right);
        lc124Result = Math.max(lc124Result, thisNode);
        return Math.max(val, Math.max(val + left, val + right));
    }

    // LC113
    List<List<Integer>> lc113Result;

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        lc113Result = new ArrayList<>();
        lc113Helper(root, targetSum, new ArrayList<>());
        return lc113Result;
    }

    private void lc113Helper(TreeNode root, int target, List<Integer> path) {
        if (root == null) return;
        path.add(root.val);
        if (root.left == null && root.right == null && target - root.val == 0) {
            lc113Result.add(new ArrayList<>(path));
        }
        lc113Helper(root.left, target - root.val, path);
        lc113Helper(root.right, target - root.val, path);
        path.remove(path.size() - 1);
    }

    // LC666
    int lc666Result = 0;
    Integer[] lc666Vals = new Integer[1 << 5];

    public int pathSum(int[] nums) {
        for (int i : nums) {
            int level = ((i / 100) % 10) - 1; // zero-based
            int ith = ((i / 10) % 10) - 1; // zero-based
            int id = (1 << level) - 1 + ith;
            lc666Vals[id] = i % 10;
        }
        lc666Helper(0, 0);
        return lc666Result;
    }

    private void lc666Helper(int id, int sum) {
        if (id >= (1 << 5)) return;
        if (lc666Vals[id] == null) return;
        sum += lc666Vals[id];
        if (lc666Vals[id * 2 + 1] == null && lc666Vals[id * 2 + 2] == null) {
            lc666Result += lc666Vals[id];
        } else {
            lc666Helper(id * 2 + 1, sum);
            lc666Helper(id * 2 + 2, sum);
        }
    }


    // LC508
    public int[] findFrequentTreeSum(TreeNode root) {
        Map<Integer, Integer> freq = new HashMap<>();
        lc508Helper(root, freq);
        List<Integer> result = new ArrayList<>();
        int maxFreq = 0;
        for (Map.Entry<Integer, Integer> e : freq.entrySet()) {
            if (e.getValue() > maxFreq) {
                maxFreq = e.getValue();
                result.clear();
                result.add(e.getKey());
            } else if (e.getValue() == maxFreq) {
                result.add(e.getKey());
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    private int lc508Helper(TreeNode root, Map<Integer, Integer> freq) {
        if (root == null) return 0;
        int left = lc508Helper(root.left, freq);
        int right = lc508Helper(root.right, freq);
        int sum = root.val + left + right;
        freq.put(sum, freq.getOrDefault(sum, 0) + 1);
        return sum;
    }

    // LC624
    public int maxDistance(List<List<Integer>> arrays) {
        int n = arrays.size(), result = -1;
        int min = arrays.get(0).get(0), max = arrays.get(0).get(arrays.get(0).size() - 1);
        for (int i = 1; i < n; i++) {
            List<Integer> a = arrays.get(i);
            int curMin = a.get(0), curMax = a.get(a.size() - 1);
            result = Math.max(result, Math.abs(curMax - min));
            result = Math.max(result, Math.abs(max - curMin));
            min = Math.min(min, curMin);
            max = Math.max(max, curMax);
        }
        return result;
    }

    // LC841
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size(), ctr = 0;
        Deque<Integer> stack = new LinkedList<>();
        stack.push(0);
        boolean[] visited = new boolean[n];
        while (!stack.isEmpty()) {
            int p = stack.pop();
            if (visited[p]) continue;
            ctr++;
            visited[p] = true;
            for (int next : rooms.get(p)) {
                if (!visited[next]) stack.push(next);
            }
        }
        return ctr == n;
    }

    // LC592
    public String fractionAddition(String expression) {
        long num = 0l, den = 1l; // 初始化 0/1
        if (expression.charAt(0) != '-') expression = '+' + expression;
        char[] ca = expression.toCharArray();
        int idx = 0, n = ca.length;
        while (idx < n) {
            long sign = ca[idx] == '+' ? 1l : -1l;
            idx++;
            int numLeft = idx;
            while ((idx + 1) < n && ca[idx + 1] != '/') idx++;
            int numRight = idx;
            int curNum = Integer.valueOf(expression.substring(numLeft, numRight + 1));
            idx += 2;
            int denLeft = idx;
            while ((idx + 1) < n && (ca[idx + 1] != '+' && ca[idx + 1] != '-')) idx++;
            int denRight = idx;
            int curDen = Integer.valueOf(expression.substring(denLeft, denRight + 1));
            idx++;


            // 处理通分
            long tmpDen = den * curDen;
            long tmpNum = num * curDen + sign * (curNum * den);
            long gcd = gcd(tmpDen, tmpNum);
            tmpDen /= gcd;
            tmpNum /= gcd;
            den = tmpDen;
            num = tmpNum;
        }
        return (num * den < 0l ? "-" : "") + Math.abs(num) + "/" + Math.abs(den);
    }

    private long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC318
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] bitmask = new int[n];
        for (int i = 0; i < n; i++) {
            int mask = 0;
            for (char c : words[i].toCharArray()) {
                mask |= 1 << (c - 'a');
            }
            bitmask[i] = mask;
        }
        int max = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((bitmask[i] & bitmask[j]) == 0) {
                    max = Math.max(words[i].length() * words[j].length(), max);
                }
            }
        }
        return max;
    }
}


// LC928 Try Tarjan O(n)
class Lc928Tarjan {
    int n;
    int[] low, timestamp, groupSize, spreadSize, save, virusTag, finalParent;
    List<List<Integer>> mtx; // 邻接表
    int timing;
    int spreadTiming;
    int maxSaveCount = Integer.MIN_VALUE;
    int currentRoot;
    int result = -1;

    public int minMalwareSpread(int[][] graph, int[] virus) {
        Arrays.sort(virus);
        build(graph, virus);
        for (int i = 0; i < n; i++) {
            if (timestamp[i] == -1) {
                currentRoot = i;
                timing = 0; // 注意我们给每一个连通分量分配一个全新的计时器(从0开始)
                spreadTiming = 0; // 感染数量也是
                tarjan(i, i);
            }
        }

        for (int i = 0; i < n; i++) {
            // **** 父块的处理, 很关键
            if (spreadSize[finalParent[i]] == spreadSize[i]) {
                save[i] += groupSize[finalParent[i]] - groupSize[i];
            }
            if (virusTag[i] == 1 && save[i] > maxSaveCount) {
                result = i;
                maxSaveCount = save[i];
            }
        }
        return result;
    }

    private void tarjan(int cur, int parent) {
        // 借用 Tarjan 求 **割点** 的算法流程。 注意此处不是真的求割点, 所以不需要统计直接孩子的数量

        low[cur] = timestamp[cur] = ++timing; // timing 是遇到一个新节点就自增
        spreadTiming += virusTag[cur]; // spreadTiming 是遇到一个新的病毒节点才自增

        finalParent[cur] = currentRoot;
        groupSize[cur] = 1;
        spreadSize[cur] = virusTag[cur];

        for (int next : mtx.get(cur)) {
            if (next == parent) continue;

            int thisMomentTiming = timing;
            int thisMomentSpreadTiming = spreadTiming;

            if (timestamp[next] == -1) {
                tarjan(next, cur);
            }

            int deltaTiming = timing - thisMomentTiming;
            int deltaSpreadTiming = spreadTiming - thisMomentSpreadTiming;

            // 判断next开始的路径能不能回到cur, 标准Tarjan求割点的做法。用以判断next开始的子图是不是独立子图
            if (low[next] >= timestamp[cur]) {
                if (deltaSpreadTiming == 0) { // 说明经过这一点next之后没有新的节点被感染, 也即如果cur消失后, 能够多拯救多少节点
                    save[cur] += deltaTiming; // DFS完这个子图, delta(timing) 即后序遍历到的节点个数
                }
                groupSize[cur] += deltaTiming;
                spreadSize[cur] += deltaSpreadTiming;
            }

            low[cur] = Math.min(low[cur], low[next]);
        }
    }


    private void build(int[][] graph, int[] virus) {
        n = graph.length;
        low = new int[n];
        timestamp = new int[n];
        groupSize = new int[n];
        spreadSize = new int[n];
        virusTag = new int[n];
        finalParent = new int[n];
        save = new int[n];
        Arrays.fill(low, -1);
        Arrays.fill(timestamp, -1);

        for (int i : virus) virusTag[i] = 1;

        mtx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && graph[i][j] == 1) {
                    mtx.get(i).add(j);
                }
            }
        }
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

class BIT {
    int len;
    int[] tree;

    public BIT(int n) {
        len = n;
        tree = new int[n + 1];
    }

    public BIT(int[] arr) {
        len = arr.length;
        tree = new int[len + 1];

        for (int i = 0; i < len; i++) {
            int one = i + 1;
            tree[one] += arr[i];
            int nextOne = one + lowbit(one);
            if (nextOne <= len) tree[nextOne] += tree[one];
        }
    }

    public void set(int zero, int val) {
        int delta = val - get(zero);
        update(zero, delta);
    }

    public void update(int zero, int delta) {
        updateOne(zero + 1, delta);
    }

    public int sumRange(int left, int right) {
        return sumOne(right + 1) - sumOne(left);
    }

    public int get(int zero) {
        return sumOne(zero + 1) - sumOne(zero);
    }

    public int sumOne(int one) {
        int result = 0;
        while (one > 0) {
            result += tree[one];
            one -= lowbit(one);
        }
        return result;
    }

    public void updateOne(int one, int delta) {
        while (one <= len) {
            tree[one] += delta;
            one += lowbit(one);
        }
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}