import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.pathSum(new int[]{113, 215, 221}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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