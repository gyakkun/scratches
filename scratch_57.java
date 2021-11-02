import java.util.*;

// LC1320 Hard TLE
class Lc1320 {
    int[][] alphabetIdx;
    int[][] distance;
    int min = Integer.MAX_VALUE;

    public int minimumDistance(String word) {
        init();
        // 遍历两个手指的初始位置
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
                helper(0, i, j, word, 0);
            }
        }
        return min;
    }


    private void helper(int targetCharIdx, int curLeft, int curRight, String word, int curSteps) {
        if (targetCharIdx == word.length()) {
            min = Math.min(min, curSteps);
            return;
        }
        if (curSteps > min) return;
        // 选择左手到下一个字母
        int leftToNext = distance[charConvert(curLeft)][word.charAt(targetCharIdx)];
        helper(targetCharIdx + 1, word.charAt(targetCharIdx) - 'A', curRight, word, curSteps + leftToNext);

        // 选择右手到下一个字母
        int rightToNext = distance[charConvert(curRight)][word.charAt(targetCharIdx)];
        helper(targetCharIdx + 1, curLeft, word.charAt(targetCharIdx) - 'A', word, curSteps + rightToNext);
    }

    private char charConvert(int code) {
        return (char) ('A' + code);
    }

    private int[] idxConvert(int code) {
        return new int[]{code / 6, code % 6};
    }

    private void init() {
        alphabetIdx = new int[256][2];
        distance = new int[256][256];
        alphabetIdx['A'] = new int[]{0, 0};
        alphabetIdx['B'] = new int[]{0, 1};
        alphabetIdx['C'] = new int[]{0, 2};
        alphabetIdx['D'] = new int[]{0, 3};
        alphabetIdx['E'] = new int[]{0, 4};
        alphabetIdx['F'] = new int[]{0, 5};
        alphabetIdx['G'] = new int[]{1, 0};
        alphabetIdx['H'] = new int[]{1, 1};
        alphabetIdx['I'] = new int[]{1, 2};
        alphabetIdx['J'] = new int[]{1, 3};
        alphabetIdx['K'] = new int[]{1, 4};
        alphabetIdx['L'] = new int[]{1, 5};
        alphabetIdx['M'] = new int[]{2, 0};
        alphabetIdx['N'] = new int[]{2, 1};
        alphabetIdx['O'] = new int[]{2, 2};
        alphabetIdx['P'] = new int[]{2, 3};
        alphabetIdx['Q'] = new int[]{2, 4};
        alphabetIdx['R'] = new int[]{2, 5};
        alphabetIdx['S'] = new int[]{3, 0};
        alphabetIdx['T'] = new int[]{3, 1};
        alphabetIdx['U'] = new int[]{3, 2};
        alphabetIdx['V'] = new int[]{3, 3};
        alphabetIdx['W'] = new int[]{3, 4};
        alphabetIdx['X'] = new int[]{3, 5};
        alphabetIdx['Y'] = new int[]{4, 0};
        alphabetIdx['Z'] = new int[]{4, 1};
        alphabetIdx['['] = new int[]{4, 2};
        alphabetIdx['\\'] = new int[]{4, 3};
        alphabetIdx[']'] = new int[]{4, 4};
        alphabetIdx['^'] = new int[]{4, 5};

        for (int i = 'A'; i <= '^'; i++) {
            for (int j = 'A'; j <= '^'; j++) {
                distance[i][j] = distance[j][i] =
                        Math.abs(alphabetIdx[i][0] - alphabetIdx[j][0])
                                + Math.abs(alphabetIdx[i][1] - alphabetIdx[j][1]);
            }
        }

    }

}

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        Lc1320 lc1320 = new Lc1320();

        System.out.println(lc1320.minimumDistance("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }


    // LC237
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    // LC1462
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        // prerequisites[k] : [i,j] , i before j
        // queries[k]: [i,j] is i before j
        boolean[][] reachable = new boolean[numCourses][numCourses];
        for (int[] pq : prerequisites) {
            reachable[pq[0]][pq[1]] = true;
        }
        for (int i = 0; i < numCourses; i++) {
            Deque<Integer> q = new LinkedList<>();
            boolean[] visited = new boolean[numCourses];
            q.offer(i);
            while (!q.isEmpty()) {
                int p = q.poll();
                if (visited[p]) continue;
                visited[p] = true;
                if (i != p) reachable[i][p] = true;
                for (int next = 0; next < numCourses; next++) {
                    if (reachable[p][next]) {
                        q.offer(next);
                    }
                }
            }
        }
        List<Boolean> result = new ArrayList<>(queries.length);
        for (int[] q : queries) {
            result.add(reachable[q[0]][q[1]]);
        }
        return result;
    }

    // LC1615
    public int maximalNetworkRank(int n, int[][] roads) {
        Map<Integer, Set<Integer>> m = new HashMap<>(n);
        int result = 0;
        for (int i = 0; i < n; i++) m.put(i, new HashSet<>());
        for (int[] r : roads) {
            m.get(r[0]).add(r[1]);
            m.get(r[1]).add(r[0]);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int cross = m.get(i).contains(j) ? 1 : 0;
                int rank = m.get(i).size() + m.get(j).size() - cross;
                result = Math.max(rank, result);
            }
        }
        return result;
    }

    // LC854 ** 很妙的DFS
    Map<String, Map<String, Integer>> lc854Memo = new HashMap<>();

    public int kSimilarity(String s1, String s2) {
        if (s1.equals("")) return 0;
        if (lc854Memo.containsKey(s1) && lc854Memo.get(s1).containsKey(s2)) return lc854Memo.get(s1).get(s2);
        lc854Memo.putIfAbsent(s1, new HashMap<>());
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) == s2.charAt(0)) {
                if (i == 0) {
                    result = Math.min(result, kSimilarity(s1.substring(1), s2.substring(1)));
                } else {
                    // 把 i 换到第一个来
                    result = Math.min(result, 1 + kSimilarity(s1.substring(1, i) + s1.charAt(0) + s1.substring(i + 1), s2.substring(1)));
                }
            }
        }
        lc854Memo.get(s1).put(s2, result);
        return result;
    }

    // LC1944 ** 单调栈
    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] result = new int[n];
        Deque<Integer> stack = new LinkedList<>(); // 递减栈
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty()) {
                result[i]++;
                if (heights[i] > heights[stack.peek()]) {
                    stack.pop();
                } else {
                    break;
                }
            }
            stack.push(i);
        }
        return result;
    }

    // LC944
    public int minDeletionSize(String[] strs) {
        int result = 0;
        outer:
        for (int i = 0; i < strs[0].length(); i++) {
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].charAt(i) < strs[j - 1].charAt(i)) {
                    result++;
                    continue outer;
                }
            }
        }
        return result;
    }

    // LC885
    public int[][] spiralMatrixIII(int rows, int cols, int rStart, int cStart) {
        int[][] result = new int[rows * cols][];
        int ctr = 0, total = rows * cols;
        int[][] directions = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int dIdx = 0;
        int r = rStart, c = cStart;
        result[ctr++] = new int[]{r, c};
        int steps = 1;
        while (ctr != total) {
            for (int i = 0; i < steps; i++) {
                r += directions[dIdx][0];
                c += directions[dIdx][1];
                if (r >= 0 && r < rows && c >= 0 && c < cols && ctr < total) {
                    result[ctr++] = new int[]{r, c};
                }
            }
            if (dIdx == 1 || dIdx == 3) steps++;
            dIdx = (dIdx + 1) % 4;
        }
        return result;
    }

    // LC1703 ** from solution 非常严格的数学证明 学不来
    // https://leetcode-cn.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/solution/de-dao-lian-xu-k-ge-1-de-zui-shao-xiang-lpa9i/
    public int minMoves(int[] nums, int k) {
        int sum = 0, n = nums.length, result = Integer.MAX_VALUE;
        for (int i : nums) sum += i;
        // 1. 找出所有1的位置
        int[] oneIdx = new int[sum];
        int ctr = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                oneIdx[ctr++] = i;
            }
        }

        // 2. 构造 g 函数, 构造g函数的累加函数(前缀和)
        int[] g = new int[sum];
        for (int i = 0; i < sum; i++) {
            g[i] = oneIdx[i] - i;
        }
        int[] gPrefix = new int[sum + 1];
        for (int i = 0; i < sum; i++) gPrefix[i + 1] = gPrefix[i] + g[i];

        // 3. 滑窗
        for (int i = 0; i + k <= g.length; i++) {
            int mid = (i + i + k - 1) / 2;
            int q = g[mid];
            int possible = (2 * (mid - i) - k + 1) * q + (gPrefix[i + k] - gPrefix[mid + 1]) - (gPrefix[mid] - gPrefix[i]);
            result = Math.min(result, possible);
        }
        return result;
    }

    // LC1576
    public String modifyString(String s) {
        StringBuilder sb = new StringBuilder();
        int n = s.length();
        char[] ca = s.toCharArray();
        outer:
        for (int i = 0; i < n; i++) {
            if (Character.isLetter(ca[i])) {
                sb.append(ca[i]);
                continue;
            }
            Set<Character> invalid = new HashSet<>(3);
            if (i - 1 >= 0) {
                invalid.add(sb.charAt(sb.length() - 1));
            }
            if (i + 1 < n && ca[i + 1] != '?') {
                invalid.add(ca[i + 1]);
            }
            for (int j = 0; j < 26; j++) {
                char target = (char) ('a' + j);
                if (!invalid.contains(target)) {
                    sb.append(target);
                    continue outer;
                }
            }
        }
        return sb.toString();
    }

    // LC1998 **
    public boolean gcdSort(int[] nums) {
        DSUArray dsu = new DSUArray((int) 1e5);
        for (int i : nums) {
            if (dsu.contains(i)) continue;
            dsu.add(i);
            int factor = 2;
            int victim = i;
            // 分解质因数
            while (victim != 0) {
                while (victim % factor == 0) {
                    dsu.add(factor);
                    dsu.merge(factor, i);
                    victim /= factor;
                }
                factor++;
                if (factor > victim) break;
            }
        }

        int[] sorted = Arrays.copyOf(nums, nums.length);
        Arrays.sort(sorted);

        for (int i = 0; i < nums.length; i++) {
            if (sorted[i] != nums[i]) {
                if (!dsu.isConnected(sorted[i], nums[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    // LC575
    public int distributeCandies(int[] candyType) {
        int n = candyType.length, result = 0;
        Arrays.sort(candyType);
        for (int i = 0; i < n; ) {
            while (i + 1 < n && candyType[i] == candyType[i + 1]) i++;
            result++;
            if (result >= n / 2) return n / 2;
            i++;
        }
        return result;
    }

    // LC260 **
    public int[] singleNumber(int[] nums) {
        long xor = 0;
        for (long i : nums) xor ^= i;
        long lsb = xor & (-xor);
        long a = 0, b = 0;
        for (int i : nums) {
            if (((long) (i) & lsb) != 0) {
                a ^= i;
            } else {
                b ^= i;
            }
        }
        return new int[]{(int) a, (int) b};
    }

    // LC1911 **
    // from https://codeforces.com/contest/1420/submission/93658399
    public long maxAlternatingSum(int[] nums) {
        // 偶数下标之和减奇数下标之和
        int n = nums.length;
        long[][] dp = new long[n + 2][2];
        // dp[n][0/1] 表示选前n个数做子序列时候, 第n个数字下标作为偶/奇数时候的最大值
        for (int i = 0; i < n; i++) {
            dp[i + 1][0] = Math.max(dp[i][0]/*不选这个数*/, dp[i][1] + nums[i]/*选这个数, 从上一个奇数结尾的状态转移过来*/);
            dp[i + 1][1] = Math.max(dp[i][1], dp[i][0] - nums[i]);
        }
        return Math.max(dp[n][0], dp[n][1]);
    }

    // LC1102 **
    public int maximumMinimumPathDSU(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int start = 0, end = m * n - 1;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        DSUArray dsu = new DSUArray(m * n + 2);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o[2]));
        for (int i = 1; i < end; i++) {
            pq.offer(new int[]{i / n, i % n, grid[i / n][i % n]});
        }
        dsu.add(start);
        dsu.add(end);
        while (!pq.isEmpty() && !dsu.isConnected(start, end)) {
            int[] p = pq.poll();
            int r = p[0], c = p[1], val = p[2];
            int id = r * n + c;
            dsu.add(id);
            for (int[] d : direction) {
                int nr = r + d[0], nc = c + d[1];
                int nid = nr * n + nc;
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && dsu.contains(nid)) {
                    dsu.merge(id, nid);
                    min = Math.min(min, val);
                }
            }
        }
        return min;
    }

    public int maximumMinimumPath(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        Set<Integer> possibleSet = new HashSet<>();
        possibleSet.add(min);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] <= min) {
                    possibleSet.add(grid[i][j]);
                }
            }
        }
        List<Integer> possibleList = new ArrayList<>(possibleSet);
        Collections.sort(possibleList);
        int lo = 0, hi = possibleList.size() - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            int victim = possibleList.get(mid);
            boolean[][] visited = new boolean[m][n];
            if (lc1102Helper(0, 0, grid, visited, victim)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return possibleList.get(lo);
    }

    private boolean lc1102Helper(int r, int c, int[][] grid, boolean[][] visited, int bound) {
        if (r == grid.length - 1 && c == grid[0].length - 1) return true;
        visited[r][c] = true;
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] d : direction) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && !visited[nr][nc] && grid[nr][nc] >= bound) {
                if (lc1102Helper(nr, nc, grid, visited, bound)) return true;
            }
        }
        return false;
    }


    // LC565
    public int arrayNesting(int[] nums) {
        int n = nums.length, max = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int count = 1, cur = i;
                visited[i] = true;
                while (!visited[nums[cur]]) {
                    visited[nums[cur]] = true;
                    count++;
                    cur = nums[cur];
                }
                max = Math.max(count, max);
            }
        }
        return max;
    }

    // LC1145 ** 贪心策略: 选x周围的三个节点, 统计两个子图节点数量
    Map<Integer, TreeNode> valNodeMap = new HashMap<>();
    Map<TreeNode, TreeNode> fatherMap = new HashMap<>();

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            valNodeMap.put(p.val, p);
            n = Math.max(n, p.val);
            if (p.left != null) {
                fatherMap.put(p.left, p);
                q.offer(p.left);
            }
            if (p.right != null) {
                fatherMap.put(p.right, p);
                q.offer(p.right);
            }
        }

        TreeNode xNode = valNodeMap.get(x);
        TreeNode[] choices = new TreeNode[]{getFather(xNode), getLeft(xNode), getRight(xNode)};
        for (TreeNode y : choices) {
            if (y != null) {
                if (lc1145Helper(n, x, xNode, y)) return true;
            }
        }
        return false;
    }

    private boolean lc1145Helper(int n, int x, TreeNode rivalFirstChoice, TreeNode y) {
        Deque<TreeNode> q = new LinkedList<>();
        // BFS, 统计邻接节点数量
        boolean[] visited = new boolean[n + 1];
        visited[x] = true;
        q.offer(y);
        int myCount = getTreeNodeCount(q, visited);

        Arrays.fill(visited, false);
        visited[y.val] = true;
        q.clear();
        q.offer(rivalFirstChoice);
        int rivalCount = getTreeNodeCount(q, visited);

        return rivalCount < myCount;
    }

    private int getTreeNodeCount(Deque<TreeNode> q, boolean[] visited) {
        int count = 0;
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            if (visited[p.val]) continue;
            visited[p.val] = true;
            count++;
            TreeNode f = getFather(p), l = getLeft(p), r = getRight(p);
            if (f != null && !visited[f.val]) q.offer(f);
            if (l != null && !visited[l.val]) q.offer(l);
            if (r != null && !visited[r.val]) q.offer(r);
        }
        return count;
    }

    private TreeNode getFather(TreeNode root) {
        return fatherMap.get(root);
    }

    private TreeNode getLeft(TreeNode root) {
        return root.left;
    }

    private TreeNode getRight(TreeNode root) {
        return root.right;
    }


    // LC1983
    public int widestPairOfIndices(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int prefix1 = 0, prefix2 = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix1 += nums1[i];
            prefix2 += nums2[i];
            int diff = prefix1 - prefix2;
            if (map.containsKey(diff)) {
                result = Math.max(result, i - map.get(diff));
            } else {
                map.put(diff, i);
            }
        }
        return result;
    }

    // LC335 **
    public boolean isSelfCrossing(int[] distance) {
        for (int i = 3; i < distance.length; i++) {
            if (i >= 3
                    && distance[i] >= distance[i - 2]
                    && distance[i - 1] <= distance[i - 3])
                return true;
            else if (i >= 4
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 1] == distance[i - 3])
                return true;
            else if (i >= 5
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 5] + distance[i - 1] >= distance[i - 3]
                    && distance[i - 2] > distance[i - 4]
                    && distance[i - 3] > distance[i - 1])
                return true;
        }
        return false;
    }

    // LC869
    public boolean reorderedPowerOf2(int n) {
        if (n == 0) return false;
        List<Integer> power2List = new ArrayList<>(31);
        for (int i = 0; i < 31; i++) power2List.add(1 << i);
        int[][] freqList = new int[31][10];
        for (int i = 0; i < 31; i++) {
            int power2 = power2List.get(i);
            int[] freq = new int[10];
            while (power2 != 0) {
                freq[power2 % 10]++;
                power2 /= 10;
            }
            freqList[i] = freq;
        }
        int[] thisFreq = new int[10];
        int dummy = n;
        while (dummy != 0) {
            thisFreq[dummy % 10]++;
            dummy /= 10;
        }
        outer:
        for (int i = 0; i < 31; i++) {
            for (int j = 0; j < 10; j++) {
                if (freqList[i][j] != thisFreq[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    // JZOF II 086 LC131
    List<List<String>> lc131Result;
    List<String> lc131Tmp;

    public String[][] partition(String s) {
        lc131Result = new ArrayList<>();
        lc131Tmp = new ArrayList<>();
        int n = s.length();
        boolean[][] judge = new boolean[n][n];
        char[] ca = s.toCharArray();
        for (int i = 0; i < n; i++) judge[i][i] = true;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left + len - 1 < n; left++) {
                if (len == 2) {
                    judge[left][left + 1] = ca[left] == ca[left + 1];
                } else if (judge[left + 1][left + len - 1 - 1] && ca[left] == ca[left + len - 1]) {
                    judge[left][left + len - 1] = true;
                }
            }
        }
        lc131Helper(0, judge, s);
        String[][] resArr = new String[lc131Result.size()][];
        for (int i = 0; i < lc131Result.size(); i++) {
            resArr[i] = lc131Result.get(i).toArray(new String[lc131Result.get(i).size()]);
        }
        return resArr;
    }

    private void lc131Helper(int idx, boolean[][] judge, String s) {
        if (idx == judge.length) {
            lc131Result.add(new ArrayList<>(lc131Tmp));
            return;
        }
        for (int len = 1; idx + len - 1 < judge.length; len++) {
            if (judge[idx][idx + len - 1]) {
                lc131Tmp.add(s.substring(idx, idx + len));
                lc131Helper(idx + len, judge, s);
                lc131Tmp.remove(lc131Tmp.size() - 1);
            }
        }
    }


    // LC792 ** 桶思想
    public int numMatchingSubseqBucket(String s, String[] words) {
        int result = 0;
        Map<Character, List<List<Character>>> bucket = new HashMap<>();
        for (String w : words) {
            bucket.putIfAbsent(w.charAt(0), new LinkedList<>());
            List<Character> bucketItem = new LinkedList<>();
            for (char c : w.toCharArray()) bucketItem.add(c);
            bucket.get(w.charAt(0)).add(bucketItem);
        }
        for (char c : s.toCharArray()) {
            Set<Character> set = new HashSet<>(bucket.keySet());
            for (char key : set) {
                if (c != key) continue;
                List<List<Character>> items = bucket.get(key);
                ListIterator<List<Character>> it = items.listIterator();
                while (it.hasNext()) {
                    List<Character> seq = it.next();
                    it.remove();
                    seq.remove(0);
                    if (seq.size() == 0) result++;
                    else {
                        bucket.putIfAbsent(seq.get(0), new LinkedList<>());
                        if (seq.get(0) == key) {
                            it.add(seq);
                        } else {
                            bucket.get(seq.get(0)).add(seq);
                        }
                    }
                }
                if (bucket.get(key).size() == 0) bucket.remove(key);
            }
        }
        return result;
    }

    // LC1055 **
    public int shortestWay(String source, String target) {
        int tIdx = 0, result = 0;
        char[] cs = source.toCharArray(), ct = target.toCharArray();
        while (tIdx < ct.length) {
            int sIdx = 0;
            int pre = tIdx;
            while (tIdx < ct.length && sIdx < cs.length) {
                if (ct[tIdx] == cs[sIdx]) tIdx++;
                sIdx++;
            }
            if (tIdx == pre) return -1;
            result++;
        }
        return result;
    }


    // LC1689
    public int minPartitions(String n) {
        int max = 0;
        for (char c : n.toCharArray()) {
            max = Math.max(max, c - '0');
        }
        return max;
    }

    // LC1962
    public int minStoneSum(int[] piles, int k) {
        int[] freq = new int[10001];
        for (int i : piles) freq[i]++;
        int sum = 0;
        for (int i = 10000; i >= 0; i--) {
            if (freq[i] == 0) continue;
            if (k > 0) {
                int minusTime = Math.min(k, freq[i]);
                freq[i] -= minusTime;
                freq[i - i / 2] += minusTime;
                k -= minusTime;
            }
            sum += i * freq[i];
        }
        return sum;
    }

    // LC301 **
    Set<String> lc301Result = new HashSet<>();

    public List<String> removeInvalidParentheses(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        // 多余的左右括号个数, 注意右括号多余当且仅当左边左括号不够匹配的时候
        int leftToRemove = 0, rightToRemove = 0;
        for (char c : ca) {
            if (c == '(') leftToRemove++;
            else if (c == ')') {
                if (leftToRemove == 0) rightToRemove++;
                else leftToRemove--;
            }
        }
        lc301Helper(0, ca, leftToRemove, rightToRemove, 0, 0, new StringBuilder());
        return new ArrayList<>(lc301Result);
    }

    private void lc301Helper(int curIdx, char[] ca,
            /*待删的左括号数*/
                             int leftToRemove, int rightToRemove,
            /*已删的左括号数*/
                             int leftCount, int rightCount,
                             StringBuilder sb) {
        if (curIdx == ca.length) {
            if (leftToRemove == 0 && rightToRemove == 0) {
                lc301Result.add(sb.toString());
            }
            return;
        }

        char c = ca[curIdx];
        if (c == '(' && leftToRemove > 0) { // 无视当前左括号
            lc301Helper(curIdx + 1, ca, leftToRemove - 1, rightToRemove, leftCount, rightCount, sb);
        }
        if (c == ')' && rightToRemove > 0) { // 无视当前右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove - 1, leftCount, rightCount, sb);
        }

        sb.append(c);
        if (c != '(' && c != ')') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount, sb);
        } else if (c == '(') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount + 1, rightCount, sb);
        } else if (c == ')' && rightCount < leftCount) { // 只有当当前已选择的左括号比右括号多才在此步选右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount + 1, sb);
        }
        sb.deleteCharAt(sb.length() - 1);
    }

    // LC1957
    public String makeFancyString(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        int n = ca.length;
        for (int i = 0; i < n; ) {
            int curIdx = i;
            char cur = ca[i];
            while (i + 1 < n && ca[i + 1] == cur) i++;
            int count = Math.min(i - curIdx + 1, 2);
            for (int j = 0; j < count; j++) {
                sb.append(cur);
            }
            i++;
        }
        return sb.toString();
    }

    // LC1540
    public boolean canConvertString(String s, String t, int k) {
        if (s.length() != t.length()) return false;
        // 第i次操作(从1算) 可以将s种之前未被操作过的下标j(从1算)的char+i
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        List<Integer> shouldChangeIdx = new ArrayList<>();
        for (int i = 0; i < cs.length; i++) {
            if (cs[i] != ct[i]) shouldChangeIdx.add(i);
        }
        int[] minSteps = new int[shouldChangeIdx.size()];
        for (int i = 0; i < shouldChangeIdx.size(); i++) {
            char sc = cs[shouldChangeIdx.get(i)], tc = ct[shouldChangeIdx.get(i)];
            minSteps[i] = (tc - 'a' + 26 - (sc - 'a')) % 26;
        }
        int[] freq = new int[27];
        for (int i : minSteps) freq[i]++;
        int max = 0;
        for (int i = 1; i <= 26; i++) {
            max = Math.max(max, i + (freq[i] - 1) * 26);
        }
        return max <= k;
    }

    // LC266
    public boolean canPermutePalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int oddCount = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddCount++;
        return oddCount <= 1;
    }

    // LC409
    public int longestPalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int even = 0, oddMax = 0, odd = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 0) even += freq[i];
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddMax = Math.max(oddMax, freq[i]);
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) odd += freq[i] - 1;
        if (oddMax == 0) return even;
        return odd + even + 1;
    }

    // LC1266
    public int minTimeToVisitAllPoints(int[][] points) {
        int x = points[0][0], y = points[0][1];
        int result = 0;
        for (int i = 1; i < points.length; i++) {
            int nx = points[i][0], ny = points[i][1];
            int deltaX = Math.abs(nx - x), deltaY = Math.abs(ny - y);
            int slash = Math.min(deltaX, deltaY);
            int line = Math.max(deltaX, deltaY) - slash;
            result += line + slash;
            x = nx;
            y = ny;
        }
        return result;
    }

    // LC1416
    Integer[] lc1416Memo;

    public int numberOfArrays(String s, int k) {
        int n = s.length();
        lc1416Memo = new Integer[n + 1];
        return lc1416Helper(0, s, k);
    }

    private int lc1416Helper(int cur, String s, int k) {
        final long mod = 1000000007l;
        if (cur == s.length()) return 1;
        if (lc1416Memo[cur] != null) return lc1416Memo[cur];
        int len = 1;
        long result = 0;
        while (cur + len <= s.length()) {
            long num = Long.parseLong(s.substring(cur, cur + len));
            if (String.valueOf(num).length() != len) break;
            if (num > k) break;
            if (num < 1) break;
            result += lc1416Helper(cur + len, s, k);
            result %= mod;
            len++;
        }
        return lc1416Memo[cur] = (int) (result % mod);
    }

    // LC1844
    public String replaceDigits(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            if (i % 2 == 0) sb.append(ca[i]);
            else sb.append((char) (ca[i - 1] + (ca[i] - '0')));
        }
        return sb.toString();
    }

    //
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

class DSUArray {
    int[] father;
    int[] rank;
    int size;

    public DSUArray(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray() {
        this.size = Integer.MAX_VALUE >> 16;
        father = new int[Integer.MAX_VALUE >> 16];
        rank = new int[Integer.MAX_VALUE >> 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public boolean contains(int i) {
        if (i >= this.size || i < 0) return false;
        return father[i] != -1;
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            rank[root]++;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }


}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

