import java.beans.Customizer;
import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.minCost(7,
                new int[]{1, 3, 4, 5}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1547 ** 另一种切绳子 注意边界处理
    Integer[][] memo = new Integer[105][105];
    int[] cuts;
    int ropeLen;

    public int minCost(int n, int[] cuts) {
        this.ropeLen = n;
        Arrays.sort(cuts);
        this.cuts = cuts;
        int result = Integer.MAX_VALUE / 2;
        for (int i = 0; i < cuts.length; i++) {
            result = Math.min(result, helper(0, i - 1) + helper(i + 1, cuts.length - 1));
        }
        result += n;
        return result;
    }

    private int helper(int start, int end) {
        if (start > end) return 0;
        if (memo[start][end] != null) return memo[start][end];
        int result = Integer.MAX_VALUE / 2;
        for (int i = start; i <= end; i++) {
            result = Math.min(result, helper(start, i - 1) + helper(i + 1, end));
        }
        result += (end == cuts.length - 1 ? ropeLen : cuts[end + 1]) - (start == 0 ? 0 : cuts[start - 1]);
        return memo[start][end] = result;
    }

    // LC711
    public int numDistinctIslands2(int[][] grid) {
        int result = 0;
        Map<Integer, Set<String>> countHashMap = new HashMap<>();
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                List<Integer> area = new ArrayList<>();
                if (!visited[i][j]) {
                    Deque<Integer> q = new LinkedList<>();
                    q.offer(i * 50 + j);
                    while (!q.isEmpty()) {
                        int p = q.pop();
                        int r = p / 50, c = p % 50;
                        if (visited[r][c]) continue;
                        visited[r][c] = true;
                        area.add(p);
                        for (int[] d : directions) {
                            int nr = r + d[0], nc = c + d[1];
                            if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc] && grid[nr][nc] == 1) {
                                q.push(nr * 50 + nc);
                            }
                        }
                    }
                    countHashMap.putIfAbsent(area.size(), new HashSet<>());
                    Set<String> allHash = getHash(area);
                    boolean count = true;
                    for (String s : allHash) {
                        if (countHashMap.get(area.size()).contains(s)) {
                            count = false;
                            break;
                        }
                    }
                    if (count) {
                        countHashMap.get(area.size()).addAll(allHash);
                        result++;
                    }
                }
            }
        }
        return result;
    }

    private Set<String> getHash(List<Integer> area) {
        Set<String> result = new HashSet<>(4);
        int[][] origArea = new int[50][50];
        Collections.sort(area);
        // 取得极左点, 极右点坐标, 归一化到左上角
        int leftMost = 50, upMost = 50;
        for (int p : area) {
            int r = p / 50, c = p % 50;
            origArea[r][c] = 1;
        }

        // 然后旋转四次取哈希
        for (int i = 0; i < 4; i++) {
            rotate(origArea);
            int[][] normalized = normalize(origArea);
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 50; k++) {
                    if (normalized[j][k] == 0) continue;
                    l.add(j * 50 + k);
                }
            }
            result.add(String.join(",", l.stream().map(String::valueOf).collect(Collectors.toList())));
        }

        {
            // 左右翻转
            int[][] leftRight = foldLeft(origArea);
            int[][] normalized = normalize(leftRight);
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 50; k++) {
                    if (normalized[j][k] == 0) continue;
                    l.add(j * 50 + k);
                }
            }
            result.add(String.join(",", l.stream().map(String::valueOf).collect(Collectors.toList())));
        }

        {
            // 上下
            int[][] leftRight = foldUp(origArea);
            int[][] normalized = normalize(leftRight);
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 50; k++) {
                    if (normalized[j][k] == 0) continue;
                    l.add(j * 50 + k);
                }
            }
            result.add(String.join(",", l.stream().map(String::valueOf).collect(Collectors.toList())));
        }

        return result;
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) { // 先上下 再斜对角线(左上, 右下)
            for (int j = 0; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = tmp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }

    public int[][] foldLeft(int[][] matrix) {
        int n = matrix.length;
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) result[i][j] = matrix[i][j];
        for (int i = 0; i < n / 2; i++) { // 先上下 再斜对角线(左上, 右下)
            for (int j = 0; j < n; j++) {
                int tmp = result[i][j];
                result[i][j] = result[n - 1 - i][j];
                result[n - 1 - i][j] = tmp;
            }
        }
        return result;
    }

    public int[][] foldUp(int[][] matrix) {
        int n = matrix.length;
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) result[i][j] = matrix[i][j];
        for (int i = 0; i < n; i++) { // 先上下 再斜对角线(左上, 右下)
            for (int j = 0; j < n / 2; j++) {
                int tmp = result[i][j];
                result[i][j] = result[i][n - 1 - j];
                result[i][n - 1 - j] = tmp;
            }
        }
        return result;
    }

    private int[][] normalize(int[][] mtx) {
        int m = mtx.length;
        int left = m, up = m;
        int[][] result = new int[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (mtx[i][j] == 0) continue;
                left = Math.min(left, j);
                up = Math.min(up, i);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (mtx[i][j] == 0) continue;
                result[i - up][j - left] = 1;
            }
        }
        return result;
    }


    // LC980
    class Lc980 {
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int[][] grid;
        int result = 0;
        int start, end;

        public int uniquePathsIII(int[][] grid) {
            this.grid = grid;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == 1) {
                        start = i * grid[0].length + j;
                    }
                    if (grid[i][j] == 2) {
                        end = i * grid[0].length + j;
                    }
                }
            }
            backtrack(start / grid[0].length, start % grid[0].length);
            return result;
        }

        private void backtrack(int r, int c) {
            if (r == end / grid[0].length && c == end % grid[0].length) {
                int zeroCount = 0;
                for (int i = 0; i < grid.length; i++) {
                    for (int j = 0; j < grid[0].length; j++) {
                        if (grid[i][j] == 0) zeroCount++;
                        if (zeroCount > 0) return;
                    }
                }
                result++;
                return;
            }
            grid[r][c] = -1;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && grid[nr][nc] != -1) {
                    backtrack(nr, nc);
                }
            }
            grid[r][c] = 0;
        }
    }

    // LC423
    // characteristic letter:
    // x: six
    // w: two
    // z: zero
    // g: eight
    // u: four
    // -----------
    // s: seven
    // v: five
    // o: one
    // i: nine
    // t: three
    public String originalDigits(String s) {
        String[] digitString = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        int[][] seq = {{'z', 0}, {'w', 2}, {'u', 4}, {'x', 6}, {'g', 8}, {'o', 1}, {'t', 3}, {'s', 7}, {'v', 5}, {'i', 9}};
        char[][] digitBet = new char[10][];
        for (int i = 0; i < 10; i++) digitBet[i] = digitString[i].toCharArray();
        int[] freq = new int[128], number = new int[10];
        for (char c : s.toCharArray()) freq[c]++;
        for (int[] pair : seq) {
            int f = freq[pair[0]];
            for (char j : digitBet[pair[1]]) freq[j] -= f;
            number[pair[1]] += f;
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < number[i]; j++) {
                sb.append(i);
            }
        }
        return sb.toString();
    }

    // Interview 17.13
    class Interview17_13 {
        Integer[] memo = new Integer[1001];
        Set<String> set = new HashSet<>();
        String sentence;

        public int respace(String[] dictionary, String sentence) {
            this.sentence = sentence;
            for (String w : dictionary) set.add(new StringBuilder(w).reverse().toString());
            return helper(sentence.length());
        }

        private int helper(int idx) {
            if (idx <= 0) return 0;
            if (memo[idx] != null) return memo[idx];
            StringBuilder sb = new StringBuilder();
            int result = idx;
            for (int i = idx - 1; i >= 0; i--) {
                sb.append(sentence.charAt(i));
                if (set.contains(sb.toString())) {
                    result = Math.min(result, helper(i));
                } else {
                    result = Math.min(result, idx - i + helper(i));
                }
            }
            return memo[idx] = result;
        }
    }

    // LC1413
    public int minStartValue(int[] nums) {
        int result = 0, prefix = 0;
        for (int i = 0; i < nums.length; i++) {
            prefix += nums[i];
            if (prefix < 1) {
                result += 1 - prefix;
                prefix += 1 - prefix;
            }
        }
        return Math.max(1, result);
    }

    // LC2079
    public int wateringPlants(int[] plants, int capacity) {
        int result = 0, cur = capacity;
        for (int i = 0; i < plants.length; i++) {
            if (cur < plants[i]) {
                result += 2 * i + 1;
                cur = capacity - plants[i];
            } else {
                result += 1;
                cur -= plants[i];
            }
        }
        return result;
    }

    // LC396 推公式
    public int maxRotateFunction(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        int init = 0, n = nums.length;
        for (int i = 0; i < n; i++) init += i * nums[i];
        int max = init;
        int cur = init;
        for (int i = 1; i < n; i++) {
            int next = cur + sum - n * nums[n - i];
            max = Math.max(next, max);
            cur = next;
        }
        return max;
    }

    // LC1728 花式TLE 判题有问题???
    class Lc1728 {
        final int TIE = 0, MOUSE_WIN = 1, CAT_WIN = 2;
        char[][] mtx;
        int catJump, mouseJump, foodR, foodC;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Integer[][][][][] memo = new Integer[8][8][8][8][505];

        public boolean canMouseWin(String[] grid, int catJump, int mouseJump) {
            int m = grid.length, n = grid[0].length();
            this.catJump = catJump;
            this.mouseJump = mouseJump;
            mtx = new char[m][];
            for (int i = 0; i < m; i++) mtx[i] = grid[i].toCharArray();
            int[] foodIdx = {-1, -1}, catIdx = {-1, -1}, mouseIdx = {-1, -1};
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (mtx[i][j] == 'C') catIdx = new int[]{i, j};
                    if (mtx[i][j] == 'M') mouseIdx = new int[]{i, j};
                    if (mtx[i][j] == 'F') foodIdx = new int[]{i, j};
                }
            }
            foodR = foodIdx[0];
            foodC = foodIdx[1];
            return helper(mouseIdx[0], mouseIdx[1], catIdx[0], catIdx[1], 0) == MOUSE_WIN;
        }

        private int helper(int mouseR, int mouseC, int catR, int catC, int globalSteps) {
            boolean isMouse = globalSteps % 2 == 0;
            if (isMouse && globalSteps >= 500) {
                return TIE;
            }
            if (catR == foodR && catC == foodC) {
                return CAT_WIN;
            }
            if (mouseR == catR && mouseC == catC) {
                return CAT_WIN;
            }
            if (mouseR == foodR && mouseC == foodC) {
                return MOUSE_WIN;
            }
            if (memo[mouseR][mouseC][catR][catC][globalSteps] != null)
                return memo[mouseR][mouseC][catR][catC][globalSteps];

            if (isMouse) {
                boolean catCanWin = true;
                for (int[] d : directions) {
                    int nr = mouseR - d[0], nc = mouseC - d[1];
                    int steps = -1;
                    while (nr + d[0] >= 0 && nr + d[0] < mtx.length && nc + d[1] >= 0 && nc + d[1] < mtx[0].length
                            && mtx[nr + d[0]][nc + d[1]] != '#'
                            && steps + 1 <= mouseJump) {
                        nr += d[0];
                        nc += d[1];
                        steps++;
                        int nextStepResult = helper(nr, nc, catR, catC, globalSteps + 1);
                        if (nextStepResult == MOUSE_WIN) {
                            return memo[mouseR][mouseC][catR][catC][globalSteps] = MOUSE_WIN;
                        } else if (nextStepResult != CAT_WIN) {
                            catCanWin = false;
                        }
                    }
                }
                if (catCanWin) memo[mouseR][mouseC][catR][catC][globalSteps] = CAT_WIN;
                return memo[mouseR][mouseC][catR][catC][globalSteps] = TIE;
            } else {
                boolean mouseCanWin = true;
                for (int[] d : directions) {
                    int nr = catR - d[0], nc = catC - d[1];
                    int steps = -1;
                    while (nr + d[0] >= 0 && nr + d[0] < mtx.length && nc + d[1] >= 0 && nc + d[1] < mtx[0].length
                            && mtx[nr + d[0]][nc + d[1]] != '#'
                            && steps + 1 <= catJump) {
                        nr += d[0];
                        nc += d[1];
                        steps++;
                        int nextStepResult = helper(mouseR, mouseC, nr, nc, globalSteps + 1);
                        if (nextStepResult == CAT_WIN) {
                            return memo[mouseR][mouseC][catR][catC][globalSteps] = CAT_WIN;
                        } else if (nextStepResult != MOUSE_WIN) {
                            mouseCanWin = false;
                        }
                    }
                }
                if (mouseCanWin) return memo[mouseR][mouseC][catR][catC][globalSteps] = MOUSE_WIN;
                return memo[mouseR][mouseC][catR][catC][globalSteps] = TIE;
            }
        }
    }

    // LC156
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        // 原来的左子节点变成新的根节点
        // 原来的根节点变成新的右子节点
        // 原来的右子节点变成新的左子节点
        // 题目保证如果存在右节点就存在左节点
        // 逆否: 如果不存在左节点, 就不存在右节点, 即如果左节点为空, 则一定是叶子
        if (root == null) return null;
        if (root.left == null) return root; // 叶子
        TreeNode origLeft = root.left, origRight = root.right;
        TreeNode newRoot = upsideDownBinaryTree(root.left);
        upsideDownBinaryTree(root.right);
        root.left = null;
        root.right = null;
        origLeft.right = root;
        origLeft.left = origRight;
        return newRoot;
    }

    // LC915
    public int partitionDisjoint(int[] nums) {
        // 左侧的最大值小于右侧的最小值
        int n = nums.length;
        int[] maxLeft = new int[n], minRight = new int[n];
        maxLeft[0] = nums[0];
        for (int i = 1; i < n; i++) maxLeft[i] = Math.max(nums[i], maxLeft[i - 1]);
        minRight[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) minRight[i] = Math.min(nums[i], minRight[i + 1]);
        for (int i = 0; i <= n - 2; i++) {
            if (maxLeft[i] <= minRight[i + 1]) return i + 1;
        }
        return -1;
    }

    // LC2065
    int lc2065Result = 0;

    // 限时在图中行走并回到出发城市, 每个城市可以经过不止一次, 路径耗时, 城市有值, 值只取一次, 选值最高的路径
    public int maximalPathQuality(int[] values, int[][] edges, int maxTime) {
        int n = values.length;
        // 构造邻接表
        List<List<int[]>> mtx = new ArrayList<>();
        int[] freq = new int[n];
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int[] e : edges) {
            mtx.get(e[0]).add(new int[]{e[1], e[2]});
            mtx.get(e[1]).add(new int[]{e[0], e[2]});
        }
        if (mtx.get(0).size() == 0) return values[0];

        // 先用Dijkstra预处理各个城市到 0 的最短时间, 方便剪枝
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1])); //[城市, 距离]
        int[] shortest = new int[n];
        Arrays.fill(shortest, -1);
        pq.offer(new int[]{0, 0});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int idx = p[0], distance = p[1];
            if (shortest[idx] != -1) continue;
            shortest[idx] = distance;
            for (int[] next : mtx.get(idx)) {
                if (shortest[next[0]] != -1) continue;
                pq.offer(new int[]{next[0], distance + next[1]});
            }
        }

        freq[0] = 1;
        lc2065Helper(0, values[0], maxTime, freq, mtx, shortest, values);
        return lc2065Result;
    }

    private void lc2065Helper(int cur, int gain, int remainTime, int[] freq, List<List<int[]>> mtx, int[] shortest, int[] values) {
        if (cur == 0) {
            lc2065Result = Math.max(lc2065Result, gain);
        }
        if (freq[cur] == 0) gain += values[cur];
        freq[cur]++;
        for (int[] next : mtx.get(cur)) {
            if (remainTime < next[1]) continue;
            if (remainTime - next[1] < shortest[next[0]]) continue;
            lc2065Helper(next[0], gain, remainTime - next[1], freq, mtx, shortest, values);
        }
        freq[cur]--; // 记得复位
    }

    // LC1474
    public ListNode deleteNodes(ListNode head, int m, int n) {
        if (head == null) return null;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = dummy;
        int idx = 0;
        while (idx < m && cur != null) {
            cur = cur.next;
            idx++;
        }
        if (cur == null) return dummy.next;
        ListNode partOneLast = cur;
        idx = 0;
        while (idx < n && cur != null) {
            cur = cur.next;
            idx++;
        }
        if (cur == null) {
            partOneLast.next = null;
            return dummy.next;
        }
        partOneLast.next = deleteNodes(cur.next, m, n);
        return dummy.next;
    }

    // LCP29 **
    public int orchestraLayout(int num, int xPos, int yPos) {
        // 找[x,y]是第几个位置 (mod9意义下)
        // 找第几圈
        long round = Math.min(Math.min(xPos, num - 1 - xPos), Math.min(yPos, num - 1 - yPos));
        long seq = 4 * (num - round) * round;
        long sideLen = num - 2 * round - 1, xInit = round, yInit = round;

        if (xPos == xInit && yPos < yInit + sideLen) {
            seq += ((long) yPos + 1 - yInit);
        } else {
            seq += sideLen;
            if (yPos == yInit + sideLen && xPos < xInit + sideLen) {
                seq += ((long) xPos + 1 - xInit);
            } else {
                seq += sideLen;
                if (xPos == xInit + sideLen && yPos > yInit) {
                    seq += (yInit + sideLen + 1 - yPos);
                } else {
                    seq += (sideLen + xInit + sideLen + 1 - xPos);
                }
            }
        }
        seq %= 9;
        return (int) (seq == 0 ? 9 : seq);
    }


    // LC1852
    public int[] distinctNumbers(int[] nums, int k) {
        int[] result = new int[nums.length - k + 1];
        int[] freq = new int[100001];
        int typeCount = 0;
        for (int i = 0; i < k; i++) {
            if (freq[nums[i]] == 0) typeCount++;
            freq[nums[i]]++;
        }
        result[0] = typeCount;
        for (int i = k; i < nums.length; i++) {
            if (freq[nums[i - k]] == 1) typeCount--;
            freq[nums[i - k]]--;
            if (freq[nums[i]] == 0) typeCount++;
            freq[nums[i]]++;
            result[i - k + 1] = typeCount;
        }
        return result;
    }

    // LC1183 **
    public int maximumNumberOfOnes(int width, int height, int sideLength, int maxOnes) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(); // pq存的是 如果 左上角sideLen x sideLen 区域的某一个点放'1'后, 其在整个矩阵的相应不冲突位置可以放1的总个数
        // ** 遇到小的就出队
        for (int i = 0; i < sideLength; i++) {
            for (int j = 0; j < sideLength; j++) {
                // 考虑11 x 11 的矩阵, 小正方形大小是5 x 5, 里面最多放1个'1',
                // 如果[0,0]的位置放一个, 则[0,5] [0,10] 可以放, 且row 0放满
                // 如果[0,1]放一个, [0,6]再放一个就放满
                // 可放列的个数为 (width - i - 1) / sideLen + 1
                // 然后乘上可放的行数(个数计算方法一样)
                pq.offer((((width - i - 1) / sideLength) + 1) * (((height - j - 1) / sideLength) + 1));
                while (pq.size() > maxOnes) pq.poll();
            }
        }
        int result = 0;
        while (!pq.isEmpty()) result += pq.poll();
        return result;
    }

    // LC1130 ** 区间DP
    Integer[][] lc1130Memo = new Integer[41][41];

    public int mctFromLeafValues(int[] arr) {
        return lc1130Helper(0, arr.length - 1, arr);
    }

    private int lc1130Helper(int start, int end, int[] arr) {
        if (start == end) return 0;
        if (lc1130Memo[start][end] != null) return lc1130Memo[start][end];
        int result = Integer.MAX_VALUE;
        for (int i = start; i < end; i++) {
            // first max
            int firstMax = arr[start];
            for (int j = start; j <= i; j++) {
                firstMax = Math.max(arr[j], firstMax);
            }
            int secondMax = arr[i + 1];
            for (int j = i + 1; j <= end; j++) {
                secondMax = Math.max(arr[j], secondMax);
            }
            result = Math.min(result, firstMax * secondMax + lc1130Helper(start, i, arr) + lc1130Helper(i + 1, end, arr));
        }
        return lc1130Memo[start][end] = result;
    }

    // LC1349
    Integer[][] lc1349Memo = new Integer[9][1 << 8];

    public int maxStudents(char[][] seats) {
        int m = seats.length, n = seats[0].length;
        int[] availMask = new int[m];
        for (int i = 0; i < m; i++) {
            int tmpMask = 0;
            for (int j = 0; j < n; j++) {
                if (seats[i][j] == '.') {
                    tmpMask |= (1 << j);
                }
            }
            availMask[i] = tmpMask;
        }
        return lc1349Helper(0, 0, availMask, m, n);
    }


    // 当前行i能填的最大学生mask 只和当前行可填的mask(availMask[i]) 与 前一行 prevRowMask有关
    private int lc1349Helper(int curRow, int prevRowMask, int[] availMask, int m, int n) {
        if (curRow == m) return 0;
        if (lc1349Memo[curRow][prevRowMask] != null) return lc1349Memo[curRow][prevRowMask];
        int fullSet = availMask[curRow];
        int result = -1;
        // 构造当前行可坐学生位置的所有非空子集
        outer:
        for (int subset = fullSet; subset > 0; subset = (subset - 1) & fullSet) {
            for (int i = 0; i < n; i++) {
                if (((subset >> i) & 1) == 0) continue; // 这个位置不能坐人 跳过
                // 校验左, 右, 左前方, 右前方
                // 左、左前方
                if (i > 0) {
                    int left = i - 1;
                    if (((subset >> left) & 1) == 1) continue outer;
                    if (((prevRowMask >> left) & 1) == 1) continue outer;
                }
                // 右, 右前方
                if (i < n - 1) {
                    int right = i + 1;
                    if (((subset >> right) & 1) == 1) continue outer;
                    if (((prevRowMask >> right) & 1) == 1) continue outer;
                }
            }
            // 校验没问题了, 进入下一行
            result = Math.max(result, Integer.bitCount(subset) + lc1349Helper(curRow + 1, subset, availMask, m, n));
        }
        // 这一行不坐人(空集)
        result = Math.max(result, lc1349Helper(curRow + 1, 0, availMask, m, n));
        return lc1349Memo[curRow][prevRowMask] = result;
    }

    // LC1874
    public int minProductSum(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int result = 0, n = nums1.length;
        for (int i = 0; i < n; i++) {
            result += nums1[i] * nums2[n - 1 - i];
        }
        return result;
    }

    // LC594
    public int findLHS(int[] nums) {
        Arrays.sort(nums);
        int prevLen = 0, prev = -1, result = 0, idx = 0, n = nums.length;
        boolean init = false;
        while (idx < n) {
            int cur = nums[idx];
            int left = idx;
            while (idx + 1 < n && nums[idx + 1] == nums[idx]) idx++;
            int right = idx;
            int len = right - left + 1;
            if (init && cur - prev == 1) result = Math.max(result, len + prevLen);
            init = true;
            prevLen = len;
            prev = cur;
            idx++;
        }
        return result;
    }

    // LC2033
    public int minOperations(int[][] grid, int x) {
        int mod = grid[0][0] % x, m = grid.length, n = grid[0].length, max = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] % x != mod) return -1;
                max = Math.max(grid[i][j], max);
            }
        }
        int[] freq = new int[max + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                freq[grid[i][j]]++;
            }
        }
        int result = Integer.MAX_VALUE / 2;
        for (int i = 1; i <= max; i++) {
            if (freq[i] != 0) {
                int tmp = 0;
                // 以 i 为基准
                // 左边要加多少次
                for (int j = 1; j < i; j++) {
                    tmp += freq[j] * ((i - j) / x);
                }
                for (int j = i + 1; j <= max; j++) {
                    tmp += freq[j] * ((j - i) / x);
                }
                result = Math.min(result, tmp);
            }
        }
        return result;
    }

    // LC1537
    Long[][] lc1537Memo;
    int[][] lc1537Nums;
    Map<Integer, Integer>[] lc1537Rm;

    public int maxSum(int[] nums0, int[] nums1) {
        lc1537Rm = new Map[2];
        lc1537Rm[0] = new HashMap<>();
        lc1537Rm[1] = new HashMap<>();
        for (int i = 0; i < nums0.length; i++) lc1537Rm[0].put(nums0[i], i);
        for (int i = 0; i < nums1.length; i++) lc1537Rm[1].put(nums1[i], i);
        lc1537Nums = new int[][]{nums0, nums1};
        lc1537Memo = new Long[2][Math.max(nums0.length, nums1.length) + 1];
        return (int) (Math.max(lc1537Helper(0, 0), lc1537Helper(1, 0)) % 1000000007l);
    }

    private long lc1537Helper(int whichArr, int curIdx) {
        int[] arr = lc1537Nums[whichArr];
        if (curIdx == arr.length) return 0;
        if (lc1537Memo[whichArr][curIdx] != null) return lc1537Memo[whichArr][curIdx];
        long result = Integer.MIN_VALUE / 2;
        result = Math.max(result, (long) arr[curIdx] + lc1537Helper(whichArr, curIdx + 1));
        if (lc1537Rm[1 - whichArr].containsKey(arr[curIdx])) {
            result = Math.max(result, (long) arr[curIdx] + lc1537Helper(1 - whichArr, lc1537Rm[1 - whichArr].get(arr[curIdx]) + 1));
        }
        return lc1537Memo[whichArr][curIdx] = result;
    }

    // LC1541 **
    public int minInsertions(String s) {
        // 平衡条件:
        //   一个'('配两个')'
        //   '(' 配的两个')' 必须在对应的'(' 后面
        // 插入任意'(',')' 使得s平衡
        // 返回最少插入次数
        int left = 0, result = 0, idx = 0, n = s.length();
        while (idx < n) {
            if (s.charAt(idx) == '(') {
                left++;
                idx++;
            } else if (s.charAt(idx) == ')') {
                if (left > 0) {
                    left--;
                } else {
                    result++;
                }
                if (idx + 1 < n && s.charAt(idx + 1) == ')') {
                    idx += 2;
                } else {
                    result++;
                    idx++;
                }
            }
        }
        result += left * 2;
        return result;
    }

    // LC366
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        while (!(root.left == null && root.right == null)) {
            List<Integer> tmp = new ArrayList<>();
            lc366Helper(root, tmp);
            result.add(tmp);
        }
        result.add(Arrays.asList(root.val));
        return result;
    }

    private void lc366Helper(TreeNode node, List<Integer> tmp) {
        if (node == null) return;
        if (node.left != null) {
            if (node.left.left == null && node.left.right == null) {
                tmp.add(node.left.val);
                node.left = null;
            } else {
                lc366Helper(node.left, tmp);
            }
        }
        if (node.right != null) {
            if (node.right.left == null && node.right.right == null) {
                tmp.add(node.right.val);
                node.right = null;
            } else {
                lc366Helper(node.right, tmp);
            }
        }
    }

    // LC796
    public boolean rotateString(String s, String goal) {
        if (s.length() != goal.length()) return false;
        if (s.equals(goal)) return true;
        for (int i = 1; i < s.length(); i++) {
            if ((s.substring(i) + s.substring(0, i)).equals(goal)) return true;
        }
        return false;
    }

    // LC1945
    public int getLucky(String s, int k) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            sb.append((int) (1 + c - 'a'));
        }
        String digitStr = sb.toString();
        int sum = 0;
        for (int i = 0; i < k; i++) {
            sum = 0;
            for (char c : digitStr.toCharArray()) {
                sum += (c - '0');
            }
            digitStr = String.valueOf(sum);
        }
        return sum;
    }

    // LC964 **
    Map<Integer, Integer> lc964Memo;

    public int leastOpsExpressTarget(int x, int target) {
        if (x == target) return 0;
        lc964Memo = new HashMap<>();
        return lc964Helper(target, 1, x);
    }

    private int lc964Helper(int target, int power, int x) {
        if (target < x) {
            return Math.min(2 * target - 1, 2 * (x - target));
            // t:2, x:3 :   3/3 + 3/3   or  3 - 3/3
            // t:1, x:3 :   3/3         or  3 - 3/3 - 3/3
        }
        if (lc964Memo.containsKey(target)) return lc964Memo.get(target);
        int result = Integer.MAX_VALUE / 2;
        long nextSum = (long) Math.pow(x, power + 1);
        if (nextSum == target) {
            lc964Memo.put(target, 1);
            return 1; // 多一个乘号
        }
        if (nextSum < target) {
            result = Math.min(result, 1 + lc964Helper(target, power + 1, x)); // 加这个乘号, 继续往下递归
        } else if (nextSum > target) {
            // 是正着取还是反着取
            // 如target=90, x=10, power=1, nextSum = 100, nextSum>target
            // 正着取: next target = 90 - 10 = 80, power reset to 1
            // 即 + 10,加一个加号, 然后剩下的80交给递归
            result = Math.min(result, 1 + lc964Helper(target - (int) Math.pow(x, power), 1, x)); // 加一个加号

            // next target: 100 - 90 = 10, 变成 + 100 - 10 , power reset to 1
            // 即 ... * 10 - (...), 后面括号部分交给递归
            if (nextSum - target < target) {
                // 反着取: 100 - 90 < 90, 为什么这样判断? 防止爆栈的依据是?
                // 考虑 x=10, power = 1, target = 40, 100 - 40 > 40, 这时候反着取需要100-6*10, 即 * 10 - 10 - 10..., 共消耗7个符号
                // 而正着取显然只消耗4个符号。 正着取总是可行的, 然而反着取可能会使递归规模无限扩大
                // 所以这里先行判断, 避免爆栈
                // 又比如 x=10, power=1, target = 50,此时正取+10...+10 共消耗5个符号, 反取 *10 - 10 -10 -10... 共消耗 6个符号
                // 所以边界使 nextSum - target < target, 取不到等号
                result = Math.min(result, 2 + lc964Helper((int) (nextSum - target), 1, x)); // 加一个乘号, 一个减号
            }
        }
        lc964Memo.put(target, result);
        return result;
    }

    // LC1612
    class Lc1612 {
        public boolean checkEquivalence(Node root1, Node root2) {
            int[] freq1 = new int[26], freq2 = new int[26];
            eval(root1, freq1);
            eval(root2, freq2);
            for (int i = 0; i < 26; i++) {
                if (freq1[i] != freq2[i]) return false;
            }
            return true;
        }

        private void eval(Node root, int[] freq) {
            if (root == null) return;
            if (root.val == '+') {
                eval(root.left, freq);
                eval(root.right, freq);
                return;
            }
            freq[root.val - 'a']++;
        }

        class Node {
            char val;
            Node left;
            Node right;

            Node() {
                this.val = ' ';
            }

            Node(char val) {
                this.val = val;
            }

            Node(char val, Node left, Node right) {
                this.val = val;
                this.left = left;
                this.right = right;
            }
        }
    }

    // LC1171
    ListNode victim = null;

    public ListNode removeZeroSumSublists(ListNode head) {
        victim = head;
        while (handle(victim)) {
            ;
        }
        return victim;
    }

    private boolean handle(ListNode head) {
        ListNode dummy = new ListNode(0);
        Map<Integer, ListNode> prefixSumNodeMap = new HashMap<>();
        dummy.next = head;
        ListNode it = dummy;
        int sum = 0;
        while (it != null) {
            sum += it.val;
            if (prefixSumNodeMap.containsKey(sum)) {
                ListNode prev = prefixSumNodeMap.get(sum);
                prev.next = it.next;
                victim = dummy.next;
                return true;
            }
            prefixSumNodeMap.put(sum, it);
            it = it.next;
        }
        victim = dummy.next;
        return false;
    }

    // LC397
    public int integerReplacement(int n) {
        if (n == Integer.MAX_VALUE) return 32; // 下面奇数那一步会溢出, 为了不升精度, 干脆特判
        if (n == 0) return Integer.MAX_VALUE / 2;
        if (n == 1) return 0;
        if (n % 2 == 1) return 1 + Math.min(integerReplacement(n + 1), integerReplacement(n - 1));
        return 1 + integerReplacement(n / 2);
    }

    // LC1313
    public int[] decompressRLElist(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        int limit = n / 2;
        for (int i = 0; i < limit; i++) {
            int freq = nums[2 * i], val = nums[2 * i + 1];
            for (int j = 0; j < freq; j++) {
                result.add(val);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC1971
    public boolean validPath(int n, int[][] edges, int start, int end) {
        List<List<Integer>> mtx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int[] e : edges) {
            mtx.get(e[0]).add(e[1]);
            mtx.get(e[1]).add(e[0]);
        }
        boolean[] visited = new boolean[n];
        Deque<Integer> stack = new LinkedList<>();
        stack.push(start);
        while (!stack.isEmpty()) {
            int p = stack.pop();
            if (visited[p]) continue;
            if (p == end) return true;
            visited[p] = true;
            for (int next : mtx.get(p)) {
                if (!visited[next]) stack.push(next);
            }
        }
        return false;
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
