package moe.nyamori.test.historical;

import java.util.*;
import java.util.stream.IntStream;

class scratch_69 {
    public static void main(String[] args) {
        scratch_69 s = new scratch_69();
        System.err.println(s.totalFruit(new int[]{0, 1, 2, 2}));
    }

    // LC904 Again
    public int totalFruit(int[] fruits) {
        List<Integer> selectedTypes = new ArrayList<>();
        int left = 0, right = 0, n = fruits.length, result = 0;
        while (right < n) {
            int currentType = fruits[right];
            if (selectedTypes.contains(currentType)) {
                right++;
            } else if (selectedTypes.size() == 2) {
                int prevType = fruits[right - 1];
                int i = right - 1;
                while (i >= left && fruits[i] == prevType) {
                    i--;
                }
                left = i + 1;
                selectedTypes.set(0, prevType);
                selectedTypes.set(1, currentType);
            } else if (selectedTypes.size() < 2) {
                right++;
                selectedTypes.add(currentType);
            }
            result = Math.max(result, right - left);
        }
        return result;
    }

    // LC1891
    public int maxLength(int[] ribbons, int k) {
        int maxLen = Arrays.stream(ribbons).max().getAsInt();
        int lo = 0, hi = maxLen;
        int result = 0;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (helper(mid, ribbons) >= k) {
                result = Math.max(result, mid);
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }

    private int helper(int target, int[] ribbons) {
        int result = 0;
        for (int i : ribbons) result += i / target;
        return result;
    }

    // LC856 ** Don't be lazy!
    public int scoreOfParentheses(String s) {
        // () - 1
        // ()() - 1 + 1
        // (()) - 2 * 1
        // ()(()) - 1 + 2 * 1
        char[] ca = s.toCharArray();
        int n = ca.length;
        if (n == 2 && ca[0] == '(' && ca[1] == ')') {
            return 1;
        } else {
            int stack = 0, firstLeftIdx = -1, score = 0;
            for (int i = 0; i < n; i++) {
                if (stack == 0) {
                    if (firstLeftIdx < i) {
                        firstLeftIdx = i;
                    }
                }
                if (ca[i] == '(') {
                    stack++;
                } else {
                    stack--;
                }
                if (stack == 0) {
                    String substr = s.substring(firstLeftIdx, i + 1);
                    if (substr.length() == 2 && substr.charAt(0) == '(' && substr.charAt(1) == ')') {
                        score += 1;
                    } else {
                        score += 2 * scoreOfParentheses(substr.substring(1, substr.length() - 1));
                    }
                }

            }
            return score;
        }
    }

    // Interview 17.19 Hard **
    public int[] missingTwo(int[] nums) {
        int xorSum = 0, n = nums.length + 2;
        for (int i : nums) xorSum ^= i;
        for (int i = 1; i <= n; i++) xorSum ^= i;
        int lsb = xorSum & -xorSum;
        if (xorSum == Integer.MIN_VALUE) lsb = xorSum; // 0x8000000
        int r1 = 0, r2 = 0;
        for (int i : nums) {
            if ((i & lsb) == 0) {
                r1 ^= i;
            } else {
                r2 ^= i;
            }
        }
        for (int i = 1; i <= n; i++) {
            if ((i & lsb) == 0) {
                r1 ^= i;
            } else {
                r2 ^= i;
            }
        }
        return new int[]{r1, r2};
    }

    // LC827 Hard
    public int largestIsland(int[][] grid) {
        int n = grid.length;
        int result = 0;
        DSUArray69 dsu = new DSUArray69(n * n + 1);
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 0) continue;
                int idx = r * n + c;
                dsu.add(idx);
                for (int[] d : dirs) {
                    int nr = r + d[0], nc = c + d[1];
                    int nidx = nr * n + nc;
                    if (nr < 0 || nc < 0 || nc >= n || nr >= n || grid[nr][nc] == 0) {
                        continue;
                    }
                    dsu.add(nidx);
                    dsu.merge(idx, nidx);
                }
            }
        }
        Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
        if (allGroups.size() == 0) return 1;
        result = allGroups.values().stream().map(i -> i.size()).max(Comparator.naturalOrder()).get();
        // Probe

        for (Map.Entry<Integer, Set<Integer>> group : allGroups.entrySet()) {
            int father = group.getKey();
            Set<Integer> members = group.getValue();
            int thisGroupSize = members.size();
            for (int idx : members) {
                int r = idx / n, c = idx % n;
                for (int[] outerD : dirs) {
                    int nr = r + outerD[0], nc = c + outerD[1];
                    int nidx = nr * n + nc;
                    if (nr < 0 || nc < 0 || nc >= n || nr >= n || grid[nr][nc] == 1) {
                        continue;
                    }
                    result = Math.max(result, thisGroupSize + 1);
                    Set<Integer> connectableFathers = new HashSet<>();
                    connectableFathers.add(father);
                    for (int[] innerD : dirs) {
                        int nnr = nr + innerD[0], nnc = nc + innerD[1];
                        int nnidx = nnr * n + nnc;
                        if (nnr < 0 || nnc < 0 || nnc >= n || nnr >= n || grid[nnr][nnc] == 0) {
                            continue;
                        }
                        int besideFather = dsu.find(nnidx);
                        connectableFathers.add(besideFather);
                    }
                    int mergedSize = connectableFathers.stream().map(allGroups::get).map(Set::size).reduce((a, b) -> a + b).get() + 1;
                    result = Math.max(result, mergedSize);
                }
            }
        }
        return result;
    }

    // LC782 ** Hard
    public int movesToChessboard(int[][] board) {
        if (judgeBoard(board)) return 0;
        int m = board.length, n = board[0].length;
        int fullmask = (1 << m) - 1;
        int oneCount = 0, zeroCount = 0;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                oneCount += board[i][j];
                zeroCount += 1 - board[i][j];
            }
        }
        if (Math.abs(oneCount - zeroCount) > 1) return -1;

        // Judge if there's only two kinds of row
        Map<Integer, Integer> rowTypeSet = new HashMap<>();
        Map<Integer, Integer> colTypeSet = new HashMap<>();
        for (int i = 0; i < m; i++) {
            int rowMask = 0, colMask = 0;
            for (int j = 0; j < n; j++) {
                rowMask |= (board[i][j] << j);
                colMask |= (board[j][i] << j);
            }
            rowTypeSet.put(rowMask, rowTypeSet.getOrDefault(rowMask, 0) + 1);
            colTypeSet.put(colMask, colTypeSet.getOrDefault(colMask, 0) + 1);
            if (rowTypeSet.keySet().size() > 2 || colTypeSet.keySet().size() > 2) return -1;
        }
        int rt1 = rowTypeSet.keySet().stream().findFirst().get();
        if (!rowTypeSet.keySet().contains(fullmask ^ rt1)) return -1;
        if (m % 2 == 0 && Integer.bitCount(fullmask) - Integer.bitCount(rt1) != Integer.bitCount(rt1)) return -1;
        if (m % 2 == 1 && Math.abs(Integer.bitCount(fullmask) - Integer.bitCount(rt1) - Integer.bitCount(rt1)) > 1)
            return -1;

        int ct1 = colTypeSet.keySet().stream().findFirst().get();
        if (!colTypeSet.keySet().contains(fullmask ^ ct1)) return -1;
        if (m % 2 == 0 && Integer.bitCount(fullmask) - Integer.bitCount(ct1) != Integer.bitCount(ct1)) return -1;
        if (m % 2 == 1 && Math.abs(Integer.bitCount(fullmask) - Integer.bitCount(ct1) - Integer.bitCount(ct1)) > 1)
            return -1;

        int rowMoves = getMoves(rt1, rowTypeSet.get(rt1), m);
        int colMoves = getMoves(ct1, colTypeSet.get(ct1), m);
        if (rowMoves == -1 || colMoves == -1) return -1;
        return colMoves + rowMoves;
    }

    private boolean judgeBoard(int[][] board) {
        int m = board.length, n = board[0].length;
        final int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[0].length; c++) {
                for (int[] d : dir) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                        if (board[nr][nc] == board[r][c]) return false;
                    }
                }
            }
        }
        return true;
    }

    public int getMoves(int mask, int count, int n) {
        int ones = Integer.bitCount(mask);
        if ((n & 1) == 1) {
            /* 如果 n 为奇数，则每一行中 1 与 0 的数目相差为 1，且满足相邻行交替 */
            if (Math.abs(n - 2 * ones) != 1 || Math.abs(n - 2 * count) != 1) {
                return -1;
            }
            if (ones == (n >> 1)) {
                /* 以 0 为开头的最小交换次数 */
                return n / 2 - Integer.bitCount(mask & 0xAAAAAAAA);
            } else {
                return (n + 1) / 2 - Integer.bitCount(mask & 0x55555555);
            }
        } else {
            /* 如果 n 为偶数，则每一行中 1 与 0 的数目相等，且满足相邻行交替 */
            if (ones != (n >> 1) || count != (n >> 1)) {
                return -1;
            }
            /* 找到行的最小交换次数 */
            int count0 = n / 2 - Integer.bitCount(mask & 0xAAAAAAAA);
            int count1 = n / 2 - Integer.bitCount(mask & 0x55555555);
            return Math.min(count0, count1);
        }
    }


    // LC655
    public List<List<String>> printTree(TreeNode69 root) {
        int layer = 0;
        Deque<TreeNode69> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                TreeNode69 p = q.poll();
                if (p.left != null) {
                    q.offer(p.left);
                }
                if (p.right != null) {
                    q.offer(p.right);
                }
            }
        }
        int colNum = (1 << layer) - 1;
        List<List<String>> result = new ArrayList<>(layer);
        for (int i = 0; i < layer; i++) {
            List<String> thisLayer = new ArrayList<>(colNum);
            for (int j = 0; j < colNum; j++) {
                thisLayer.add("");
            }
            result.add(thisLayer);
        }
        lc655Helper(root, 0, 0, colNum - 1, result);
        return result;
    }

    private void lc655Helper(TreeNode69 node, int layer, int from, int to, List<List<String>> result) {
        if (node == null) return;
        List<String> thisLayer = result.get(layer);
        int idx = (from + to) / 2;
        thisLayer.set(idx, "" + node.val);
        lc655Helper(node.left, layer + 1, from, idx - 1, result);
        lc655Helper(node.right, layer + 1, idx + 1, to, result);
    }

    // LC1455
    public int isPrefixOfWord(String sentence, String searchWord) {
        return IntStream.range(0, sentence.split(" ").length).filter(i -> sentence.split(" ")[i].startsWith(searchWord)).findFirst().orElse(-1);
    }

    // LC1450
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        int n = startTime.length, result = 0;
        for (int i = 0; i < n; i++) {
            int start = startTime[i], end = endTime[i];
            if (start <= queryTime && queryTime <= end) result++;
        }
        return result;
    }

    // LC1422
    public int maxScore(String s) {
        int one = 0;
        char[] ca = s.toCharArray();
        for (char c : ca) {
            one += c - '0';
        }
        int result = 0, zero = 0;
        for (int i = 0; i < ca.length - 1; i++) {
            char c = ca[i];
            int score = 0;
            if (c == '0') {
                zero++;
            } else {
                one--;
            }
            score += zero;
            score += one;
            result = Math.max(result, score);
        }
        return result;
    }

    // LC1417
    public String reformat(String s) {
        char[] ca = s.toCharArray();
        StringBuilder dsb = new StringBuilder(), lsb = new StringBuilder();
        for (char c : ca) {
            if (Character.isDigit(c)) {
                dsb.append(c);
            } else if (Character.isLetter(c)) {
                lsb.append(c);
            }
        }
        if (Math.abs(dsb.length() - lsb.length()) > 1) return "";
        StringBuilder longSb = dsb.length() > lsb.length() ? dsb : lsb;
        StringBuilder shortSb = longSb == dsb ? lsb : dsb;
        StringBuilder result = new StringBuilder();
        int i = 0;
        for (; i < shortSb.length(); i++) {
            result.append(longSb.charAt(i));
            result.append(shortSb.charAt(i));
        }
        if (i <= longSb.length() - 1) result.append(longSb.charAt(i));
        return result.toString();
    }

    // LC636
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] result = new int[n];
        int mutexFid = -1, mutexStartTime = -1;
        Deque<Integer> stack = new LinkedList<>();
        for (String log : logs) {
            String[] split = log.split(":");
            int fid = Integer.parseInt(split[0]), timestamp = Integer.parseInt(split[2]);
            String type = split[1];
            if (type.equals("start")) {
                if (!stack.isEmpty()) {
                    result[mutexFid] += timestamp - mutexStartTime;
                }
                stack.push(fid);
                mutexFid = fid;
                mutexStartTime = timestamp;
            } else {
                timestamp++;
                result[fid] += timestamp - mutexStartTime;
                stack.pop();
                if (!stack.isEmpty()) {
                    mutexFid = stack.peek();
                    mutexStartTime = timestamp;
                }
            }
        }
        return result;
    }

    static final class Log implements Comparable<Log> {
        int fid = -1, timestamp = -1;
        Event type = Event.INVALID;

        enum Event {
            START, END, INVALID
        }

        public Log(int fid, int timestamp, Event type) {
            this.fid = fid;
            this.timestamp = timestamp;
            this.type = type;
        }

        @Override
        public int compareTo(Log another) {
            return this.timestamp - another.timestamp;
        }
    }

    // LC623
    public TreeNode69 addOneRow(TreeNode69 root, int val, int depth) {
        if (depth == 1) {
            TreeNode69 newRoot = new TreeNode69(val);
            newRoot.left = root;
            return newRoot;
        }
        if (root == null) return null;
        int layer = 0;
        Deque<TreeNode69> q = new LinkedList<>();
        List<TreeNode69> parent = new ArrayList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            layer++;
            boolean parentFlag = layer == depth - 1;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                TreeNode69 p = q.poll();
                if (parentFlag) {
                    parent.add(p);
                }
                if (p.left != null) {
                    q.offer(p.left);
                }
                if (p.right != null) {
                    q.offer(p.right);
                }
            }
            if (parentFlag) break;
        }

        for (TreeNode69 t : parent) {
            TreeNode69 origLeft = t.left, origRight = t.right;
            t.left = new TreeNode69(val);
            t.right = new TreeNode69(val);
            t.left.left = origLeft;
            t.right.right = origRight;
        }

        return root;
    }

    // LC1403
    public List<Integer> minSubsequence(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
        }
        int threshold = (sum / 2) + 1;
        List<Integer> result = new ArrayList<>();
        Arrays.sort(nums);
        int tmp = 0;
        for (int i = n - 1; i >= 0; i--) {
            tmp += nums[i];
            result.add(nums[i]);
            if (tmp >= threshold) {
                return result;
            }
        }
        return result;
    }

    // LC1161
    public int maxLevelSum(TreeNode69 root) {
        LinkedList<TreeNode69> q = new LinkedList<>();
        int layer = 0, result = -1, maxSum = Integer.MIN_VALUE;
        q.offer(root);
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size(), layerSum = 0;
            for (int i = 0; i < qs; i++) {
                TreeNode69 p = q.poll();
                layerSum += p.val;
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
            if (layerSum > maxSum) {
                result = layer;
                maxSum = layerSum;
            }
        }
        return result;
    }

    // LC952
    public int largestComponentSize(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        Map<Integer, Set<Integer>> factorNumMap = new HashMap<>();
        for (int i = n - 1; i >= 0; i--) {
            int victim = nums[i];
            int sqrt = (int) Math.sqrt(victim) + 1;
            for (int j = 1; j <= sqrt; j++) {
                if (victim % j == 0) {
                    int anotherFactor = victim / j;
                    factorNumMap.putIfAbsent(j, new HashSet<>());
                    factorNumMap.putIfAbsent(anotherFactor, new HashSet<>());
                    factorNumMap.get(j).add(victim);
                    factorNumMap.get(anotherFactor).add(victim);
                }
            }
        }
        DSUArray69 dsu = new DSUArray69(100001);
        for (Map.Entry<Integer, Set<Integer>> e : factorNumMap.entrySet()) {
            if (e.getKey() == 1) continue;
            Set<Integer> sharingEdge = e.getValue();
            Integer root = sharingEdge.stream().findFirst().get();
            for (int i : sharingEdge) {
                dsu.add(i);
                dsu.merge(i, root);
            }
        }
        Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
        int result = 0;
        for (Map.Entry<Integer, Set<Integer>> e : allGroups.entrySet()) {
            result = Math.max(result, e.getValue().size());
        }
        return result;
    }

    // LC444 JZOF II 115
    public boolean sequenceReconstruction(int[] nums, int[][] sequences) {
        int n = nums.length;
        List<Integer>[] outEdge = new List[n + 1];
        int[] indegree = new int[n + 1];
        BitSet bs = new BitSet(n + 1);
        for (int[] s : sequences) {
            Integer prev = null;
            for (int cur : s) {
                bs.set(cur);
                if (prev != null) {
                    indegree[cur]++;
                    if (outEdge[prev] == null) {
                        outEdge[prev] = new ArrayList<>();
                    }
                    outEdge[prev].add(cur);
                }
                prev = cur;
            }
        }
        if (bs.cardinality() != n) return false;
        Deque<Integer> q = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
                break;
            }
        }
        if (q.size() != 1) return false;
        List<Integer> topo = new ArrayList<>(n + 1);
        while (!q.isEmpty()) {
            int qs = q.size();
            if (qs > 1) return false;
            int p = q.poll();
            topo.add(p);
            if (outEdge[p] == null) continue;
            for (int next : outEdge[p]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        if (topo.size() != n) return false;
        for (int i = 0; i < n; i++) {
            if (topo.get(i) != nums[i]) return false;
        }
        return true;
    }

    // LC1184
    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start == destination) return 0;
        int forward = 0, backward = 0, startPoint = destination > start ? start : destination,
                endPoint = destination > startPoint ? destination : start, total = Arrays.stream(distance).sum();

        for (int i = startPoint; i < endPoint; i++) {
            forward += distance[i];
        }
        backward = total - forward;
        return Math.min(backward, forward);
    }

    // LC558 ** Quad Tree
    public Node69 intersect(Node69 quadTree1, Node69 quadTree2) {
        if (quadTree1.isLeaf) {
            if (quadTree1.val) {
                return new Node69() {{
                    val = true;
                    isLeaf = true;
                }};
            }
            return new Node69(quadTree2.val, quadTree2.isLeaf, quadTree2.topLeft, quadTree2.topRight, quadTree2.bottomLeft, quadTree2.bottomRight);
        }
        if (quadTree2.isLeaf) {
            return intersect(quadTree2, quadTree1);
        }
        Node69 o1 = intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node69 o2 = intersect(quadTree1.topRight, quadTree2.topRight);
        Node69 o3 = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node69 o4 = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node69() {{
                val = o1.val;
                isLeaf = true;
            }};
        }
        return new Node69(false, false, o1, o2, o3, o4);
    }

}

// Definition for a QuadTree node.
class Node69 {
    public boolean val;
    public boolean isLeaf; // true means all value of the 4 corner is the same
    public Node69 topLeft;
    public Node69 topRight;
    public Node69 bottomLeft;
    public Node69 bottomRight;

    public Node69() {
    }

    public Node69(boolean _val, boolean _isLeaf, Node69 _topLeft, Node69 _topRight, Node69 _bottomLeft, Node69 _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
}


class DisjointSetUnion69<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion69() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(T i, T j) {
        if (!father.containsKey(i) || !father.containsKey(j)) return false;
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(T i) {
        return father.containsKey(i);
    }

}


class DSUArray69 {
    int[] father;
    int[] rank;
    int size;

    public DSUArray69(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray69() {
        this.size = 1 << 16;
        father = new int[1 << 16];
        rank = new int[1 << 16];
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

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (int i = 0; i < size; i++) {
            if (father[i] != -1) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
        }
        return result;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }

    public int getSelfGroupSize(int x) {
        return rank[find(x)];
    }

}


class TreeNode69 {
    int val;
    TreeNode69 left;
    TreeNode69 right;

    TreeNode69() {
    }

    TreeNode69(int val) {
        this.val = val;
    }

    TreeNode69(int val, TreeNode69 left, TreeNode69 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}