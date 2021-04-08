import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        System.err.println(s.exist(new char[][]
                {{'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a'}}, "aaaaaaaaaaaaa"));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    private boolean lc79Result = false;

    // LC79 可优化的地方: 回溯函数传target的目标下标, 避免字符串直接比较
    public boolean exist(char[][] board, String word) {
        boolean[][] visited;
        if (word.length() > (board.length * board[0].length)) return false;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                visited = new boolean[board.length][board[0].length];
                visited[i][j] = true;
                if (("" + board[i][j]).equals(word)) return true;
                if (lc79Backtrack(board, word, i, j, "" + board[i][j], visited, -1)) return true;
            }
        }
        return false;
    }

    private boolean lc79Backtrack(char[][] board, String word, int curRow, int curCol,
                                  String curWord, boolean[][] visited, int direct) { // 0123 - 上下左右
        if (curWord.length() > word.length()) {
            return false;
        }
        StringBuffer sb = new StringBuffer(curWord);
        // 上下左右
        if (curRow - 1 >= 0 && !visited[curRow - 1][curCol] && direct != 1) {
            if (sb.append(board[curRow - 1][curCol]).toString().equals(word)) {
                return true;
            }
            visited[curRow - 1][curCol] = true;
            if (lc79Backtrack(board, word, curRow - 1, curCol, sb.toString(), visited, 0)) {
                return true;
            }
            visited[curRow - 1][curCol] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curRow + 1 < board.length && !visited[curRow + 1][curCol] && direct != 0) {
            if (sb.append(board[curRow + 1][curCol]).toString().equals(word)) {
                return true;
            }
            visited[curRow + 1][curCol] = true;
            if (lc79Backtrack(board, word, curRow + 1, curCol, sb.toString(), visited, 1)) {
                return true;
            }
            visited[curRow + 1][curCol] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curCol - 1 >= 0 && !visited[curRow][curCol - 1] && direct != 3) {
            if (sb.append(board[curRow][curCol - 1]).toString().equals(word)) {
                return true;
            }
            visited[curRow][curCol - 1] = true;
            if (lc79Backtrack(board, word, curRow, curCol - 1, sb.toString(), visited, 2)) {
                return true;
            }
            visited[curRow][curCol - 1] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curCol + 1 < board[0].length && !visited[curRow][curCol + 1] && direct != 2) {
            if (sb.append(board[curRow][curCol + 1]).toString().equals(word)) {
                return true;
            }
            visited[curRow][curCol + 1] = true;
            if (lc79Backtrack(board, word, curRow, curCol + 1, sb.toString(), visited, 3)) {
                return true;
            }
            visited[curRow][curCol + 1] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        return false;
    }

    // LC153
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            boolean leftAscFlag = false, rightAscFlag = false; // [l,mid] [mid,r]
            if (nums[l] < nums[mid]) leftAscFlag = true;
            if (nums[mid] < nums[r]) rightAscFlag = true;

            if (leftAscFlag && rightAscFlag) return nums[l];
            else if (leftAscFlag) {
                l = mid;
            } else if (rightAscFlag) {
                r = mid;
            } else {
                break;
            }
        }
        return Math.min(nums[l], nums[r]);
    }

    // LC76
    public String minWindow(String s, String t) {
        int l = -1, r = Integer.MAX_VALUE;
        int minLen;
        Map<Character, Integer> m = new HashMap<>();
        Map<Character, Integer> tMap = new HashMap<>();
        Map<Character, Integer> tCount;

        for (char c : t.toCharArray()) {
            tMap.put(c, tMap.getOrDefault(c, 0) + 1);
        }
        tCount = new HashMap<>(tMap);

        for (int i = 0; i < 26; i++) {
            m.put((char) ('a' + i), 0);
            m.put((char) ('A' + i), 0);
        }

        for (int i = 0; i < s.length(); i++) {
            if (l == -1 && tCount.containsKey(s.charAt(i))) {
                l = i;
            }
            if (tCount.size() == 1 && tCount.get(tCount.keySet().iterator().next()) == 1 && tCount.containsKey(s.charAt(i))) {
                r = i;
                break;
            }
            if (tCount.containsKey(s.charAt(i))) {
                tCount.put(s.charAt(i), tCount.get(s.charAt(i)) - 1);
                if (tCount.get(s.charAt(i)) == 0) {
                    tCount.remove(s.charAt(i));
                }
            }
        }
        if (l == -1 || r == Integer.MAX_VALUE) return "";

        minLen = r - l + 1;
        String result = s.substring(l, r + 1);
        if (t.length() == 1) return t;

        for (int i = l; i <= r; i++) {
            m.put(s.charAt(i), m.get(s.charAt(i)) + 1);
        }
        if (!checkLC76(m, tMap)) return ""; // 没必要这里check?

        while (l < r && l < s.length()) {
            m.put(s.charAt(l), m.get(s.charAt(l)) - 1);
            if (checkLC76(m, tMap)) {
                l++;
                if (r - l + 1 < minLen) {
                    result = s.substring(l, r + 1);
                    minLen = r - l + 1;
                }
            } else {
                m.put(s.charAt(l), m.get(s.charAt(l)) + 1);
                if (r + 1 < s.length()) {
                    m.put(s.charAt(r + 1), m.get(s.charAt(r + 1)) + 1);
                    r++;
                } else {
                    break;
                }
            }
        }
        return result;
    }

    private boolean checkLC76(Map<Character, Integer> m, Map<Character, Integer> tMap) {
        for (char c : tMap.keySet()) {
            if (m.get(c) < tMap.get(c)) {
                return false;
            }
        }
        return true;
    }

    // LC69 二分
    public int mySqrt(int x) {
        int r = 46340; // == sqrt(Integer.MAX_VALUE)
        int l = 0;
        while (l < r) {
            int mid = (l + r + 1) / 2;
            if (mid * mid <= x) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    // LC66
    public int[] plusOne(int[] digits) {
        int carry = 0;
        for (int i = digits.length - 1; i >= 0; i--) {
            if (i == digits.length - 1) {
                digits[i] += 1;
            } else {
                digits[i] += carry;
                carry = 0;
            }
            if (digits[i] >= 10) {
                carry = 1;
                digits[i] -= 10;
            }
        }
        if (carry == 1) {
            int[] result = new int[digits.length + 1];
            for (int i = 0; i < digits.length; i++) {
                result[i + 1] = digits[i];
            }
            result[0] = 1;
            return result;
        }
        return digits;
    }

    // LC55
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i = 0; i < n; ++i) {
            if (i <= rightmost) {
                rightmost = Math.max(rightmost, i + nums[i]);
                if (rightmost >= n - 1) {
                    return true;
                }
            }
        }
        return false;
    }

    // LC55 AC
    public boolean canJumpAC(int[] nums) {
        boolean[] reachable = new boolean[nums.length];
        boolean[] visited = new boolean[nums.length];
        reachable[0] = true;
        for (int i = 0; i < nums.length; i++) {

            if (!reachable[i]) continue;
            int j = i + nums[i];

            while (j < nums.length) {
                if (visited[j]) break;
                reachable[j] = true;
                visited[j] = true;
                j += nums[j];
                if (nums[j] == 0) break;
                if (reachable[nums.length - 1]) return true;
            }

            for (int k = nums[i]; k >= 1; k--) {
                if (i + k > nums.length) continue;
                reachable[i + k] = true;
                if (reachable[nums.length - 1]) return true;
            }
        }
        return reachable[nums.length - 1];
    }

    // LC55 并查集TLE
    public boolean canJumpTLE(int[] nums) {
        DSUArray dsu = new DSUArray();
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j <= nums[i]; j++) {
                dsu.add(i + j);
                dsu.merge(i, i + j);
            }
            if (dsu.isConnected(0, nums.length - 1)) return true;
        }
        return dsu.isConnected(0, nums.length - 1);
    }

    // LC81
    public boolean search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return nums[0] == target;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[l] == nums[mid] && nums[mid] == nums[r]) {
                ++l;
                --r;
            } else if (nums[l] <= nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }

    // LC1006
    public int clumsy(int N) {
        char[] ops = new char[]{'*', '/', '+', '-'};
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < N; i++) {
            sb.append(N - i);
            sb.append(ops[i % 4]);
        }
        sb.deleteCharAt(sb.length() - 1);
        return calculate(sb.toString());
    }

    public int calculate(String s) {
        return (int) evalRPN(toRPN(decodeExpression(s)));
    }

    public List<String> toRPN(List<String> express) {
        List<String> rpn = new LinkedList<>();
        Deque<String> stack = new LinkedList<>();
        Set<String> notNumber = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
            add("(");
            add(")");
        }};
        String tmp;
        for (String token : express) {
            if (!notNumber.contains(token)) {
                rpn.add(token);
            } else if (token.equals("(")) {
                stack.push(token);
            } else if (token.equals(")")) {
                while (!(tmp = stack.pop()).equals("(")) {
                    rpn.add(tmp);
                }

            } else {
                while (!stack.isEmpty() && getOperPriority(stack.peek()) >= getOperPriority(token)) {
                    rpn.add(stack.pop());
                }
                stack.push(token);
            }
        }
        while (!stack.isEmpty()) {
            rpn.add(stack.pop());
        }
        return rpn;
    }

    private List<String> decodeExpression(String express) {
        express = express.replaceAll("\\ ", "");
        express = express.replaceAll("\\(\\+", "(0+");
        express = express.replaceAll("\\(\\-", "(0-");
        express = express.replaceAll("\\((\\d+\\.?\\d*)\\)", "$1");
        List<String> result = new LinkedList<>();
        int i = 0;
        StringBuffer sb;
        do {
            if ((express.charAt(i) < '0' || express.charAt(i) > '9') && express.charAt(i) != '.') {
                result.add(express.charAt(i) + "");
                i++;
            } else {
                sb = new StringBuffer();
                while (i < express.length() && ((express.charAt(i) >= '0' && express.charAt(i) <= '9') || express.charAt(i) == '.')) {
                    sb.append(express.charAt(i));
                    i++;
                }
                result.add(sb.toString());
            }
        } while (i < express.length());
        return result;
    }

    private int getOperPriority(String oper) {
        switch (oper) {
            case "+":
            case "-":
                return 1;
            case "*":
            case "/":
                return 2;
            default:
                return -1;
        }
    }

    // LC150 逆波兰表达式
    public int evalRPN(List<String> tokens) {
        Deque<String> stack = new LinkedList<>();
        stack.push("0");
        Set<String> oper = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
        }};
        for (String token : tokens) {
            if (oper.contains(token)) {
                int a = Integer.parseInt(stack.pop());
                int b = Integer.parseInt(stack.pop());
                int tmp;
                switch (token) {
                    case "+":
                        tmp = a + b;
                        break;
                    case "-":
                        tmp = b - a;
                        break;
                    case "/":
                        tmp = b / a;
                        break;
                    case "*":
                        tmp = a * b;
                        break;
                    default:
                        tmp = 0;
                }
                stack.push(String.valueOf(tmp));
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
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

class DisjointSetUnion {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(int i) {
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
    public int find(int i) {
        //先找到根 再压缩
        int root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            int origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iFather = find(i);
        int jFather = find(j);
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

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (Integer i : father.keySet()) {
            int f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<Integer> s = new HashSet<Integer>();
        for (Integer i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

}