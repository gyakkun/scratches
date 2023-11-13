package moe.nyamori.test.historical;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class scratch_37 {
    public static void main(String[] args) {
        scratch_37 s = new scratch_37();
        long timing = System.currentTimeMillis();

//        System.err.println(s.findWords(new char[][]{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}}, new String[]{"oath", "pea", "eat", "rain"}));
        System.err.println(s.findWords(new char[][]{{'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}}
                , new String[]{"lllllll", "fffffff", "ssss", "s", "rr", "xxxx", "ttt", "eee", "ppppppp", "iiiiiiiii", "xxxxxxxxxx", "pppppp", "xxxxxx", "yy", "jj", "ccc", "zzz", "ffffffff", "r", "mmmmmmmmm", "tttttttt", "mm", "ttttt", "qqqqqqqqqq", "z", "aaaaaaaa", "nnnnnnnnn", "v", "g", "ddddddd", "eeeeeeeee", "aaaaaaa", "ee", "n", "kkkkkkkkk", "ff", "qq", "vvvvv", "kkkk", "e", "nnn", "ooo", "kkkkk", "o", "ooooooo", "jjj", "lll", "ssssssss", "mmmm", "qqqqq", "gggggg", "rrrrrrrrrr", "iiii", "bbbbbbbbb", "aaaaaa", "hhhh", "qqq", "zzzzzzzzz", "xxxxxxxxx", "ww", "iiiiiii", "pp", "vvvvvvvvvv", "eeeee", "nnnnnnn", "nnnnnn", "nn", "nnnnnnnn", "wwwwwwww", "vvvvvvvv", "fffffffff", "aaa", "p", "ddd", "ppppppppp", "fffff", "aaaaaaaaa", "oooooooo", "jjjj", "xxx", "zz", "hhhhh", "uuuuu", "f", "ddddddddd", "zzzzzz", "cccccc", "kkkkkk", "bbbbbbbb", "hhhhhhhhhh", "uuuuuuu", "cccccccccc", "jjjjj", "gg", "ppp", "ccccccccc", "rrrrrr", "c", "cccccccc", "yyyyy", "uuuu", "jjjjjjjj", "bb", "hhh", "l", "u", "yyyyyy", "vvv", "mmm", "ffffff", "eeeeeee", "qqqqqqq", "zzzzzzzzzz", "ggg", "zzzzzzz", "dddddddddd", "jjjjjjj", "bbbbb", "ttttttt", "dddddddd", "wwwwwww", "vvvvvv", "iii", "ttttttttt", "ggggggg", "xx", "oooooo", "cc", "rrrr", "qqqq", "sssssss", "oooo", "lllllllll", "ii", "tttttttttt", "uuuuuu", "kkkkkkkk", "wwwwwwwwww", "pppppppppp", "uuuuuuuu", "yyyyyyy", "cccc", "ggggg", "ddddd", "llllllllll", "tttt", "pppppppp", "rrrrrrr", "nnnn", "x", "yyy", "iiiiiiiiii", "iiiiii", "llll", "nnnnnnnnnn", "aaaaaaaaaa", "eeeeeeeeee", "m", "uuu", "rrrrrrrr", "h", "b", "vvvvvvv", "ll", "vv", "mmmmmmm", "zzzzz", "uu", "ccccccc", "xxxxxxx", "ss", "eeeeeeee", "llllllll", "eeee", "y", "ppppp", "qqqqqq", "mmmmmm", "gggg", "yyyyyyyyy", "jjjjjj", "rrrrr", "a", "bbbb", "ssssss", "sss", "ooooo", "ffffffffff", "kkk", "xxxxxxxx", "wwwwwwwww", "w", "iiiiiiii", "ffff", "dddddd", "bbbbbb", "uuuuuuuuu", "kkkkkkk", "gggggggggg", "qqqqqqqq", "vvvvvvvvv", "bbbbbbbbbb", "nnnnn", "tt", "wwww", "iiiii", "hhhhhhh", "zzzzzzzz", "ssssssssss", "j", "fff", "bbbbbbb", "aaaa", "mmmmmmmmmm", "jjjjjjjjjj", "sssss", "yyyyyyyy", "hh", "q", "rrrrrrrrr", "mmmmmmmm", "wwwww", "www", "rrr", "lllll", "uuuuuuuuuu", "oo", "jjjjjjjjj", "dddd", "pppp", "hhhhhhhhh", "kk", "gggggggg", "xxxxx", "vvvv", "d", "qqqqqqqqq", "dd", "ggggggggg", "t", "yyyy", "bbb", "yyyyyyyyyy", "tttttt", "ccccc", "aa", "eeeeee", "llllll", "kkkkkkkkkk", "sssssssss", "i", "hhhhhh", "oooooooooo", "wwwwww", "ooooooooo", "zzzz", "k", "hhhhhhhh", "aaaaa", "mmmmm"}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC212 Hard, 多线程版本效果不好
    ConcurrentMap<String, Integer> lc212Result;
    HashSet<String> lc212ResultStr;
    int boardRow;
    int boardCol;

    public List<String> findWords(char[][] board, String[] words) {
        Trie35 trie = new Trie35();
        for (String word : words) {
            trie.insert(word);
        }
        lc212Result = new ConcurrentHashMap<>();
        boardRow = board.length;
        boardCol = board[0].length;
        ForkJoinPool fjp = new ForkJoinPool();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                lc79 job = new lc79(board, trie, i, j, "" + board[i][j], new boolean[boardRow][boardCol], -1);
                fjp.submit(job);
            }
        }
        while (fjp.getActiveThreadCount() != 0) {
            ;
        }
        return new LinkedList<>(lc212Result.keySet());
    }

    public class lc79 extends RecursiveAction {

        char[][] board;
        Trie35 trie;
        int curRow;
        int curCol;
        String curWord;
        boolean[][] visited;
        int direct;
        ReentrantReadWriteLock rwl;
        Lock rLVisited;
        Lock wLVisited;

        public lc79(char[][] board, Trie35 trie, int curRow, int curCol, String curWord, boolean[][] visited, int direct) {
            this.board = board;
            this.trie = trie;
            this.curRow = curRow;
            this.curCol = curCol;
            this.curWord = curWord;
            this.visited = visited;
            this.direct = direct;
        }

        public lc79(char[][] board, Trie35 trie, int curRow, int curCol, String curWord, boolean[][] visited, int direct, Lock rLVisited, Lock wLVisited) {
            this.board = board;
            this.trie = trie;
            this.curRow = curRow;
            this.curCol = curCol;
            this.curWord = curWord;
            this.visited = visited;
            this.direct = direct;
            this.rLVisited = rLVisited;
            this.wLVisited = wLVisited;
        }

        @Override
        protected void compute() {
            if (direct == -1) {
                this.rwl = new ReentrantReadWriteLock();
                this.rLVisited = rwl.readLock();
                this.wLVisited = rwl.writeLock();
                try {
                    wLVisited.lock();
                    visited[curRow][curCol] = true;
                } finally {
                    wLVisited.unlock();
                }
            }
            if (!trie.startsWith(curWord)) {
                return;
            }
            if (trie.search(curWord)) {
                lc212Result.putIfAbsent(curWord, 1);
            }

            int[][] options = new int[][]{{curRow - 1, curCol, 1, 0}, {curRow + 1, curCol, 0, 1}, {curRow, curCol - 1, 3, 2}, {curRow, curCol + 1, 2, 3}};
            for (int[] option : options) {
                if (checkLegalPosition(option[0], option[1]) && direct != option[2]) {
                    try {
                        wLVisited.lock();
                        visited[option[0]][option[1]] = true;
                    } finally {
                        wLVisited.unlock();
                    }

                    lc79 job = new lc79(board, trie, option[0], option[1], curWord + board[option[0]][option[1]], visited, option[3], this.rLVisited, this.wLVisited);
                    job.fork();

                    try {
                        wLVisited.lock();
                        visited[option[0]][option[1]] = false;
                    } finally {
                        wLVisited.unlock();
                    }
                }
            }

            return;
        }

        private boolean checkLegalPosition(int row, int col) {
            boolean flag1 = (row >= 0 && row < boardRow && col >= 0 && col < boardCol);
            if (!flag1) return false;
            boolean flag2 = false;
            try {
                rLVisited.lock();
                flag2 = visited[row][col];
            } finally {
                rLVisited.unlock();
            }
            return flag1 && !flag2;
        }
    }

    private void lc79Backtrack(char[][] board, Trie35 trie, int curRow, int curCol,
                               String curWord, boolean[][] visited, int direct) { // 0123 - 上下左右
        if (!trie.startsWith(curWord)) {
            return;
        }
        if (trie.search(curWord)) {
            lc212ResultStr.add(curWord);
        }
        StringBuffer sb = new StringBuffer(curWord);
        // 上下左右
        if (curRow - 1 >= 0 && !visited[curRow - 1][curCol] && direct != 1) {
            sb.append(board[curRow - 1][curCol]);
            visited[curRow - 1][curCol] = true;
            lc79Backtrack(board, trie, curRow - 1, curCol, sb.toString(), visited, 0);
            visited[curRow - 1][curCol] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curRow + 1 < board.length && !visited[curRow + 1][curCol] && direct != 0) {
            sb.append(board[curRow + 1][curCol]);
            visited[curRow + 1][curCol] = true;
            lc79Backtrack(board, trie, curRow + 1, curCol, sb.toString(), visited, 1);
            visited[curRow + 1][curCol] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curCol - 1 >= 0 && !visited[curRow][curCol - 1] && direct != 3) {
            sb.append(board[curRow][curCol - 1]);
            visited[curRow][curCol - 1] = true;
            lc79Backtrack(board, trie, curRow, curCol - 1, sb.toString(), visited, 2);
            visited[curRow][curCol - 1] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        if (curCol + 1 < board[0].length && !visited[curRow][curCol + 1] && direct != 2) {
            sb.append(board[curRow][curCol + 1]);
            visited[curRow][curCol + 1] = true;
            lc79Backtrack(board, trie, curRow, curCol + 1, sb.toString(), visited, 3);
            visited[curRow][curCol + 1] = false;
            sb.deleteCharAt(sb.length() - 1);
        }
        return;
    }


    // LC204
    public int countPrimes(int n) {
        int MAX_SIZE = 348514;
        boolean[] visit = new boolean[n + 1];
        int[] prime = new int[Math.max(MAX_SIZE, n + 1)];
        int ctr = 0;
        for (int i = 2; i <= n; i++) {
            if (!visit[i]) {
                prime[ctr++] = i;
            }
            for (int j = 0; j < ctr; j++) {
                if (i * prime[j] > n) break;
                visit[i * prime[j]] = true;
                if (i % prime[j] == 0) break;
            }
        }
        return ctr;
    }

    // LC200
    public int numIslands(char[][] grid) {
        int rowNum = grid.length;
        int colNum = grid[0].length;
        int totalPoints = rowNum * colNum;
        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                if (grid[i][j] == '1') {
                    dsu.add(i * colNum + j);
                    if (i - 1 > 0 && grid[i - 1][j] == '1') {
                        dsu.add((i - 1) * colNum + j);
                        dsu.merge(i * colNum + j, (i - 1) * colNum + j);
                    }
                    if (i + 1 < rowNum && grid[i + 1][j] == '1') {
                        dsu.add((i + 1) * colNum + j);
                        dsu.merge(i * colNum + j, (i + 1) * colNum + j);
                    }
                    if (j - 1 > 0 && grid[i][j - 1] == '1') {
                        dsu.add(i * colNum + j - 1);
                        dsu.merge(i * colNum + j, i * colNum + j - 1);
                    }
                    if (j + 1 < colNum && grid[i][j + 1] == '1') {
                        dsu.add(i * colNum + j + 1);
                        dsu.merge(i * colNum + j, i * colNum + j + 1);
                    }
                }
            }
        }
        return dsu.getNumOfGroups();
    }

    // LC938
    int lc938Sum = 0;

    public int rangeSumBST(TreeNode8 root, int low, int high) {
        rangeSumBSTHelper(root, low, high);
        return lc938Sum;
    }

    private void rangeSumBSTHelper(TreeNode8 root, int low, int high) {
        if (root == null) return;
        if (root.val < low) {
            rangeSumBSTHelper(root.right, low, high);
        } else if (root.val >= low && root.val <= high) {
            lc938Sum += root.val;
            rangeSumBSTHelper(root.left, low, high);
            rangeSumBSTHelper(root.right, low, high);
        } else if (root.val > high) {
            rangeSumBSTHelper(root.left, low, high);
        }
    }

    // LC198
    public int rob(int[] nums) {
        int[][] dp = new int[nums.length][2];
        // dp[i][0] 表示不rob得到的最大金额, dp[i][1]表示rob得到得最大金额
        if (nums.length == 1) return nums[0];
        if (nums.length == 2) return Math.max(nums[0], nums[1]);
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        dp[1][0] = nums[0];
        dp[1][1] = nums[1];
        for (int i = 2; i < nums.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }
        return Math.max(dp[nums.length - 1][0], dp[nums.length - 1][1]);
    }

    // LC172 建议纸笔做点草稿
    public int trailingZeroes(int n) {
        int result = 0;
        while (n != 0) {
            n /= 5;
            result += n;
        }
        return result;
    }

    // LC171
    public int titleToNumber(String columnTitle) {
        int result = 0;
        for (int i = 0; i < columnTitle.length(); i++) {
            result = 26 * result + (columnTitle.charAt(i) - 'A' + 1);
        }
        return result;
    }

    // LC169
    public int majorityElement(int[] nums) {
        int half = nums.length / 2;
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : nums) {
            m.put(i, m.getOrDefault(i, 0) + 1);
            if (m.get(i) > half) {
                return i;
            }
        }
        return -1;
    }

    // LC166 almost Solution
    public String fractionToDecimal(int numerator, int denominator) {
        long num = numerator;
        long den = denominator;
        num = Math.abs(num);
        den = Math.abs(den);
        String left = String.valueOf(num / den);
        if ((numerator < 0 && denominator > 0) || (numerator > 0 && denominator < 0)) left = "-" + left;
        long remainder = num % den;
        StringBuffer sb = new StringBuffer(left);
        sb.append(".");
        if (remainder == 0) {
            return left;
        }
        Map<Long, Integer> map = new HashMap<>();
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

    // LC162
    public int findPeakElement(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1]) return i;
        }
        return nums.length - 1;
    }

    // LC04.04
    Map<TreeNode8, Integer> treeHeight = new HashMap<>();
    boolean lc0404Flag = false;

    public boolean isBalanced(TreeNode8 root) {
        checkHeight(root);
        return !lc0404Flag;
    }

    private int checkHeight(TreeNode8 root) {
        if (root == null) return 0;
        if (treeHeight.containsKey(root)) return treeHeight.get(root);
        int height = Math.max(checkHeight(root.left), checkHeight(root.right)) + 1;
        if (Math.abs(checkHeight(root.left) - checkHeight(root.right)) > 1) lc0404Flag = true;
        treeHeight.put(root, height);
        return height;
    }


    // LC152 乘积最大子数组
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] dpMax = Arrays.copyOf(nums, n); // dpMax[i] 表示以nums[i] 结尾的最大子数组的积
        int[] dpMin = Arrays.copyOf(nums, n); // dpMin 表示以nums[i]结尾的最小乘积
        for (int i = 1; i < n; i++) {
            dpMax[i] = Math.max(Math.max(dpMax[i - 1] * nums[i], dpMin[i - 1] * nums[i]), nums[i]);
            dpMin[i] = Math.min(Math.min(dpMin[i - 1] * nums[i], dpMax[i - 1] * nums[i]), nums[i]);
        }
        return Arrays.stream(dpMax).max().getAsInt();
    }

    // LC149 Hard
    public int maxPoints(int[][] points) {
        Map<Double, Integer> slash = new HashMap<>();
        int result = 1;
        for (int i = 0; i < points.length - 1; i++) {
            slash.clear();
            int horizon = 1;
            int dup = 0;
            int tmpMax = 1;
            for (int j = i + 1; j < points.length; j++) {
                if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
                    dup++;
                } else if (points[i][1] == points[j][1]) {
                    horizon++;
                    tmpMax = Math.max(tmpMax, horizon);
                } else {
                    double k;
                    if (points[i][0] == points[j][0]) {
                        k = 0d;
                    } else {
                        k = ((double) (points[i][0] - points[j][0])) / ((double) (points[i][1] - points[j][1]));
                    }
                    slash.put(k, slash.getOrDefault(k, 1) + 1);
                    tmpMax = Math.max(tmpMax, slash.get(k));
                }
            }
            result = Math.max(result, tmpMax + dup);
        }
        return result;
    }

    // LC1011 Solution
    public int shipWithinDaysSolution(int[] weights, int D) {
        // 确定二分查找左右边界
        int left = Arrays.stream(weights).max().getAsInt(), right = Arrays.stream(weights).sum();
        while (left < right) {
            int mid = (left + right) / 2;
            // need 为需要运送的天数
            // cur 为当前这一天已经运送的包裹重量之和
            int need = 1, cur = 0;
            for (int weight : weights) {
                if (cur + weight > mid) {
                    ++need;
                    cur = 0;
                }
                cur += weight;
            }
            if (need <= D) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // LC1011 141ms
    public int shipWithinDays(int[] weights, int D) {
        int total = 0;
        int maxSingle = Integer.MIN_VALUE;
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : weights) {
            total += i;
            maxSingle = Math.max(maxSingle, i);
            ts.add(total);
        }
        int average = (int) Math.ceil((double) total / (double) D);
        int left = Math.max(maxSingle, average), right = average + maxSingle;
        while (left < right) {
            int midCap = left + (right - left) / 2;
            int days = howManyDays(D, total, ts, midCap);
            if (days <= D) {
                right = midCap;
            } else {
                left = midCap + 1;
            }
        }
        return left;
    }

    private int howManyDays(int D, int total, TreeSet<Integer> ts, int cap) {
        int totalCtr;
        int dCtr;
        totalCtr = 0;
        for (dCtr = 0; dCtr < D; dCtr++) {
            Integer floor = ts.floor(totalCtr + cap);
            if (floor != null && totalCtr < total) {
                totalCtr = floor;
            } else {
                break;
            }
        }
        if (totalCtr == total) {
            return dCtr;
        }
        return Integer.MAX_VALUE;
    }

    // LC148
    public ListNode37 sortList(ListNode37 head) {
        return sortListHelper(head, null);
    }

    private ListNode37 sortListHelper(ListNode37 head, ListNode37 tail) {
        if (head == null) return null;
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode37 fast = head, slow = head;
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }

        ListNode37 left = sortListHelper(head, slow);
        ListNode37 right = sortListHelper(slow, tail);
        ListNode37 newHead = sortMerge(left, right);
        return newHead;
    }

    private ListNode37 sortMerge(ListNode37 l1, ListNode37 l2) {
        ListNode37 dummy = new ListNode37(-1);
        ListNode37 tmp = dummy, p1 = l1, p2 = l2;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                tmp.next = p1;
                p1 = p1.next;
            } else {
                tmp.next = p2;
                p2 = p2.next;
            }
            tmp = tmp.next;
        }
        if (p1 != null) {
            tmp.next = p1;
        } else if (p2 != null) {
            tmp.next = p2;
        }
        return dummy.next;
    }

    // LC140
    private List<String> lc140Result = new LinkedList<>();
    private int longestWordLen = 0;

    // LC140
    public List<String> wordBreak140(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        for (String word : wordSet) {
            longestWordLen = Math.max(longestWordLen, word.length());
        }
        wordBreak140Backtrack(s, wordSet, 0, new LinkedList<>());
        return lc140Result;
    }

    private void wordBreak140Backtrack(String s, Set<String> wordSet, int curIdx, List<String> curList) {
        if (curIdx == s.length()) {
            lc140Result.add(String.join(" ", curList));
            return;
        }
        for (int i = 1; i <= longestWordLen; i++) {
            if (curIdx + i <= s.length() && wordSet.contains(s.substring(curIdx, curIdx + i))) {
                curList.add(s.substring(curIdx, curIdx + i));
                wordBreak140Backtrack(s, wordSet, curIdx + i, curList);
                curList.remove(curList.size() - 1);
            }
        }
    }

    // LC139
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] reachable = new boolean[n + 1];
        reachable[0] = true;
        Set<String> wordSet = new HashSet<>(wordDict);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (reachable[i] && wordSet.contains(s.substring(i, j))) {
                    reachable[j] = true;
                }
            }
        }
        return reachable[n];
    }

    // LC138
    public Node37 copyRandomList(Node37 head) {
        if (head == null) {
            return head;
        }
        Map<Node37, Node37> map = new HashMap<>();
        Node37 node = head;
        while (node != null) {
            Node37 temp = new Node37(node.val);
            map.put(node, temp);
            node = node.next;
        }
        node = head;
        while (node != null) {
            map.get(node).next = map.get(node.next);
            map.get(node).random = map.get(node.random);
            node = node.next;
        }
        return map.get(head);
    }

}

// LC146
class LRUCache146 extends LinkedHashMap<Integer, Integer> {
    private int capacity;

    public LRUCache146(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    // 这个可不写
    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */


// LC138
class Node37 {
    int val;
    Node37 next;
    Node37 random;

    public Node37(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}


class ListNode37 {
    int val;
    ListNode37 next;

    ListNode37() {
    }

    ListNode37(int val) {
        this.val = val;
    }

    ListNode37(int val, ListNode37 next) {
        this.val = val;
        this.next = next;
    }
}

class TreeNode37 {
    int val;
    TreeNode8 left;
    TreeNode8 right;

    TreeNode37() {
    }

    TreeNode37(int val) {
        this.val = val;
    }

    TreeNode37(int val, TreeNode8 left, TreeNode8 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class DisjointSetUnion37 {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion37() {
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

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}

class Trie37 {
    TrieNode35 root;

    /**
     * Initialize your data structure here.
     */
    public Trie37() {
        root = new TrieNode35();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode35 former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {

            if (former.val == '#') former.val = word.charAt(i);

            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) {
                former.sibling = new TrieNode35(word.charAt(i));
                former = former.sibling;
            }
            if (former.child == null) former.child = new TrieNode35();
            former = former.child;
        }

        if (former.val == '#') former.val = word.charAt(i);

        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) {
            former.sibling = new TrieNode35(word.charAt(i));
            former = former.sibling;
        }
        if (former.child == null) former.child = new TrieNode35();
        former.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode35 former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {
            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) return false;
        return former.isEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode35 former = root;
        int i;
        for (i = 0; i < prefix.length() - 1; i++) {
            former = former.searchSibling(prefix.charAt(i));
            if (former.val != prefix.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(prefix.charAt(i));
        if (former.val != prefix.charAt(i)) return false;
        return true;
    }
}

class TrieNode37 {
    Character val;
    Boolean isEnd;
    TrieNode37 child;
    TrieNode37 sibling;

    public TrieNode37() {
        this.val = '#';
        this.isEnd = false;
    }

    public TrieNode37(Character c) {
        this.val = c;
        this.isEnd = false;
    }

    public TrieNode37 searchSibling(Character c) {
        TrieNode37 former = this;
        while (former.sibling != null) {
            if (former.val == c) return former;
            former = former.sibling;
        }
        return former;
    }

    public TrieNode37 searchChildren(Character c) {
        TrieNode37 former = this;
        while (former.child != null) {
            if (former.val == c) return former;
            former = former.child;
        }
        return former;
    }
}