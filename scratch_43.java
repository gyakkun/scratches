import javafx.util.*;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.err.println(s.smallestDifference(new int[]{1, 2, 11, 15}, new int[]{4, 12, 19, 23, 127, 11}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC Interview 16.06
    public int smallestDifference(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        long smallestDifference = Math.abs((long) a[0] - (long) b[0]);


        int aPtr = 0;
        int bPtr = 0;
        while (aPtr < a.length && bPtr < b.length) {
            // 移动aptr 还是移动bptr?
            // a<b, a右移 直到 a>=b
            while (aPtr < a.length && bPtr < b.length && a[aPtr] <= b[bPtr]) {
                smallestDifference = Math.min(smallestDifference, Math.abs((long) a[aPtr] - (long) b[bPtr]));
                aPtr++;
                if (smallestDifference == 0) return 0;
            }

            while (aPtr < a.length && bPtr < b.length && a[aPtr] >= b[bPtr]) {
                smallestDifference = Math.min(smallestDifference, Math.abs((long) a[aPtr] - (long) b[bPtr]));
                bPtr++;
                if (smallestDifference == 0) return 0;
            }
        }

        return (int) smallestDifference;
    }

    // LC326
    public boolean isPowerOfThree(int n) {
        int upper = (int) (Math.log(Integer.MAX_VALUE) / Math.log(3));
        int max3Power = (int) (Math.pow(3, upper));
        return n > 0 && max3Power % n == 0;
    }

    // LC342 Power of 4
    public boolean isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }

    // LC1074
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int rowNum = matrix.length;
        int colNum = matrix[0].length;
        int[][] prefix = new int[rowNum + 1][colNum + 1];
        for (int i = 1; i <= rowNum; i++) {
            for (int j = 1; j <= colNum; j++) {
                prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }

        int result = 0;

        // 枚举
        for (int i = 1; i <= rowNum; i++) {
            for (int j = 1; j <= colNum; j++) {

                // 长x宽  = i * j
                for (int row = 0; row <= rowNum - i; row++) {
                    for (int col = 0; col <= colNum - j; col++) {

                        // 四个点 upLeft upRight downLeft downRight
                        int upLeftRow = row;
                        int upLeftCol = col;
                        int upRightRow = row;
                        int upRightCol = col + j;
                        int downLeftRow = row + i;
                        int downLeftCol = col;
                        int downRightRow = row + i;
                        int downRightCol = col + j;

                        int sum = prefix[downRightRow][downRightCol] - prefix[downLeftRow][downLeftCol] - prefix[upRightRow][upRightCol] + prefix[upLeftRow][upLeftCol];
                        if (sum == target) {
                            result++;
                        }

                    }
                }
            }
        }
        return result;

    }

    // LC1629
    // LeetCode 设计了一款新式键盘，正在测试其可用性。测试人员将会点击一系列键（总计 n 个），每次一个。
    //
    // 给你一个长度为 n 的字符串 keysPressed ，其中 keysPressed[i] 表示测试序列中第 i 个被按下的键。releaseTimes 是一个升序排列的列表，其中 releaseTimes[i] 表示松开第 i 个键的时间。字符串和数组的 下标都从 0 开始 。第 0 个键在时间为 0 时被按下，接下来每个键都 恰好 在前一个键松开时被按下。
    //
    // 测试人员想要找出按键 持续时间最长 的键。第 i 次按键的持续时间为 releaseTimes[i] - releaseTimes[i - 1] ，第 0 次按键的持续时间为 releaseTimes[0] 。
    //
    // 注意，测试期间，同一个键可以在不同时刻被多次按下，而每次的持续时间都可能不同。
    //
    // 请返回按键 持续时间最长 的键，如果有多个这样的键，则返回 按字母顺序排列最大 的那个键。
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        int n = keysPressed.length();
        char[] cArr = keysPressed.toCharArray();
        long[] timeCount = new long[26];
        timeCount[cArr[0] - 'a'] += releaseTimes[0];
        for (int i = 1; i < n; i++) {
            timeCount[cArr[i] - 'a'] = Math.max((long) releaseTimes[i] - (long) releaseTimes[i - 1], timeCount[cArr[i] - 'a']);
        }
        long max = timeCount[0];
        char ans = 'a';
        for (int i = 1; i < 26; i++) {
            if (timeCount[i] >= max) {
                max = timeCount[i];
                ans = (char) ('a' + i);
            }
        }
        return ans;
    }

    // LC472 这都行???
    List<String> lc472Result = new LinkedList<>();
    Trie lc472Trie;

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        lc472Trie = new Trie();
        for (String word : words) {
            if (!word.equals("")) {
                lc472Trie.insert(word);
            }
        }
        for (String word : words) {
            if (!word.equals("")) {
                if (lc472Helper(word, 0, 1, 0)) {
                    lc472Result.add(word);
                }
            }
        }
        return lc472Result;
    }

    private boolean lc472Helper(String word, int idx, int cur, int ctr) {
        if (cur >= word.length()) {
            if (lc472Trie.search(word.substring(idx)) && ctr >= 1) {
                return true;
            }
            return false;
        }
        if (!lc472Trie.startsWith(word.substring(idx, cur))) {
            return false;
        }
        if (lc472Trie.search(word.substring(idx, cur)) && lc472Helper(word, cur, cur + 1, ctr + 1)) {
            return true;
        }
        return lc472Helper(word, idx, cur + 1, ctr);
    }

    // 标准蓄水池抽样算法, 通常nums的长度很大, 或是只是一个链表头节点, 未知总长度
    public int[] reservoirSampling(int[] nums, int m) {
        int[] result = new int[m];
        int total = 0;
        int count = 0;
        Random r = new Random();
        for (int i : nums) {
            total++;
            if (count < m) {
                result[count++] = i;
            } else {
                int ran = r.nextInt(total);
                if (ran >= 0 && ran < m) {
                    result[ran] = i;
                }
            }
        }
        return result;
    }

    // LC398
    public int pick(int target, int[] nums) {
        Random r = new Random();
        int count = 0, result = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == target) {
                count++;
                if (r.nextInt(count) == 0) { // 蓄水池大小为1, 落在[0,1)范围的数即只有0
                    result = i;
                }
            }
        }
        return result;
    }

    // LC382 流中的随机抽样问题, 蓄水池算法
    ListNode h;
    ListNode dummy = new ListNode(-1);
    Random r = new Random();
    int len = -1;

    public int getRandom() {
        // 蓄水池算法, 以1/n的概率保留第n个数, 每个数的期望概率都是1/len
        int reserve = 0;
        ListNode cur = h;
        int count = 0;
        while (cur != null) {
            count++;
            int ran = r.nextInt(count);
            if (ran == 0) {
                reserve = cur.val;
            }
            cur = cur.next;
        }
        return reserve;
    }

    public int getRandomMy() {
        int nth;
        if (len == -1) {
            nth = r.nextInt();
        } else {
            nth = r.nextInt(len);
        }

        ListNode ptr = h;
        int ctr = 1;
        while (ptr.next != null && ctr <= nth) {
            ptr = ptr.next;
            ctr++;
        }
        if (ptr.next == null) {
            len = ctr;
        }
        return ptr.val;

    }

    // LC1298 Learn from Solution
    public int maxCandiesS(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        int n = status.length;
        int ans = 0;
        boolean[] hasBox = new boolean[n];
        boolean[] canOpen = new boolean[n];
        boolean[] visited = new boolean[n];
        Deque<Integer> boxQueue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (status[i] == 1) {
                canOpen[i] = true;
            }
        }
        for (int i : initialBoxes) {
            hasBox[i] = true;
            if (canOpen[i]) {
                boxQueue.offer(i);
                visited[i] = true;
                ans += candies[i];
            }
        }
        while (!boxQueue.isEmpty()) {
            int frontBoxIdx = boxQueue.poll();
            for (int key : keys[frontBoxIdx]) {
                canOpen[key] = true;
                if (!visited[key] && hasBox[key]) {
                    boxQueue.offer(key);
                    visited[key] = true;
                    ans += candies[key];
                }
            }
            for (int box : containedBoxes[frontBoxIdx]) {
                hasBox[box] = true;
                if (!visited[box] && canOpen[box]) {
                    boxQueue.offer(box);
                    visited[box] = true;
                    ans += candies[box];
                }
            }
        }
        return ans;
    }

    // LC1298 Hard BFS
    // 给你n个盒子，每个盒子的格式为[status, candies, keys, containedBoxes]，其中：
    //
    // 状态字status[i]：整数，如果box[i]是开的，那么是 1，否则是 0。
    // 糖果数candies[i]: 整数，表示box[i] 中糖果的数目。
    // 钥匙keys[i]：数组，表示你打开box[i]后，可以得到一些盒子的钥匙，每个元素分别为该钥匙对应盒子的下标。
    // 内含的盒子containedBoxes[i]：整数，表示放在box[i]里的盒子所对应的下标。
    // 给你一个initialBoxes 数组，表示你现在得到的盒子，你可以获得里面的糖果，也可以用盒子里的钥匙打开新的盒子，还可以继续探索从这个盒子里找到的其他盒子。
    //
    // 请你按照上述规则，返回可以获得糖果的 最大数目。
    //
    // 每个盒子最多被一个盒子包含。
    public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        Set<Integer> acquiredKey = new HashSet<>();
        Deque<Integer> boxVisitQueue = new LinkedList<>();
//        Set<Integer> visited = new HashSet<>();
        int[] noKeyCounter = new int[status.length];
        int ans = 0;
        for (int i : initialBoxes) {
            boxVisitQueue.offer(i);
        }
        while (!boxVisitQueue.isEmpty()) {
            int tmpBoxIdx = boxVisitQueue.poll();
//            if (visited.contains(tmpBoxIdx)) {
//                continue;
//            }
            if (status[tmpBoxIdx] == 1 || acquiredKey.contains(tmpBoxIdx)) {
                ans += candies[tmpBoxIdx];
                for (int j : containedBoxes[tmpBoxIdx]) {
                    boxVisitQueue.offer(j);
                }
                for (int j : keys[tmpBoxIdx]) {
                    acquiredKey.add(j);
                }
//                visited.add(tmpBoxIdx);
            } else {
                noKeyCounter[tmpBoxIdx]++;
                if (noKeyCounter[tmpBoxIdx] > status.length) {
                    break;
                }
                boxVisitQueue.offer(tmpBoxIdx);
            }
        }
        return ans;
    }

    // LC477 Solution
    public int totalHammingDistance(int[] nums) {
        int ans = 0, n = nums.length;
        for (int i = 0; i < 30; i++) {
            int c = 0;
            for (int val : nums) {
                c += (val >> i) & 1;
            }
            ans += c * (n - c);
        }
        return ans;
    }

    // LC874
    public int robotSim(int[] commands, int[][] obstacles) {
        // -2 左转90度
        // -1 右转90度
        int x = 0, y = 0;
        int direct = 0; // 0 - 北, 1 - 东 , 2 - 南, 3 - 西
        int[][] step = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int ans = 0;
        Set<Pair<Integer, Integer>> obSet = new HashSet<>();
        for (int[] i : obstacles) {
            obSet.add(new Pair<>(i[0], i[1]));
        }
        for (int c : commands) {
            if (c < 0) {
                if (c == -2) {
                    direct = (direct + 4 - 1) % 4;
                } else if (c == -1) {
                    direct = (direct + 1) % 4;
                }
            } else {
                for (int j = 0; j < c; j++) {
                    if (!obSet.contains(new Pair<>(x + step[direct][0], y + step[direct][1]))) {
                        x += step[direct][0];
                        y += step[direct][1];
                        ans = Math.max(ans, x * x + y * y);
                    } else {
                        break;
                    }
                }
            }
        }
        return ans;
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

class Trie {
    Map<String, Boolean> m;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        m = new HashMap<>();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        for (int i = 0; i < word.length(); i++) {
            m.putIfAbsent(word.substring(0, i + 1), false);
        }
        m.put(word, true);
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        return m.getOrDefault(word, false);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        return m.containsKey(prefix);
    }
}