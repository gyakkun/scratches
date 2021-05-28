import javafx.util.*;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        int[] candies = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

        System.err.println(s.findAllConcatenatedWordsInADict(new String[]{"cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"}));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC472 这都行???
    List<String> lc472Result = new LinkedList<>();
    Trie lc472Trie;

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        lc472Trie = new Trie();
        Arrays.sort(words);
        for (String word : words) {
            lc472Trie.insert(word);
        }
        for (String word : words) {
            if (lc472Helper(word, 0, 1, 0)) {
                lc472Result.add(word);
            }
        }
        return lc472Result;
    }

    private boolean lc472Helper(String word, int idx, int cur, int ctr) {
        if (cur >= word.length()) {
            if (lc472Trie.search(word.substring(idx, word.length())) && ctr >= 1) {
                return true;
            }
            return false;
        }
        if (!lc472Trie.startsWith(word.substring(idx, cur))) {
            return false;
        }
        if (lc472Trie.search(word.substring(idx, cur))) {
            return lc472Helper(word, cur, cur + 1, ctr + 1) || lc472Helper(word, idx, cur + 1, ctr);
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