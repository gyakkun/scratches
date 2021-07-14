import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        WordFilter wf = new WordFilter(new String[]{"apple"});
        System.out.println(wf.f("a", "e"));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1818
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        int n = nums1.length;
        final long mod = 1_000_000_007;
        int[] absDiff = new int[n];
        long result = 0;
        TreeSet<Integer> nums1Ts = new TreeSet<>();
        Map<Integer, Set<Integer>> diffIdxMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            nums1Ts.add(nums1[i]);
            absDiff[i] = Math.abs(nums1[i] - nums2[i]);
            diffIdxMap.putIfAbsent(absDiff[i], new HashSet<>());
            diffIdxMap.get(absDiff[i]).add(i);
            result += absDiff[i];
        }
        int maxReduce = 0;
        Iterator<Integer> it = diffIdxMap.keySet().iterator();
        while (it.hasNext()) {
            int next = it.next();
            Set<Integer> nums2Idxes = diffIdxMap.get(next);
            for (int i : nums2Idxes) {
                Integer ceiling = nums1Ts.ceiling(nums2[i]);
                Integer floor = nums1Ts.floor(nums2[i]);
                if (ceiling != null) {
                    maxReduce = Math.max(maxReduce, Math.abs(nums1[i] - nums2[i]) - Math.abs(ceiling - nums2[i]));
                }
                if (floor != null) {
                    maxReduce = Math.max(maxReduce, Math.abs(nums1[i] - nums2[i]) - Math.abs(floor - nums2[i]));
                }
            }
        }
        result -= maxReduce;
        return (int) (result % mod);
    }

    // LC218
    public List<List<Integer>> getSkyline(int[][] buildings) {
        // buildings[i] = [left_i,right_i,height_i]
        // event = [x, y, in/out]
        List<List<Integer>> result = new ArrayList<>();
        final int IN = -1, OUT = 1;
        // X轴靠左的优先, IN事件优先, IN事件高度相同时候高度越高(越负,越小)的优先
        PriorityQueue<int[]> events = new PriorityQueue<>((o1, o2) -> o1[0] == o2[0] ? o1[2] * o1[1] - o2[2] * o2[1] : o1[0] - o2[0]);
        for (int[] b : buildings) {
            events.offer(new int[]{b[0], b[2], IN});
            events.offer(new int[]{b[1], b[2], OUT});
        }
        TreeMap<Integer, Integer> height = new TreeMap<>();
        height.put(0, 1);
        int formerMaxHeight = 0;
        while (!events.isEmpty()) {
            int[] tuple = events.poll();
            if (tuple[2] == IN) {
                height.put(tuple[1], height.getOrDefault(tuple[1], 0) + 1);
            } else if (tuple[2] == OUT) {
                height.put(tuple[1], height.get(tuple[1]) - 1);
                if (height.get(tuple[1]) == 0) height.remove(tuple[1]);
            }
            int maxHeight = height.lastKey();
            if (maxHeight != formerMaxHeight) {
                formerMaxHeight = maxHeight;
                result.add(Arrays.asList(tuple[0], maxHeight));
            }
        }
        return result;
    }

    // JZOF 56 **
    public int[] singleNumbers(int[] nums) {
        int victimXor = 0;
        for (int i : nums) victimXor ^= i;
        int div = 1;
        for (int i = 0; i < Integer.SIZE; i++, div <<= 1) {
            if ((div & victimXor) == div) break;
        }
        int groupA = 0, groupB = 0;
        for (int i : nums) {
            if ((i & div) == div) {
                groupA ^= i;
            } else {
                groupB ^= i;
            }
        }
        return new int[]{groupA, groupB};
    }

    // LC318
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] bitArr = new int[n];
        int max = 0;
        for (int i = 0; i < n; i++) {
            for (char c : words[i].toCharArray()) {
                bitArr[i] |= 1 << (c - 'a');
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((bitArr[i] & bitArr[j]) == 0) {
                    max = Math.max(words[i].length() * words[j].length(), max);
                }
            }
        }
        return max;
    }

    // LC1451
    public String arrangeWords(String text) {
        String[] arr = text.toLowerCase().split(" ");
        Integer[] relPos = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) relPos[i] = i;
        Arrays.sort(relPos, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return arr[o1].length() == arr[o2].length() ? o1 - o2 : arr[o1].length() - arr[o2].length();
            }
        });
        StringBuilder sb = new StringBuilder(text.length() + 2);
        for (int i : relPos) {
            sb.append(arr[i]);
            sb.append(" ");
        }
        sb.setCharAt(0, Character.toUpperCase(sb.charAt(0)));
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    // LC1869
    public boolean checkZeroOnes(String s) {
        int[] count = new int[]{0, 0};
        char[] cArr = s.toCharArray();
        char cur = cArr[0];
        int tmpCounter = 1;
        count[cArr[0] - '0'] = 1;
        for (int i = 1; i < cArr.length; i++) {
            if (cur != cArr[i]) {
                cur = cArr[i];
                tmpCounter = 1;
            } else {
                tmpCounter++;
            }
            count[cArr[i] - '0'] = Math.max(count[cArr[i] - '0'], tmpCounter);
        }
        return count[1] > count[0];
    }

    // LC274 275 **
    public int hIndex(int[] citations) {
        int n = citations.length;
        Arrays.sort(citations);
        int h = 0;
        for (int i = n - 1; i >= 0 && citations[i] > h; h++, i--) ;
        return h;
    }

    // LC467 **
    public int findSubstringInWraproundString(String p) {
        String alphabet = "abcdefghijklmnopqrstuvwxyz";
        char[] al = alphabet.toCharArray();
        int[] dp = new int[26];
        char[] cArr = p.toCharArray();
        dp[cArr[0] - 'a'] = 1;
        int k = 1;
        for (int i = 1; i < cArr.length; i++) {
            if (al[(cArr[i - 1] - 'a' + 1) % 26] == cArr[i]) {
                k++;
            } else {
                k = 1;
            }
            dp[cArr[i] - 'a'] = Math.max(dp[cArr[i] - 'a'], k);
        }
        int sum = 0;
        for (int i : dp) sum += i;
        return sum;
    }

    // LC1048
    Map<String, Integer> lc1048Map;
    Map<Integer, Set<String>> lc1048tm;

    public int longestStrChain(String[] words) {
        lc1048Map = new HashMap<>();
        lc1048tm = new TreeMap<>((o1, o2) -> o2 - o1);
        for (String w : words) {
            lc1048Map.put(w, 1);
            lc1048tm.putIfAbsent(w.length(), new HashSet<>());
            lc1048tm.get(w.length()).add(w);
        }
        // Hint: For each word in order of length, for each word2 which is word with one character removed, length[word2] = max(length[word2], length[word] + 1).
        Iterator<Integer> it = lc1048tm.keySet().iterator();
        int max = 1;
        while (it.hasNext()) {
            for (String w : lc1048tm.get(it.next())) {
                max = Math.max(max, lc1048Helper(w));
            }
        }
        return max;
    }

    private int lc1048Helper(String word) {
        for (int i = 1; i <= word.length(); i++) {
            String removed = word.substring(0, i - 1) + word.substring(i);
            if (lc1048Map.containsKey(removed)) {
                int lw2 = Math.max(lc1048Map.get(removed), lc1048Map.get(word) + 1);
                lc1048Map.put(removed, lw2);
            }
        }
        return lc1048Map.get(word);
    }

    // Interview 17.10 ** 摩尔投票算法
    public int majorityElement(int[] nums) {
        int count = 0;
        int major = -1;
        for (int i : nums) {
            if (count == 0) {
                major = i;
                count = 1;
            } else {
                if (i == major) {
                    count++;
                } else {
                    count--;
                }
            }
        }
        if (count <= 0) return -1;
        count = 0;
        for (int i : nums) {
            if (i == major) count++;
        }
        if (count > (nums.length / 2)) return major;
        return -1;
    }

    // LC1732
    public int largestAltitude(int[] gain) {
        int cur = 0;
        int max = 0;
        for (int i : gain) {
            cur += i;
            max = Math.max(max, cur);
        }
        return max;
    }
}

// LC745 **
class WordFilter {

    TrieNode trie;

    public WordFilter(String[] words) {
        trie = new TrieNode();
        for (int i = 0; i < words.length; i++) {
            String tmpWord = words[i] + "#";
            for (int j = 0; j < words[i].length(); j++) {
                TrieNode cur = trie;
                cur.idx = i;
                for (int k = j; k < 2 * tmpWord.length() - 1; k++) {
                    char x = tmpWord.charAt(k % tmpWord.length()); // (apple#) 循环两次, 终止于第二次到达#前
                    if (cur.children.get(x) == null) {
                        cur.children.put(x, new TrieNode());
                    }
                    cur = cur.children.get(x);
                    cur.idx = i; // 这里这个node自然存的就是下标最大的值
                }
            }
        }
    }

    public int f(String prefix, String suffix) {
        TrieNode cur = trie;
        for (char letter : (suffix + "#" + prefix).toCharArray()) {
            if (cur.children.get(letter) == null) return -1;
            cur = cur.children.get(letter);
        }
        return cur.idx;
    }

    class TrieNode {
        Map<Character, TrieNode> children;
        int idx;

        public TrieNode() {
            children = new HashMap<>();
            idx = 0;
        }
    }

}

// LC732
class MyCalendarThree {
    TreeMap<Integer, Integer> delta; // 相当于差分数组


    public MyCalendarThree() {
        delta = new TreeMap<>();
    }

    public int book(int start, int end) {
        delta.put(start, delta.getOrDefault(start, 0) + 1);
        delta.put(end, delta.getOrDefault(end, 0) - 1);

        int active = 0;
        int max = 0;
        for (int i : delta.values()) {
            active += i;
            max = Math.max(active, max);
        }
        return max;
    }
}

// LC731 **
class MyCalendarTwo {

    TreeMap<Integer, Integer> delta; // 相当于差分数组

    public MyCalendarTwo() {
        delta = new TreeMap<>();
    }

    public boolean book(int start, int end) {
        delta.put(start, delta.getOrDefault(start, 0) + 1);
        delta.put(end, delta.getOrDefault(end, 0) - 1);

        int active = 0;
        for (int i : delta.values()) {
            active += i;
            if (active >= 3) {
                delta.put(start, delta.get(start) - 1);
                delta.put(end, delta.get(end) + 1);
                if (delta.get(start) == 0) {
                    delta.remove(start);
                }
                return false;
            }
        }
        return true;
    }
}

// LC729
class MyCalendar {

    TreeMap<Integer, Integer> tm;

    public MyCalendar() {
        tm = new TreeMap<>();
    }

    // 前闭后开
    public boolean book(int start, int end) {
        Integer prev = tm.floorKey(start);
        Integer next = tm.ceilingKey(start);
        if ((prev == null || tm.get(prev) <= start) && (next == null || end <= tm.get(next))) {
            tm.put(start, end);
            return true;
        }
        return false;
    }
}

// LC677
class MapSum {
    Trie trie;
    Map<String, Integer> m;

    /**
     * Initialize your data structure here.
     */
    public MapSum() {
        trie = new Trie();
        m = new HashMap<>();
    }

    public void insert(String key, int val) {
        m.put(key, val);
        trie.addWord(key);
    }

    public int sum(String prefix) {
        if (!trie.beginWith(prefix)) return 0;
        return m.getOrDefault(prefix, 0) + dfs(prefix);
    }

    private int dfs(String prefix) {
        if (prefix.length() > 50) return 0;
        int result = 0;
        for (int i = 0; i < 26; i++) {
            String newPrefix = prefix + (char) (i + 'a');
            if (!trie.beginWith(newPrefix)) continue;
            if (trie.search(newPrefix)) result += m.get(newPrefix);
            result += dfs(newPrefix);
        }
        return result;
    }
}

// LC676 比较慢
class MagicDictionary {

    Trie trie;
    Set<Integer> length;

    /**
     * Initialize your data structure here.
     */
    public MagicDictionary() {
        trie = new Trie();
        length = new HashSet<>();
    }


    public void buildDict(String[] dictionary) {
        for (String word : dictionary) {
            trie.addWord(word);
            length.add(word.length());
        }
    }

    public boolean search(String searchWord) {
        if (!length.contains(searchWord.length())) return false;

        for (int i = 0; i < searchWord.length(); i++) {
            String prefix = searchWord.substring(0, i);
            if (prefix.length() != 0 && !trie.beginWith(prefix)) break;
            char curChar = searchWord.charAt(i);
            String suffix = searchWord.substring(i + 1);
            for (int j = 0; j < 26; j++) {
                if (curChar == (char) ('a' + j)) continue;
                String newWord = prefix + (char) ('a' + j) + suffix;
                if (trie.search(newWord)) return true;
            }
        }
        return false;
    }
}

// LC449 BST Serialize / Deserialized
class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        preOrder(root, sb);
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    private void preOrder(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#,");
            return;
        }
        sb.append(root.val + ",");
        preOrder(root.left, sb);
        preOrder(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] split = data.split(",");
        List<String> list = new LinkedList<>();
        for (String s : split) list.add(s);
        return preOrderDes(list);
    }

    private TreeNode preOrderDes(List<String> list) {
        if (list.get(0).equals("#")) {
            list.remove(0);
            return null;
        }
        TreeNode root = new TreeNode(Integer.valueOf(list.get(0)));
        list.remove(0);
        root.left = preOrderDes(list);
        root.right = preOrderDes(list);
        return root;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// LC432 没有全O(1), 插入删除O(logN) ; 全O(1)要手写双向链表
class AllOne {
    Map<String, Integer> map;
    TreeMap<Integer, Set<String>> treeMap;

    /**
     * Initialize your data structure here.
     */
    public AllOne() {
        map = new HashMap<>();
        treeMap = new TreeMap<>();
    }

    /**
     * Inserts a new key <Key> with value 1. Or increments an existing key by 1.
     */
    public void inc(String key) {
        if (map.containsKey(key)) {
            treeMap.get(map.get(key)).remove(key);
            if (treeMap.get(map.get(key)).size() == 0) treeMap.remove(map.get(key));
            map.put(key, map.get(key) + 1);
            treeMap.putIfAbsent(map.get(key), new HashSet<>());
            treeMap.get(map.get(key)).add(key);
        } else {
            map.put(key, 1);
            treeMap.putIfAbsent(1, new HashSet<>());
            treeMap.get(1).add(key);
        }
    }

    /**
     * Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
     */
    public void dec(String key) {
        if (map.containsKey(key)) {
            treeMap.get(map.get(key)).remove(key);
            if (treeMap.get(map.get(key)).size() == 0) treeMap.remove(map.get(key));
            map.put(key, map.get(key) - 1);
            if (map.get(key) == 0) {
                map.remove(key);
            } else {
                treeMap.putIfAbsent(map.get(key), new HashSet<>());
                treeMap.get(map.get(key)).add(key);
            }
        }
    }

    /**
     * Returns one of the keys with maximal value.
     */
    public String getMaxKey() {
        try {
            Integer maxVal = treeMap.lastKey();
            if (maxVal == null) return "";
            Iterator<String> it = treeMap.get(maxVal).iterator();
            return it.next();
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Returns one of the keys with Minimal value.
     */
    public String getMinKey() {
        try {
            Integer minVal = treeMap.firstKey();
            if (minVal == null) return "";
            Iterator<String> it = treeMap.get(minVal).iterator();
            return it.next();
        } catch (Exception e) {
            return "";
        }
    }
}

// LC381 ** from Solution
class RandomizedCollection {
    List<Integer> nums;
    Map<Integer, Set<Integer>> idx;
    Random r;

    /**
     * Initialize your data structure here.
     */
    public RandomizedCollection() {
        r = new Random();
        idx = new HashMap<>();
        nums = new ArrayList<>();
    }

    /**
     * Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
     */
    public boolean insert(int val) {
        idx.putIfAbsent(val, new HashSet<>());
        nums.add(val);
        idx.get(val).add(nums.size() - 1);
        return idx.get(val).size() == 1;
    }

    /**
     * Removes a value from the collection. Returns true if the collection contained the specified element.
     */
    public boolean remove(int val) {
        if (!idx.containsKey(val)) {
            return false;
        }
        Iterator<Integer> it = idx.get(val).iterator();
        int possibleIdx = it.next();
        int lastNum = nums.get(nums.size() - 1);
        nums.set(possibleIdx, lastNum);
        idx.get(val).remove(possibleIdx);
        idx.get(lastNum).remove(nums.size() - 1);
        if (possibleIdx != nums.size() - 1) { // 注意合理值判断, 如果取到的待删除下标恰好是列表的最后一个数的下标, 则不需要录入新的下标信息
            idx.get(lastNum).add(possibleIdx);
        }
        nums.remove(nums.size() - 1);
        if (idx.get(val).size() == 0) idx.remove(val);
        return true;
    }

    /**
     * Get a random element from the collection.
     */
    public int getRandom() {
        int idx = r.nextInt(nums.size());
        return nums.get(idx);
    }
}

// LC981
class TimeMap {
    Map<String, TreeMap<Integer, String>> m;

    /**
     * Initialize your data structure here.
     */
    public TimeMap() {
        m = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        m.putIfAbsent(key, new TreeMap<>());
        m.get(key).put(timestamp, value);
    }

    public String get(String key, int timestamp) {
        if (!m.containsKey(key)) return "";
        return m.get(key).floorEntry(timestamp) == null ? "" : m.get(key).floorEntry(timestamp).getValue();
    }
}

// LC355
class Twitter {

    Map<Integer, Set<Integer>> follow;
    Map<Integer, List<Pair<Long, Integer>>> tweets;
    long time;

    /**
     * Initialize your data structure here.
     */
    public Twitter() {
        time = 0;
        follow = new HashMap<>();
        tweets = new HashMap<>();
    }

    /**
     * Compose a new tweet.
     */
    public void postTweet(int userId, int tweetId) {
        if (!follow.containsKey(userId)) follow.put(userId, new HashSet<>());
        if (!tweets.containsKey(userId)) tweets.put(userId, new LinkedList<>());

        tweets.get(userId).add(0, new Pair<>(getTime(), tweetId));
    }

    /**
     * Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
     */
    public List<Integer> getNewsFeed(int userId) {
        if (!follow.containsKey(userId)) follow.put(userId, new HashSet<>());
        if (!tweets.containsKey(userId)) tweets.put(userId, new LinkedList<>());
        PriorityQueue<Pair<Long, Integer>> pq = new PriorityQueue<>(Comparator.comparingLong(o -> o.getKey()));
        final int maxSize = 10;
        for (Pair<Long, Integer> t : tweets.get(userId)) {
            if (pq.size() < maxSize) {
                pq.offer(t);
            }
        }
        for (int fid : follow.get(userId)) {
            for (Pair<Long, Integer> t : tweets.get(fid)) {
                if (pq.size() < maxSize) {
                    pq.offer(t);
                } else {
                    if (pq.peek().getKey() < t.getKey()) {
                        pq.poll();
                        pq.offer(t);
                    } else {
                        break;
                    }
                }
            }
        }
        List<Integer> result = new LinkedList<>();
        while (!pq.isEmpty()) {
            result.add(0, pq.poll().getValue());
        }
        return result;
    }

    /**
     * Follower follows a followee. If the operation is invalid, it should be a no-op.
     */
    public void follow(int followerId, int followeeId) {
        if (!follow.containsKey(followerId)) follow.put(followerId, new HashSet<>());
        if (!follow.containsKey(followeeId)) follow.put(followeeId, new HashSet<>());
        if (!tweets.containsKey(followerId)) tweets.put(followerId, new LinkedList<>());
        if (!tweets.containsKey(followeeId)) tweets.put(followeeId, new LinkedList<>());

        follow.get(followerId).add(followeeId);
    }

    /**
     * Follower unfollows a followee. If the operation is invalid, it should be a no-op.
     */
    public void unfollow(int followerId, int followeeId) {
        if (!follow.containsKey(followerId)) return;
        follow.get(followerId).remove(followeeId);
    }

    private long getTime() {
        return time++;
    }
}

// LC715 ** from Solution
class RangeModule {
    TreeSet<Pair<Integer, Integer>> ts;

    public RangeModule() {
        ts = new TreeSet<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                return o1.getValue() == o2.getValue() ? o1.getKey() - o2.getKey() : o1.getValue() - o2.getValue();
            }
        });
    }

    public void addRange(int left, int right) {
        Iterator<Pair<Integer, Integer>> it = ts.tailSet(new Pair<>(0, left), false).iterator();
        while (it.hasNext()) {
            Pair<Integer, Integer> r = it.next();
            if (right < r.getKey()) break;
            left = Math.min(r.getKey(), left);
            right = Math.max(r.getValue(), right);
            it.remove();
            ts.remove(r);
        }
        ts.add(new Pair<>(left, right));
    }

    public boolean queryRange(int left, int right) {
        Pair<Integer, Integer> r = ts.higher(new Pair<>(0, left));
        return r != null && r.getKey() <= left && right <= r.getValue();
    }

    public void removeRange(int left, int right) {
        Iterator<Pair<Integer, Integer>> it = ts.tailSet(new Pair<>(0, left), true).iterator();
        List<Pair<Integer, Integer>> toAdd = new LinkedList<>();
        while (it.hasNext()) {
            Pair<Integer, Integer> r = it.next();
            if (r.getKey() > right) break;
            if (r.getKey() < left) toAdd.add(new Pair<>(r.getKey(), left));
            if (r.getValue() > right) toAdd.add(new Pair<>(right, r.getValue()));
            it.remove();
            ts.remove(r);
        }
        for (Pair<Integer, Integer> p : toAdd) ts.add(p);
    }
}

// LC352
class SummaryRanges {

    TreeSet<Pair<Integer, Integer>> leftSide;
    TreeSet<Pair<Integer, Integer>> rightSide;

    /**
     * Initialize your data structure here.
     */
    public SummaryRanges() {
        leftSide = new TreeSet<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                return o1.getKey().compareTo(o2.getKey());
            }
        });
        rightSide = new TreeSet<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });
    }

    public void addNum(int val) {
        Pair<Integer, Integer> p = new Pair<>(val, val);
        Pair<Integer, Integer> lsf = leftSide.floor(p);
        Pair<Integer, Integer> rsc = rightSide.ceiling(p);
        if (lsf != null && rsc != null && lsf.getValue() + 1 == val && rsc.getKey() - 1 == val) {
            // merge lsf rsc
            leftSide.remove(rsc);
            leftSide.remove(lsf);
            rightSide.remove(rsc);
            rightSide.remove(lsf);
            Pair<Integer, Integer> n = new Pair<>(lsf.getKey(), rsc.getValue());
            leftSide.add(n);
            rightSide.add(n);
        } else if (lsf != null && lsf.getValue() + 1 == val) {
            leftSide.remove(lsf);
            rightSide.remove(lsf);
            Pair<Integer, Integer> n = new Pair<>(lsf.getKey(), val);
            leftSide.add(n);
            rightSide.add(n);
        } else if (rsc != null && rsc.getKey() - 1 == val) {
            leftSide.remove(rsc);
            rightSide.remove(rsc);
            Pair<Integer, Integer> n = new Pair<>(val, rsc.getValue());
            leftSide.add(n);
            rightSide.add(n);
        } else if ((lsf != null && val <= lsf.getValue()) || (rsc != null && val >= rsc.getKey())) {
            ;
        } else {
            leftSide.add(p);
            rightSide.add(p);
        }

    }

    public int[][] getIntervals() {
        int[][] result = new int[leftSide.size()][];
        int ctr = 0;
        for (Pair<Integer, Integer> p : leftSide) {
            result[ctr++] = new int[]{p.getKey(), p.getValue()};
        }
        return result;
    }
}

// LC211
class WordDictionary {

    Set<String> s;
    Trie trie;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        s = new HashSet<>();
        trie = new Trie();
    }

    public void addWord(String word) {
        s.add(word);
        trie.addWord(word);
    }

    public boolean search(String word) {
        return searchHelper("", word);
    }

    private boolean searchHelper(String prefix, String suffix) {
        if (suffix.equals("")) {
            return s.contains(prefix);
        }
        StringBuilder prefixSb = new StringBuilder(prefix);
        for (int i = 0; i < suffix.length(); i++) {
            if (suffix.charAt(i) != '.') {
                prefixSb.append(suffix.charAt(i));
            } else {
                for (int j = 0; j < 26; j++) {
                    prefixSb.append((char) ('a' + j));
                    if (!trie.beginWith(prefixSb.toString())) { // 用Trie剪枝
                        prefixSb.deleteCharAt(prefixSb.length() - 1);
                        continue;
                    }
                    String suf = suffix.substring(i + 1);
                    if (searchHelper(prefixSb.toString(), suf)) return true;
                    prefixSb.deleteCharAt(prefixSb.length() - 1);
                }
                return false;
            }
        }
        return s.contains(prefixSb.toString());
    }
}

class Trie {
    Map<String, Boolean> m;

    public Trie() {
        m = new HashMap<>();
    }

    public void addWord(String word) {
        for (int i = 1; i < word.length(); i++) {
            if (!m.getOrDefault(word.substring(0, i), false)) {
                m.put(word.substring(0, i), false);
            }
        }
        m.put(word, true);
    }

    public boolean search(String word) {
        return m.getOrDefault(word, false);
    }

    public boolean beginWith(String word) {
        return m.containsKey(word);
    }
}

// JZOF 59
class MaxQueue {

    Deque<Integer> q;
    Deque<Integer> dq;

    public MaxQueue() {
        q = new LinkedList<>();
        dq = new LinkedList<>();
    }

    public int max_value() {
        if (q.size() == 0) return -1;
        return dq.peekFirst();
    }

    public void push_back(int value) {
        while (!dq.isEmpty() && dq.peekLast() < value) {
            dq.pollLast();
        }
        dq.offer(value);
        q.offer(value);
    }

    public int pop_front() {
        if (q.isEmpty()) return -1;
        int victim = q.poll();
        if (victim == dq.peekFirst()) dq.pollFirst();
        return victim;
    }
}

// LC460
class LFUCache {
    Map<Integer, Node> keyValue;
    TreeMap<Integer, LinkedList<Node>> freqMap;
    int capacity;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        keyValue = new HashMap<>();
        freqMap = new TreeMap<>();
    }

    public int get(int key) {
        if (capacity == 0) return -1;
        if (!keyValue.containsKey(key)) return -1;
        Node n = keyValue.get(key);
        int oldFreq = n.freq;
        int newFreq = oldFreq + 1;
        int value = n.value;
        freqMap.get(oldFreq).remove(n); // 这步应该是O(n)的
        if (freqMap.get(oldFreq).size() == 0) freqMap.remove(oldFreq);
        n.freq = newFreq;
        freqMap.putIfAbsent(newFreq, new LinkedList<>());
        freqMap.get(newFreq).offerLast(n);
        keyValue.put(key, n);
        return value;
    }

    public void put(int key, int value) {
        if (capacity == 0) return;
        if (!keyValue.containsKey(key)) {
            if (keyValue.size() == capacity) {
                int smallestFreq = freqMap.firstKey();
                Node victim = freqMap.get(smallestFreq).pollFirst();
                keyValue.remove(victim.key);
                if (freqMap.get(smallestFreq).size() == 0) {
                    freqMap.remove(smallestFreq);
                }
            }
            Node newcomer = new Node(1, key, value);
            LinkedList<Node> freqOneList = freqMap.getOrDefault(1, new LinkedList<>());
            freqOneList.offerLast(newcomer);
            keyValue.put(key, newcomer);
            freqMap.put(1, freqOneList);
        } else {
            Node n = keyValue.get(key);
            int oldFreq = n.freq;
            int newFreq = oldFreq + 1;
            freqMap.get(oldFreq).remove(n); // 这步应该是O(n)的
            if (freqMap.get(oldFreq).size() == 0) freqMap.remove(oldFreq);
            n.freq = newFreq;
            n.value = value;
            freqMap.putIfAbsent(newFreq, new LinkedList<>());
            freqMap.get(newFreq).offerLast(n);
            keyValue.put(key, n);
        }
    }

    class Node {
        int freq, key, value;

        public Node(int freq, int key, int value) {
            this.freq = freq;
            this.key = key;
            this.value = value;
        }
    }
}

// LC225
class MyStack {
    Queue<Integer> q;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
        q = new LinkedList<>();
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        int size = q.size();
        q.offer(x);
        for (int i = 0; i < size; i++) {
            q.offer(q.poll());
        }
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        return q.poll();
    }

    /**
     * Get the top element.
     */
    public int top() {
        return q.peek();
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return q.isEmpty();
    }
}