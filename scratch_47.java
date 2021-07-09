import javafx.util.Pair;
import org.bouncycastle.asn1.cmc.PopLinkWitnessV2;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        Twitter t = new Twitter();
        t.postTweet(1, 1);
        t.getNewsFeed(1);
        t.follow(2, 1);
        t.getNewsFeed(2);
        t.unfollow(2, 1);
        t.getNewsFeed(2);


//        System.out.println(s.largestAltitude(new int[]{-4, -3, -2, -1, 4, 3, 2}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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