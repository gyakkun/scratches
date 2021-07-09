import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        SummaryRanges sr = new SummaryRanges();
        sr.addNum(1);
        sr.addNum(9);
        sr.addNum(2);
        sr.getIntervals();
//        System.out.println(s.largestAltitude(new int[]{-4, -3, -2, -1, 4, 3, 2}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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