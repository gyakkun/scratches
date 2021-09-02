import javafx.util.Pair;

import java.util.Comparator;
import java.util.*;
import java.util.function.Function;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
//        System.out.println("a".substring(0));
        System.out.println(s.palindromePairs(new String[]{"a", ""}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC336 **
    public List<List<Integer>> palindromePairs(String[] words) {
        Set<Pair<Integer, Integer>> result = new HashSet<>();
        int wLen = words.length;
        String[] rWords = new String[wLen];
        Map<String, Integer> rWordIdx = new HashMap<>();
        for (int i = 0; i < wLen; i++) {
            rWords[i] = new StringBuilder(words[i]).reverse().toString();
            rWordIdx.put(rWords[i], i);
        }
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            int len = cur.length();
            if (len == 0) continue;
            for (int j = 0; j <= len; j++) { // 注意边界, 为了取到空串, 截取长度可以去到len, 同时为了去重用到Set<Pair<>>
                if (checkPal(cur, j, len)) {
                    int leftId = rWordIdx.getOrDefault(cur.substring(0, j), -1);
                    if (leftId != -1 && leftId != i) {
                        result.add(new Pair<>(i, leftId));
                    }
                }
                if (checkPal(cur, 0, j)) {
                    int rightId = rWordIdx.getOrDefault(cur.substring(j), -1);
                    if (rightId != -1 && rightId != i) {
                        result.add(new Pair<>(rightId, i));
                    }
                }
            }
        }
        List<List<Integer>> listResult = new ArrayList<>(result.size());
        for (Pair<Integer, Integer> p : result) {
            listResult.add(Arrays.asList(p.getKey(), p.getValue()));
        }
        return listResult;
    }

    private boolean checkPal(String s, int startIdx, int endIdxExclude) {
        if (startIdx > endIdxExclude) return false;
        if (startIdx == endIdxExclude) return true;
        int len = endIdxExclude - startIdx;
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(startIdx + i) != s.charAt(endIdxExclude - 1 - i)) return false;
        }
        return true;
    }

    // LC747
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int[] idxMap = new int[101];
        for (int i = 0; i < nums.length; i++) {
            idxMap[nums[i]] = i;
        }
        Arrays.sort(nums);
        if (nums[nums.length - 1] >= nums[nums.length - 2] * 2) return idxMap[nums[nums.length - 1]];
        return -1;
    }

    // LC1224
    public int maxEqualFreq(int[] nums) {
        Map<Integer, Integer> numFreqMap = new HashMap<>();
        for (int i : nums) {
            numFreqMap.put(i, numFreqMap.getOrDefault(i, 0) + 1);
        }
        Map<Integer, Set<Integer>> freqIntSetMap = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : numFreqMap.entrySet()) {
            freqIntSetMap.putIfAbsent(e.getValue(), new HashSet<>());
            freqIntSetMap.get(e.getValue()).add(e.getKey());
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            // 当前元素
            int curEle = nums[i];

            // 看该不该删除

            // 情况1: freqMap.keySet.size > 2 此时删除哪个都没用
            if (freqIntSetMap.keySet().size() > 2) {
                continue;
            } else if (freqIntSetMap.keySet().size() == 2) {
                // 情况2: size == 2 时候, 看哪个set 的size ==1
                Iterator<Integer> it = freqIntSetMap.keySet().iterator();
                int freq1 = it.next(), freq2 = it.next();
                int smallFreq = freq1 < freq2 ? freq1 : freq2;
                int largeFreq = smallFreq == freq1 ? freq2 : freq1;
                Set<Integer> smallFreqSet = freqIntSetMap.get(smallFreq), largeFreqSet = freqIntSetMap.get(largeFreq);
                // 如果两个set都有超过一个元素, 则删除哪个元素都没用
                if (smallFreqSet.size() != 1 && largeFreqSet.size() != 1) {
                    continue;
                } else {
                    Set<Integer> oneEleSet = smallFreqSet.size() == 1 ? smallFreqSet : largeFreqSet;
                    Set<Integer> anotherSet = oneEleSet == smallFreqSet ? largeFreqSet : smallFreqSet;

                    int oneEle = oneEleSet.iterator().next();
                    int eleFreq = numFreqMap.get(oneEle);
                    int anotherFreq = eleFreq == smallFreq ? largeFreq : smallFreq;

                    // 情况1： 这个元素的当前频率是1
                    if (eleFreq == 1) return i + 1;
                        // 情况2: 当前元素的频率比另一个频率大1
                    else if (eleFreq == anotherFreq + 1) return i + 1;
                        // 特判一下 111 22 这种情况, 即两个freq的set的大小都是1
                        // 前面只判断了2不能删除, 没有判断1能不能删除, 此处补充判断一次
                    else if (anotherSet.size() == 1) {
                        if (anotherFreq == 1) return i + 1;
                        else if (anotherFreq == eleFreq + 1) return i + 1;
                    }
                    // 否则没办法 只能删除当前元素
                }
            }

            // 若没有找到该删除的 就删除当前元素
            int curFreq = numFreqMap.get(curEle);
            int nextFreq = curFreq - 1;
            numFreqMap.put(curEle, nextFreq);
            freqIntSetMap.get(curFreq).remove(nums[i]);
            if (freqIntSetMap.get(curFreq).size() == 0) freqIntSetMap.remove(curFreq);
            if (nextFreq != 0) {
                freqIntSetMap.putIfAbsent(nextFreq, new HashSet<>());
                freqIntSetMap.get(nextFreq).add(nums[i]);
            } else {
                numFreqMap.remove(nums[i]);
            }

        }
        return nums.length;
    }

    // JZOF 22
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // LC1705
    public int eatenApples(int[] apples, int[] days) {
        // pq 存数对 [i,j], i表示苹果数量, j表示过期时间
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        int n = apples.length;
        int result = 0;
        int i = 0;
        do {
            if (i < n) {
                if (apples[i] == 0 && days[i] == 0) {
                    ;
                } else if (apples[i] != 0) {
                    pq.offer(new int[]{apples[i], days[i] + i});
                }
            }
            if (!pq.isEmpty()) {
                int[] entry = null;
                do {
                    int[] p = pq.poll();
                    if (i >= p[1]) continue;
                    entry = p;
                    break;
                } while (!pq.isEmpty());
                if (entry != null) {
                    entry[0]--;
                    result++;
                    if (entry[0] > 0) pq.offer(entry);
                }
            }
            i++;
        } while (!pq.isEmpty() || i < n);
        return result;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Trie {
    Trie[] children = new Trie[26];
    boolean isEnd = false;

    public void addWord(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new Trie();
            }
            cur = cur.children[c - 'a'];
        }
        cur.isEnd = true;
    }

    public boolean startsWith(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return true;
    }

    public boolean search(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return cur.isEnd;
    }
}