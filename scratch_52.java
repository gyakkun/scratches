import java.util.Comparator;
import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.maxEqualFreq(new int[]{2, 2, 1, 1, 5, 3, 3, 5}));
        System.out.println(s.maxEqualFreq(new int[]{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5}));
        System.out.println(s.maxEqualFreq(new int[]{1, 1, 1, 2, 2, 2}));
        System.out.println(s.maxEqualFreq(new int[]{10, 2, 8, 9, 3, 8, 1, 5, 2, 3, 7, 6}));
        System.out.println(s.maxEqualFreq(new int[]{1, 1, 1, 2, 2, 2, 3, 3, 3}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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
                ;
            } else if (freqIntSetMap.keySet().size() == 2) {
                // 情况2: size == 2 时候, 看哪个set 的size ==1
                Iterator<Integer> it = freqIntSetMap.keySet().iterator();
                int freq1 = it.next(), freq2 = it.next();
                int smallFreq = freq1 < freq2 ? freq1 : freq2;
                int largeFreq = smallFreq == freq1 ? freq2 : freq1;
                Set<Integer> smallFreqSet = freqIntSetMap.get(smallFreq), largeFreqSet = freqIntSetMap.get(largeFreq);
                if (smallFreqSet.size() != 1 && largeFreqSet.size() != 1) {
                    ;
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
                        // 特判一下 111 22 这种情况
                    else if (anotherSet.size() == 1) {
                        if (anotherFreq == 1) return i + 1;
                        else if (anotherFreq == eleFreq + 1) return i + 1;
                    }
                    // 否则没办法
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