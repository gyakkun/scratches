import javafx.util.Pair;

import java.math.BigInteger;
import java.util.*;
import java.util.List;
import java.util.function.Function;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.minTaps(7, new int[]{1, 2, 1, 0, 2, 1, 0, 1}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1326
    public int minTaps(int n, int[] ranges) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i <= n; i++) {
            tm.put(i - ranges[i], Math.max(tm.getOrDefault(i - ranges[i], Integer.MIN_VALUE), i + ranges[i]));
        }
        int result = Integer.MAX_VALUE;
        loop:
        for (Map.Entry<Integer, Integer> i : tm.entrySet()) { // 从i开始
            if (i.getKey() > 0) break;
            LinkedList<Map.Entry<Integer, Integer>> candidateQueue = new LinkedList<>();
            candidateQueue.add(i);
            while (candidateQueue.getLast().getValue() < n) {
                Map.Entry<Integer, Integer> last = candidateQueue.getLast();
                NavigableMap<Integer, Integer> intersect = tm.subMap(last.getKey(), false, last.getValue(), true);
                if (intersect.isEmpty()) break loop;
                Map.Entry<Integer, Integer> candidate = null;
                int rightMost = last.getValue();
                for (Map.Entry<Integer, Integer> j : intersect.entrySet()) {
                    if (j.getValue() > rightMost) {
                        candidate = j;
                        rightMost = j.getValue();
                    }
                }
                if (candidate == null) break;
                candidateQueue.add(candidate);
            }
            if (candidateQueue.getLast().getValue() < n) break;
            result = Math.min(result, candidateQueue.size());
            if (result == 1) return 1;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }
}