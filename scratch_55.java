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
        List<int[]> intvs = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            intvs.add(new int[]{i - ranges[i], i + ranges[i]});
        }
        Collections.sort(intvs, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);
        TreeSet<int[]> ts = new TreeSet<>(Comparator.comparingInt(o -> o[0]));
        for (int[] i : intvs) ts.add(i);
        int result = Integer.MAX_VALUE;
        loop: for (int[] i : intvs) { // 从i开始
            if (i[0] > 0) break;
            List<int[]> accepted = new ArrayList<>();
            accepted.add(i);
            while (accepted.get(accepted.size() - 1)[1] < n) {
                int[] last = accepted.get(accepted.size() - 1);
                NavigableSet<int[]> intersect = ts.subSet(new int[]{last[0], 0}, false, new int[]{last[1], 0}, true);
                if (intersect.isEmpty()) break loop;
                int[] candidate = null;
                int rightMost = last[1];
                for (int[] j : intersect) {
                    if (j[1] > rightMost) {
                        candidate = j;
                        rightMost = j[1];
                    }
                }
                if (candidate == null) break;
                accepted.add(candidate);
            }
            if (accepted.get(accepted.size() - 1)[1] < n) break;
            result = Math.min(result, accepted.size());
            if (result == 1) return 1;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }
}