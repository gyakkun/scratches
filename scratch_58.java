import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.largestValsFromLabels(new int[]{9, 8, 8, 7, 6},
                new int[]{0, 0, 0, 1, 1},
                3,
                3));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1090
    public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
        int n = values.length;
        List<int[]> idxLabelSet = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            idxLabelSet.add(new int[]{i, labels[i], values[i]});
        }
        Collections.sort(idxLabelSet, Comparator.comparingInt(o -> -o[2]));
        int totalCount = 0, sum = 0;
        Map<Integer, Integer> labelFreq = new HashMap<>();
        for (int[] p : idxLabelSet) {
            int idx = p[0], label = p[1], val = p[2];
            if (labelFreq.getOrDefault(label, 0) == useLimit) continue;
            if (totalCount == numWanted) break;
            totalCount++;
            sum += val;
            labelFreq.put(label, labelFreq.getOrDefault(label, 0) + 1);
        }
        return sum;
    }
}