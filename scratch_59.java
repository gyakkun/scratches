import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {

    }

    // LC630 ** Hard
    public int scheduleCourse(int[][] courses) {
        // [duration, lastDay]
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o[0]));
        Arrays.sort(courses, Comparator.comparingInt(o -> o[1]));
        int totalTime = 0;
        for (int[] i : courses) {
            if (totalTime + i[0] <= i[1]) {
                pq.offer(i);
                totalTime += i[0];
            } else {
                if (pq.isEmpty()) continue;
                int[] p = pq.peek();
                if (p[0] > i[0]) {
                    int delta = p[0] - i[0];
                    pq.poll();
                    pq.offer(i);
                    totalTime -= delta;
                }
            }
        }
        return pq.size();
    }
}

// LC911
class TopVotedCandidate {
    TreeMap<Integer, Integer> timeCanMap = new TreeMap<>();

    public TopVotedCandidate(int[] persons, int[] times) {
        int n = persons.length;
        int[] canFreqMap = new int[n + 1];
        TreeMap<Integer, Integer> freqCanMap = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            int oldFreq = canFreqMap[persons[i]];
            int newFreq = oldFreq + 1;
            canFreqMap[persons[i]] = newFreq;
            freqCanMap.put(newFreq, persons[i]);
            timeCanMap.put(times[i], freqCanMap.lastEntry().getValue());
        }
    }

    public int q(int t) {
        return timeCanMap.floorEntry(t).getValue();
    }
}