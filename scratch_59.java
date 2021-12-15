import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {

    }

    // LC851 **
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        int[] result = new int[n], indegree = new int[n];
        List<List<Integer>> mtx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int i = 0; i < n; i++) result[i] = i;
        for (int[] r : richer) { // r: [a,b] a is richer than b
            mtx.get(r[0]).add(r[1]); // a->b, first a (richer) then b
            indegree[r[1]]++;
        }
        Deque<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i++) if (indegree[i] == 0) q.offer(i);
        // result[x] = y, means that among people who are equal or richer than x, y is the quietest one
        while (!q.isEmpty()) {
            int p = q.poll();
            for (int next : mtx.get(p)) {
                if (quiet[result[p]] < quiet[result[next]]) {
                    result[next] = result[p];
                }
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        return result;
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