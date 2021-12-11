import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {

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