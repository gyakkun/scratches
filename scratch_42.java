import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        int[] arr = new int[]{1, 4, 2, 3};
        int k = 4;
        System.err.println(
                s.letterCasePermutation("12345")
        );
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC784
    public List<String> letterCasePermutation(String S) {
        List<String> result = new ArrayList<>();
        List<Integer> letterIdx = new ArrayList<>();
        for (int i = 0; i < S.length(); i++) {
            if (Character.isLetter(S.charAt(i))) {
                letterIdx.add(i);
            }
        }
        char[] lowerCase = new char[]{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
        char[] upperCase = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
        int maxMask = 1 << letterIdx.size();
        for (int mask = 0; mask < maxMask; mask++) {
            StringBuffer sb = new StringBuffer(S);
            for (int i = 0; i < letterIdx.size(); i++) {
                // 1 变 0 不变
                if (((mask >> i) & 1) == 1) {
                    int idx = letterIdx.get(i);
                    char c = S.charAt(idx);
                    sb.setCharAt(idx, Character.isLowerCase(c) ? upperCase[c - 'a'] : lowerCase[c - 'A']);
                }
            }
            result.add(sb.toString());
        }

        return result;
    }

    // LC480
    PriorityQueue<Long> maxHeap;
    PriorityQueue<Long> minHeap;

    public double[] medianSlidingWindow(int[] nums, int k) {
        minHeap = new PriorityQueue<>(Comparator.comparingLong(o -> o));
        maxHeap = new PriorityQueue<>(Comparator.comparingLong(o -> -o));

        int len = nums.length;
        double[] result = new double[len - k + 1];
        for (int i = 0; i < k; i++) {
            addNum(nums[i]);
        }
        for (int i = 0; i < len - k + 1; i++) {
            if (minHeap.size() == maxHeap.size()) {
                long sum = (minHeap.peek() + maxHeap.peek());
                result[i] = ((double) (sum)) / 2;
            } else {
                result[i] = (double) maxHeap.peek();
            }
            removeNum(nums[i]);
            if (i + k < len) {
                addNum(nums[i + k]);
            }
        }
        return result;

    }

    private void addNum(long i) {
        if (maxHeap.isEmpty() || i < maxHeap.peek()) {
            maxHeap.offer(i);
        } else {
            minHeap.offer(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    private void removeNum(long i) {
        if (maxHeap.contains(i)) {
            maxHeap.remove(i);
        } else {
            minHeap.remove(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

}