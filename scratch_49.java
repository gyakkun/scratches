import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.findNumberOfLIS(new int[]{2, 1}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC673 **
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        int[] dp = new int[n], count = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(count, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[i] <= dp[j]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
            }
        }
        int max = Arrays.stream(dp).max().getAsInt();
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == max) {
                result += count[i];
            }
        }
        return result;
    }
}

// Interview 03.06
class AnimalShelf {
    int seq = 0;
    // type 0-cat 1-dog
    final int CAT = 0, DOG = 1;
    Map<Integer, Integer> idSeqMap = new HashMap<>();
    Deque<Integer> catQueue = new LinkedList<>();
    Deque<Integer> dogQueue = new LinkedList<>();

    public AnimalShelf() {

    }

    public void enqueue(int[] a) {
        // a[0] = id, a[1] = type
        int sequence = getSeq();
        idSeqMap.put(a[0], sequence);
        if (a[1] == CAT) {
            catQueue.offer(a[0]);
        } else {
            dogQueue.offer(a[0]);
        }
    }

    public int[] dequeueAny() {
        if (catQueue.isEmpty() && dogQueue.isEmpty()) {
            return new int[]{-1, -1};
        } else if (catQueue.isEmpty() && !dogQueue.isEmpty()) {
            return dequeueDog();
        } else if (!catQueue.isEmpty() && dogQueue.isEmpty()) {
            return dequeueCat();
        } else if (idSeqMap.get(catQueue.peek()) < idSeqMap.get(dogQueue.peek())) {
            return dequeueCat();
        } else {
            return dequeueDog();
        }

    }

    public int[] dequeueDog() {
        if (dogQueue.isEmpty()) return new int[]{-1, -1};
        int polledDogId = dogQueue.poll();
        idSeqMap.remove(polledDogId);
        return new int[]{polledDogId, DOG};
    }

    public int[] dequeueCat() {
        if (catQueue.isEmpty()) return new int[]{-1, -1};
        int polledCatId = catQueue.poll();
        idSeqMap.remove(polledCatId);
        return new int[]{polledCatId, CAT};
    }

    private int getSeq() {
        return seq++;
    }
}
