import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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
        if(dogQueue.isEmpty()) return new int[]{-1, -1};
        int polledDogId = dogQueue.poll();
        idSeqMap.remove(polledDogId);
        return new int[]{polledDogId, DOG};
    }

    public int[] dequeueCat() {
        if(catQueue.isEmpty()) return new int[]{-1, -1};
        int polledCatId = catQueue.poll();
        idSeqMap.remove(polledCatId);
        return new int[]{polledCatId, CAT};
    }

    private int getSeq() {
        return seq++;
    }
}
