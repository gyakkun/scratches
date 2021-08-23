import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println();

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1223
    public List<String> removeSubfolders(String[] folder) {
        Set<String> prefix = new HashSet<>();
        Arrays.sort(folder, Comparator.comparingInt(o -> o.length()));
        for (String f : folder) {
            int ptr = 0;
            while (ptr < f.length()) {
                int last = ptr + 1;
                while (last != f.length() && f.charAt(last) != '/') last++;
                if (prefix.contains(f.substring(0, last))) break;
                ptr = last;
            }
            if (ptr == f.length()) prefix.add(f);
        }
        return new ArrayList<>(prefix);
    }
}

// LC 1226 哲学家进餐
class DiningPhilosophers {

    ReentrantLock[] locks = new ReentrantLock[]{
            new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock()
    };

    ReentrantLock pickBoth = new ReentrantLock();


    public DiningPhilosophers() {

    }

    // call the run() method of any runnable to execute its code
    public void wantsToEat(int philosopher,
                           Runnable pickLeftFork,
                           Runnable pickRightFork,
                           Runnable eat,
                           Runnable putLeftFork,
                           Runnable putRightFork) throws InterruptedException {
        int left = (philosopher + 1) % 5, right = philosopher;

        pickBoth.lock();

        locks[left].lock();
        locks[right].lock();
        pickLeftFork.run();
        pickRightFork.run();

        pickBoth.unlock();

        eat.run();

        putLeftFork.run();
        putRightFork.run();

        locks[left].unlock();
        locks[right].unlock();
    }
}