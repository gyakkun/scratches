import java.util.ArrayList;
import java.util.List;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.countArrangement(2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC526 **
    List<Integer>[] lc526Match;
    boolean[] lc526Visited;
    int lc526Result;

    public int countArrangement(int n) {
        lc526Match = new List[n + 1];
        lc526Visited = new boolean[n + 1];
        for (int i = 1; i <= n; i++) {
            lc526Match[i] = new ArrayList<>();
            for (int j = 1; j <= n; j++) {
                if (i % j == 0 || j % i == 0) {
                    lc526Match[i].add(j);
                }
            }
        }
        lc526Backtrack(1, n);
        return lc526Result;
    }

    public void lc526Backtrack(int index, int n) {
        if (index == n + 1) {
            lc526Result++;
            return;
        }
        for (int x : lc526Match[index]) {
            if (!lc526Visited[x]) {
                lc526Visited[x] = true;
                lc526Backtrack(index + 1, n);
                lc526Visited[x] = false;
            }
        }
    }
}