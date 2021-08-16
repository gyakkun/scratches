class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.countArrangement(2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC526 **
    boolean[] lc526Visited;
    int lc526Result;

    public int countArrangement(int n) {
        lc526Visited = new boolean[n + 1];
        lc526Backtrack(1, n);
        return lc526Result;
    }

    public void lc526Backtrack(int index, int n) {
        if (index == n + 1) {
            lc526Result++;
            return;
        }
        for (int i = 1; i <= n; i++) {
            if (!lc526Visited[i] && (i % index == 0 || index % i == 0)) {
                lc526Visited[i] = true;
                lc526Backtrack(index + 1, n);
                lc526Visited[i] = false;
            }
        }
    }
}