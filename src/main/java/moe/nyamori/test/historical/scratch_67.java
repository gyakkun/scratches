package moe.nyamori.test.historical;

import java.io.*;
import java.util.*;

class Main67 {
    static List<Integer>[] reachable;
    static BitSet visited;
    final static int MAX = 123457;

    // PRIME OJ 1003
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        String line;
        line = br.readLine();
        int total = Integer.parseInt(line);

        for (int i = 0; i < total; i++) {
            line = br.readLine();
            String[] mn = line.split(" ");
            int m = Integer.parseInt(mn[0]), n = Integer.parseInt(mn[1]);
            reachable = new List[MAX];
            visited = new BitSet(MAX);
            for (int j = 0; j < n; j++) {
                line = br.readLine();
                String[] uv = line.split(" ");
                int u = Integer.parseInt(uv[0]), v = Integer.parseInt(uv[1]);
                if (reachable[u] == null) reachable[u] = new LinkedList<>();
                reachable[u].add(v);
            }

            // BFS
            Deque<Integer> q = new LinkedList<>();
            q.offer(0);
            while (!q.isEmpty()) {
                int p = q.poll();
                if (visited.get(p)) continue;
                visited.set(p);
                if (reachable[p] == null) continue;
                for (int next : reachable[p]) {
                    if (visited.get(next)) continue;
                    q.offer(next);
                }
            }
            bw.write("" + (m - visited.cardinality() + 1) + "\n");
        }
        bw.flush();
        bw.close();
        br.close();
    }
}