import java.util.*;

class Solution {
    public static void main(String[] args) {
        var s = new Solution();
        var i = s.countPairs(11, new int[][]{{5, 0}, {1, 0}, {10, 7}, {9, 8}, {7, 2}, {1, 3}, {0, 2}, {8, 5}, {4, 6}, {4, 2}});
        System.err.println(i);
    }

    public long countPairs(int n, int[][] edges) {
        var dsu = new DSU();
        for (int i = 0; i < n; i++) dsu.add(i);
        for (int[] i : edges) {
            dsu.merge(i[0], i[1]);
        }
        List<Integer> groups = dsu.getAllGroups().values().stream().map(Set::size).toList();
        long sum = 0L;
        for (int i : groups) sum += i;
        long len = groups.size();
        long res = 0L;
        for (int i = 0; i < len; i++) {
            long the = groups.get(i);
            res += the * (sum - the);
            sum -= the;
        }
        return res;
    }
}

class DSU {
    Map<Integer, Integer> parent = new HashMap<>();
    Map<Integer, Integer> rank = new HashMap<>();

    public boolean add(int i) {
        if (parent.containsKey(i)) return false;
        parent.put(i, i);
        rank.put(i, 1);
        return true;
    }

    public int find(int i) { // find the root parent
        int root = i;
        int tmp;
        while ((tmp = parent.get(root)) != root) {
            root = tmp;
        }
        int ptr = i;
        while ((tmp = parent.get(ptr)) != root) { // Compress route
            parent.put(ptr, root);
            rank.put(root, rank.get(root) + 1); // merge by higher ranking
            ptr = tmp;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iParent = find(i);
        int jParent = find(j);
        if (iParent == jParent) return false;
        int iPRank = rank.get(iParent);
        int jPRank = rank.get(jParent);
        if (iPRank >= jPRank) {
            parent.put(jParent, iParent);
            rank.put(iParent, rank.get(iParent) + rank.get(jParent));
        } else {
            parent.put(iParent, jParent);
            rank.put(jParent, rank.get(iParent) + rank.get(jParent));
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        // Find all roots
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (int i : parent.keySet()) {
            int p = find(i);
            Set<Integer> s = res.computeIfAbsent(p, j -> new HashSet<>());
            s.add(i);
        }
        return res;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }
}
