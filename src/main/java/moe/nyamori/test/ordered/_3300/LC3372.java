package moe.nyamori.test.ordered._3300;

import java.util.*;

public class LC3372 {
    public int[] maxTargetNodes(int[][] edges1, int[][] edges2, int k) {
        var numNodes1 = edges1.length + 1; // in a tree, there are n nodes and (n-1) edges
        var numNodes2 = edges2.length + 1;
        // preprocess the edges
        var em1 = new HashMap<Integer, List<Integer>>();
        var em2 = new HashMap<Integer, List<Integer>>();
        for (var e : edges1) {
            em1.computeIfAbsent(e[0], i -> new ArrayList<>()).add(e[1]);
            em1.computeIfAbsent(e[1], i -> new ArrayList<>()).add(e[2]);
        }
        for (var e : edges2) {
            em2.computeIfAbsent(e[0], i -> new ArrayList<>()).add(e[1]);
            em2.computeIfAbsent(e[1], i -> new ArrayList<>()).add(e[2]);
        }

        // we only care about those node pairs whose distance is less than k (k<=1000)
        // the problem is how we index the [distance, node-paris]
        // one node can have 1000 edges at most, and in case they are all less than k,
        // then there will be 1e6 elements to store
        var ndc1 = new HashMap<Integer, TreeMap<Integer, Integer>>();
        var ndc2 = new HashMap<Integer, TreeMap<Integer, Integer>>();
        for (int i = 0; i < numNodes1; i++) {
            // we use dsu + lca tarjan to calculate the distance of each node in edge1 and edge2
            // we store the result int a [node, distance, count] map,
            // and for further query (how many nodes from myself are within distance k),
            // we can use treemap for [distance, count] part
            // also distance should be within k (<=1000)
            ndc1.put(i, new TreeMap<>());
            var dsu = new DSUMap();
            // do dfs here

        }
        return null;
    }

    private Interim getInterim(Map<Integer, List<Integer>> neighbour, int numNodes) {
        // let's do a dfs with stack
        var stack = new LinkedList<Integer>();
        var root = neighbour.keySet().stream().findAny().get();
        var visited = new boolean[numNodes];
        stack.push(root);
        while (!stack.isEmpty()) {
            var cur = stack.pop();
            visited[cur] = true;
            // post-order, root last
            for (var n : neighbour.getOrDefault(cur, Collections.emptyList())) {
                stack.push(n);

            }
        }
        return null;
    }

    record Interim(
            int root, // node 0 by default
            int[] height, // height[root] = 0
            int[] lca,
            Map<Integer,TreeMap<Integer,Integer>> ndc // [node, distance, count]
    ) {
    }

    class DSUArray {
        int size;
        int[] parent;
        int[] rank;

        DSUArray(int size) {
            this.size = size;
            parent = new int[size];
            rank = new int[size];
            Arrays.fill(parent, -1);
            Arrays.fill(rank, -1);
        }

        DSUArray() {
            this(1 << 16);
        }

        private boolean checkNode(int v) {
            return v >= 0 && v < size;
        }

        void add(int v) {
            if (!checkNode(v)) return;
            parent[v] = parent[v] == -1 ? v : parent[v];
            rank[v] = rank[v] == -1 ? 1 : rank[v];
        }

        int find(int v) {
            if (!checkNode(v)) return -1;
            var root = v;
            while (checkNode(root) && root != parent[root]) {
                root = parent[root];
            }
            if (root == -1) return -1;
            // route compress
            var cur = v;
            while (parent[cur] != root) {
                var origParent = parent[cur];
                parent[cur] = root;
                rank[root]++;
                cur = origParent;
            }
            return root;
        }

        boolean merge(int v1, int v2) {
            if (!checkNode(v1) || !checkNode(v2)) return false;
            var r1 = find(v1);
            var r2 = find(v2);
            if (r1 == r2) return false;
            if (rank[r1] > rank[r2]) {
                // merge r2 to r1
                parent[r2] = r1;
                rank[r1] += rank[r2];
            } else {
                parent[r1] = r2;
                rank[r2] += rank[r1];
            }
            return true;
        }

        boolean isConnected(int v1, int v2) {
            if (!checkNode(v1) || !checkNode(v2)) return false;
            return find(v1) == find(v2);
        }

        boolean contains(int v) {
            if (!checkNode(v)) return false;
            return parent[v] != -1 && rank[v] != -1;
        }

        Map<Integer, Set<Integer>> getAllGroups() {
            var res = new HashMap<Integer, Set<Integer>>();
            for (int i = 0; i < size; i++) {
                res.computeIfAbsent(find(i), j -> new HashSet<>()).add(i);
            }
            return res;
        }

        int getNumOfGroups() {
            return getAllGroups().size();
        }

    }

    class DSUMap {

        Map<Integer, Integer> father;
        Map<Integer, Integer> rank;

        public DSUMap() {
            father = new HashMap<>();
            rank = new HashMap<>();
        }

        public void add(int i) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.putIfAbsent(i, i);
            rank.putIfAbsent(i, 1);
        }

        // 找父亲, 路径压缩
        public int find(int i) {
            //先找到根 再压缩
            int root = i;
            while (father.get(root) != root) {
                root = father.get(root);
            }
            // 找到根, 开始对一路上的子节点进行路径压缩
            while (father.get(i) != root) {
                int origFather = father.get(i);
                father.put(i, root);
                // 更新秩, 按照节点数
                rank.put(root, rank.get(root) + 1);
                i = origFather;
            }
            return root;
        }

        public boolean merge(int i, int j) {
            int iFather = find(i);
            int jFather = find(j);
            if (iFather == jFather) return false;
            // 按秩合并
            if (rank.get(iFather) >= rank.get(jFather)) {
                father.put(jFather, iFather);
                rank.put(iFather, rank.get(jFather) + rank.get(iFather));
            } else {
                father.put(iFather, jFather);
                rank.put(jFather, rank.get(jFather) + rank.get(iFather));
            }
            return true;
        }

        public boolean isConnected(int i, int j) {
            return find(i) == find(j);
        }

        public Map<Integer, Set<Integer>> getAllGroups() {
            Map<Integer, Set<Integer>> result = new HashMap<>();
            // 找出所有根
            for (Integer i : father.keySet()) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
            return result;
        }

        public int getNumOfGroups() {
            Set<Integer> s = new HashSet<Integer>();
            for (Integer i : father.keySet()) {
                s.add(find(i));
            }
            return s.size();
        }

        public boolean contains(int i) {
            return father.containsKey(i);
        }

        public int getPointCount() {
            return father.size();
        }

    }
}
