import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.sequenceReconstruction(new int[]{4, 1, 5, 2, 6, 3},
                new int[][]{{5, 2, 6, 3}, {4, 1, 5, 2}}));

    }

    // LC952
    public int largestComponentSize(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        Map<Integer, Set<Integer>> factorNumMap = new HashMap<>();
        for (int i = n - 1; i >= 0; i--) {
            int victim = nums[i];
            int sqrt = (int) Math.sqrt(victim) + 1;
            for (int j = 1; j <= sqrt; j++) {
                if (victim % j == 0) {
                    int anotherFactor = victim / j;
                    factorNumMap.putIfAbsent(j, new HashSet<>());
                    factorNumMap.putIfAbsent(anotherFactor, new HashSet<>());
                    factorNumMap.get(j).add(victim);
                    factorNumMap.get(anotherFactor).add(victim);
                }
            }
        }
        DSUArray dsu = new DSUArray(100001);
        for (Map.Entry<Integer, Set<Integer>> e : factorNumMap.entrySet()) {
            if (e.getKey() == 1) continue;
            Set<Integer> sharingEdge = e.getValue();
            Integer root = sharingEdge.stream().findFirst().get();
            for (int i : sharingEdge) {
                dsu.add(i);
                dsu.merge(i, root);
            }
        }
        Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
        int result = 0;
        for (Map.Entry<Integer, Set<Integer>> e : allGroups.entrySet()) {
            result = Math.max(result, e.getValue().size());
        }
        return result;
    }

    // LC444 JZOF II 115
    public boolean sequenceReconstruction(int[] nums, int[][] sequences) {
        int n = nums.length;
        List<Integer>[] outEdge = new List[n + 1];
        int[] indegree = new int[n + 1];
        BitSet bs = new BitSet(n + 1);
        for (int[] s : sequences) {
            Integer prev = null;
            for (int cur : s) {
                bs.set(cur);
                if (prev != null) {
                    indegree[cur]++;
                    if (outEdge[prev] == null) {
                        outEdge[prev] = new ArrayList<>();
                    }
                    outEdge[prev].add(cur);
                }
                prev = cur;
            }
        }
        if (bs.cardinality() != n) return false;
        Deque<Integer> q = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
                break;
            }
        }
        if (q.size() != 1) return false;
        List<Integer> topo = new ArrayList<>(n + 1);
        while (!q.isEmpty()) {
            int qs = q.size();
            if (qs > 1) return false;
            int p = q.poll();
            topo.add(p);
            if (outEdge[p] == null) continue;
            for (int next : outEdge[p]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        if (topo.size() != n) return false;
        for (int i = 0; i < n; i++) {
            if (topo.get(i) != nums[i]) return false;
        }
        return true;
    }

    // LC1184
    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start == destination) return 0;
        int forward = 0, backward = 0, startPoint = destination > start ? start : destination,
                endPoint = destination > startPoint ? destination : start, total = Arrays.stream(distance).sum();

        for (int i = startPoint; i < endPoint; i++) {
            forward += distance[i];
        }
        backward = total - forward;
        return Math.min(backward, forward);
    }

    // LC558 ** Quad Tree
    public Node intersect(Node quadTree1, Node quadTree2) {
        if (quadTree1.isLeaf) {
            if (quadTree1.val) {
                return new Node() {{
                    val = true;
                    isLeaf = true;
                }};
            }
            return new Node(quadTree2.val, quadTree2.isLeaf, quadTree2.topLeft, quadTree2.topRight, quadTree2.bottomLeft, quadTree2.bottomRight);
        }
        if (quadTree2.isLeaf) {
            return intersect(quadTree2, quadTree1);
        }
        Node o1 = intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node o2 = intersect(quadTree1.topRight, quadTree2.topRight);
        Node o3 = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node o4 = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node() {{
                val = o1.val;
                isLeaf = true;
            }};
        }
        return new Node(false, false, o1, o2, o3, o4);
    }

}

// Definition for a QuadTree node.
class Node {
    public boolean val;
    public boolean isLeaf; // true means all value of the 4 corner is the same
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;

    public Node() {
    }

    public Node(boolean _val, boolean _isLeaf, Node _topLeft, Node _topRight, Node _bottomLeft, Node _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
}


class DisjointSetUnion<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
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

    public boolean isConnected(T i, T j) {
        if (!father.containsKey(i) || !father.containsKey(j)) return false;
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(T i) {
        return father.containsKey(i);
    }

}


class DSUArray {
    int[] father;
    int[] rank;
    int size;

    public DSUArray(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray() {
        this.size = 1 << 16;
        father = new int[1 << 16];
        rank = new int[1 << 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public boolean contains(int i) {
        if (i >= this.size || i < 0) return false;
        return father[i] != -1;
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (int i = 0; i < size; i++) {
            if (father[i] != -1) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
        }
        return result;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }

    public int getSelfGroupSize(int x) {
        return rank[find(x)];
    }

}