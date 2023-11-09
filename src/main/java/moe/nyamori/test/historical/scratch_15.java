package moe.nyamori.test.historical;

import java.util.*;

class scratch_15 {
    public static void main(String[] args) {
        findCriticalAndPseudoCriticalEdges(6,
                new int[][]
                        {
                                {0, 1, 1},
                                {1, 2, 1},
                                {0, 2, 1},
                                {2, 3, 4},
                                {3, 4, 2},
                                {3, 5, 2},
                                {4, 5, 2}
                        });

    }

    public static int removeStones(int[][] stones) {
        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        for (int[] s : stones) {
            dsu.add(s[0] + 10000);
            dsu.add(s[1]);
            dsu.merge(s[0] + 10000, s[1]);
        }
//        for (int i = 0; i < stones.length; i++) {s
//            for (int j = 0; j < stones.length; j++) {
//                if(stones[i][0]==stones[j][0]){
//                    ufs.merge(stones[i][1],stones[j][1]);
//                }
//                if(stones[i][1]==stones[j][1]){
//                    ufs.merge(stones[i][0], stones[j][0]);
//                }
//            }
//        }
        return stones.length - dsu.getAllGroups().size();
    }


    public int maxProfit(int[] prices) {
        int k = 2;
        int len = prices.length;
        int[][] dp2 = new int[k + 1][2];
        for (int i = 0; i < k; i++) {
            dp2[i][0] = 0;
            dp2[i][1] = -prices[0];

        }
        for (int i = 1; i < len; i++) {
            for (int j = k; j > 0; j--) {
                dp2[j][0] = Math.max(dp2[j][0], dp2[j][1] + prices[i]);
                dp2[j][1] = Math.max(dp2[j][1], dp2[j - 1][0] - prices[i]);
            }
        }
        return dp2[k][0];
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        Map<String, Integer> mailToAccountIndexMap = new HashMap<>();
        Map<Integer, Set<String>> accountIndexToMailSetMap = new HashMap<>();
        for (int i = 0; i < accounts.size(); i++) {
            dsu.add(i);
            for (int j = 1; j < accounts.get(i).size(); j++) {
                if (!mailToAccountIndexMap.containsKey(accounts.get(i).get(j))) {
                    mailToAccountIndexMap.put(accounts.get(i).get(j), i);
                } else {
                    dsu.merge(i, mailToAccountIndexMap.get(accounts.get(i).get(j)));
                }
            }
        }

        Map<Integer, Set<Integer>> groups = dsu.getAllGroups();

        for (int i : groups.keySet()) {
            accountIndexToMailSetMap.put(i, new HashSet<>());
            for (int j : groups.get(i)) {
                for (int k = 1; k < accounts.get(j).size(); k++) {
                    accountIndexToMailSetMap.get(i).add(accounts.get(j).get(k));
                }
            }
        }

        List<List<String>> result = new ArrayList<>();
        for (int i : accountIndexToMailSetMap.keySet()) {
            List<String> tmp = new LinkedList<>();
            for (String m : accountIndexToMailSetMap.get(i)) {
                tmp.add(m);
            }
            Collections.sort(tmp);
//            tmp.sort(Comparator.naturalOrder());
            tmp.add(0, accounts.get(i).get(0));
            result.add(tmp);
        }

        return result;
    }

    public int minCostConnectPoints(int[][] points) {
        int n = points.length, res = 0, i = 0, connected = 0, next = 0;
        int[] dist = new int[n];
        boolean[] visited = new boolean[n];
        Arrays.fill(dist, 10000000);
        while (++connected < n) {
            i = next;
            visited[i] = true;
            dist[i] = Integer.MAX_VALUE;
            for (int j = 0; j < n; ++j) {
                if (!visited[j]) {
                    dist[j] = Math.min(dist[j], ManhattanDistance(points[i], points[j]));
                    next = dist[j] < dist[next] ? j : next;
                }
            }
            res += dist[next];
        }
        return res;
    }

    public int minCostConnectPointsPrim(int[][] points) {
        int n = points.length, result = 0, minDist, distJ, connected = 0, i = 0, next = 0;
        Set<Integer> visited = new HashSet<>();
        while (++connected < n) {
            i = next;
            visited.add(i);
            minDist = Integer.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                if (!visited.contains(j)) {
                    distJ = ManhattanDistance(points[i], points[j]);
                    minDist = Math.min(minDist, distJ);
                    next = minDist == distJ ? j : next;
                }
            }
            result += minDist;
        }
        return result;
    }

    public int minCostConnectPointsKruskal(int[][] points) {
        int n = points.length;
        List<Edge> edgeList = new ArrayList<>();
        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                edgeList.add(new Edge(i, j, ManhattanDistance(points[i], points[j])));
            }
        }
        edgeList.sort(Comparator.comparingInt(o -> o.len));
        int totalNumOfEdges = 0, result = 0;
        for (Edge e : edgeList) {
            dsu.add(e.A);
            dsu.add(e.B);
            if (dsu.merge(e.A, e.B)) {
                totalNumOfEdges++;
                result += e.len;
            }
            if (totalNumOfEdges == (n - 1)) {
                return result;
            }
        }
        return result;
    }

    public int ManhattanDistance(int[] A, int[] B) {
        return Math.abs(A[0] - B[0]) + Math.abs(A[1] - B[1]);
    }

    public static List<List<Integer>> findCriticalAndPseudoCriticalEdges(int n, int[][] edges) {
        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> criticalEdge = new ArrayList<>();
        List<Integer> nonCriticalEdge = new ArrayList<>();
        List<Edge> le = new ArrayList<>();
        for (int i = 0; i < edges.length; i++) {
            le.add(new Edge(edges[i][0], edges[i][1], edges[i][2], i));
//            le.add(new Edge(edges[i][1], edges[i][0], edges[i][2], i));
        }
        le.sort(Comparator.comparingInt((o) -> o.len));
        int totalNumOfEdges = 0, mstWeight = 0;
        for (Edge e : le) {
            dsu.add(e.A);
            dsu.add(e.B);
            if (dsu.merge(e.A, e.B)) {
                totalNumOfEdges++;
                mstWeight += e.len;
            }
            if (totalNumOfEdges == (n - 1)) {
                break;
            }
        }

        // 得到最小生成树的总权重

        int allEdgeNum = le.size();
        for (int i = 0; i < allEdgeNum; i++) {
            DisjointSetUnion14 tmpDsu = new DisjointSetUnion14();
            DisjointSetUnion14 tmpDsuForNonCritical = new DisjointSetUnion14();
            int tmpTotalNumOfEdges = 0, tmpMstWeight = 0;
            int tmpTotalNumOfEdgesForNonCritical = 0, tmpMstWeightForNonCritical = 0;
            tmpDsuForNonCritical.add(le.get(i).A);
            tmpDsuForNonCritical.add(le.get(i).B);
            tmpDsuForNonCritical.merge(le.get(i).A, le.get(i).B);
            tmpTotalNumOfEdgesForNonCritical++;
            tmpMstWeightForNonCritical += (le.get(i).len);
            boolean criticalMstFinished = false, nonCriticalMstFinished = false;
            for (int j = 0; j < allEdgeNum; j++) {
                if (j == i) continue;
                tmpDsu.add(le.get(j).A);
                tmpDsu.add(le.get(j).B);
                tmpDsuForNonCritical.add(le.get(j).A);
                tmpDsuForNonCritical.add(le.get(j).B);
                if (!criticalMstFinished && tmpDsu.merge(le.get(j).A, le.get(j).B)) {
                    tmpTotalNumOfEdges++;
                    tmpMstWeight += le.get(j).len;
                }
                if (!nonCriticalMstFinished && tmpDsuForNonCritical.merge(le.get(j).A, le.get(j).B)) {
                    tmpTotalNumOfEdgesForNonCritical++;
                    tmpMstWeightForNonCritical += le.get(j).len;
                }
                if (tmpTotalNumOfEdges == (n - 1)) {
                    criticalMstFinished = true;
                }
                if (tmpTotalNumOfEdgesForNonCritical == (n - 1)) {
                    nonCriticalMstFinished = true;
                }
                if (criticalMstFinished && nonCriticalMstFinished) {
                    break;
                }
            }
            if (tmpMstWeightForNonCritical == mstWeight) {
                nonCriticalEdge.add(le.get(i).indexOfEdge);
            }
            if (tmpMstWeight > mstWeight) {
                criticalEdge.add(le.get(i).indexOfEdge);
            }
            if (!criticalMstFinished) {
                criticalEdge.add(le.get(i).indexOfEdge);
            }
        }

        Set<Integer> tmpNonCritical = new HashSet<>(nonCriticalEdge);
//        Set<Integer> tmpCritical = new HashSet<>(criticalEdge);
//        Set<Integer> tmpJiaoJi = new HashSet<>();

        tmpNonCritical.removeAll(criticalEdge);
        nonCriticalEdge = new ArrayList<>(tmpNonCritical);


        result.add(criticalEdge);
        result.add(nonCriticalEdge);
        return result;

    }

}

class Edge {
    // index of point A, B
    int A;
    int B;

    // len of edge
    int len;

    int indexOfEdge;

    public Edge(int A, int B, int len) {
        this.A = A;
        this.B = B;
        this.len = len;
        this.indexOfEdge = -1;
    }

    public Edge(int A, int B, int len, int indexOfEdge) {
        this.A = A;
        this.B = B;
        this.len = len;
        this.indexOfEdge = indexOfEdge;
    }

}


class DisjointSetUnion15 {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion15() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(int i) {
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

}

class Solution15 {
    public int maxSubarraySumCircular(int[] A) {
        int N = A.length;

        int ans = A[0], cur = A[0];
        for (int i = 1; i < N; ++i) {
            cur = A[i] + Math.max(cur, 0);
            ans = Math.max(ans, cur);
        }

        // ans is the answer for 1-interval subarrays.
        // Now, let's consider all 2-interval subarrays.
        // For each i, we want to know
        // the maximum of sum(A[j:]) with j >= i+2

        // rightsums[i] = A[i] + A[i+1] + ... + A[N-1]
        int[] rightsums = new int[N];
        rightsums[N - 1] = A[N - 1];
        for (int i = N - 2; i >= 0; --i)
            rightsums[i] = rightsums[i + 1] + A[i];

        // maxright[i] = max_{j >= i} rightsums[j]
        int[] maxright = new int[N];
        maxright[N - 1] = A[N - 1];
        for (int i = N - 2; i >= 0; --i)
            maxright[i] = Math.max(maxright[i + 1], rightsums[i]);

        int leftsum = 0;
        for (int i = 0; i < N - 2; ++i) {
            leftsum += A[i];
            ans = Math.max(ans, leftsum + maxright[i + 2]);
        }
        return ans;
    }
}

class MedianFinderSort {

    List<Integer> l;

    /**
     * initialize your data structure here.
     */
    public MedianFinderSort() {
        l = new ArrayList<>();
    }

    public void addNum(int num) {
        l.add(num);
    }

    public double findMedian() {

        Collections.sort(l);
        if (l.size() % 2 == 0) {
            int left = l.get(l.size() / 2 - 1);
            int right = l.get(l.size() / 2);
            return 0.5d * ((double) left + (double) right);
        }
        return l.get((l.size() - 1) / 2);
    }
}

class MedianFinderHeap {

    /**
     * initialize your data structure here.
     */

    private PriorityQueue<Integer> minHeap;
    private PriorityQueue<Integer> maxHeap;

    public MedianFinderHeap() {

        //initial minHeap and maxHeap;
        minHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1 - o2;
            }
        });
        maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });

    }

    public void addNum(int num) {
        if (maxHeap.isEmpty() || num < maxHeap.peek()) {
            maxHeap.offer(num);
        } else {
            minHeap.offer(num);
        }

        // check both sides size, re-balance their size
        // ensure maxHeap.size()-minHeap.size == 0 | 1
        if (maxHeap.size() == minHeap.size() + 2) {
            // 拉平
            minHeap.offer(maxHeap.poll());
        }
        if (minHeap.size() == maxHeap.size() + 1) {
            // 交换
            maxHeap.offer(minHeap.poll());
        }

    }

    public double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            double median = (maxHeap.peek() + minHeap.peek()) / 2.0;
            return median;
        } else {
            return maxHeap.peek();
        }
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */