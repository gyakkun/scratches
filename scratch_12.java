import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

class Dijkstra {
    /*
     * 参数adjMatrix:为图的权重矩阵，权值为-1的两个顶点表示不能直接相连
     * 函数功能：返回顶点0到其它所有顶点的最短距离，其中顶点0到顶点0的最短距离为0
     */
    public int[] getShortestPaths(int[][] adjMatrix) {
        int[] result = new int[adjMatrix.length];   //用于存放顶点0到其它顶点的最短距离
        boolean[] used = new boolean[adjMatrix.length];  //用于判断顶点是否被遍历
        used[0] = true;  //表示顶点0已被遍历
        for (int i = 1; i < adjMatrix.length; i++) {
            result[i] = adjMatrix[0][i];
            used[i] = false;
        }

        for (int i = 1; i < adjMatrix.length; i++) {
            int min = Integer.MAX_VALUE;    //用于暂时存放顶点0到i的最短距离，初始化为Integer型最大值
            int k = 0;
            for (int j = 1; j < adjMatrix.length; j++) {  //找到顶点0到其它顶点中距离最小的一个顶点
                if (!used[j] && result[j] != -1 && min > result[j]) {
                    min = result[j];
                    k = j;
                }
            }
            used[k] = true;    //将距离最小的顶点，记为已遍历
            for (int j = 1; j < adjMatrix.length; j++) {  //然后，将顶点0到其它顶点的距离与加入中间顶点k之后的距离进行比较，更新最短距离
                if (!used[j]) {  //当顶点j未被遍历时
                    //首先，顶点k到顶点j要能通行；这时，当顶点0到顶点j的距离大于顶点0到k再到j的距离或者顶点0无法直接到达顶点j时，更新顶点0到顶点j的最短距离
                    if (adjMatrix[k][j] != -1 && (result[j] > min + adjMatrix[k][j] || result[j] == -1))
                        result[j] = min + adjMatrix[k][j];
                }
            }
        }
        return result;
    }

    public int[] getShortestPathsPQ(int[][] adj) {
        // 假设源点为0
        // 邻接矩阵, 用-1表示不可达
        // 先统一将-1转换为INF 方便处理
        final int INF = Integer.MAX_VALUE / 2;
        int n = adj.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (adj[i][j] == -1) {
                    adj[i][j] = INF;
                }
            }
        }
        boolean[] visited = new boolean[n];
        int[] result = new int[n];
        Arrays.fill(result, INF);
        result[0] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1])); // [下标, 到源点距离]
        pq.offer(new int[]{0, 0});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int curIdx = p[0], distanceFromCurIdxToSrc = p[1];
            if (visited[curIdx]) continue;
            visited[curIdx] = true;

            for (int next = 0; next < n; next++) {
                if (next != curIdx && adj[curIdx][next] != INF && !visited[next]) {
                    // 如果下一跳前到源点的距离大于 下一跳 途径当前点 再到源点的距离
                    if (result[next] > distanceFromCurIdxToSrc + adj[curIdx][next]) {
                        result[next] = distanceFromCurIdxToSrc + adj[curIdx][next];
                        pq.offer(new int[]{next, result[next]});
                    }
                }
            }
        }
        return result;

    }


    public static void main(String[] args) {
        Dijkstra test = new Dijkstra();
        int[][] adjMatrix = {{0, 6, 3, -1, -1, -1},
                {6, 0, 2, 5, -1, -1},
                {3, 2, 0, 3, 4, -1},
                {-1, 5, 3, 0, 2, 3},
                {-1, -1, 4, 2, 0, 5},
                {-1, -1, -1, 3, 5, 0}};
        long timing = System.currentTimeMillis();
        int[] result = test.getShortestPaths(adjMatrix);
        timing = System.currentTimeMillis() - timing;
        System.out.println("顶点0到图中所有顶点之间的最短距离为：");
        for (int i = 0; i < result.length; i++) {
            System.out.print(result[i] + " ");
        }
        System.err.println("\r\nTIMING: " + timing + "ms");
    }
}