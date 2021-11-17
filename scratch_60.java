import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println();

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC592
    public String fractionAddition(String expression) {
        long num = 0l, den = 1l; // 初始化 0/1
        if (expression.charAt(0) != '-') expression = '+' + expression;
        char[] ca = expression.toCharArray();
        int idx = 0, n = ca.length;
        while (idx < n) {
            int sign = ca[idx] == '+' ? 1 : -1;
            idx++;
            int numLeft = idx;
            while ((idx + 1) < n && ca[idx + 1] != '/') idx++;
            int numRight = idx;
            int curNum = Integer.valueOf(expression.substring(numLeft, numRight + 1));
            idx += 2;
            int denLeft = idx;
            while ((idx + 1) < n && (ca[idx + 1] != '+' && ca[idx + 1] != '-')) idx++;
            int denRight = idx;
            int curDen = Integer.valueOf(expression.substring(denLeft, denRight + 1));
            idx++;


            // 处理通分
            long tmpDen = den * curDen;
            long tmpNum = num * curDen + sign * (curNum * den);
            long gcd = gcd(tmpDen, tmpNum);
            tmpDen /= gcd;
            tmpNum /= gcd;
            den = tmpDen;
            num = tmpNum;
        }
        return (num * den < 0l ? "-" : "") + Math.abs(num) + "/" + Math.abs(den);
    }

    private long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC318
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] bitmask = new int[n];
        for (int i = 0; i < n; i++) {
            int mask = 0;
            for (char c : words[i].toCharArray()) {
                mask |= 1 << (c - 'a');
            }
            bitmask[i] = mask;
        }
        int max = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((bitmask[i] & bitmask[j]) == 0) {
                    max = Math.max(words[i].length() * words[j].length(), max);
                }
            }
        }
        return max;
    }
}


// LC928 Try Tarjan O(n)
class Lc928Tarjan {
    int n;
    int[] low, timestamp, groupSize, spreadSize, save, virusTag, finalParent;
    List<List<Integer>> mtx; // 邻接表
    int timing;
    int spreadTiming;
    int maxSaveCount = Integer.MIN_VALUE;
    int currentRoot;
    int result = -1;

    public int minMalwareSpread(int[][] graph, int[] virus) {
        Arrays.sort(virus);
        build(graph, virus);
        for (int i = 0; i < n; i++) {
            if (timestamp[i] == -1) {
                currentRoot = i;
                timing = 0; // 注意我们给每一个连通分量分配一个全新的计时器(从0开始)
                spreadTiming = 0; // 感染数量也是
                tarjan(i, i);
            }
        }

        for (int i = 0; i < n; i++) {
            // **** 父块的处理, 很关键
            if (spreadSize[finalParent[i]] == spreadSize[i]) {
                save[i] += groupSize[finalParent[i]] - groupSize[i];
            }
            if (virusTag[i] == 1 && save[i] > maxSaveCount) {
                result = i;
                maxSaveCount = save[i];
            }
        }
        return result;
    }

    private void tarjan(int cur, int parent) {
        // 借用 Tarjan 求 **割点** 的算法流程。 注意此处不是真的求割点, 所以不需要统计直接孩子的数量

        low[cur] = timestamp[cur] = ++timing; // timing 是遇到一个新节点就自增
        spreadTiming += virusTag[cur]; // spreadTiming 是遇到一个新的病毒节点才自增

        finalParent[cur] = currentRoot;
        groupSize[cur] = 1;
        spreadSize[cur] = virusTag[cur];

        for (int next : mtx.get(cur)) {
            if (next == parent) continue;

            int thisMomentTiming = timing;
            int thisMomentSpreadTiming = spreadTiming;

            if (timestamp[next] == -1) {
                tarjan(next, cur);
            }

            int deltaTiming = timing - thisMomentTiming;
            int deltaSpreadTiming = spreadTiming - thisMomentSpreadTiming;

            // 判断next开始的路径能不能回到cur, 标准Tarjan求割点的做法。用以判断next开始的子图是不是独立子图
            if (low[next] >= timestamp[cur]) {
                if (deltaSpreadTiming == 0) { // 说明经过这一点next之后没有新的节点被感染, 也即如果cur消失后, 能够多拯救多少节点
                    save[cur] += deltaTiming; // DFS完这个子图, delta(timing) 即后序遍历到的节点个数
                }
                groupSize[cur] += deltaTiming;
                spreadSize[cur] += deltaSpreadTiming;
            }

            low[cur] = Math.min(low[cur], low[next]);
        }
    }


    private void build(int[][] graph, int[] virus) {
        n = graph.length;
        low = new int[n];
        timestamp = new int[n];
        groupSize = new int[n];
        spreadSize = new int[n];
        virusTag = new int[n];
        finalParent = new int[n];
        save = new int[n];
        Arrays.fill(low, -1);
        Arrays.fill(timestamp, -1);

        for (int i : virus) virusTag[i] = 1;

        mtx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && graph[i][j] == 1) {
                    mtx.get(i).add(j);
                }
            }
        }
    }
}
