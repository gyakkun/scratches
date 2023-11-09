package moe.nyamori.test.historical;


import kotlin.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class scratch_64 {
    public static void main(String[] args) {
        scratch_64 s = new scratch_64();
        long timing = System.currentTimeMillis();

        System.out.println(s.canReorderDoubled(new int[]{1, 2, 1, -8, 8, -4, 4, -4, 2, -2}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1305
    List<Integer> result = new ArrayList<>();

    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        inorder(root1);
        inorder(root2);
        Collections.sort(result);
        return result;
    }

    private void inorder(TreeNode node) {
        if (node == null) return;
        if (node.left != null) inorder(node.left);
        result.add(node.val);
        if (node.right != null) inorder(node.right);
    }

    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // LC954
    public boolean canReorderDoubled(int[] arr) {
        long zeroCount = Arrays.stream(arr).boxed().filter(i -> i == 0).count();
        if (zeroCount % 2 == 1) return false;
        TreeMap<Integer, Integer> posCollect = new TreeMap<>() {{
            putAll(Arrays.stream(arr).boxed().filter(i -> i > 0).collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1))));
        }};
        TreeMap<Integer, Integer> negCollect = new TreeMap<>() {{
            putAll(Arrays.stream(arr).boxed().filter(i -> i < 0).map(i -> -i).collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1))));
        }};
        if (posCollect.values().stream().mapToInt(Integer::valueOf).sum() % 2 == 1
                || negCollect.values().stream().mapToInt(Integer::valueOf).sum() % 2 == 1) {
            return false;
        }
        if (!helper(posCollect)) return false;
        if (!helper(negCollect)) return false;

        return true;
    }

    private boolean helper(TreeMap<Integer, Integer> posCollect) {
        while (!posCollect.isEmpty()) {
            int largestKey = posCollect.descendingKeySet().first();
            if (largestKey % 2 == 1) return false;
            int largestValue = posCollect.get(largestKey);
            int halfKey = largestKey / 2;
            int halfValue = posCollect.getOrDefault(halfKey, Integer.MAX_VALUE);
            if (halfValue == Integer.MAX_VALUE || halfValue < largestValue) return false;
            posCollect.remove(largestKey);
            if (halfValue == largestValue) {
                posCollect.remove(halfKey);
            } else {
                posCollect.put(halfKey, halfValue - largestValue);
            }
        }
        return true;
    }


    // LC1606
    public List<Integer> busiestServers(int k, int[] arrival, int[] load) {
        if (arrival.length <= k) {
            return IntStream.range(0, arrival.length).boxed().collect(Collectors.toList());
        }
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i.getFirst())); // <在什么时刻重新空闲, 是第几个服务器>
        TreeSet<Integer> ts = new TreeSet<>(); // 空闲服务器列表
        int[] count = new int[k];
        final Integer[] max = {Integer.MIN_VALUE / 2};
        List<Integer> result = new ArrayList<>();
        IntStream.range(0, k).forEachOrdered(ts::add);
        IntStream.range(0, arrival.length).forEachOrdered(i -> {
            while (!pq.isEmpty() && pq.peek().getFirst() <= arrival[i]) {
                Pair<Integer, Integer> p = pq.poll();
                ts.add(p.getSecond());
            }
            if (ts.isEmpty()) {
                return;
            }
            Integer nextServer = ts.ceiling(i % k);
            if (nextServer == null)
                nextServer = ts.first(); // 如果没有比这个i%k大的编号的服务器, 则说明已经只能从头开始找了, 而ts不为空, 所以总能找到编号最小的服务器响应请求
            ts.remove(nextServer);
            pq.offer(new Pair<>(arrival[i] + load[i], nextServer));
            count[nextServer]++;

            if (count[nextServer] > max[0]) {
                max[0] = count[nextServer];
                result.clear();
                result.add(nextServer);
            } else if (count[nextServer] == max[0]) {
                result.add(nextServer);
            }
        });

        return result;
    }

    // LC2024 ** Sliding Window 滑动窗口
    public int maxConsecutiveAnswers(String answerKey, int k) {
        int n = answerKey.length(), result = 0;
        char[] ca = answerKey.toCharArray();
        int[] zeroOne = new int[n];
        for (int i = 0; i < n; i++) zeroOne[i] = ca[i] == 'T' ? 1 : 0;
        // Check 0 among 1
        int sum = 0;
        for (int left = 0, right = 0; right < n; right++) {
            sum += 1 - zeroOne[right];
            while (sum > k) {
                sum -= 1 - zeroOne[left];
                left++;
            }
            result = Math.max(result, right - left + 1);
        }
        sum = 0;
        for (int left = 0, right = 0; right < n; right++) {
            sum += zeroOne[right];
            while (sum > k) {
                sum -= zeroOne[left];
                left++;
            }
            result = Math.max(result, right - left + 1);
        }
        return result;
    }


    // LC2028
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int m = rolls.length;
        int curSum = Arrays.stream(rolls).sum();
        int totalCount = m + n;
        int remain = mean * totalCount - curSum;
        if (remain > 6 * n || remain < n) return new int[]{};
        int avg = remain / n;
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = avg;
            remain -= avg;
            if (i != n - 1) avg = remain / (n - i - 1);
        }
        return result;
    }

    // 220327 LYJJ
    int backwardOne;
    int[] op;
    Integer[][][] memo;

    public int minOnes(int[] arr, int m) { // op: 0 - &, 1 - | ,2 - ^
        int n = arr.length + 1;
        backwardOne = m;
        op = arr;
        memo = new Integer[2][n][1 << m];
        return Ly220327Helper(1, n - 1, 0);
    }

    private int Ly220327Helper(int target, int numIdx, int mask) {
        boolean isBackwardMOne = ((mask >> (backwardOne - 1)) & 1) == 1;
        boolean isBackwardMMinusOneOne = ((mask >> (backwardOne - 2)) & 1) == 1;
        if (numIdx == 1) { // 边界条件, 轮到正数第二个数(numIdx==1), 此时只剩下一个运算符
            switch (op[0]) {
                case 0:
                    if (target == 1) { // 此时只能两侧各填1, 所以前m个数和前m-1个数都不能是1, 否则返回极大值
                        if (isBackwardMOne || isBackwardMMinusOneOne)
                            return Integer.MAX_VALUE / 2;
                        return 2;
                    } else if (target == 0) {
                        return 0;
                    }
                case 1: // 这里或运算和异或运算所要判断的情形是一致的
                case 2:
                    if (target == 1) { // 前m和前m-1不能同时是1
                        if (isBackwardMOne && isBackwardMMinusOneOne)
                            return Integer.MAX_VALUE / 2;
                        return 1;
                    } else if (target == 0) {
                        return 0;
                    }
            }
        }
        if (memo[target][numIdx][mask] != null) return memo[target][numIdx][mask];
        int result = -1;
        int newMaskWithOne = ((mask << 1) | 1) & ((1 << backwardOne) - 1);
        int newMaskWithZero = ((mask << 1) | 0) & ((1 << backwardOne) - 1);
        switch (op[numIdx - 1]) {
            case 0:
                if (target == 1) { // 此时两侧都要填1, 只要第前m个数是1, 就返回极大值
                    if (isBackwardMOne) return Integer.MAX_VALUE / 2;
                    result = 1 + Ly220327Helper(1, numIdx - 1, newMaskWithOne);
                } else if (target == 0) { // (0,0),(0,1),(1,0) 中最小的
                    result = Math.min(
                            Math.min(
                                    0 + Ly220327Helper(0, numIdx - 1, newMaskWithZero),
                                    0 + Ly220327Helper(1, numIdx - 1, newMaskWithZero)
                            ),
                            1 + Ly220327Helper(0, numIdx - 1, newMaskWithOne)
                    );
                }
                break;
            case 1:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + Ly220327Helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0),(1,1) 中最小的
                        result = Math.min(
                                Math.min(
                                        0 + Ly220327Helper(1, numIdx - 1, newMaskWithZero),
                                        1 + Ly220327Helper(0, numIdx - 1, newMaskWithOne)),
                                1 + Ly220327Helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                } else if (target == 0) { // 意味着两侧都要填0
                    result = 0 + Ly220327Helper(0, numIdx - 1, newMaskWithZero);
                }
                break;
            case 2:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + Ly220327Helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0)中最小的
                        result = Math.min(
                                0 + Ly220327Helper(1, numIdx - 1, newMaskWithZero),
                                1 + Ly220327Helper(0, numIdx - 1, newMaskWithOne)
                        );

                    }
                } else if (target == 0) { // (0,0), (1,1)
                    if (isBackwardMOne) {
                        result = 0 + Ly220327Helper(0, numIdx - 1, newMaskWithZero);
                    } else {
                        result = Math.min(
                                0 + Ly220327Helper(0, numIdx - 1, newMaskWithZero),
                                1 + Ly220327Helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                }
        }
        return memo[target][numIdx][mask] = result;
    }


    // LC682
    public int calPoints(String[] ops) {
        List<Integer> points = new ArrayList<>(ops.length);
        for (String o : ops) {
            char op = o.charAt(0);
            if (Character.isDigit(op) || op == '-') {
                points.add(Integer.parseInt(o));
            } else {
                switch (op) {
                    case '+':
                        points.add(points.get(points.size() - 1) + points.get(points.size() - 2));
                        break;
                    case 'D':
                        points.add(points.get(points.size() - 1) * 2);
                        break;
                    case 'C':
                        points.remove(points.size() - 1);
                        break;
                }
            }
        }
        return points.stream().reduce(0, Integer::sum);
    }

    // LC2038
    public boolean winnerOfGame(String colors) {
        int a = 0, b = 0, n = colors.length();
        char[] ca = colors.toCharArray();
        for (int i = 0; i < n; i++) {
            if (i - 1 >= 0 && i + 1 < n && ca[i - 1] == ca[i] && ca[i + 1] == ca[i]) {
                switch (ca[i]) {
                    case 'A':
                        a++;
                        continue;
                    case 'B':
                        b++;
                        continue;
                }
            }
        }
        return a > b;
    }

    // LC2039
    public int networkBecomesIdle(int[][] edges, int[] patience) {
        final int n = patience.length, INF = Integer.MAX_VALUE / 2;
        // Dijkstra first
        Map<Integer, List<Integer>> m = new HashMap<>(n);
        for (int i = 0; i < n; i++) m.put(i, new ArrayList<>());
        for (int[] e : edges) {
            m.get(e[0]).add(e[1]);
            m.get(e[1]).add(e[0]);
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i[1]));
        pq.offer(new int[]{0, 0});
        Set<Integer> visited = new HashSet<>(n);
        int[] shortestDistanceToZero = new int[n];
        Arrays.fill(shortestDistanceToZero, INF);
        shortestDistanceToZero[0] = 0;
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int cur = p[0], distanceFromCurToZero = p[1];
            if (visited.contains(cur)) continue;
            visited.add(cur);
            for (int next : m.get(cur)) {
                if (cur != next && !visited.contains(next)) {
                    if (shortestDistanceToZero[next] > distanceFromCurToZero + 1) {// 1 为邻接点间的距离
                        shortestDistanceToZero[next] = distanceFromCurToZero + 1;
                        pq.offer(new int[]{next, distanceFromCurToZero + 1});
                    }
                }
            }
        }

        int result = 0;
        for (int i = 0; i < n; i++) {
            if (patience[i] == 0) continue;
            // rtt: 2 x shortest distance
            result = Math.max(result, ((2 * shortestDistanceToZero[i] - 1) / patience[i]) * patience[i] + 2 * shortestDistanceToZero[i]);
            //                                                        ^ ^ -1 是因为第一秒才开始检查, 下面的错误做法认为从第0秒就开始检查

            // Wrong:
            // if ((2 * shortestDistanceToZero[i]) % patience[i] == 0) {
            //     result = Math.max(result, ((2 * shortestDistanceToZero[i] / patience[i]) - 1) + 2 * shortestDistanceToZero[i]);
            // } else {
            //    result = Math.max(result, (2 * shortestDistanceToZero[i] / patience[i]) * patience[i] + 2 * shortestDistanceToZero[i]);
            //}
        }
        return result + 1;
    }

    // LC2044
    public int countMaxOrSubsets(int[] nums) {
        int n = nums.length;
        int max = Integer.MIN_VALUE, maxCount = 0;
        for (int mask = 1; mask < (1 << n); mask++) {
            int tmp = 0;
            for (int j = 0; j < n; j++) {
                if (((mask >> j) & 1) == 1) {
                    tmp |= nums[j];
                }
            }
            if (tmp > max) {
                max = tmp;
                maxCount = 1;
            } else if (tmp == max) {
                maxCount++;
            }
        }
        return maxCount;
    }
}