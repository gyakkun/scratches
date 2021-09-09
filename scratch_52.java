import javafx.util.Pair;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.*;
import java.util.function.Function;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

//        System.out.println(s.removeInterval(new int[][]{{0, 2}, {3, 4}, {5, 7}}, new int[]{1, 6}));
        System.out.println(s.removeInterval(new int[][]{{-10, 10}, {50, 60}}, new int[]{-100, 100}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1272
    public List<List<Integer>> removeInterval(int[][] intervals, int[] toBeRemoved) {
        List<List<Integer>> result = new ArrayList<>();
        for (int[] i : intervals) {
            int left = i[0], right = i[1];
            if (right <= toBeRemoved[0] || left >= toBeRemoved[1]) {
                result.add(Arrays.asList(left, right));
                continue;
            }
            int intersectionRight = Math.min(right, toBeRemoved[1]);
            int intersectionLeft = Math.max(left, toBeRemoved[0]);

            if (intersectionLeft == left && intersectionRight == right) { // 交集竟是我自己
                continue;
            }
            if (intersectionLeft == left) {
                result.add(Arrays.asList(intersectionRight, right));
            } else if (intersectionRight == right) {
                result.add(Arrays.asList(left, intersectionLeft));
            } else { // 中间挖空
                result.add(Arrays.asList(left, intersectionLeft));
                result.add(Arrays.asList(intersectionRight, right));
            }
        }

        return result;
    }

    // LC1964 Hard ** LIS 变体
    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        // 离散化
        Set<Integer> set = new HashSet<>();
        for (int i : obstacles) set.add(i);
        List<Integer> l = new ArrayList<>(set);
        Collections.sort(l);
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < l.size(); i++) {
            m.put(l.get(i), i);
        }

        int[] result = new int[obstacles.length];
        BIT bit = new BIT(l.size() + 5);
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i < obstacles.length; i++) {
            int o = m.get(obstacles[i]);
            if (tm.isEmpty() || tm.lastKey() <= o) {
                tm.put(o, tm.getOrDefault(o, 0) + 1);
                bit.update(o, 1);
            } else {
                Integer higer = tm.higherKey(o);
                tm.put(higer, tm.get(higer) - 1);
                if (tm.get(higer) == 0) tm.remove(higer);
                tm.put(o, tm.getOrDefault(o, 0) + 1);
                bit.update(higer, -1);
                bit.update(o, 1);
            }
            result[i] = bit.sumRange(0, o);
        }
        return result;
    }

    // LC68 Hard
    public List<String> fullJustify(String[] words, int maxWidth) {
        Deque<String> wordQueue = new LinkedList<>();
        for (String w : words) wordQueue.offer(w);
        List<String> line = new ArrayList<>();
        List<String> result = new ArrayList<>();
        while (!wordQueue.isEmpty()) {
            // 当前行词的长度 + 下一个词的长度 + 最少空格个数 <= maxWidth
            while (!wordQueue.isEmpty() && lineWordLen(line) + wordQueue.peekFirst().length() + line.size() <= maxWidth) {
                line.add(wordQueue.pollFirst());
            }

            // 处理空格个数
            StringBuilder sb = new StringBuilder();
            // 除了最后一行之外不应该在右侧添加空格
            if (!wordQueue.isEmpty()) {
                // 植树问题, 间隙个数
                int intervalNum = line.size() - 1;
                if (intervalNum == 0) {
                    sb.append(line.get(0));
                    while (sb.length() < maxWidth) sb.append(" ");
                } else {
                    int totalSpace = maxWidth - lineWordLen(line);
                    int lowerBound = totalSpace / intervalNum;
                    int remain = totalSpace - lowerBound * intervalNum;

                    for (int i = 0; i < line.size() - 1; i++) {
                        sb.append(line.get(i));
                        for (int j = 0; j < lowerBound; j++) {
                            sb.append(" ");
                        }
                        if (remain != 0) {
                            sb.append(" ");
                            remain--;
                        }
                    }
                    sb.append(line.get(line.size() - 1));
                }

            } else {
                for (int i = 0; i < line.size() - 1; i++) {
                    sb.append(line.get(i));
                    sb.append(" ");
                }
                sb.append(line.get(line.size() - 1));
                while (sb.length() < maxWidth) {
                    sb.append(" ");
                }
            }
            result.add(sb.toString());
            line = new ArrayList<>();
        }
        return result;
    }

    private int lineWordLen(List<String> line) {
        return line.stream().mapToInt(o -> o.length()).sum();
    }

    // LC1647
    public int minDeletions(String s) {
        int result = 0;
        int[] freq = new int[26];
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
        }
        Arrays.sort(freq);
        for (int i = 24; i >= 0; i--) {
            if (freq[i] == 0) break;
            if (freq[i] >= freq[i + 1]) {
                int delta = Math.min(freq[i], Math.max(0, freq[i] - freq[i + 1]) + 1);
                freq[i] -= delta;
                result += delta;
            }
        }
        return result;
    }

    // LC862 Try TreeMap / Binary Search
    public int shortestSubarrayTS(int[] nums, int lowerBound) {
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        List<Pair<Integer, Integer>> stack = new ArrayList<>(); // <前缀和, 前缀和下标>
        // 缺少一个StackArray这样的Stack实现, 万幸ArrayList也不是不能用
        int right = 0, result = Integer.MAX_VALUE;
        while (right <= n) {
            // 单调栈, 同时需要O(1)/O(log(n))的寻址, TreeMap和普通的ArrayList都可以
            while (!stack.isEmpty() && stack.get(stack.size() - 1).getKey() >= prefix[right]) {
                stack.remove(stack.size() - 1);
            }

            if (!stack.isEmpty()) {
                int target = prefix[right] - lowerBound;
                int lo = 0, hi = stack.size() - 1;
                while (lo < hi) { // 相当于求floor
                    int mid = lo + (hi - lo + 1) / 2;
                    if (stack.get(mid).getKey() <= target) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                if (stack.get(lo).getKey() > target) {
                    // 无效下界
                } else {
                    result = Math.min(result, right - stack.get(lo).getValue());
                }
            }

            stack.add(new Pair<>(prefix[right], right));
            // 如果是treeMap put的话, 开始还担心重复key问题, 但实际上不需要担心, 因为如果key重复,
            // 则最小长度已经在这一轮循环中得到, 下一轮循环得到的更大的长度不会更新result
            // TreeMap最主要的缺点是寻址(lastKey)是O(log(n))的, 比单纯ArrayList的O(1)寻址慢
            right++;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }

    // LC862 ** Hard 单调队列
    public int shortestSubarray(int[] nums, int lowerBound) {
        int n = nums.length;
        Deque<Integer> dq = new LinkedList<>(); // 存的是prefix的下标, 存储一个单调增的前缀和下标, 队首最小
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        int right = 0, result = Integer.MAX_VALUE;
        while (right <= n) {
            while (!dq.isEmpty() && prefix[right] <= prefix[dq.peekLast()]) {
                dq.pollLast();
            }
            while (!dq.isEmpty() && prefix[right] - prefix[dq.peekFirst()] >= lowerBound) {
                result = Math.min(result, right - dq.pollFirst());
            }
            dq.offerLast(right);
            right++;
        }
        return result == Integer.MAX_VALUE ? -1 : result;

        // 顺带一提 如果是整个数组都是正数 前一个方法是正确的
        // int minLeftIdx = -1, minRightIdx = -1, minLen = Integer.MAX_VALUE / 2, curSum = 0;
        // for (int left = 0, right = 0; right < nums.length; right++) {
        //     curSum += nums[right];
        //     while (curSum > k && curSum - nums[left] >= k) curSum -= nums[left++];
        //     if (curSum >= k) {
        //         int len = right - left + 1;
        //         if (len < minLen) {
        //             minLen = len;
        //             minLeftIdx = left;
        //             minRightIdx = right;
        //         }
        //     }
        // }
        // if(minLeftIdx==-1&&minRightIdx==-1) return -1;
        // return minLen;
    }

    // LC420 ** Hard
    public int strongPasswordChecker(String password) {
        // 规则: 1) 小写字母、大写字母、数字至少各一个
        //      2) 不能有3个或以上连续的相同字符
        // 返回: 到符合规则为止的最少修改次数, 增删改都算一次修改
        char[] ca = password.toCharArray();
        int lc = 1, uc = 1, digit = 1;
        for (char c : ca) {
            if (Character.isLowerCase(c)) lc = 0;
            if (Character.isUpperCase(c)) uc = 0;
            if (Character.isDigit(c)) digit = 0;
        }
        int missing = lc + uc + digit;
        int ptr = 1, dupCount = 1;
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o % 3));
        while (ptr < ca.length) {
            if (ca[ptr - 1] == ca[ptr]) {
                dupCount++;
            } else {
                if (dupCount > 2) {
                    pq.offer(dupCount);
                }
                dupCount = 1;
            }
            ptr++;
        }
        if (dupCount > 2) pq.offer(dupCount);

        if (ca.length < 6) {
            return Math.max(6 - ca.length, missing);
        } else {
            int len = ca.length, replace = 0, delete = 0, result = 0;
            while (len > 20 && !pq.isEmpty()) { //  优先删除连续段中长度为3的倍数(模3=0)的连续段中的元素, 因为剩余的连续段中, 长度模3=0的替换效率最低
                int p = pq.poll();              //  考虑以下算例: 有 4 3 3 三个连续段, 总长度为22, 如果删除4,3中的一个元素, 剩余连续段 3 3, 仍需替换2次
                //  如果删除3,3中的各一个元素, 剩余连续段4, 只需替换1次
                //  又比如 6 6 4 三个连续段, 总长度22, 如果删除6,4 中的一个元素, 剩余连续段6 5 3, 仍需替换4次
                //  若删除 6 6 中的一个元素, 剩余连续段 5 5 4, 只需替换3次
                result++;  // 删除p的一个元素, 操作次数+1
                len--;
                if (p - 1 > 2) pq.offer(p - 1);
            }
            if (len > 20) { // 如果连续段都没了, 但还是太长了, 只能删除, 再替换缺失的元素
                result += len - 20 + missing;
            } else { // 如果还有连续段
                while (!pq.isEmpty()) {
                    int p = pq.poll();
                    replace += p / 3; // 处理一次需要替换 len/3个字符
                }
                result += Math.max(replace, missing); // 如果替换个数只有1个, 而缺失元素有2个, 则将一个连续元素替换, 然后再增加一个缺失元素, 也就是两次操作
                // 其他情况都是需要替换的更多, 直接取最大值
            }
            return result;
        }
    }

    // Interview 16.03 Hard 判断线段有无交点 写得太丑陋了
    public double[] intersection(int[] start1, int[] end1, int[] start2, int[] end2) {
        // 1 计算斜率 判断是否平行
        boolean parallel = false;
        boolean[] vertical = new boolean[3], horizontal = new boolean[3];
        double k1 = Integer.MAX_VALUE / 2, k2 = Integer.MAX_VALUE / 2;
        double b1 = Integer.MAX_VALUE / 2, b2 = Integer.MAX_VALUE / 2;
        double[] xRange1 = new double[2], xRange2 = new double[2];
        double[] yRange1 = new double[2], yRange2 = new double[2];
        xRange1[0] = start1[0] < end1[0] ? start1[0] : end1[0];
        xRange1[1] = start1[0] > end1[0] ? start1[0] : end1[0];
        xRange2[0] = start2[0] < end2[0] ? start2[0] : end2[0];
        xRange2[1] = start2[0] > end2[0] ? start2[0] : end2[0];

        yRange1[0] = start1[1] < end1[1] ? start1[1] : end1[1];
        yRange1[1] = start1[1] > end1[1] ? start1[1] : end1[1];
        yRange2[0] = start2[1] < end2[1] ? start2[1] : end2[1];
        yRange2[1] = start2[1] > end2[1] ? start2[1] : end2[1];
        // 1) 判断是否平行于Y轴

        if (start1[0] - end1[0] == 0) vertical[1] = true;
        if (start2[0] - end2[0] == 0) vertical[2] = true;
        if (start1[1] - end1[1] == 0) horizontal[1] = true;
        if (start2[1] - end2[1] == 0) horizontal[2] = true;
        if (vertical[1] && vertical[2]) parallel = true;
        if (horizontal[1] && horizontal[2]) parallel = true;

        if (!parallel) {
            if (!vertical[1]) {
                k1 = (start1[1] - end1[1] + 0d) / (start1[0] - end1[0] + 0d);
                b1 = start1[1] - k1 * start1[0]; // b = y1-kx1 = y2-kx2
            }
            if (!vertical[2]) {
                k2 = (start2[1] - end2[1] + 0d) / (start2[0] - end2[0] + 0d);
                b2 = start2[1] - k2 * start2[0];
            }
            if (k1 == k2) { // 平行
                parallel = true;
            }
        }

        if (parallel) {
            if (vertical[1] && vertical[2]) {
                if (start1[0] == start2[0]) {
                    // 判断有无重叠
                    if (yRange1[0] < yRange2[0]) {
                        if (yRange1[1] < yRange2[0]) {
                            return new double[]{};
                        } else {
                            return new double[]{start1[0], yRange2[0]};
                        }
                    } else if (yRange1[0] > yRange2[0]) {
                        if (yRange2[1] < yRange1[0]) {
                            return new double[]{};
                        } else {
                            return new double[]{start1[0], yRange1[0]};
                        }
                    } else {
                        return new double[]{start1[0], yRange1[0]};
                    }
                } else {
                    return new double[]{};
                }
            }
            if (horizontal[1] && horizontal[2]) {
                if (start1[1] == start2[1]) { // 两线水平, 距离Y轴方向相等即重叠
                    // 判断有无重叠
                    if (xRange1[0] < xRange2[0]) {
                        if (xRange1[1] < xRange2[0]) {
                            return new double[]{};
                        } else {
                            return new double[]{xRange2[0], start1[1]};
                        }
                    } else if (xRange1[0] > xRange2[0]) {
                        if (xRange2[1] < xRange1[0]) {
                            return new double[]{};
                        } else {
                            return new double[]{xRange1[0], start1[1]};
                        }
                    } else {
                        return new double[]{xRange1[0], start1[1]};
                    }
                } else {
                    return new double[]{};
                }
            }
            if (b1 != b2) { // 截距不同必定无交点
                return new double[]{};
            }
            // 返回x最小的点, 都一样小则返回y最小的点, 因为两条斜线的四个点不可能x都一样, 所以根据x判断即可

            // 判断有无重叠
            if (xRange1[0] < xRange2[0]) {
                if (xRange1[1] < xRange2[0]) {
                    return new double[]{};
                } else {
                    return new double[]{xRange2[0], k1 * xRange2[0] + b1};
                }
            } else if (xRange1[0] > xRange2[0]) {
                if (xRange2[1] < xRange1[0]) {
                    return new double[]{};
                } else {
                    return new double[]{xRange1[0], k1 * xRange1[0] + b1};
                }
            } else {
                return new double[]{xRange1[0], start1[1]};
            }

        } else {
            // 若不平行
            if (vertical[1] || vertical[2]) {
                if (vertical[1]) {
                    double y = start1[0] * k2 + b2;
                    if (y >= yRange1[0] && y <= yRange1[1] && y >= yRange2[0] && y <= yRange2[1]) {
                        return new double[]{start1[0], y};
                    } else {
                        return new double[]{};
                    }
                } else {
                    double y = start2[0] * k1 + b1;
                    if (y >= yRange1[0] && y <= yRange1[1] && y >= yRange2[0] && y <= yRange2[1]) {
                        return new double[]{start2[0], y};
                    } else {
                        return new double[]{};
                    }
                }
            }
            if (horizontal[1] || horizontal[2]) {
                if (horizontal[1]) {
                    double x = (start1[1] - b2) / k2;
                    if (x >= xRange1[0] && x <= xRange1[1] && x >= xRange2[0] && x <= xRange2[1]) {
                        return new double[]{x, start1[1]};
                    } else {
                        return new double[]{};
                    }
                } else {
                    double x = (start2[1] - b1) / k1;
                    if (x >= xRange1[0] && x <= xRange1[1] && x >= xRange2[0] && x <= xRange2[1]) {
                        return new double[]{x, start2[1]};
                    } else {
                        return new double[]{};
                    }
                }
            }
            double x = (b2 - b1) / (k1 - k2);
            double y = k1 * x + b1;
            if (x >= xRange1[0] && x <= xRange1[1] && x >= xRange2[0] && x <= xRange2[1]
                    && y >= yRange1[0] && y <= yRange1[1] && y >= yRange2[0] && y <= yRange2[1]) {
                return new double[]{x, y};
            } else {
                return new double[]{};
            }
        }
    }

    // LC502 Hard
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return profits[o1] == profits[o2] ? capital[o1] - capital[o2] : profits[o2] - profits[o1];
            }
        });
        Set<Integer> notAllow = new HashSet<>();
        for (int i = 0; i < profits.length; i++) {
            if (capital[i] <= w) {
                pq.offer(i);
            } else {
                notAllow.add(i);
            }
        }
        while (k != 0 && !pq.isEmpty()) {
            w += profits[pq.poll()];
            k--;
            Iterator<Integer> it = notAllow.iterator();
            while (it.hasNext()) {
                int next = it.next();
                if (capital[next] <= w) {
                    pq.offer(next);
                    it.remove();
                }
            }
        }
        return w;
    }

    // LC906
    public int superpalindromesInRange(String left, String right) {
        long l = Long.valueOf(left), r = Long.valueOf(right);
        String rightLeftHalf = right.substring(0, (right.length() + 1) / 2);
        long upperBound = (long) Math.sqrt(Long.valueOf(rightLeftHalf)) + 1;
        List<Long> result = new ArrayList<>();
        if (l <= 1) result.add(1l);
        if (4l >= l && 4 <= r) result.add(4l);
        if (9l >= l && 9 <= r) result.add(9l);
        for (int i = 1; i <= upperBound; i++) {
            // 构造回文数
            String str = String.valueOf(i);
            // 如果长度大于等于2, 可以不重复最右侧的数
            if (str.length() > 1) {
                String pal = str + new StringBuilder(str.substring(0, str.length() - 1)).reverse().toString();
                long test = Long.valueOf(pal);
                long testPow2 = test * test;
                if (checkPal(testPow2) && testPow2 >= l && testPow2 <= r) {
                    result.add(testPow2);
                }
            }
            String pal = str + new StringBuilder(str).reverse().toString();
            long test = Long.valueOf(pal);
            long testPow2 = test * test;
            if (checkPal(testPow2) && testPow2 >= l && testPow2 <= r) {
                result.add(testPow2);
            }
        }
        return result.size();
    }

    private boolean checkPal(long num) {
        if (num < 10) return true;
        List<Integer> l = new ArrayList<>();
        while (num != 0) {
            l.add((int) (num % 10));
            num /= 10;
        }
        int mid = l.size() / 2, len = l.size();
        for (int i = 0; i < mid; i++) {
            if (l.get(i) != l.get(len - 1 - i)) return false;
        }
        return true;
    }

    // LC370
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] result = new int[length];
        BIT bit = new BIT(length);
        for (int[] u : updates) {
            bit.update(u[0], u[2]);
            bit.update(u[1] + 1, -u[2]);
        }
        result[0] = bit.get(0);
        for (int i = 1; i < length; i++) {
            result[i] = result[i - 1] + bit.get(i);
        }
        return result;
    }

    // LC1265
    public void printLinkedListInReverse(ImmutableListNode head) {
        if (head == null) return;
        printLinkedListInReverse(head.getNext());
        head.printValue();
    }

    interface ImmutableListNode {
        public void printValue(); // print the value of this node.

        public ImmutableListNode getNext(); // return the next node.
    }

    ;

    // LC1660
    Map<TreeNode, TreeNode> lc1660Parent;

    public TreeNode correctBinaryTree(TreeNode root) {
        lc1660Parent = new HashMap<>();
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            if (lc1660Parent.get(p.right) != null) {
                TreeNode victim = p;
                TreeNode vp = lc1660Parent.get(victim);
                if (vp.left == victim) vp.left = null;
                else if (vp.right == victim) vp.right = null;
                break;
            }
            if (p.left != null) {
                q.offer(p.left);
                lc1660Parent.put(p.left, p);
            }
            if (p.right != null) {
                q.offer(p.right);
                lc1660Parent.put(p.right, p);
            }
        }
        return root;
    }

    // LCP 34 ** 树形DP
    int lcp34Limit;

    public int maxValue(TreeNode root, int k) {
        lcp34Limit = k;
        int[] result = lcp34Dfs(root);
        return Arrays.stream(result).max().getAsInt();
    }

    private int[] lcp34Dfs(TreeNode root) {
        int[] dp = new int[lcp34Limit + 1];
        if (root == null) return dp;
        int[] left = lcp34Dfs(root.left);
        int[] right = lcp34Dfs(root.right);
        // 不染 root, 两个子树都可以染色, 数量不限
        dp[0] = Arrays.stream(left).max().getAsInt() + Arrays.stream(right).max().getAsInt();

        // 染色 root, 则 左+右最多 = limit-1
        for (int i = 1; i <= lcp34Limit; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = Math.max(dp[i], root.val + left[j] + right[i - 1 - j]);
            }
        }
        return dp;
    }

    // LC1882
    public int[] assignTasks(int[] servers, int[] tasks) {
        // servers : 权重
        // tasks: 任务耗时
        // 第i秒时才可以开始执行tasks[i]任务

        int[] result = new int[tasks.length];

        // availServer: [权重, 下标]
        PriorityQueue<int[]> availServer = new PriorityQueue<>((o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o1[0] - o2[0]);
        // working: [结束时间, 下标]
        PriorityQueue<int[]> working = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
        // availTask : [下标, 耗时]
        Deque<int[]> availTask = new LinkedList<>();
        for (int i = 0; i < servers.length; i++) {
            availServer.offer(new int[]{servers[i], i});
        }
        int taskPtr = 0, timing = 0;
        while (taskPtr < tasks.length || !availTask.isEmpty()) {
            if (taskPtr < tasks.length) {
                availTask.offer(new int[]{taskPtr, tasks[taskPtr]});
                taskPtr++;
                // 先执行退出working队列检查
                while (!working.isEmpty() && working.peek()[0] <= timing) {
                    int[] happy = working.poll();
                    availServer.offer(new int[]{servers[happy[1]], happy[1]});
                }
                // 再执行进入working队列操作 (顺序很重要!!!)
                while (!availServer.isEmpty() && !availTask.isEmpty()) {
                    int[] victim = availServer.poll();
                    int[] workload = availTask.poll();
                    working.offer(new int[]{workload[1] + timing, victim[1]});
                    result[workload[0]] = victim[1];
                }
                timing++;
            } else {
                // 此时全部任务都已经进入可用任务队列, 队列中仍有元素,  等待working 中的工作完成
                if (!working.isEmpty()) {
                    int[] workTop = working.poll();
                    timing = workTop[0];
                    availServer.offer(new int[]{servers[workTop[1]], workTop[1]});
                    while (!working.isEmpty() && working.peek()[0] <= timing) {
                        int[] happy = working.poll();
                        availServer.offer(new int[]{servers[happy[1]], happy[1]});
                    }
                }
                while (!availServer.isEmpty() && !availTask.isEmpty()) {
                    int[] victim = availServer.poll();
                    int[] workload = availTask.poll();
                    working.offer(new int[]{workload[1] + timing, victim[1]});
                    result[workload[0]] = victim[1];
                }
            }
        }
        return result;
    }

    // JZOF II 088
    public int minCostClimbingStairs(int[] cost) {
        int[] sum = new int[]{cost[0], cost[1], 0};
        for (int i = 2; i < cost.length; i++) {
            sum[i % 3] = Math.min(sum[(i - 2) % 3] + cost[i], sum[(i - 1) % 3] + cost[i]);
        }
        return Math.min(sum[(cost.length - 1) % 3], sum[(cost.length - 2) % 3]);
    }

    // JZOF II 017 LC76 **
    public String minWindow(String s, String t) {
        int m = s.length(), n = t.length(), overlap = 0;
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        String result = "";
        int[] freq = new int[256];
        for (char c : ct) freq[c]--;
        for (int left = 0, right = 0; right < m; right++) {
            freq[cs[right]]++;
            if (freq[cs[right]] <= 0) overlap++;
            while (overlap == n && freq[cs[left]] > 0) freq[cs[left++]]--;
            if (overlap == n) {
                if (result.equals("") || result.length() > right - left + 1) {
                    result = s.substring(left, right + 1);
                }
            }
        }
        return result;
    }

    // LC1088
    int lc1088Result = 0;
    int[] lc1088Valid = new int[]{0, 1, 6, 8, 9};
    int lc1088Limit;

    public int confusingNumberII(int n) {
        lc1088Limit = n;
        lc1088Backtrack(0);
        return lc1088Result;
    }

    private void lc1088Backtrack(long cur) {
        if (lc1088Check(cur)) lc1088Result++;
        for (int i : lc1088Valid) {
            cur = cur * 10 + i;
            if (cur >= 1 && cur <= lc1088Limit) lc1088Backtrack(cur);
            cur /= 10;
        }
    }

    // LC1056
    private boolean lc1088Check(long num) {
        long reversed = 0, orig = num;
        while (num != 0) {
            long next = num % 10;
            // LC1056 Part
            // boolean flag = false;
            // for (long v : lc1088Valid) if (next == v) flag = true;
            // if (!flag) return false;
            if (next == 9) next = 6;
            else if (next == 6) next = 9;
            reversed = reversed * 10 + next;
            num /= 10;
        }
        return reversed != orig;
    }

    // LC1599
    public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
        int waiting = 0, maxProfit = -1, curProfit = 0, cstPtr = 0, boardPtr = 0, board = 0, maxCount = 0, rotateCount = 0;
        while (waiting != 0 || cstPtr != customers.length) {
            if (cstPtr < customers.length) waiting += customers[cstPtr++];
            if (waiting > 4) {
                board = 4;
            } else {
                board = waiting;
            }
            waiting -= board;
            rotateCount++;
            curProfit += board * boardingCost - runningCost;
            if (curProfit > maxProfit) {
                maxProfit = curProfit;
                maxCount = rotateCount;
            }
        }
        if (maxCount < 4) maxProfit -= (4 - maxCount) * runningCost;
        return maxProfit < 0 ? -1 : maxCount;
    }

    // LC1221
    public int balancedStringSplit(String s) {
        char[] ca = s.toCharArray();
        int diff = 0, count = 0, ptr = 0;
        while (ptr != ca.length) {
            if (ca[ptr++] == 'L') diff++;
            else diff--;
            if (diff == 0) count++;
        }
        return count;
    }

    // LC749 The implementation is long.
    public int containVirus(int[][] isInfected) {
        int m = isInfected.length, n = isInfected[0].length;
        Function<int[], Boolean> checkLegalPos = new Function<int[], Boolean>() {
            @Override
            public Boolean apply(int[] pos) {
                return pos[0] >= 0 && pos[0] < m && pos[1] >= 0 && pos[1] < n;
            }
        };
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int totalCount = m * n, stillInfectCount = 0, wallCount = 0, isolatedCount = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                stillInfectCount += isInfected[i][j];
            }
        }

        while (stillInfectCount != 0) {
            DisjointSetUnion dsu = new DisjointSetUnion();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int idx = i * n + j;
                    if (isInfected[i][j] == 1) {
                        dsu.add(idx);
                        for (int[] dir : directions) {
                            int nr = i + dir[0], nc = j + dir[1];
                            int nIdx = nr * n + nc;
                            if (checkLegalPos.apply(new int[]{nr, nc}) && isInfected[nr][nc] == 1) {
                                dsu.add(nIdx);
                                dsu.merge(idx, nIdx);
                            }
                        }
                    }
                }
            }
            Map<Integer, Set<Integer>> allGroups = dsu.getAllGroups();
            Set<Integer> victim = null;
            int maxCountInfect = -1;
            for (Set<Integer> virus : allGroups.values()) {
                boolean[][] visited = new boolean[m][n];
                int countInfect = 0;
                for (int idx : virus) {
                    int r = idx / n, c = idx % n;
                    for (int[] dir : directions) {
                        int nr = r + dir[0], nc = c + dir[1];
                        if (checkLegalPos.apply(new int[]{nr, nc})) {
                            if (!visited[nr][nc] && isInfected[nr][nc] == 0) {
                                visited[nr][nc] = true;
                                countInfect++;
                            }
                        }
                    }
                }
                if (countInfect > maxCountInfect) {
                    maxCountInfect = countInfect;
                    victim = virus;
                }
            }
            isolatedCount += victim.size();
            int wallCountDiff = 0;
            boolean[][] wallVisited = new boolean[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (!wallVisited[i][j] && isInfected[i][j] == 0) {
                        for (int[] dir : directions) {
                            int nr = i + dir[0], nc = j + dir[1];
                            int nIdx = nr * n + nc;
                            if (checkLegalPos.apply(new int[]{nr, nc}) && victim.contains(nIdx)) {
                                wallCountDiff++;
                            }
                        }
                        wallVisited[i][j] = true;
                    }
                }
            }
            wallCount += wallCountDiff;
            for (int idx : victim) {
                int r = idx / n, c = idx % n;
                isInfected[r][c] = 2; // 用2表示已被隔离
            }
            for (Set<Integer> virus : allGroups.values()) {
                if (virus != victim) {
                    for (int idx : virus) {
                        int r = idx / n, c = idx % n;
                        for (int[] dir : directions) {
                            int nr = r + dir[0], nc = c + dir[1];
                            if (checkLegalPos.apply(new int[]{nr, nc})) {
                                if (isInfected[nr][nc] == 0) {
                                    isInfected[nr][nc] = 1;
                                }
                            }
                        }
                    }
                }
            }
            stillInfectCount = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (isInfected[i][j] == 1) stillInfectCount++;
                }
            }
            if (stillInfectCount == totalCount - isolatedCount) return wallCount;
        }
        return wallCount;
    }

    // LC501
    Map<Integer, Integer> lc501Freq;

    public int[] findMode(TreeNode root) {
        lc501Freq = new HashMap<>();
        lc501Dfs(root);
        List<Integer> k = new ArrayList<>(lc501Freq.keySet());
        k.sort(Comparator.comparingInt(o -> -lc501Freq.get(o)));
        int maxFreq = lc501Freq.get(k.get(0));
        int end = 0;
        for (int i : k) {
            if (lc501Freq.get(i) != maxFreq) break;
            end++;
        }
        return k.subList(0, end).stream().mapToInt(Integer::valueOf).toArray();
    }

    private void lc501Dfs(TreeNode root) {
        if (root == null) return;
        lc501Freq.put(root.val, lc501Freq.getOrDefault(root.val, 0) + 1);
        lc501Dfs(root.left);
        lc501Dfs(root.right);
    }

    // LCP 12 TBD 参考 LC410
    public int minTime(int[] time, int m) {
        // m 天 完成 time.length 题, 按顺序做题, 每天耗时最长的一题可以不计入耗时, 求最长的一天的耗时
        int lo = 0, hi = Arrays.stream(time).sum();
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (lcp12Helper(time, mid) <= m) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int lcp12Helper(int[] nums, int segLen) {
        int count = 1, sum = 0, curMax = 0; // 多维护一个当前最大值, 判断是否大于segLen的时候先减去当前最大值
        for (int i : nums) {
            if (sum + i - Math.max(i, curMax) > segLen) {
                sum = i;
                curMax = i;
                count++;
            } else {
                sum += i;
                curMax = Math.max(curMax, i);
            }
        }
        return count;
    }

    // LC410 二分
    public int splitArrayBS(int[] nums, int m) {
        int n = nums.length;
        int lo = Arrays.stream(nums).max().getAsInt();
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        int hi = prefix[n];
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (lc410CountSeg(prefix, mid) <= m) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private int lc410CountSeg(int[] prefix, int segLen) {
        int count = 0, ptr = 0;
        int len = prefix.length - 1;
        while (ptr < len) {
            int lo = ptr, hi = len;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                if (prefix[mid] <= prefix[ptr] + segLen) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            count++;
            ptr = lo;
        }
        return count;
    }

    // LC410 Hard Minmax, 极小化极大, DFS
    Integer[][] lc410Memo;

    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        lc410Memo = new Integer[n + 1][m + 1];
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        return lc410Helper(prefix, 0, m);
    }

    private int lc410Helper(int[] prefix, int begin, int leftSegNum) {
        if (leftSegNum == 1) {
            return prefix[prefix.length - 1] - prefix[begin];
        }

        if (lc410Memo[begin][leftSegNum] != null) return lc410Memo[begin][leftSegNum];
        int result = 0x3f3f3f3f;
        int len = prefix.length - 1;

        // 理想值: 即平均分配所有划分的和, 实际的最小值总是大于等于理想值, 所以可以找到小于等于理想值的第一个下标开始枚举
        int ideal = (prefix[len + 1] - prefix[begin]) / leftSegNum;
        int lo = begin, hi = len;

        // 找到小于等于ideal的第一个下标
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (prefix[mid] <= prefix[begin] + ideal) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        // 枚举下一子数组的开始下标i
        // 初始化 i = begin+1 是因为子数组的长度最少为1
        // i<= nums.length-(segNum-1): 保证剩下的子数组至少分配到有1个数
        // for (int i = begin+1; i <= len - (leftSegNum - 1); i++) {
        for (int i = lo; i <= len - (leftSegNum - 1); i++) {
            if (lo == 0) continue;
            int sum = prefix[i] - prefix[begin];
            int maxSum = Math.max(sum, lc410Helper(prefix, i, leftSegNum - 1));
            result = Math.min(result, maxSum);
        }

        return lc410Memo[begin][leftSegNum] = result;
    }

    // Interview 01.02
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int[] freq = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            freq[s1.charAt(i) - 'a']++;
            freq[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) if (freq[i] != 0) return false;
        return true;
    }

    // LC1263 Hard
    public int minPushBox(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> q = new LinkedList<>();
        boolean[][][][] visited = new boolean[m][n][m][n];
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int[] box = new int[]{-1, -1}, target = new int[]{-1, -1}, self = new int[]{-1, -1};
        Function<int[], Boolean> checkLegalPos = new Function<int[], Boolean>() {
            @Override
            public Boolean apply(int[] pos) {
                return pos[0] >= 0 && pos[0] < m && pos[1] >= 0 && pos[1] < n && grid[pos[0]][pos[1]] != '#';
            }
        };
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                switch (grid[i][j]) {
                    case 'S':
                        self = new int[]{i, j};
                        break;
                    case 'T':
                        target = new int[]{i, j};
                        break;
                    case 'B':
                        box = new int[]{i, j};
                    default:
                        continue;
                }
            }
        }
        // [selfRow, selfCol, boxRow, boxCol]
        int[] initState = new int[]{self[0], self[1], box[0], box[1]};
        q.offer(initState);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]][p[2]][p[3]]) continue;
                visited[p[0]][p[1]][p[2]][p[3]] = true;
                int selfRow = p[0], selfCol = p[1];
                int boxRow = p[2], boxCol = p[3];
                if (boxRow == target[0] && boxCol == target[1]) return layer;

                // 去到箱子的旁边推箱子
                // 1. 确定箱子四周是否是障碍物, 找出立足点和相对方向上的目标点
                // 2. 确定是否有路径到这些立足点, 此时应把箱子本身也视作障碍物
                // 3. 若有路到立足点, 向队列推入[箱子位置, 新箱子位置](玩家到了箱子的位置)

                // 立足点和箱子目标位置的delta_row,delta_col可以通过简单取负数得到(因为两个delta必有一个是0)
                List<Pair<int[], int[]>> legalStandPointTargetPosList = new ArrayList<>();
                for (int[] dir : directions) {
                    int[] standPoint = new int[]{boxRow + dir[0], boxCol + dir[1]};
                    int[] targetPos = new int[]{boxRow - dir[0], boxCol - dir[1]};
                    if (checkLegalPos.apply(standPoint) && checkLegalPos.apply(targetPos)) {
                        legalStandPointTargetPosList.add(new Pair<>(standPoint, targetPos));
                    }
                }

                for (Pair<int[], int[]> pair : legalStandPointTargetPosList) {
                    boolean[][] innerVisited = new boolean[m][n];
                    int[] innerTarget = pair.getKey();
                    int[] innerStartPoint = new int[]{selfRow, selfCol};
                    Deque<int[]> innerQ = new LinkedList<>();
                    boolean canReach = false;
                    // 这里可以用offer/poll 做bfs, 也可以用push/pop 做dfs
                    innerQ.offer(innerStartPoint);
                    while (!innerQ.isEmpty()) {
                        int[] innerP = innerQ.poll();
                        if (innerP[0] == innerTarget[0] && innerP[1] == innerTarget[1]) {
                            canReach = true;
                            break;
                        }
                        if (innerVisited[innerP[0]][innerP[1]]) continue;
                        innerVisited[innerP[0]][innerP[1]] = true;
                        for (int[] dir : directions) {
                            int nextRow = innerP[0] + dir[0], nextCol = innerP[1] + dir[1];
                            int[] next = new int[]{nextRow, nextCol};
                            if (checkLegalPos.apply(next) && !(nextRow == boxRow && nextCol == boxCol) && !innerVisited[nextRow][nextCol]) {
                                innerQ.offer(next);
                            }
                        }
                    }
                    if (canReach) {
                        // 若推得动, 则此时玩家位置变为原箱子位置, 箱子位置变为targetPos(即pair.getValue())
                        q.offer(new int[]{boxRow, boxCol, pair.getValue()[0], pair.getValue()[1]});
                    }
                }
            }
        }
        return -1;
    }

    // LCP10 Hard **
    public double minimalExecTime(TreeNode root) {
        double[] result = betterDfs(root);
        return result[1];
    }

    // 返回[任务总时间, 最短执行时间]
    private double[] betterDfs(TreeNode root) {
        if (root == null) return new double[]{0, 0};
        double[] left = betterDfs(root.left);
        double[] right = betterDfs(root.right);
        return new double[]{
                left[0] + right[0] + root.val,
                root.val + Math.max(Math.max(left[1], right[1]), (left[0] + right[0]) / 2d)
        };
    }

    // LC1103
    public int[] distributeCandies(int candies, int num_people) {
        int[] result = new int[num_people];
        int ptr = 0;
        while (candies != 0) {
            if (candies >= ptr + 1) {
                result[ptr % num_people] += ptr + 1;
                candies -= ptr + 1;
                ptr++;
            } else {
                result[ptr % num_people] += candies;
                break;
            }
        }
        return result;
    }

    // LC1719 Hard **

    // DFS
    Map<Integer, Set<Integer>> lc1719DfsMap;

    public int checkWaysDfs(int[][] pairs) {
        lc1719DfsMap = new HashMap<>();
        for (int[] p : pairs) {
            lc1719DfsMap.putIfAbsent(p[0], new HashSet<>());
            lc1719DfsMap.putIfAbsent(p[1], new HashSet<>());
            lc1719DfsMap.get(p[0]).add(p[1]);
            lc1719DfsMap.get(p[1]).add(p[0]);
        }
        int numEle = lc1719DfsMap.size();
        List<Integer> validNodes = new ArrayList<>(lc1719DfsMap.keySet());
        validNodes.sort(Comparator.comparingInt(o -> -lc1719DfsMap.get(o).size()));
        if (lc1719DfsMap.get(validNodes.get(0)).size() != numEle - 1) return 0;
        int rootNode = validNodes.get(0);
        return lc1719Dfs(rootNode);
    }

    private int lc1719Dfs(int root) {
        Set<Integer> subNode = new HashSet<>(lc1719DfsMap.get(root)); // root的children, 记subnode
        lc1719DfsMap.get(root).clear();
        for (int c : subNode) {
            lc1719DfsMap.get(c).remove(root);
        }
        boolean multi = false;
        List<Integer> subNodeList = new ArrayList<>(subNode);
        subNodeList.sort(Comparator.comparingInt(o -> -lc1719DfsMap.get(o).size()));
        for (int c : subNodeList) {
            // subNode 的 children
            for (int snc : lc1719DfsMap.get(c)) {
                if (!subNode.contains(snc)) return 0;
            }
            if (lc1719DfsMap.get(c).size() == subNode.size() - 1) { // -1是因为之前remove了root
                multi = true;
            }
            int result = lc1719Dfs(c);
            if (result == 0) return 0;
            if (result == 2) multi = true;
        }
        return multi ? 2 : 1;
    }

    public int checkWays(int[][] pairs) {
        final int maxSize = 501;
        int result = 1;
        int[] parent = new int[maxSize];
        Map<Integer, List<Integer>> mtx = new HashMap<>();
        for (int[] p : pairs) {
            mtx.putIfAbsent(p[0], new ArrayList<>());
            mtx.putIfAbsent(p[1], new ArrayList<>());
            mtx.get(p[0]).add(p[1]);
            mtx.get(p[1]).add(p[0]);
            parent[p[0]] = parent[p[1]] = -1;
        }
        int numEle = mtx.size();
        List<Integer> validEle = new ArrayList<>(mtx.keySet());
        validEle.sort(Comparator.comparingInt(o -> -mtx.get(o).size()));
        if (mtx.get(validEle.get(0)).size() != numEle - 1) return 0;
        boolean[] visited = new boolean[maxSize];
        for (int u : validEle) { // 按照关系数倒序排序, 按此顺序遍历后遍历到的只能是前面的子节点
            for (int v : mtx.get(u)) {
                if (mtx.get(u).size() == mtx.get(v).size()) result = 2;
                if (!visited[v]) {
                    // parent[v] 是 当前已知的v最近的父节点, 如果p[u]!=p[v],
                    // 说明v有比u更近的父节点(否则在p[v]未更新前, p[v]应该和p[u]拥有相同的最近父节点
                    // p[v]!=p[u] 说明v在不从属于u的一支中已被更新, 而u又是v一个可能的父节点, 说明v会存在两个入度, 此时无解
                    if (parent[u] != parent[v]) return 0;
                    parent[v] = u;
                }
            }
            visited[u] = true; // 将确定了所有可能孩子的u加入已访问
        }
        return result;
    }

    // LC798 Hard ** 差分数组 学习分析方法
    // https://leetcode-cn.com/problems/smallest-rotation-with-highest-score/solution/chai-fen-shu-zu-by-sssz-qdut/
    public int bestRotation(int[] nums) {
        // 数组向左平移轮转
        int n = nums.length;
        int[] diff = new int[n + 1];
        // [0,Math.min(i,i-arr[i])], [i+1,Math.min(i-arr[i]+arr.len),arr.len-1)] 为可以得分的范围
        for (int i = 0; i < n; i++) {
            diff[0]++;
            int r1 = Math.min(i, i - nums[i]) + 1;
            if (r1 >= 0)
                diff[r1]--;
            diff[i + 1]++;
            int r2 = Math.min(i - nums[i] + n, n - 1) + 1;
            if (r2 >= 0 && r2 <= n)
                diff[r2]--;
        }
        int accumulate = 0, max = 0, maxIdx = -1;
        for (int i = 0; i < n; i++) {
            accumulate += diff[i];
            if (accumulate > max) {
                max = accumulate;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Interview 17.14
    public int[] smallestK(int[] arr, int k) {
        quickSelect.topK(arr, arr.length - k);
        return Arrays.copyOfRange(arr, 0, k);
    }

    // Interview 08.08 全排列
    List<String> iv0808Result;

    public String[] permutation(String S) {
        iv0808Result = new ArrayList<>();
        iv0808Dfs(S.toCharArray(), 0);
        return iv0808Result.toArray(new String[iv0808Result.size()]);
    }

    private void iv0808Dfs(char[] ca, int cur) {
        if (cur == ca.length) {
            iv0808Result.add(new String(ca));
        }
        Set<Character> set = new HashSet<>();
        for (int i = cur; i < ca.length; i++) {
            if (!set.contains(ca[i])) {
                set.add(ca[i]);
                char tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
                iv0808Dfs(ca, cur + 1);
                tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
            }
        }
    }

    // LC905
    public int[] sortArrayByParity(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] % 2 == 1) {
                int tmp = nums[right];
                nums[right] = nums[left];
                nums[left] = tmp;
                left--;
                right--;
            }
            left++;
        }
        return nums;
    }

    // JZOF II 056 LC653
    List<Integer> lc653List = new ArrayList<>();

    public boolean findTarget(TreeNode root, int k) {
        lc653Inorder(root);
        int left = 0, right = lc653List.size() - 1;
        while (left < right) {
            if (lc653List.get(left) + lc653List.get(right) > k) {
                right--;
            } else if (lc653List.get(left) + lc653List.get(right) < k) {
                left++;
            } else {
                return true;
            }
        }
        return false;
    }

    private void lc653Inorder(TreeNode root) {
        if (root == null) return;
        lc653Inorder(root.left);
        lc653List.add(root.val);
        lc653Inorder(root.right);
    }

    // LC1611 Hard ** Reverse Gray Code
    public int minimumOneBitOperations(int n) {
        // Function<Integer, Integer> gray = orig -> orig ^ (orig >> 1);
        int orig = 0;
        while (n != 0) {
            orig ^= n;
            n >>= 1;
        }
        return orig;
    }

    // LC468
    public String validIPAddress(String IP) {
        final String NEITHER = "Neither", IPV4 = "IPv4", IPV6 = "IPv6";
        if (checkIpv4(IP)) return IPV4;
        if (checkIpv6(IP)) return IPV6;
        return NEITHER;
    }

    private boolean checkIpv4(String ip) {
        String[] groups = ip.split("\\.", -1);
        if (groups.length != 4) return false;
        for (String w : groups) {
            for (char c : w.toCharArray()) if (!Character.isDigit(c)) return false;
            if (w.length() > 3 || w.length() == 0) return false;
            if (w.charAt(0) == '0' && w.length() != 1) return false;
            if (Integer.valueOf(w) > 255 || Integer.valueOf(w) < 0) return false;
        }
        return true;
    }

    private boolean checkIpv6(String ip) {
        String[] groupsByTwoColon = ip.split("::", -1);
        if (groupsByTwoColon.length > 2) return false; // 注意LC题目这里要改成>1, 因为题目不允许双冒号表示
        String[] groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length > 8) return false;
        for (String w : groupsByOneColon) {
            if (w.length() > 4) return false;
            String wlc = w.toLowerCase();
            for (char c : wlc.toCharArray()) {
                if (!Character.isLetter(c) && !Character.isDigit(c)) return false;
                if (Character.isLetter(c) && c > 'f') return false;
            }
        }
        int numColon = groupsByOneColon.length - 1;
        if (numColon != 7) {
            StringBuilder sb = new StringBuilder();
            int remain = 7 - numColon + 1;
            for (int i = 0; i < remain; i++) {
                sb.append(":0");
            }
            sb.append(":");
            ip = ip.replaceFirst("::", sb.toString());
        }
        groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length != 8) return false;
        long ipv6Left = 0, ipv6Right = 0;
        for (int i = 0; i < 8; i++) {
            String w = groupsByOneColon[i];
            int part = 0;
            if (w.equals("")) {
                if (i == 7) return false;
            } else {
                part = Integer.parseInt(w, 16);
            }
            if (i < 4) {
                ipv6Left = (ipv6Left << 16) | part;
            } else {
                ipv6Right = (ipv6Right << 16) | part;
            }
        }
        return true;
    }

    // LC751 **
    public List<String> ipToCIDR(String ip, int n) {
        final int INT_MASK = 0xffffffff;
        long start = ipStrToInt(ip);
        long end = start + n - 1;
        List<String> result = new ArrayList<>();
        while (n > 0) {
            int numTrailingZero = Long.numberOfTrailingZeros(start);
            int mask = 0, bitsInCidr = 1;
            while (bitsInCidr < n && mask < numTrailingZero) {
                bitsInCidr <<= 1;
                mask++;
            }
            if (bitsInCidr > n) {
                bitsInCidr >>= 1;
                mask--;
            }
            result.add(longToIpStr(start) + "/" + (32 - mask));
            start += bitsInCidr;
            n -= bitsInCidr;
        }
        return result;
    }

    private int bitLength(long x) {
        if (x == 0) return 1;
        return Long.SIZE - Long.numberOfLeadingZeros(x);
    }

    private String longToIpStr(long ipLong) {
        return String.format("%d.%d.%d.%d", (ipLong >> 24) & 0xff, (ipLong >> 16) & 0xff, (ipLong >> 8) & 0xff, ipLong & 0xff);
    }

    private long ipStrToInt(String ip) {
        String[] split = ip.split("\\.");
        long result = 0;
        for (String s : split) {
            int i = Integer.valueOf(s);
            result = (result << 8) | i;
        }
        return result;
    }


    // LC356
    public boolean isReflected(int[][] points) {
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        for (int[] p : points) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
        }
        int midXTimes2 = (minX + maxX);
        Set<Pair<Integer, Integer>> all = new HashSet<>();
        for (int[] p : points) {
            all.add(new Pair<>(p[0], p[1]));
        }
        Set<Pair<Integer, Integer>> s = new HashSet<>();
        for (Pair<Integer, Integer> p : all) {
            if ((double) p.getKey() == ((double) midXTimes2 / 2)) continue;
            if (!s.remove(new Pair<>(p.getKey(), p.getValue()))) {
                s.add(new Pair<>(midXTimes2 - p.getKey(), p.getValue()));
            }
        }
        return s.size() == 0;
    }

    // LC336 **
    public List<List<Integer>> palindromePairs(String[] words) {
        Set<Pair<Integer, Integer>> result = new HashSet<>();
        int wLen = words.length;
        String[] rWords = new String[wLen];
        Map<String, Integer> rWordIdx = new HashMap<>();
        for (int i = 0; i < wLen; i++) {
            rWords[i] = new StringBuilder(words[i]).reverse().toString();
            rWordIdx.put(rWords[i], i);
        }
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            int len = cur.length();
            if (len == 0) continue;
            for (int j = 0; j <= len; j++) { // 注意边界, 为了取到空串, 截取长度可以去到len, 同时为了去重用到Set<Pair<>>
                if (checkPal(cur, j, len)) {
                    int leftId = rWordIdx.getOrDefault(cur.substring(0, j), -1);
                    if (leftId != -1 && leftId != i) {
                        result.add(new Pair<>(i, leftId));
                    }
                }
                if (checkPal(cur, 0, j)) {
                    int rightId = rWordIdx.getOrDefault(cur.substring(j), -1);
                    if (rightId != -1 && rightId != i) {
                        result.add(new Pair<>(rightId, i));
                    }
                }
            }
        }
        List<List<Integer>> listResult = new ArrayList<>(result.size());
        for (Pair<Integer, Integer> p : result) {
            listResult.add(Arrays.asList(p.getKey(), p.getValue()));
        }
        return listResult;
    }

    private boolean checkPal(String s, int startIdx, int endIdxExclude) {
        if (startIdx > endIdxExclude) return false;
        if (startIdx == endIdxExclude) return true;
        int len = endIdxExclude - startIdx;
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(startIdx + i) != s.charAt(endIdxExclude - 1 - i)) return false;
        }
        return true;
    }

    // LC747
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int[] idxMap = new int[101];
        for (int i = 0; i < nums.length; i++) {
            idxMap[nums[i]] = i;
        }
        Arrays.sort(nums);
        if (nums[nums.length - 1] >= nums[nums.length - 2] * 2) return idxMap[nums[nums.length - 1]];
        return -1;
    }

    // LC1224
    public int maxEqualFreq(int[] nums) {
        Map<Integer, Integer> numFreqMap = new HashMap<>();
        for (int i : nums) {
            numFreqMap.put(i, numFreqMap.getOrDefault(i, 0) + 1);
        }
        Map<Integer, Set<Integer>> freqIntSetMap = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : numFreqMap.entrySet()) {
            freqIntSetMap.putIfAbsent(e.getValue(), new HashSet<>());
            freqIntSetMap.get(e.getValue()).add(e.getKey());
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            // 当前元素
            int curEle = nums[i];

            // 看该不该删除

            // 情况1: freqMap.keySet.size > 2 此时删除哪个都没用
            if (freqIntSetMap.keySet().size() > 2) {
                continue;
            } else if (freqIntSetMap.keySet().size() == 2) {
                // 情况2: size == 2 时候, 看哪个set 的size ==1
                Iterator<Integer> it = freqIntSetMap.keySet().iterator();
                int freq1 = it.next(), freq2 = it.next();
                int smallFreq = freq1 < freq2 ? freq1 : freq2;
                int largeFreq = smallFreq == freq1 ? freq2 : freq1;
                Set<Integer> smallFreqSet = freqIntSetMap.get(smallFreq), largeFreqSet = freqIntSetMap.get(largeFreq);
                // 如果两个set都有超过一个元素, 则删除哪个元素都没用
                if (smallFreqSet.size() != 1 && largeFreqSet.size() != 1) {
                    continue;
                } else {
                    Set<Integer> oneEleSet = smallFreqSet.size() == 1 ? smallFreqSet : largeFreqSet;
                    Set<Integer> anotherSet = oneEleSet == smallFreqSet ? largeFreqSet : smallFreqSet;

                    int oneEle = oneEleSet.iterator().next();
                    int eleFreq = numFreqMap.get(oneEle);
                    int anotherFreq = eleFreq == smallFreq ? largeFreq : smallFreq;

                    // 情况1： 这个元素的当前频率是1
                    if (eleFreq == 1) return i + 1;
                        // 情况2: 当前元素的频率比另一个频率大1
                    else if (eleFreq == anotherFreq + 1) return i + 1;
                        // 特判一下 111 22 这种情况, 即两个freq的set的大小都是1
                        // 前面只判断了2不能删除, 没有判断1能不能删除, 此处补充判断一次
                    else if (anotherSet.size() == 1) {
                        if (anotherFreq == 1) return i + 1;
                        else if (anotherFreq == eleFreq + 1) return i + 1;
                    }
                    // 否则没办法 只能删除当前元素
                }
            }

            // 若没有找到该删除的 就删除当前元素
            int curFreq = numFreqMap.get(curEle);
            int nextFreq = curFreq - 1;
            numFreqMap.put(curEle, nextFreq);
            freqIntSetMap.get(curFreq).remove(nums[i]);
            if (freqIntSetMap.get(curFreq).size() == 0) freqIntSetMap.remove(curFreq);
            if (nextFreq != 0) {
                freqIntSetMap.putIfAbsent(nextFreq, new HashSet<>());
                freqIntSetMap.get(nextFreq).add(nums[i]);
            } else {
                numFreqMap.remove(nums[i]);
            }

        }
        return nums.length;
    }

    // JZOF 22
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // LC1705
    public int eatenApples(int[] apples, int[] days) {
        // pq 存数对 [i,j], i表示苹果数量, j表示过期时间
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        int n = apples.length;
        int result = 0;
        int i = 0;
        do {
            if (i < n) {
                if (apples[i] == 0 && days[i] == 0) {
                    ;
                } else if (apples[i] != 0) {
                    pq.offer(new int[]{apples[i], days[i] + i});
                }
            }
            if (!pq.isEmpty()) {
                int[] entry = null;
                do {
                    int[] p = pq.poll();
                    if (i >= p[1]) continue;
                    entry = p;
                    break;
                } while (!pq.isEmpty());
                if (entry != null) {
                    entry[0]--;
                    result++;
                    if (entry[0] > 0) pq.offer(entry);
                }
            }
            i++;
        } while (!pq.isEmpty() || i < n);
        return result;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Trie {
    Trie[] children = new Trie[26];
    boolean isEnd = false;

    public void addWord(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new Trie();
            }
            cur = cur.children[c - 'a'];
        }
        cur.isEnd = true;
    }

    public boolean startsWith(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return true;
    }

    public boolean search(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return cur.isEnd;
    }
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

class quickSelect {
    static Random r = new Random();

    public static int topK(int[] arr, int topK) {
        return helper(arr, 0, arr.length - 1, topK);
    }

    private static Integer helper(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }
        int pivotVal = arr[start];
        int left = start, right = end;
        while (left < right) {
            while (left < right && arr[right] > pivotVal) {
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivotVal) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivotVal;
        if (left == arr.length - topK) return arr[left];
        Integer leftResult = helper(arr, start, left - 1, topK);
        if (leftResult != null) return leftResult;
        Integer rightResult = helper(arr, right + 1, end, topK);
        if (rightResult != null) return rightResult;
        return null;
    }
}

// LC635 TBD
class LogSystem {
    final SimpleDateFormat sdf = new SimpleDateFormat("yyyy:MM:dd:HH:mm:ss");
    final TreeSet<Integer> MAGIC_NUMBER = new TreeSet<Integer>() {{
        add(Calendar.YEAR);
        add(Calendar.MONTH);
        add(Calendar.DAY_OF_MONTH);
        add(Calendar.HOUR_OF_DAY);
        add(Calendar.MINUTE);
        add(Calendar.SECOND);
    }};

    TreeMap<Long, Integer> tm = new TreeMap<>();

    public void put(int id, String timestamp) {
        try {
            long ts = sdf.parse(timestamp).getTime();
            tm.put(ts, id);
        } catch (Exception e) {

        }
    }

    public List<Integer> retrieve(String start, String end, String granularity) {
        try {
            long sts = granHelper(start, granularity, false), ets = granHelper(end, granularity, true);
            List<Integer> result = new ArrayList<>();
            for (Map.Entry<Long, Integer> e : tm.subMap(sts, true, ets, false).entrySet()) {
                result.add(e.getValue());
            }
            return result;
        } catch (Exception e) {
            return null;
        }
    }

    private long granHelper(String timestamp, String gran, boolean isRight) throws ParseException {
        Date ts = sdf.parse(timestamp);
        int granMagic = granStrToMagic(gran);
        Calendar cal = Calendar.getInstance();
        cal.setTime(ts);
        for (int mn : MAGIC_NUMBER.tailSet(granMagic, false)) {
            if (mn <= Calendar.DAY_OF_MONTH) {
                if (mn == Calendar.MONTH) {
                    cal.set(mn, 0);
                } else {
                    cal.set(mn, 1);
                }
            } else {
                cal.set(mn, 0);
            }
        }
        if (isRight) {
            cal.add(granMagic, 1);
        }
        return cal.getTimeInMillis();
    }

    private int granStrToMagic(String gran) {
        switch (gran) {
            case "Year":
                return Calendar.YEAR;
            case "Month":
                return Calendar.MONTH;
            case "Day":
                return Calendar.DAY_OF_MONTH;
            case "Hour":
                return Calendar.HOUR_OF_DAY;
            case "Minute":
                return Calendar.MINUTE;
            case "Second":
                return Calendar.SECOND;
        }
        return -1;
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

    public boolean contains(int i) {
        return father.containsKey(i);
    }

}

class BIT {
    int[] tree;
    int len;

    public BIT(int len) {
        this.len = len;
        this.tree = new int[len + 1];
    }

    public BIT(int[] arr) {
        this.len = arr.length;
        this.tree = new int[len + 1];
        for (int i = 0; i < arr.length; i++) {
            int oneBasedIdx = i + 1;
            tree[oneBasedIdx] += arr[i];
            int nextOneBasedIdx = oneBasedIdx + lowbit(oneBasedIdx);
            if (nextOneBasedIdx <= len) tree[nextOneBasedIdx] += tree[oneBasedIdx];
        }
    }

    public void set(int idxZeroBased, int val) {
        int delta = val - get(idxZeroBased);
        update(idxZeroBased, delta);
    }

    public int get(int idxZeroBased) {
        return sumOneBased(idxZeroBased + 1) - sumOneBased(idxZeroBased);
    }

    public void update(int idxZeroBased, int delta) {
        updateOneBased(idxZeroBased + 1, delta);
    }

    public int sumRange(int left, int right) {
        return sumOneBased(right + 1) - sumOneBased(left);
    }

    public void updateOneBased(int idxOneBased, int delta) {
        while (idxOneBased <= len) {
            tree[idxOneBased] += delta;
            idxOneBased += lowbit(idxOneBased);
        }
    }

    public int sumOneBased(int idxOneBased) {
        int sum = 0;
        while (idxOneBased > 0) {
            sum += tree[idxOneBased];
            idxOneBased -= lowbit(idxOneBased);
        }
        return sum;
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}


// LC1756 fetch: O(log(n)*log(n))
class MRUQueue {

    BIT bit = new BIT(4002);
    int[] arr = new int[4002];
    int tail = -1;

    public MRUQueue(int n) {
        tail = n;
        for (int i = 1; i <= n; i++) {
            arr[i] = i;
            bit.updateOneBased(i, 1);
        }
    }

    public int fetch(int k) {
        int lo = 1, hi = tail;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (bit.sumOneBased(mid) >= k) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        arr[tail + 1] = arr[lo];
        arr[lo] = 0;
        bit.updateOneBased(tail + 1, 1);
        bit.updateOneBased(lo, -1);
        tail++;
        return arr[tail];
    }
}

class MaxHeap<E extends Comparable<E>> {
    ArrayList<E> arr = new ArrayList<>();

    public int size() {
        return arr.size();
    }

    public boolean isEmpty() {
        return arr.size() == 0;
    }

    public E peek() {
        if (isEmpty()) throw new NoSuchElementException("Queue is empty!");
        return arr.get(0);
    }

    public E poll() {
        if (isEmpty()) throw new NoSuchElementException("Queue is empty!");
        E result = arr.get(0);
        arr.set(0, arr.get(arr.size() - 1));
        arr.remove(arr.size() - 1);
        siftDown(0);
        return result;
    }

    public void offer(E ele) {
        arr.add(ele);
        siftUp(arr.size() - 1);
    }

    private void siftUp(int tail) {
        // 如果比自己的父节点大, 则交换上浮
        while (tail > 0) {
            int parent = parentIdx(tail);
            if (arr.get(tail).compareTo(arr.get(parent)) > 0) {
                E tmp = arr.get(parent);
                arr.set(parent, arr.get(tail));
                arr.set(tail, tmp);
                tail = parent;
            } else {
                return;
            }
        }
    }

    private void siftDown(int root) {
        // 如果比自己的孩子小, 则交换下沉
        while (leftChildIdx(root) < arr.size()) {
            int child = leftChildIdx(root);
            if (rightChildIdx(root) < arr.size() && arr.get(rightChildIdx(root)).compareTo(arr.get(leftChildIdx(root))) > 0) {
                child = rightChildIdx(root);
            }
            if (arr.get(root).compareTo(arr.get(child)) >= 0) return;
            E tmp = arr.get(root);
            arr.set(root, arr.get(child));
            arr.set(child, tmp);
            root = child;
        }
    }

    private int leftChildIdx(int idx) {
        return idx * 2 + 1;
    }

    private int rightChildIdx(int idx) {
        return idx * 2 + 2;
    }

    private int parentIdx(int idx) {
        if (idx == 0) throw new IllegalArgumentException("Root node has no parent!");
        return (idx - 1) / 2;
    }

}