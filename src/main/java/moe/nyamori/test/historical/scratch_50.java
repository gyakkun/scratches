package moe.nyamori.test.historical;


import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;

class scratch_50 {
    public static void main(String[] args) {
        scratch_50 s = new scratch_50();
        long timing = System.currentTimeMillis();

        RangeBit50 rbit = new RangeBit50(5);
        for (int i = 0; i < 5; i++) {
            rbit.set(i, i);
        }
        System.out.println(rbit.sumRange(0, 4));
        rbit.rangeUpdate(0, 4, 1);
        System.out.println(rbit.sumRange(0, 0));
//        System.out.println(s.compress(new char[]{'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'}));
        System.out.println(s.wordPattern("jquery",
                "jquery"));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC315
    public List<Integer> countSmaller(int[] nums) {
        final int offset = 10000;
        BIT50 bit = new BIT50(20002);
        List<Integer> result = new ArrayList<>(nums.length);
        for (int i = 0; i < nums.length; i++) result.add(0);
        bit.update(nums[nums.length - 1] + offset, 1);
        for (int i = nums.length - 2; i >= 0; i--) {
            result.set(i, bit.sumRange(0, nums[i] - 1 + offset));
            bit.update(nums[i] + offset, 1);
        }
        return result;
    }


    // LC1401 ** 相交测试 原理: 比较(矩形距离圆心最近的点到圆心的距离, 半径)
    public boolean checkOverlap(int radius, int x_center, int y_center, int x1, int y1, int x2, int y2) {
        double squareCenterX = (x1 + x2 + 0d) / 2, squareCenterY = (y1 + y2 + 0d) / 2;
        double[] vecSquareHalfLen = new double[]{x2 - squareCenterX, y2 - squareCenterY};
        double[] vecSquareCenterToCircleCenter = new double[]{Math.abs(x_center - squareCenterX), Math.abs(y_center - squareCenterY)};
        double[] vecResult = new double[]{Math.max(0, vecSquareCenterToCircleCenter[0] - vecSquareHalfLen[0]),
                Math.max(0, vecSquareCenterToCircleCenter[1] - vecSquareHalfLen[1])};
        return vecResult[0] * vecResult[0] + vecResult[1] * vecResult[1] <= radius * radius;
    }

    // JZOF II 075 LC1122
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        LinkedHashMap<Integer, Integer> count = new LinkedHashMap<>();
        for (int i : arr2) count.put(i, 0);
        int ptr = arr1.length - 1;
        for (int i = arr1.length - 1; i >= 0; i--) {
            if (!count.containsKey(arr1[i])) {
                arr1[ptr--] = arr1[i];
            } else {
                count.put(arr1[i], count.get(arr1[i]) + 1);
            }
        }
        Arrays.sort(arr1, ptr + 1, arr1.length);
        ptr = 0;
        for (Map.Entry<Integer, Integer> e : count.entrySet()) {
            for (int i = 0; i < e.getValue(); i++) {
                arr1[ptr++] = e.getKey();
            }
        }
        return arr1;
    }

    // LC290
    public boolean wordPattern(String pattern, String s) {
        Map<Character, String> map = new HashMap<>();
        Map<String, Character> reverseMap = new HashMap<>();
        String[] words = s.split(" ");
        int wIdx = 0, pIdx = 0;
        while (wIdx < words.length && pIdx < pattern.length()) {
            char c = pattern.charAt(pIdx);
            if (wIdx >= words.length) break;
            if (map.containsKey(c)) {
                if (!words[wIdx].equals(map.get(c))) return false;
            } else {
                if (reverseMap.containsKey(words[wIdx])) return false;
                reverseMap.put(words[wIdx], c);
                map.put(c, words[wIdx]);
            }
            wIdx++;
            pIdx++;
        }
        return pIdx == pattern.length() && wIdx == words.length;
    }

    // LC291
    public boolean wordPatternMatch(String pattern, String s) {
        return lc291Backtrack(pattern, 0, s, 0, new HashMap<>(), new HashMap<>());
    }

    private boolean lc291Backtrack(String p, int pIdx, String s, int sIdx, Map<Character, String> map, Map<String, Character> reverseMap) {
        for (int i = pIdx; i < p.length(); i++) {
            if (!map.containsKey(p.charAt(i))) {
                for (int j = sIdx; j <= s.length() - (p.length() - i); j++) {
                    String val = s.substring(sIdx, j + 1);
                    if (!reverseMap.containsKey(val)) {
                        map.put(p.charAt(i), val);
                        reverseMap.put(val, p.charAt(i));
                        if (lc291Backtrack(p, i + 1, s, sIdx + val.length(), map, reverseMap)) {
                            return true;
                        }
                        reverseMap.remove(val);
                        map.remove(p.charAt(i));
                    }
                }
                return false;
            } else {
                if (s.indexOf(map.get(p.charAt(i)), sIdx) != sIdx) {
                    return false;
                }
                sIdx += map.get(p.charAt(i)).length();
            }
        }
        return sIdx == s.length();
    }

    // Interview 16.16 **
    public int[] subSortStack(int[] nums) {
        if (nums.length <= 1) return new int[]{-1, -1};
        int n = nums.length;
        int start = -1, end = -1;
        int min = nums[n - 1], max = nums[0];
        for (int i = 0, j = n - 1; i < n; i++, j--) {
            // 如果排好序, 则正向遍历数会越来越大, 最大值永远是当前数
            // 如果遇到一个数小于最大值, 则从该数开始, 直到最后一个小于最大值的数, 这段都是无序的
            // 反向遍历同理
            if (nums[i] < max) {
                end = i;
            } else {
                max = nums[i];
            }

            if (nums[j] > min) {
                start = j;
            } else {
                min = nums[j];
            }
        }
        return new int[]{start, end};
    }

    // Interview 16.16 和 LC581 基本一样
    public int[] subSort(int[] array) {
        if (array.length <= 1) return new int[]{-1, -1};
        int[] orig = Arrays.copyOfRange(array, 0, array.length);
        Arrays.sort(array);
        int left = 0, right = array.length - 1;
        while (left < array.length && orig[left] == array[left]) left++;
        if (left == array.length) return new int[]{-1, -1};
        while (right >= 0 && orig[right] == array[right]) right--;
        return new int[]{left, right};
    }

    // LC1217
    public int minCostToMoveChips(int[] position) {
        int oddCtr = 0, evenCtr = 0;
        for (int i : position) {
            if (i % 2 == 0) evenCtr++;
            else oddCtr++;
        }
        return Math.min(oddCtr, evenCtr);
    }

    // LC937
    public String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String[] arr1 = o1.split(" "), arr2 = o2.split(" ");
                String id1 = arr1[0], id2 = arr2[0];
                String content1 = String.join(" ", Arrays.copyOfRange(arr1, 1, arr1.length));
                String content2 = String.join(" ", Arrays.copyOfRange(arr2, 1, arr2.length));
                boolean isDigit1 = true, isDigit2 = true;
                for (char c : content1.toCharArray()) {
                    if (!Character.isSpaceChar(c) && !Character.isDigit(c)) {
                        isDigit1 = false;
                        break;
                    }
                }
                for (char c : content2.toCharArray()) {
                    if (!Character.isSpaceChar(c) && !Character.isDigit(c)) {
                        isDigit2 = false;
                        break;
                    }
                }

                if (isDigit1 && !isDigit2) return 1;
                else if (isDigit2 && !isDigit1) return -1;
                else if (isDigit1 && isDigit2) return 0;
                else {
                    return !content1.equals(content2) ? content1.compareTo(content2) : id1.compareTo(id2);
                }
            }
        });
        return logs;
    }

    // LC1646
    public int getMaximumGenerated(int n) {
        if (n <= 1) return n;
        int[] arr = new int[n + 1];
        arr[0] = 0;
        arr[1] = 1;
        for (int i = 1; i * 2 + 1 <= n; i++) {
            arr[2 * i] = arr[i];
            arr[2 * i + 1] = arr[i] + arr[i + 1];
        }
        if ((n / 2) * 2 == n) arr[n] = arr[n / 2];
        return Arrays.stream(arr).max().getAsInt();
    }

    // LC789 ** 曼哈顿距离及推理见Solution
    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        Function<int[], Boolean> manh = ghost -> Math.abs(ghost[0] - target[0]) + Math.abs(ghost[1] - target[1]) > Math.abs(target[0]) + Math.abs(target[1]);
        for (int[] g : ghosts) if (!manh.apply(g)) return false;
        return true;
    }

    // LC443
    public int compress(char[] chars) {
        int shadowPtr = 0, ctr = 1;
        char cur = chars[0];
        for (int ptr = 1; ptr < chars.length; ptr++) {
            if (chars[ptr] == cur) {
                ctr++;
            } else {
                cur = chars[ptr];
                shadowPtr++;
                if (ctr != 1) {
                    for (char c : String.valueOf(ctr).toCharArray()) {
                        chars[shadowPtr++] = c;
                    }
                }
                chars[shadowPtr] = chars[ptr];
                ctr = 1;
            }
        }
        shadowPtr++;
        if (ctr != 1) {
            for (char c : String.valueOf(ctr).toCharArray()) {
                chars[shadowPtr++] = c;
            }
        }
        return shadowPtr;
    }


    // LC252
    public boolean canAttendMeetings(int[][] intervals) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int[] i : intervals) {
            tm.put(i[0], tm.getOrDefault(i[0], 0) + 1);
            tm.put(i[1], tm.getOrDefault(i[1], 0) - 1);
        }
        int sum = 0;
        for (int i : tm.values()) {
            sum += i;
            if (sum > 1) return false;
        }
        return true;
    }

    // LC1176
    public int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
        int result = 0, c = 0;
        for (int i = 0; i < k; i++) {
            c += calories[i];
        }
        if (c < lower) result--;
        if (c > upper) result++;
        for (int i = k; i < calories.length; i++) {
            c -= calories[i - k];
            c += calories[i];
            if (c < lower) result--;
            if (c > upper) result++;
        }
        return result;
    }

    // LC1443
    List<Set<Integer>> lc1443EdgeMtx;
    Map<Integer, Set<Integer>> lc1443Mark; // 当发现 [i,j] 这条边能够连接到苹果, 则在mark中加入标记
    List<Boolean> lc1443AppleSet;

    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        lc1443EdgeMtx = new ArrayList<>();
        lc1443Mark = new HashMap<>();
        lc1443AppleSet = hasApple;
        int[] indegree = new int[n];
        for (int i = 0; i < n; i++) lc1443EdgeMtx.add(new HashSet<>());
        for (int[] e : edges) {
            lc1443EdgeMtx.get(e[0]).add(e[1]);
            indegree[e[1]]++;
        }
        for (int i = 1; i < n; i++) {
            if (indegree[i] == 0) {
                for (int outNode : lc1443EdgeMtx.get(i)) {
                    lc1443EdgeMtx.get(outNode).add(i);
                }
                lc1443EdgeMtx.get(i).clear();
            }
        }
        lc1443IsConnectedToApple(0);
        int result = 0;
        for (int key : lc1443Mark.keySet()) {
            result += lc1443Mark.get(key).size();
        }
        return 2 * result;
    }

    private boolean lc1443IsConnectedToApple(int node) {
        boolean result = lc1443AppleSet.get(node);
        for (int next : lc1443EdgeMtx.get(node)) {
            if (lc1443IsConnectedToApple(next)) {
                result = true;
                lc1443Mark.putIfAbsent(node, new HashSet<>());
                lc1443Mark.get(node).add(next);
            }
        }
        return result;
    }

    // LC742
    public int findClosestLeaf(TreeNode50 root, int k) {
        TreeNode50 target = null;
        Deque<TreeNode50> q = new LinkedList<>();
        Set<TreeNode50> leaves = new HashSet<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode50 p = q.poll();
            if (p.left != null) {
                q.offer(p.left);
            }
            if (p.right != null) {
                q.offer(p.right);
            }
            if (p.left == null && p.right == null) leaves.add(p);
            if (p.val == k) target = p;
        }
        int minDistance = Integer.MAX_VALUE, result = -1;
        for (TreeNode50 leaf : leaves) {
            int dis = treeNodeDistance(root, leaf, target);
            if (dis < minDistance) {
                minDistance = dis;
                result = leaf.val;
            }
        }
        return result;
    }

    private int treeNodeDistance(TreeNode50 root, TreeNode50 a, TreeNode50 b) {
        TreeNode50 lca = LCARecursive(root, a, b);
        int disA = -1, disB = -1;
        Deque<TreeNode50> q = new LinkedList<>();
        q.offer(lca);
        while (!q.isEmpty()) {
            disA++;
            int qSize = q.size();
            boolean finish = false;
            for (int i = 0; i < qSize; i++) {
                TreeNode50 p = q.poll();
                if (p == a) {
                    finish = true;
                    break;
                }
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
            if (finish) break;
        }
        q.clear();
        q.offer(lca);
        while (!q.isEmpty()) {
            disB++;
            int qSize = q.size();
            boolean finish = false;
            for (int i = 0; i < qSize; i++) {
                TreeNode50 p = q.poll();
                if (p == b) {
                    finish = true;
                    break;
                }
                if (p.left != null) q.offer(p.left);
                if (p.right != null) q.offer(p.right);
            }
            if (finish) break;
        }
        return disA + disB;
    }

    private TreeNode50 LCARecursive(TreeNode50 root, TreeNode50 a, TreeNode50 b) {
        if (root == null) return null;
        if (root == a || root == b) return root;
        TreeNode50 left = LCARecursive(root.left, a, b);
        TreeNode50 right = LCARecursive(root.right, a, b);
        if (left != null && right != null) return root;
        if (left != null) return left;
        return right;
    }

    // LC590
    class Lc590 {

        // ** 迭代法 注意各种顺序
        public List<Integer> postorder(Node root) {
            LinkedList<Integer> result = new LinkedList<>();
            if (root == null) return result;
            Deque<Node> stack = new LinkedList<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                Node last = stack.pop();
                result.push(last.val);
                for (Node child : last.children) {
                    stack.push(child);
                }
            }
            return result;
        }

        public List<Integer> postorder(TreeNode50 root) {
            LinkedList<Integer> result = new LinkedList<>();
            if (root == null) return result;
            Deque<TreeNode50> stack = new LinkedList<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                TreeNode50 last = stack.pop();
                result.push(last.val);
                if (last.left != null) stack.push(last.left);
                if (last.right != null) stack.push(last.right);
            }
            return result;
        }


        class Node {
            public int val;
            public List<Node> children;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, List<Node> _children) {
                val = _val;
                children = _children;
            }
        }

    }

    // LC1598
    public int minOperations(String[] logs) {
        int stack = 0;
        for (String oper : logs) {
            if (oper.equals("./")) {
                ;
            } else if (oper.equals("../")) {
                stack = Math.max(stack - 1, 0);
            } else {
                stack++;
            }
        }
        return stack;
    }

    // LC541
    public String reverseStr(String s, int k) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        int i = 0;
        for (; i < ca.length; i += 2 * k) {
            for (int j = 0; j < k && (i + k - 1 - j) < ca.length; j++) {
                sb.append(ca[i + k - 1 - j]);
            }
            for (int j = 0; j < k && (i + k + j) < ca.length; j++) {
                sb.append(ca[i + k + j]);
            }
        }
        if (sb.length() < ca.length) {
            int remain = ca.length - sb.length();
            String r = s.substring(i - 2 * k, i - 2 * k + remain);
            if (remain < k) {
                sb.append(new StringBuilder(r).reverse());
            } else if (remain < 2 * k) {
                String shouldReverse = r.substring(0, k);
                String shouldKeep = r.substring(k);
                sb.append(new StringBuilder(shouldReverse).reverse());
                sb.append(shouldKeep);
            }
        }
        return sb.toString();
    }

    // Microsoft O(n) Time O(n) Space
    // https://leetcode-cn.com/circle/discuss/OPC9WF/
    public int[] maxCrossSum(int[] m, int[] n) {
        // m,n 是长度为l的数列, 求区间[a,b] [c,d], 使得  sum(m, a,b) -sum(n,a,b) - (sum(m,c,d) - sum(n,c,d)) 最大
        // [a,b] [c,d] 没有交集
        // 返回[a,b,c,d]
        int l = m.length, maxValue = Integer.MIN_VALUE;
        int[] mMinusN = new int[l], result = new int[0];
        for (int i = 0; i < l; i++) mMinusN[i] = m[i] - n[i];
        int[] dpMaxFromLeft = new int[l], dpMinFromLeft = new int[l], dpMaxFromRight = new int[l], dpMinFromRight = new int[l];
        int[][] dpMaxFromLeftRange = new int[l][2], dpMinFromLeftRange = new int[l][2], dpMaxFromRightRange = new int[l][2], dpMinFromRightRange = new int[l][2];
        // 左侧最大
        {
            int curSeg = mMinusN[0], left = 0, right = 0, max = mMinusN[0];
            dpMaxFromLeftRange[0] = new int[]{0, 0};
            dpMaxFromLeft[0] = mMinusN[0];
            for (int i = 1; i < l; i++) {
                curSeg = Math.max(curSeg + mMinusN[i], mMinusN[i]);
                if (curSeg != mMinusN[i]) { // 扩张了
                    right = i;
                } else { // 选择了下标为i这个数
                    left = right = i;
                }
                if (curSeg > max) {
                    max = curSeg;
                    dpMaxFromLeft[i] = max;
                    dpMaxFromLeftRange[i] = new int[]{left, right};
                } else {
                    dpMaxFromLeft[i] = dpMaxFromLeft[i - 1];
                    dpMaxFromLeftRange[i] = dpMaxFromLeftRange[i - 1];
                }
            }
        }
        // 左侧最小
        {
            int curSeg = mMinusN[0], left = 0, right = 0, min = mMinusN[0];
            dpMinFromLeftRange[0] = new int[]{0, 0};
            dpMinFromLeft[0] = mMinusN[0];
            for (int i = 1; i < l; i++) {
                curSeg = Math.min(curSeg + mMinusN[i], mMinusN[i]);
                if (curSeg != mMinusN[i]) { // 扩张了
                    right = i;
                } else { // 选择了下标为i这个数
                    left = right = i;
                }
                if (curSeg < min) {
                    min = curSeg;
                    dpMinFromLeft[i] = min;
                    dpMinFromLeftRange[i] = new int[]{left, right};
                } else {
                    dpMinFromLeft[i] = dpMinFromLeft[i - 1];
                    dpMinFromLeftRange[i] = dpMinFromLeftRange[i - 1];
                }
            }
        }
        // 右侧最大
        {
            int curSeg = mMinusN[l - 1], left = l - 1, right = l - 1, max = mMinusN[l - 1];
            dpMaxFromRightRange[l - 1] = new int[]{l - 1, l - 1};
            dpMaxFromRight[l - 1] = mMinusN[l - 1];
            for (int i = l - 2; i >= 0; i--) {
                curSeg = Math.max(curSeg + mMinusN[i], mMinusN[i]);
                if (curSeg != mMinusN[i]) { // 扩张了
                    left = i;
                } else { // 选择了下标为i这个数
                    left = right = i;
                }
                if (curSeg > max) {
                    max = curSeg;
                    dpMaxFromRight[i] = max;
                    dpMaxFromRightRange[i] = new int[]{left, right};
                } else {
                    dpMaxFromRight[i] = dpMaxFromRight[i + 1];
                    dpMaxFromRightRange[i] = dpMaxFromRightRange[i + 1];
                }
            }
        }
        // 右侧最小
        {
            int curSeg = mMinusN[l - 1], left = l - 1, right = l - 1, min = mMinusN[l - 1];
            dpMinFromRightRange[l - 1] = new int[]{l - 1, l - 1};
            dpMinFromRight[l - 1] = mMinusN[l - 1];
            for (int i = l - 2; i >= 0; i--) {
                curSeg = Math.min(curSeg + mMinusN[i], mMinusN[i]);
                if (curSeg != mMinusN[i]) { // 扩张了
                    left = i;
                } else { // 选择了下标为i这个数
                    left = right = i;
                }
                if (curSeg < min) {
                    min = curSeg;
                    dpMinFromRight[i] = min;
                    dpMinFromRightRange[i] = new int[]{left, right};
                } else {
                    dpMinFromRight[i] = dpMinFromRight[i + 1];
                    dpMinFromRightRange[i] = dpMinFromRightRange[i + 1];
                }
            }
        }
        // 左大右小
        for (int i = 0; i < l - 1; i++) {
            int tmpValue = dpMaxFromLeft[i] - dpMinFromRight[i + 1];
            if (tmpValue > maxValue) {
                maxValue = tmpValue;
                result = new int[]{dpMaxFromLeftRange[i][0], dpMaxFromLeftRange[i][1], dpMinFromRightRange[i + 1][0], dpMinFromRightRange[i + 1][1]};
            }
        }
        // 左小右大
        for (int i = 1; i < l; i++) {
            int tmpValue = dpMaxFromRight[i] - dpMinFromLeft[i - 1];
            if (tmpValue > maxValue) {
                maxValue = tmpValue;
                result = new int[]{dpMaxFromRightRange[i][0], dpMaxFromRightRange[i][1], dpMinFromLeftRange[i + 1][0], dpMinFromLeftRange[i + 1][1]};
            }
        }
        return result;
    }

    // LC991
    public int brokenCalc(int startValue, int target) {
        int result = 0;
        while (target > startValue) {
            result++;
            if (target % 2 == 1) {
                target++;
            } else {
                target /= 2;
            }
        }
        return result + startValue - target;
    }

    // LC1144
    public int movesToMakeZigzag(int[] nums) {
        if (nums.length <= 2) return 0;
        // 一开始下降
        int result1 = 0;
        for (int i = 1; i < nums.length; i += 2) {
            if (i - 1 >= 0 && i + 1 < nums.length) {
                result1 += Math.max(nums[i] - Math.min(nums[i - 1], nums[i + 1]) + 1, 0);
            } else if (i - 1 < 0) {
                result1 += Math.max(nums[i] - nums[i + 1] + 1, 0);
            } else if (i + 1 >= nums.length) {
                result1 += Math.max(nums[i] - nums[i - 1] + 1, 0);
            }
        }

        // 一开始上升
        int result2 = 0;
        for (int i = 0; i < nums.length; i += 2) {
            if (i - 1 >= 0 && i + 1 < nums.length) {
                result2 += Math.max(nums[i] - Math.min(nums[i - 1], nums[i + 1]) + 1, 0);
            } else if (i - 1 < 0) {
                result2 += Math.max(nums[i] - nums[i + 1] + 1, 0);
            } else if (i + 1 >= nums.length) {
                result2 += Math.max(nums[i] - nums[i - 1] + 1, 0);
            }
        }
        return Math.min(result1, result2);
    }

    // LC873
    public int lenLongestFibSubseq(int[] arr) {
        // arr 严格递增
        int n = arr.length;
        int[] result = new int[1001];
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < n; i++) m.put(arr[i], i);
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                int len = 2, last = i, lastButOne = j, next = arr[last] - arr[lastButOne];
                while (m.containsKey(next) && m.get(next) < lastButOne) {
                    len++;
                    last = lastButOne;
                    lastButOne = m.get(next);
                    next = arr[last] - arr[lastButOne];
                }
                if (len != 2) {
                    result[len]++;
                }
            }
        }
        for (int i = 1000; i >= 0; i--) {
            if (result[i] != 0) return i;
        }
        return 0;
    }

    // JZOF II 057
    public int[][] findContinuousSequence(int target) {
        // target = (a1 + an) * n /2
        // product = (a1 +  a1 + (n-1) *d) *n
        List<List<Integer>> result = new ArrayList<>();
        Set<Integer> a1Set = new HashSet<>();
        int product = target * 2;
        int sqrt = (int) (Math.sqrt(product));
        for (int i = 1; i <= sqrt; i++) {
            if (product % i == 0) {
                // 令i为n
                int n = i;
                int k = product / n;
                jzofii057Handle(target, result, a1Set, n, k);

                // 令product/i为n
                n = product - i;
                k = product / n;
                jzofii057Handle(target, result, a1Set, n, k);
            }
        }
        result.sort(Comparator.comparingInt(o -> o.get(0)));
        int[][] resultArr = new int[result.size()][];
        for (int i = 0; i < result.size(); i++) {
            resultArr[i] = result.get(i).stream().mapToInt(Integer::valueOf).toArray();
        }
        return resultArr;
    }

    private void jzofii057Handle(int target, List<List<Integer>> result, Set<Integer> a1Set, int n, int k) {
        if ((k - n + 1) % 2 == 0) {
            int a1 = (k - n + 1) / 2;
            if (a1 > 0 && !a1Set.contains(a1) && a1 != target) {
                a1Set.add(a1);
                List<Integer> tmp = new ArrayList<>(n);
                for (int j = 0; j < n; j++) {
                    tmp.add(a1++);
                }
                result.add(tmp);
            }
        }
    }

    // LC1754 **
    public String largestMerge(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        char[] ca1 = word1.toCharArray(), ca2 = word2.toCharArray();
        int ptr1 = 0, ptr2 = 0;
        while (ptr1 != ca1.length && ptr2 != ca2.length) {
            int compare = charSeqCompare(ca1, ptr1, ca2, ptr2);
            if (compare > 0) {
                sb.append(ca1[ptr1++]);
            } else {
                sb.append(ca2[ptr2++]);
            }
        }
        while (ptr1 != ca1.length) {
            sb.append(ca1[ptr1++]);
        }
        while (ptr2 != ca2.length) {
            sb.append(ca2[ptr2++]);
        }
        return sb.toString();
    }

    // ** 字典序比较算法
    private int charSeqCompare(char[] arr1, int startIdx1, char[] arr2, int startIdx2) {
        int idx1 = startIdx1, idx2 = startIdx2;
        while (idx1 < arr1.length && idx2 < arr2.length) {
            if (arr1[idx1] == arr2[idx2]) {
                idx1++;
                idx2++;
                continue;
            }
            return arr1[idx1] - arr2[idx2];
        }
        return arr1.length - startIdx1 - (arr2.length - startIdx2);
    }

    // LC542
    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] result = new int[m][n], directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
                    q.offer(new int[]{i, j});
                }
            }
        }
        int layer = 0;
        while (!q.isEmpty()) {
            int qSize = q.size();
            layer++;
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                for (int[] d : directions) {
                    int x = p[0] + d[0], y = p[1] + d[1];
                    if (x >= 0 && x < mat.length && y >= 0 && y < mat[0].length) {
                        if (!visited[x][y]) {
                            if (mat[x][y] == 1 && result[x][y] == 0) {
                                result[x][y] = layer;
                            }
                            q.offer(new int[]{x, y});
                        }
                    }
                }
            }
        }
        return result;
    }

    // LCP 22 **
    public int paintingPlan(int n, int k) {
        if (k == 0) return 1;
        if (k < n) return 0;
        if (k == n * n) return 1;
        // 排列数递推公式
        int[][] C = new int[n + 1][n + 1];
        int result = 0;
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) C[i][j] = 1;
                else C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
            }
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                if (i * n + j * n - i * j == k) {
                    result += C[n][i] * C[n][j];
                }
            }
        }
        return result;
    }

    // LC1119
    public String removeVowels(String s) {
        Set<Character> vowelSet = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u'));
        char[] ca = s.toCharArray();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ca.length; i++) if (!vowelSet.contains(ca[i])) sb.append(ca[i]);
        return sb.toString();
    }

    // LC345
    public String reverseVowels(String s) {
        Set<Character> vowelSet = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
        char[] ca = s.toCharArray();
        List<Character> l = new ArrayList<>();
        for (int i = 0; i < ca.length; i++) {
            if (vowelSet.contains(ca[i])) l.add(ca[i]);
        }
        int ptr = 0;
        for (int i = ca.length - 1; i >= 0; i--) {
            if (vowelSet.contains(ca[i])) ca[i] = l.get(ptr++);
        }
        return new String(ca);
    }

    // LCP 03
    public boolean robot(String command, int[][] obstacles, int x, int y) {
        Set<Pair<Integer, Integer>> trackSetInOneCycle = new HashSet<>();
        char[] ca = command.toCharArray();
        int xCount = 0, yCount = 0;
        trackSetInOneCycle.add(new Pair<>(0, 0));
        for (char c : ca) {
            if (c == 'U') yCount++;
            else xCount++;
            trackSetInOneCycle.add(new Pair<>(xCount, yCount));
        }
        int cycles = Math.min(x / xCount, y / yCount);
        if (!trackSetInOneCycle.contains(new Pair<>(x - cycles * xCount, y - cycles * yCount))) return false;

        for (int[] o : obstacles) {
            if (o[0] > x || o[1] > y) continue;
            int oCycles = Math.min(o[0] / xCount, o[1] / yCount);
            if (trackSetInOneCycle.contains(new Pair<>(o[0] - oCycles * xCount, o[1] - oCycles * yCount))) return false;
        }
        return true;
    }

    // LC1742
    public int countBalls(int lowLimit, int highLimit) {
        int[] count = new int[46];
        for (int i = lowLimit; i <= highLimit; i++) {
            count[getDigitSum(i)]++;
        }

        return Arrays.stream(count).max().getAsInt();
    }

    private int getDigitSum(int num) {
        int result = 0;
        while (num != 0) {
            result += (num % 10);
            num /= 10;
        }
        return result;
    }

    // JZOF II 058 LC729
    class MyCalendar {

        TreeSet<int[]> left;
        TreeSet<int[]> right;

        public MyCalendar() {
            left = new TreeSet<>((o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o1[0] - o2[0]);
            right = new TreeSet<>((o1, o2) -> o1[1] == o2[1] ? o1[0] - o2[0] : o1[1] - o2[1]);
        }

        // 前闭后开
        public boolean book(int start, int end) {
            int[] lQuery = new int[]{start, start};
            int[] rQuery = new int[]{end, end};
            int[] lsf = left.floor(lQuery), lsc = left.ceiling(lQuery),
                    rsf = right.floor(rQuery), rsc = right.ceiling(rQuery);

            if ((lsf == null || lsf[1] <= start)
                    && (lsc == null || lsc[0] >= end)
                    && (rsf == null || rsf[1] <= start)
                    && (rsc == null || rsc[0] >= end)) {
                int[] entry = new int[]{start, end};
                left.add(entry);
                right.add(entry);
                return true;
            }
            return false;
        }
    }

    // LC589
    class Lc589 {
        public List<Integer> preorder(Node root) {
            List<Integer> result = new ArrayList<>();
            if (root == null) return result;
            Deque<Node> q = new LinkedList<>();
            q.push(root);
            while (!q.isEmpty()) {
                Node p = q.pop();
                result.add(p.val);
                for (int i = p.children.size() - 1; i >= 0; i--) {
                    q.push(p.children.get(i));
                }
            }
            return result;
        }

        class Node {
            public int val;
            public List<Node> children;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, List<Node> _children) {
                val = _val;
                children = _children;
            }
        }
    }

    // Interview 16.18 ** 写得太复杂了！
    public boolean patternMatching(String pattern, String value) {
        char[] cp = pattern.toCharArray();
        int aCount = 0, bCount = 0, valueLen = value.length();
        int firstAIdx = pattern.indexOf('a'), firstBIdx = pattern.indexOf('b');
        for (char c : cp) {
            if (c == 'a') aCount++;
            else bCount++;
        }
        if (aCount != 0 && bCount != 0 && value.equals("")) return false;
        else if (value.equals("")) return true;
        if (bCount == 0) {
            int aLen = valueLen / aCount;
            if (aLen * aCount != valueLen) return false;
            String ap = value.substring(0, aLen);
            int ptrOnV = -aLen;
            for (int i = 0; i < aCount; i++) {
                ptrOnV = value.indexOf(ap, ptrOnV + aLen);
                if (ptrOnV == -1) return false;
            }
            return true;
        } else if (aCount == 0) {
            int bLen = valueLen / bCount;
            if (bLen * bCount != valueLen) return false;
            String ap = value.substring(0, bLen);
            int ptrOnV = -bLen;
            for (int i = 0; i < bCount; i++) {
                ptrOnV = value.indexOf(ap, ptrOnV + bLen);
                if (ptrOnV == -1) return false;
            }
            return true;
        } else {
            for (int aLen = 0; aLen <= (valueLen / aCount); aLen++) {
                int bLen = (valueLen - aLen * aCount) / bCount;
                if (bLen * bCount + aLen * aCount != valueLen) continue;
                String ap = "", bp = "";
                if (aLen == 0) {
                    bp = value.substring(0, bLen);
                    int ptr = 0, count = 0;
                    while (ptr >= 0) {
                        ptr = value.indexOf(bp, ptr);
                        if (ptr >= 0) {
                            count++;
                            ptr += bLen;
                        }
                    }
                    if (count == bCount) {
                        return true;
                    }
                    continue;
                }
                if (bLen == 0) {
                    ap = value.substring(0, aLen);
                    int ptr = 0, count = 0;
                    while (ptr >= 0) {
                        ptr = value.indexOf(ap, ptr);
                        if (ptr >= 0) {
                            count++;
                            ptr += aLen;
                        }
                    }
                    if (count == aCount) {
                        return true;
                    }
                    continue;
                }
                ap = value.substring(firstAIdx * bLen, firstAIdx * bLen + aLen);
                bp = value.substring(firstBIdx * aLen, firstBIdx * aLen + bLen);
                int ptrOnP = 0, ptrOnV = 0;
                boolean success = true;
                for (; ptrOnP < cp.length; ptrOnP++) {
                    if (cp[ptrOnP] == 'a') {
                        ptrOnV = value.indexOf(ap, ptrOnV);
                        if (ptrOnV >= 0) {
                            ptrOnV += aLen;
                        }
                    } else {
                        ptrOnV = value.indexOf(bp, ptrOnV);
                        if (ptrOnV >= 0) {
                            ptrOnV += bLen;
                        }
                    }
                    if (ptrOnV < 0) {
                        success = false;
                        break;
                    }
                }
                if (!success) continue;
                return true;
            }
        }

        return false;
    }

    // LC1100
    public int numKLenSubstrNoRepeats(String s, int k) {
        int result = 0, n = s.length();
        if (k > n) return 0;
        int[] freq = new int[26];
        char[] ca = s.toCharArray();
        for (int i = 0; i < k; i++) {
            freq[ca[i] - 'a']++;
        }
        if (lc1100Check(freq)) result++;
        for (int i = k; i < n; i++) {
            freq[ca[i - k] - 'a']--;
            freq[ca[i] - 'a']++;
            if (lc1100Check(freq)) result++;
        }
        return result;
    }

    private boolean lc1100Check(int[] freq) {
        for (int i : freq) if (i > 1) return false;
        return true;
    }

    // LC117 ** O(1) 空间
    class Lc117 {
        public Node connect(Node root) {
            if (root == null) return null;
            Node start = root;
            while (start != null) { // 思想: 在本层连接下一层的next
                Node nextStart = null, last = null; // 下一层的开始节点, 遍历到的下一层的最后一个节点
                for (Node ptr = start; ptr != null; ptr = ptr.next) {
                    if (ptr.left != null) {
                        if (last != null) {
                            last.next = ptr.left;
                        }
                        if (nextStart == null) {
                            nextStart = ptr.left;
                        }
                        last = ptr.left;
                    }
                    if (ptr.right != null) {
                        if (last != null) {
                            last.next = ptr.right;
                        }
                        if (nextStart == null) {
                            nextStart = ptr.right;
                        }
                        last = ptr.right;
                    }
                }
                start = nextStart;
            }
            return root;
        }

        class Node {
            public int val;
            public Node left;
            public Node right;
            public Node next;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, Node _left, Node _right, Node _next) {
                val = _val;
                left = _left;
                right = _right;
                next = _next;
            }
        }

    }

    // LC1737
    public int minCharacters(String a, String b) {
        // ONE TWO
        int result = Integer.MAX_VALUE / 2;
        int[] freqA = new int[26], freqB = new int[26], prefixA = new int[27], prefixB = new int[27];
        char[] ca = a.toCharArray(), cb = b.toCharArray();
        for (char c : ca) freqA[c - 'a']++;
        for (char c : cb) freqB[c - 'a']++;
        for (int i = 1; i <= 26; i++) prefixA[i] = freqA[i - 1] + prefixA[i - 1];
        for (int i = 1; i <= 26; i++) prefixB[i] = freqB[i - 1] + prefixB[i - 1];
        // 令A严格小于B, 使用一个指针指示B中最小的字母, 指针应该在b...z上移动, 当B最小的是a时, 首先考虑将B的所有a变为b
        for (int ptr = 'b'; ptr <= 'z'; ptr++) {
            char target = (char) ptr;
            int tmpMove = 0;
            // 将B中小于ptr的变为ptr
            tmpMove += prefixB[target - 'a'] - prefixB[0];
            // 将A中大于ptr的变为ptr-1
            tmpMove += prefixA[26] - prefixA[target - 'a'];
            result = Math.min(result, tmpMove);
        }
        // 令B严格小于A
        for (int ptr = 'b'; ptr <= 'z'; ptr++) {
            char target = (char) ptr;
            int tmpMove = 0;
            tmpMove += prefixA[target - 'a'] - prefixA[0];
            tmpMove += prefixB[26] - prefixB[target - 'a'];
            result = Math.min(result, tmpMove);
        }

        // THREE
        for (int i = 0; i < 26; i++) {
            int tmpMove = prefixA[26] + prefixB[26] - freqA[i] - freqB[i];
            result = Math.min(result, tmpMove);
        }

        return result;
    }

    // JZOF II 113
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> outEdge = new ArrayList<>(numCourses);
        List<Integer> result = new ArrayList<>(numCourses);
        Deque<Integer> q = new LinkedList<>();
        int[] indegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) outEdge.add(new ArrayList<>());
        for (int[] p : prerequisites) {
            outEdge.get(p[1]).add(p[0]);
            indegree[p[0]]++;
        }
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) q.offer(i);
        }
        while (!q.isEmpty()) {
            int p = q.poll();
            result.add(p);
            for (int next : outEdge.get(p)) {
                indegree[next]--;
                if (indegree[next] == 0) q.offer(next);
            }
        }
        if (result.size() != numCourses) return new int[]{};
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // Interview 17.01
    public int add(int a, int b) {
        int sum = a;
        while (b != 0) {
            int and = a & b;
            int xor = a ^ b;
            b = and << 1;
            sum = xor;
            a = xor;
        }
        return sum;
    }

    // Interview 01.08
    public void setZeroes(int[][] matrix) {
        boolean[] rowMark = new boolean[matrix.length], colMark = new boolean[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    rowMark[i] = true;
                    colMark[j] = true;
                }
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            if (rowMark[i]) {
                for (int j = 0; j < matrix[0].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < matrix[0].length; j++) {
            if (colMark[j]) {
                for (int i = 0; i < matrix.length; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    // LC1338
    public int minSetSize(int[] arr) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i : arr) freq.put(i, freq.getOrDefault(i, 0) + 1);
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -freq.getOrDefault(o, 0)));
        for (int i : freq.keySet()) {
            pq.offer(i);
        }
        int left = arr.length, half = arr.length / 2, result = 0;
        while (left > half) {
            left -= freq.get(pq.poll());
            result++;
        }
        return result;
    }

    // Interview 05.01
    public int insertBits(int N, int M, int i, int j) {
        int mask = 0;
        for (int k = i; k <= j; k++) {
            mask |= 1 << k;
        }
        mask = ~mask;
        N &= mask;
        M <<= i;
        M &= ~mask;
        N |= M;
        return N;
    }

    // LC1129 ** 可能有环 有平行边
    public int[] shortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges) {
        final int RED = 0, BLUE = 1;
        int[] result = new int[n];
        Arrays.fill(result, -1);
        result[0] = 0;
        List<List<Integer>> redOutEdge = new ArrayList<>(n), blueOutEdge = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            redOutEdge.add(new ArrayList<>());
            blueOutEdge.add(new ArrayList<>());
        }
        boolean[] redVisit = new boolean[n], blueVisit = new boolean[n];
        for (int[] re : red_edges) {
            redOutEdge.get(re[0]).add(re[1]);
        }
        for (int[] be : blue_edges) {
            blueOutEdge.get(be[0]).add(be[1]);
        }
        if (redOutEdge.get(0) == null && blueOutEdge.get(0) == null) return result;

        Deque<int[]> q = new LinkedList<>();
        for (int next : redOutEdge.get(0)) {
            q.offer(new int[]{next, RED});
        }
        for (int next : blueOutEdge.get(0)) {
            q.offer(new int[]{next, BLUE});
        }
        int layer = 0;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                // 注意剪枝时机!!!
                if (blueVisit[p[0]] && redVisit[p[0]]) continue;
                if (p[1] == RED && redVisit[p[0]]) continue;
                if (p[1] == BLUE && blueVisit[p[0]]) continue;
                if (result[p[0]] == -1) {
                    result[p[0]] = layer;
                } else {
                    result[p[0]] = Math.min(result[p[0]], layer);
                }
                if (p[1] == RED) {
                    redVisit[p[0]] = true;
                    for (int next : blueOutEdge.get(p[0])) {
                        q.offer(new int[]{next, BLUE});
                    }
                } else {
                    blueVisit[p[0]] = true;
                    for (int next : redOutEdge.get(p[0])) {
                        q.offer(new int[]{next, RED});
                    }
                }
            }
        }
        return result;
    }

    // LC1387
    Map<Integer, Integer> lc1387Memo;

    public int getKth(int lo, int hi, int k) {
        lc1387Memo = new HashMap<>();
        List<Integer> arr = new ArrayList<>(hi - lo + 1);
        for (int i = lo; i <= hi; i++) {
            arr.add(i);
        }
        arr.sort((o1, o2) -> lc1387Weight(o1) == lc1387Weight(o2) ? o1 - o2 : lc1387Weight(o1) - lc1387Weight(o2));
        return arr.get(k - 1);
    }

    private int lc1387Weight(int num) {
        if (num == 1) return 0;
        if (lc1387Memo.containsKey(num)) return lc1387Memo.get(num);
        int result = num % 2 == 1 ? lc1387Weight(3 * num + 1) + 1 : lc1387Weight(num / 2) + 1;
        lc1387Memo.put(num, result);
        return result;
    }


    // LC1513
    public int numSub(String s) {
        long oneCount = 0, mod = 1000000007;
        long result = 0;
        for (char c : s.toCharArray()) {
            if (c == '1') {
                oneCount++;
            } else {
                result += oneCount * (oneCount + 1) / 2;
                result %= mod;
                oneCount = 0;
            }
        }
        if (oneCount != 0) {
            result += oneCount * (oneCount + 1) / 2;
            result %= mod;
        }
        return (int) result;
    }

    // LC1709
    public int[] largestSubarrayBetter(int[] nums, int k) {
        int n = nums.length;
        int maxInt = 0, ptr = n - k, maxStartPoint = 0;
        while (ptr >= 0) {
            if (nums[ptr] > maxInt) {
                maxStartPoint = ptr;
                maxInt = nums[ptr];
            }
            ptr--;
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1709 没有利用数字不重复的条件
    public int[] largestSubarray(int[] nums, int k) {
        int maxStartPoint = 0, n = nums.length;
        for (int i = 1; i < n - k + 1; i++) {
            for (int j = 0; j < k; j++) {
                if (nums[maxStartPoint + j] < nums[i + j]) {
                    maxStartPoint = i;
                } else if (nums[maxStartPoint + j] == nums[i + j]) {
                    continue;
                } else {
                    break;
                }
            }
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1039 ** 几何
    Integer[][] lc1039Memo;

    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        lc1039Memo = new Integer[n + 1][n + 1];
        return lc1039Helper(values, 0, n - 1);
    }

    private int lc1039Helper(int[] values, int start, int end) {
        if (start + 1 == end) return 0;
        if (lc1039Memo[start][end] != null) return lc1039Memo[start][end];
        int result = Integer.MAX_VALUE;
        for (int i = start + 1; i < end; i++) {
            result = Math.min(result, lc1039Helper(values, start, i) + lc1039Helper(values, i, end) + values[start] * values[end] * values[i]);
        }
        return lc1039Memo[start][end] = result;
    }

    // LC128
    // https://bbs.byr.cn/n/article/Talking/6295267
    public int longestConsecutive(int[] nums) {
        Set<Integer> s = new HashSet<>();
        int result = 0;
        for (int e : nums) {
            s.add(e);
        }
        for (int i : s) {
            if (!s.contains(i - 1)) {
                int l = 1;
                while (s.contains(i + 1)) {
                    l++;
                    i++;
                }
                result = Math.max(result, l);
            }
        }
        return result;
    }


    // LC1775
    public int minOperations(int[] nums1, int[] nums2) {
        if (nums1.length * 6 < nums2.length || nums2.length * 6 < nums1.length) return -1;
        int origSum1 = Arrays.stream(nums1).sum(), origSum2 = Arrays.stream(nums2).sum();
        if (origSum1 == origSum2) return 0;
        int[] inc = new int[6], dec = new int[6];
        for (int i : nums1) {
            inc[6 - i]++;
            dec[i - 1]++;
        }
        for (int i : nums2) {
            dec[6 - i]++;
            inc[i - 1]++;
        }
        inc[0] = dec[0] = 0;
        int result = 0;
        int delta = origSum1 - origSum2;
        if (delta > 0) { // nums1 should decrease
            for (int i = 5; i >= 1; i--) {
                while (dec[i] > 0) {
                    result++;
                    dec[i]--;
                    delta -= i;
                    if (delta <= 0) return result;
                }
            }
        } else {
            for (int i = 5; i >= 1; i--) {
                while (inc[i] > 0) {
                    result++;
                    inc[i]--;
                    delta += i;
                    if (delta >= 0) return result;
                }
            }
        }
        return -1;
    }

    // LC552 HARD
    public int checkRecord(int n) {
        final int mod = 1000000007;
        long[] dp = new long[Math.max(4, n + 1)];
        dp[0] = 1;
        dp[1] = 2;
        dp[2] = 4;
        dp[3] = 7;
        for (int i = 4; i <= n; i++) {
            dp[i] = (2 * dp[i - 1] - dp[i - 4] + mod) % mod;
        }
        long result = dp[n];
        for (int i = 1; i <= n; i++) {
            result += (dp[i - 1] * dp[n - i]) % mod;
        }
        return (int) (result % mod);
    }

    // LC551
    public boolean checkRecord(String s) {
        int lCount = 0, aCount = 0;
        for (char c : s.toCharArray()) {
            if (c == 'L') {
                lCount++;
            } else {
                lCount = 0;
            }
            if (lCount >= 3) return false;
            if (c == 'A') {
                aCount++;
            }
            if (aCount >= 2) return false;
        }
        return true;
    }

    // LC1389
    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (index[i] >= result.size()) {
                int targetSize = (index[i] + 1) - result.size();
                for (int j = 0; j < targetSize; j++) {
                    result.add(-1);
                }
                result.set(index[i], nums[i]);
            } else {
                result.add(index[i], nums[i]);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC1390
    public int sumFourDivisors(int[] nums) {
        int result = 0;
        for (int n : nums) result += lc1390Helper(n);
        return result;
    }

    private int lc1390Helper(int n) {
        if (n <= 5) return 0;
        int sqrt = (int) Math.sqrt(n);
        Set<Integer> s = new HashSet<>();
        for (int i = 1; i <= sqrt; i++) {
            if (n % i == 0) {
                s.add(i);
                s.add(n / i);
            }
            if (s.size() > 4) return 0;
        }
        if (s.size() != 4) return 0;
        return s.stream().reduce((a, b) -> a + b).get();
    }

    // JZOF 06
    public int[] reversePrint(ListNode37 head) {
        // 倒置链表
        ListNode37 dummy = new ListNode37(-1);
        dummy.next = head;
        ListNode37 prev = null, cur = head;
        int count = 0;
        while (cur != null) {
            count++;
            ListNode37 origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }
        int[] result = new int[count];
        count = 0;
        cur = prev;
        while (cur != null) {
            result[count++] = cur.val;
            cur = cur.next;
        }
        return result;
    }

    // JZOF II 090
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[1], nums[0]);
        // ROB ZERO, then n-1 can't be robbed
        dp[0] = dp[1] = nums[0];
        for (int i = 2; i < n - 1; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int noZero = dp[n - 2];
        Arrays.fill(dp, 0);
        // Not rob zero, then n-1 can be robbed
        dp[1] = nums[1];
        dp[2] = Math.max(nums[1], nums[2]);
        for (int i = 3; i < n; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int zero = dp[n - 1];
        return Math.max(zero, noZero);
    }

    public TreeNode50 LCA(TreeNode50 root, TreeNode50 p, TreeNode50 q) {
        Map<TreeNode50, TreeNode50> parent = new HashMap<>();
        parent.put(root, null);
        Deque<TreeNode50> queue = new LinkedList<>();
        Set<TreeNode50> visited = new HashSet<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode50 poll = queue.poll();
            if (poll.left != null) {
                parent.put(poll.left, poll);
                queue.offer(poll.left);
            }
            if (poll.right != null) {
                parent.put(poll.right, poll);
                queue.offer(poll.right);
            }
        }
        while (p != null) {
            visited.add(p);
            p = parent.get(p);
        }
        while (q != null) {
            if (visited.contains(q)) return q;
            q = parent.get(q);
        }
        return null;
    }

    // JZOF 68
    public TreeNode50 lowestCommonAncestor(TreeNode50 root, TreeNode50 p, TreeNode50 q) {
        TreeNode50 result = root;
        while (true) {
            if (p.val > result.val && q.val > result.val) {
                result = result.right;
            } else if (p.val < result.val && q.val < result.val) {
                result = result.left;
            } else {
                break;
            }
        }
        return result;
    }

    // LC1338
    Map<TreeNode50, Long> nodeSumMap = new HashMap<>();

    public int maxProduct(TreeNode50 root) {
        final int mod = 1000000007;
        lc1338Helper(root);
        long result = 0;
        long total = nodeSumMap.get(root);
        for (TreeNode50 node : nodeSumMap.keySet()) {
            result = Math.max(result, (total - nodeSumMap.get(node)) * nodeSumMap.get(node));
        }
        return (int) (result % mod);
    }

    private long lc1338Helper(TreeNode50 root) {
        if (root == null) return 0;
        long result = root.val + lc1338Helper(root.left) + lc1338Helper(root.right);
        nodeSumMap.put(root, result);
        return result;
    }

    // LC733
    int[][] lc733Directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int origColor = image[sr][sc];
        boolean[][] visited = new boolean[image.length][image[0].length];
        lc733Helper(image, sr, sc, origColor, newColor, visited);
        return image;
    }

    private void lc733Helper(int[][] image, int x, int y, int origColor, int newColor, boolean[][] visited) {
        if (x < 0 || x >= image.length || y < 0 || y >= image[0].length || visited[x][y]) {
            return;
        }
        if (image[x][y] == origColor) {
            image[x][y] = newColor;
            visited[x][y] = true;
            for (int[] dir : lc733Directions) {
                lc733Helper(image, x + dir[0], y + dir[1], origColor, newColor, visited);
            }
        }
    }

    // JZOF 64 **
    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    // LC1282
    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, List<Integer>> sizeCountMap = new HashMap<>();
        for (int i = 0; i < groupSizes.length; i++) {
            sizeCountMap.putIfAbsent(groupSizes[i], new ArrayList<>());
            sizeCountMap.get(groupSizes[i]).add(i);
        }
        for (int gs : sizeCountMap.keySet()) {
            List<Integer> users = sizeCountMap.get(gs);
            int cur = 0;
            while (cur != users.size()) {
                result.add(users.subList(cur, cur + gs));
                cur += gs;
            }
        }
        return result;
    }

    // LC1684
    public int countConsistentStrings(String allowed, String[] words) {
        int result = 0, mask = 0;
        for (char c : allowed.toCharArray()) mask |= 1 << (c - 'a');
        for (String w : words) {
            int wm = 0;
            for (char c : w.toCharArray()) wm |= 1 << (c - 'a');
            if ((wm & mask) == wm) result++;
        }
        return result;
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

class TreeNode50 {
    int val;
    TreeNode50 left;
    TreeNode50 right;

    TreeNode50() {
    }

    TreeNode50(int val) {
        this.val = val;
    }

    TreeNode50(int val, TreeNode50 left, TreeNode50 right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class ListNode50 {
    int val;
    ListNode37 next;

    ListNode50(int x) {
        val = x;
    }
}

// LC622
class MyCircularQueue {
    int[] arr;
    int headIdx, count;

    public MyCircularQueue(int k) {
        arr = new int[k];
        headIdx = 0;
        count = 0;
    }

    public boolean enQueue(int value) {
        if (isFull()) return false;
        arr[(headIdx + count) % arr.length] = value;
        count++;
        return true;
    }

    public boolean deQueue() {
        if (isEmpty()) return false;
        arr[headIdx] = 0;
        headIdx = (headIdx + 1) % arr.length;
        count--;
        return true;
    }

    public int Front() {
        if (isEmpty()) return -1;
        return arr[headIdx];
    }

    public int Rear() {
        if (isEmpty()) return -1;
        return arr[(headIdx + count - 1 + arr.length) % arr.length];
    }

    public boolean isEmpty() {
        return count == 0;
    }

    public boolean isFull() {
        return count == arr.length;
    }
}

// LC588
class FileSystem {

    Node root;

    public FileSystem() {
        root = new Node(false, "");
    }

    public List<String> ls(String path) {
        List<String> result = new ArrayList<>();
        if (path.equals("/")) {
            for (Node n : root.nameChildMap.values()) {
                result.add(n.name);
            }
        } else {
            Node cur = getNode(path);
            if (cur.isFile) {
                result.add(cur.name);
            } else {
                for (Node c : cur.nameChildMap.values()) {
                    result.add(c.name);
                }
            }
        }
        result.sort(Comparator.naturalOrder());
        return result;
    }

    public void mkdir(String path) {
        String[] pathSplit = path.split("/");
        pathSplit = Arrays.copyOfRange(pathSplit, 1, pathSplit.length);
        Node cur = root;
        for (String n : pathSplit) {
            if (!cur.nameChildMap.containsKey(n)) {
                cur.addChild(new Node(false, n));
            }
            cur = cur.nameChildMap.get(n);
        }
    }

    public void addContentToFile(String filePath, String content) {
        String[] pathSplit = filePath.split("/");
        String fileName = pathSplit[pathSplit.length - 1];
        String directory = String.join("/", Arrays.copyOfRange(pathSplit, 0, pathSplit.length - 1));
        mkdir(directory);
        Node directoryNode = getNode(directory);
        if (!directoryNode.nameChildMap.containsKey(fileName)) {
            directoryNode.addChild(new Node(true, fileName));
        }
        Node fileNode = getNode(filePath);
        fileNode.content = fileNode.content + content;

    }

    public String readContentFromFile(String filePath) {
        Node fileNode = getNode(filePath);
        return fileNode.content;
    }

    private Node getNode(String path) {
        if (path.equals("/")) {
            return root;
        } else {
            String[] pathSplit = path.split("/");
            pathSplit = Arrays.copyOfRange(pathSplit, 1, pathSplit.length);
            Node cur = root;
            for (String n : pathSplit) {
                cur = cur.nameChildMap.get(n);
            }
            return cur;
        }
    }

    class Node {
        Map<String, Node> nameChildMap;
        boolean isFile;
        String name;
        String content;

        public Node(boolean isFile, String name) {
            this.isFile = isFile;
            this.name = name;
            if (!isFile) {
                nameChildMap = new HashMap<>();
            } else {
                this.content = "";
            }
        }

        public void addChild(Node child) {
            nameChildMap.put(child.name, child);
        }
    }
}

class RangeBit50 {
    BIT50 diff;
    BIT50 iDiff;
    int len;

    public RangeBit50(int len) {
        this.len = len;
        diff = new BIT50(len);
        iDiff = new BIT50(len);
    }

    public RangeBit50(int[] arr) {
        this.len = arr.length;
        diff = new BIT50(len);
        iDiff = new BIT50(len);
        for (int i = 0; i < arr.length; i++) {
            update(i, arr[i]);
        }
    }

    public void set(int zeroBased, int val) {
        update(zeroBased, val - get(zeroBased));
    }

    public int get(int zeroBased) {
        return sum(zeroBased + 1) - sum(zeroBased);
    }

    public void update(int zeroBased, int delta) {
        rangeUpdate(zeroBased, zeroBased, delta);
    }

    public int sumRange(int l, int r) {
        return sum(r + 1) - sum(l);
    }

    public void rangeUpdate(int l, int r, int delta) {
        rangeUpdateOneBased(l + 1, r + 1, delta);
    }

    private void updateOneBased(int oneBased, int delta) {
        int iDelta = oneBased * delta;
        diff.update(oneBased - 1, delta);
        iDiff.update(oneBased - 1, iDelta);
    }

    private void rangeUpdateOneBased(int leftOneBased, int rightOneBased, int delta) {
        updateOneBased(leftOneBased, delta);
        updateOneBased(rightOneBased + 1, -delta);
    }

    private int sum(int rightOneBased) {
        return (rightOneBased + 1) * diff.sumRange(0, rightOneBased - 1) - iDiff.sumRange(0, rightOneBased - 1);
    }

}

class BIT50 {
    int[] tree;
    int len;

    public BIT50(int len) {
        this.len = len;
        tree = new int[len + 1];
    }

    public BIT50(int[] arr) {
        this.len = arr.length;
        tree = new int[len + 1];
        for (int i = 0; i < arr.length; i++) {
            update(i, arr[i]);
        }
    }

    public int get(int zeroBased) {
        return sum(zeroBased + 1) - sum(zeroBased);
    }

    public void set(int zeroBased, int val) {
        update(zeroBased, val - get(zeroBased));
    }

    public void update(int zeroBased, int delta) {
        updateOneBased(zeroBased + 1, delta);
    }

    public int sumRange(int left, int right) {
        return sum(right + 1) - sum(left);
    }

    private void updateOneBased(int oneBased, int delta) {
        while (oneBased <= len) {
            tree[oneBased] += delta;
            oneBased += lowbit(oneBased);
        }
    }

    private int sum(int oneBasedRight) {
        int result = 0;
        while (oneBasedRight > 0) {
            result += tree[oneBasedRight];
            oneBasedRight -= lowbit(oneBasedRight);
        }
        return result;
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}