import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.transformArray(new int[]{6, 2, 3, 4}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC311 矩阵乘法
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        // axb mult bxc = axc
        int a = mat1.length, b = mat1[0].length, c = mat2[0].length;
        int[][] result = new int[a][c];
        for (int i = 0; i < a; i++) {
            for (int k = 0; k < b; k++) {
                if (mat1[i][k] == 0) continue;
                for (int j = 0; j < c; j++) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return result;
    }

    // LC259 ** Solution O(n^2)
    public int threeSumSmaller(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int result = 0;
        for (int i = 0; i < n - 2; i++) {
            result += twoSumSmaller(nums, i + 1, target - nums[i]);
        }
        return result;
    }

    private int twoSumSmaller(int[] nums, int startIdx, int target) {
        int result = 0;
        int left = startIdx, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                result += right - left;
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    // LC1243
    public List<Integer> transformArray(int[] arr) {
        int n = arr.length;
        List<Integer> prev = Arrays.stream(arr).boxed().collect(Collectors.toList());
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < n; i++) cur.add(-1);
        while (true) {
            cur = helper(prev);
            if (cur.equals(prev)) return cur;
            prev = cur;
        }
    }

    private List<Integer> helper(List<Integer> prev) {
        int n = prev.size();
        List<Integer> cur = new ArrayList<>();
        cur.add(prev.get(0));
        for (int i = 1; i < n - 1; i++) {
            // 假如一个元素小于它的左右邻居，那么该元素自增 1。
            // 假如一个元素大于它的左右邻居，那么该元素自减 1。
            if (prev.get(i) < prev.get(i - 1) && prev.get(i) < prev.get(i + 1)) {
                cur.add(prev.get(i) + 1);
            } else if (prev.get(i) > prev.get(i - 1) && prev.get(i) > prev.get(i + 1)) {
                cur.add(prev.get(i) - 1);
            } else {
                cur.add(prev.get(i));
            }
        }
        cur.add(prev.get(n - 1));
        return cur;
    }


    // Interview 17.09 LC264 UglyNumber 丑数
    public int getKthMagicNumber(int k) {
        // Prime Factor 3,5,7
        long[] factor = {3, 5, 7};
        PriorityQueue<Long> pq = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        pq.offer(1l);
        set.add(1l);
        long result = -1;
        for (int i = 0; i < k; i++) {
            long p = pq.poll();
            result = p;
            for (long f : factor) {
                if (set.add(f * p)) {
                    pq.offer(f * p);
                }
            }
        }
        return (int) result;
    }

    // LC365
    public boolean canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        Deque<int[]> q = new LinkedList<>();
        Set<Pair<Integer, Integer>> visited = new HashSet<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{jug1Capacity, jug2Capacity});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            Pair<Integer, Integer> pair = new Pair<>(p[0], p[1]);
            if (visited.contains(pair)) continue;
            visited.add(pair);
            if (p[0] == targetCapacity || p[1] == targetCapacity) return true;
            if (p[0] + p[1] == targetCapacity) return true;
            // 倒满一侧
            pair = new Pair<>(jug1Capacity, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{jug1Capacity, p[1]});
            }
            pair = new Pair<>(p[0], jug2Capacity);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], jug2Capacity});
            }
            // 倒掉一侧
            pair = new Pair<>(0, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{0, p[1]});
            }
            pair = new Pair<>(p[0], 0);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], 0});
            }
            // 一侧倒向另一侧
            if (p[0] < jug1Capacity) {
                int jug1Empty = jug1Capacity - p[0];
                int jug2ToJug1 = Math.min(p[1], jug1Empty);
                pair = new Pair<>(p[0] + jug2ToJug1, p[1] - jug2ToJug1);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] + jug2ToJug1, p[1] - jug2ToJug1});
                }
            }
            if (p[1] < jug2Capacity) {
                int jug2Empty = jug2Capacity - p[1];
                int jug1ToJug2 = Math.min(p[0], jug2Empty);
                pair = new Pair<>(p[0] - jug1ToJug2, p[1] + jug1ToJug2);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] - jug1ToJug2, p[1] + jug1ToJug2});
                }
            }
        }
        return false;
    }

    // LC439 ** Great Solution
    public String parseTernary(String expression) {
        int len = expression.length();
        int level = 0;
        for (int i = 1; i < len; i++) {
            if (expression.charAt(i) == '?') level++;
            if (expression.charAt(i) == ':') level--;
            if (level == 0) {
                return expression.charAt(0) == 'T' ?
                        parseTernary(expression.substring(2, i)) : parseTernary(expression.substring(i + 1));
            }
        }
        return expression;
    }

    // LC385
    public NestedInteger deserialize(String s) {
        NestedInteger root = new NestedInteger();
        if (s.charAt(0) != '[') {
            root.setInteger(Integer.parseInt(s));
            return root;
        }
        Deque<NestedInteger> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '[') {
                NestedInteger next = new NestedInteger();
                stack.push(next);
            } else if (c == ']') {
                NestedInteger pop = stack.pop();
                if (sb.length() != 0) {
                    pop.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                if (!stack.isEmpty()) {
                    stack.peek().add(pop);
                    continue;
                } else {
                    return pop;
                }
            } else if (c == ',') {
                NestedInteger peek = stack.peek();
                if (sb.length() != 0) {
                    peek.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                continue;
            } else {
                sb.append(c);
            }
        }
        return null;
    }
}

// LC385
class NestedInteger {
    // Constructor initializes an empty nested list.
    public NestedInteger() {

    }

    // Constructor initializes a single integer.
    public NestedInteger(int value) {

    }

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger() {
        return false;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger() {
        return -1;
    }

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value) {
        ;
    }

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni) {
        ;
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return empty list if this NestedInteger holds a single integer
    public List<NestedInteger> getList() {
        return null;
    }
}
