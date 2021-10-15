import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.canMeasureWater(3, 5, 4));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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
