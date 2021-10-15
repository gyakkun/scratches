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
