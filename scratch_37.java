import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();

        Node zero = new Node(0);
        Node one = new Node(1);
        Node two = new Node(2);
        Node three = new Node(3);
        Node four = new Node(4);

        zero.next = one;
        one.next = two;
        two.next = three;
        three.next = four;

        zero.random = four;
        one.random = three;
        two.random = null;
        three.random = one;
        four.random = zero;

        long timing = System.currentTimeMillis();
        List<String> wordDict = new ArrayList<>();
        wordDict.add("leet");
        wordDict.add("code");
        wordDict.add("l");
        wordDict.add("ee");
        wordDict.add("t");
        System.err.println(s.wordBreak140("leetcode", wordDict));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC140
    private List<String> lc140Result = new LinkedList<>();
    private int longestWordLen = 0;

    // LC140
    public List<String> wordBreak140(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        for (String word : wordSet) {
            longestWordLen = Math.max(longestWordLen, word.length());
        }
        wordBreak140Backtrack(s, wordSet, 0, new LinkedList<>());
        return lc140Result;
    }

    private void wordBreak140Backtrack(String s, Set<String> wordSet, int curIdx, List<String> curList) {
        if (curIdx == s.length()) {
            lc140Result.add(String.join(" ", curList));
            return;
        }
        for (int i = 1; i <= longestWordLen; i++) {
            if (curIdx + i <= s.length() && wordSet.contains(s.substring(curIdx, curIdx + i))) {
                curList.add(s.substring(curIdx, curIdx + i));
                wordBreak140Backtrack(s, wordSet, curIdx + i, curList);
                curList.remove(curList.size() - 1);
            }
        }
    }

    // LC139
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] reachable = new boolean[n + 1];
        reachable[0] = true;
        Set<String> wordSet = new HashSet<>(wordDict);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (reachable[i] && wordSet.contains(s.substring(i, j))) {
                    reachable[j] = true;
                }
            }
        }
        return reachable[n];
    }

    // LC138
    public Node copyRandomList(Node head) {
        if (head == null) {
            return head;
        }
        Map<Node, Node> map = new HashMap<>();
        Node node = head;
        while (node != null) {
            Node temp = new Node(node.val);
            map.put(node, temp);
            node = node.next;
        }
        node = head;
        while (node != null) {
            map.get(node).next = map.get(node.next);
            map.get(node).random = map.get(node.random);
            node = node.next;
        }
        return map.get(head);
    }

}


// LC138
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}