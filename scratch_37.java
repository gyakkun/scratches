import java.util.HashMap;
import java.util.Map;

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
        Node a = s.copyRandomList(zero);
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC138
    public Node copyRandomList(Node head) {
        Map<Node, Integer> order = new HashMap<>();
        order.put(null, -1);
        Map<Integer, Integer> randomOrder = new HashMap<>();
        int ctr = 0;
        Node cur = head;
        while (cur != null) {
            order.put(cur, ctr++);
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            randomOrder.put(order.get(cur), order.get(cur.random));
            cur = cur.next;
        }
        Node copyDummy = new Node(-1);
        Node copyPrev = copyDummy; // newHead = copyDummy.next
        cur = head;
        Map<Node, Integer> copyOrder = new HashMap<>();
        copyOrder.put(null, -1);
        Map<Integer, Node> copyOrderReverseLookup = new HashMap<>();
        copyOrderReverseLookup.put(-1, null);
        while (cur != null) {
            copyPrev.next = new Node(cur.val);
            copyPrev = copyPrev.next;
            copyOrder.put(copyPrev, order.get(cur));
            copyOrderReverseLookup.put(order.get(cur), copyPrev);
            cur = cur.next;
        }
        Node copyCur = copyDummy.next;
        while (copyCur != null) {
            copyCur.random = copyOrderReverseLookup.get(randomOrder.get(copyOrder.get(copyCur)));
            copyCur = copyCur.next;
        }
        return copyDummy.next;
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