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
        if(head==null){
            return head;
        }
        Map<Node, Node> map = new HashMap<>();
        Node node = head;
        while(node != null){
            Node temp = new Node(node.val);
            map.put(node, temp);
            node = node.next;
        }
        node = head;
        while(node!=null){
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