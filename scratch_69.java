import java.util.Arrays;

class Scratch {
    public static void main(String[] args) {

    }

    // LC1184
    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start == destination) return 0;
        int forward = 0, backward = 0, startPoint = destination > start ? start : destination,
                endPoint = destination > startPoint ? destination : start, total = Arrays.stream(distance).sum();

        for (int i = startPoint; i < endPoint; i++) {
            forward += distance[i];
        }
        backward = total - forward;
        return Math.min(backward, forward);
    }

    // LC558 ** Quad Tree
    public Node intersect(Node quadTree1, Node quadTree2) {
        if (quadTree1.isLeaf) {
            if (quadTree1.val) {
                return new Node() {{
                    val = true;
                    isLeaf = true;
                }};
            }
            return new Node(quadTree2.val, quadTree2.isLeaf, quadTree2.topLeft, quadTree2.topRight, quadTree2.bottomLeft, quadTree2.bottomRight);
        }
        if (quadTree2.isLeaf) {
            return intersect(quadTree2, quadTree1);
        }
        Node o1 = intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node o2 = intersect(quadTree1.topRight, quadTree2.topRight);
        Node o3 = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node o4 = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node() {{
                val = o1.val;
                isLeaf = true;
            }};
        }
        return new Node(false, false, o1, o2, o3, o4);
    }

}

// Definition for a QuadTree node.
class Node {
    public boolean val;
    public boolean isLeaf; // true means all value of the 4 corner is the same
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;

    public Node() {
    }

    public Node(boolean _val, boolean _isLeaf, Node _topLeft, Node _topRight, Node _bottomLeft, Node _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
}
