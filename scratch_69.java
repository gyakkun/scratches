class Scratch {
    public static void main(String[] args) {

    }

    public Node intersect(Node quadTree1, Node quadTree2) {
        Node result = new Node();
        if (quadTree1.isLeaf && quadTree2.isLeaf) {
            result.val = quadTree1.val || quadTree2.val;
            result.isLeaf = true;
        } else if (quadTree1.isLeaf || quadTree2.isLeaf) {
            Node dummy = new Node(), notLeafOne = quadTree1.isLeaf ? quadTree2 : quadTree1;
            dummy.val = quadTree1.isLeaf ? quadTree1.val : quadTree2.val;
            dummy.isLeaf = false;
            result.topLeft = intersect(dummy, notLeafOne.topLeft);
            result.topRight = intersect(dummy, notLeafOne.topRight);
            result.bottomLeft = intersect(dummy, notLeafOne.bottomLeft);
            result.bottomRight = intersect(dummy, notLeafOne.bottomRight);
            result.isLeaf = result.topLeft.val == result.topRight.val && result.bottomLeft.val == result.bottomRight.val
                    && result.bottomRight.val == result.topLeft.val;
            if (result.isLeaf) {
                result.val = result.topLeft.val;
                // result.bottomLeft = result.bottomRight = result.topLeft = result.topRight = null;
                result.bottomLeft = result.bottomRight = result.topLeft = result.topRight = null;
            }
        } else {
            result.topLeft = intersect(quadTree1.topLeft, quadTree2.topLeft);
            result.topRight = intersect(quadTree1.topRight, quadTree2.topRight);
            result.bottomLeft = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
            result.bottomRight = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
            result.isLeaf = result.topLeft.val == result.topRight.val && result.bottomLeft.val == result.bottomRight.val
                    && result.bottomRight.val == result.topLeft.val;
            if (result.isLeaf) {
                result.val = result.topLeft.val;
                // result.bottomLeft = result.bottomRight = result.topLeft = result.topRight = null;
                result.bottomLeft = result.bottomRight = result.topLeft = result.topRight = null;
            }
        }
        return result;
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
