import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        TreeNode a0 = new TreeNode(1);
        TreeNode a1 = new TreeNode(2);
        TreeNode a2 = new TreeNode(2);
        TreeNode a3 = new TreeNode(3);
        TreeNode a4 = new TreeNode(3);
        TreeNode a5 = new TreeNode(3);
        TreeNode a6 = new TreeNode(3);

        a0.left = a1;
        a0.right = a2;
        a1.left = a3;
        a1.right = a4;
        a2.left = a5;
        a2.right = a5;

        System.err.println(s.isSymmetric(a0));

    }

    // LC101
    public boolean isSymmetric(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int layer = 0;

        // 取层数
        while(!q.isEmpty()){
            layer++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                if (q.peek().left != null) {
                    q.offer(q.peek().left);
                } else {
                    ;
                }
                if (q.peek().left != null) {
                    q.offer(q.peek().right);
                } else {
                    ;
                }
                q.poll();
            }
        }

        q.clear();
        q.offer(root);
        int layerCtr = 0;

        while (!q.isEmpty()) {
            layerCtr++;
            if(layerCtr>layer) return true;
            int qLen = q.size();
            List<Integer> valList = new ArrayList<>(qLen);
            for (int i = 0; i < qLen; i++) {
                if (q.peek().left != null) {
                    q.offer(q.peek().left);
                } else {
                    q.offer(new TreeNode(19260817));
                }
                if (q.peek().left != null) {
                    q.offer(q.peek().right);
                } else {
                    q.offer(new TreeNode(19260817));
                }
                valList.add(q.poll().val);
            }
            for (int i = 0; i < valList.size(); i++) {
                if (valList.get(i) != valList.get(valList.size() - 1 - i)) {
                    return false;
                }
            }
        }
        return true;
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