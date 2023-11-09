import sun.reflect.generics.tree.Tree;

import java.util.*;

//Definition for a binary tree node.
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}


class Solution {

//    0 1 2 3 4 5 6
//            0 1 2 3 4 5 6
//
//
//            [0,1,1,0,2,2,1]
//            0 1 1 0 1 1 0
//
//            [1,3,2,2,1]
//             0 2 0 1 0

    public int[] constructArray(int n, int k) {
        int[] result = new int[n];
        result[0] = 1;
        int max = n - 1;
        int numOfDiffOne = max - k;
        for (int i = 0; i < numOfDiffOne; i++) {
            result[i + 1] = result[i] + 1;
        }
        int nextStartIdx = numOfDiffOne + 1;
        for (int i = nextStartIdx, j = 0; i < n; i += 2, j++) {
            result[i] = n - j;
        }
        for (int i = nextStartIdx + 1, j = 0; i < n; i += 2, j++) {
            result[i] = result[numOfDiffOne] + j + 1;
        }

        return result;
    }

    public int candy(int[] ratings) {
        int num = ratings.length;
        int[] left = new int[num];
        int[] right = new int[num];
        int[] max = new int[num];
        int result = 0;
        for (int i = 0; i < num - 1; i++) {
            if (ratings[i + 1] > ratings[i]) {
                left[i + 1] = left[i] + 1;
            } else {
                left[i + 1] = 0;
            }
        }
        for (int i = num - 1; i > 0; i--) {
            if (ratings[i - 1] > ratings[i]) {
                right[i - 1] = right[i] + 1;
            } else {
                right[i - 1] = 0;
            }
        }
        for (int i = 0; i < num; i++) {
            result += Math.max(right[i], left[i]);
        }
        result += num;
        return result;
    }

    public int lengthOfLongestSubstring(String s) {
        int left = 0;
        int right = 0;
        int max = 0;

        Set<Character> occ = new HashSet<>();

        for (left = 0; left < s.length(); left++) {
            if (left + max > s.length()) {
                break;
            }
            for (right = left + max; right < s.length(); right++) {
                occ.clear();
                boolean flag = true;
//                String tmp = s.substring(left, right + 1);
//                for (char c : tmp.toCharArray()) {
//                    if (occ.contains(c)) {
//                        flag = false;
//                        break;
//                    } else {
//                        occ.add(c);
//                    }
//                }
//                if (!flag) {
//                    break;
//                }
                for (int i = left; i <= right; i++) {
                    if (occ.contains(s.charAt(i))) {
                        flag = false;
                        break;
                    } else {
                        occ.add(s.charAt(i));
                    }
                }
                if (!flag) {
                    break;
                }
            }
            max = Math.max(max, right - left);
        }
        return max;
    }

    private List<List<Integer>> result = new ArrayList<>();

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<TreeNode> treeNodeList = new ArrayList<>();

        if (root != null) {
            treeNodeList.add(root);
        }
        recursive(treeNodeList, sum);
        return result;
    }

    private void recursive(List<TreeNode> tmpTreePath, int sum) {
        if (tmpTreePath.size() == 0) {
            return;
        }
        if (tmpTreePath.get(tmpTreePath.size() - 1).left != null || tmpTreePath.get(tmpTreePath.size() - 1).right != null) {
            return;
        }

        if (sumList(tmpTreePath) == sum) {

            List<Integer> tmp = new ArrayList<>();
            for (TreeNode i : tmpTreePath) {
                tmp.add(i.val);
            }
            result.add(tmp);
            return;
        } else if (sumList(tmpTreePath) > sum) {
            return;
        }

        if (tmpTreePath.get(tmpTreePath.size() - 1).left != null) {
            tmpTreePath.add(tmpTreePath.get(tmpTreePath.size() - 1).left);
            recursive(tmpTreePath, sum);
            tmpTreePath.remove(tmpTreePath.size() - 1);
        }
        if (tmpTreePath.get(tmpTreePath.size() - 1).right != null) {
            tmpTreePath.add(tmpTreePath.get(tmpTreePath.size() - 1).right);
            recursive(tmpTreePath, sum);
            tmpTreePath.remove(tmpTreePath.size() - 1);
        }

    }

    public int sumList(List<TreeNode> l) {
        int result = 0;
        for (TreeNode i : l) {
            result += i.val;
        }
        return result;
    }

    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];

        for (int i = 0; i < (word1.length() + 1); i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i < (word2.length() + 1); i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i < (word1.length() + 1); i++) {
            for (int j = 1; j < (word2.length() + 1); j++) {

                int min = Math.min(dp[i - 1][j], dp[i][j - 1]);
                min = Math.min(min, dp[i - 1][j - 1]);
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = min;
                } else {
                    dp[i][j] = min + 1;
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    public static void main(String[] args) {
        TreeNode n1 = new TreeNode(-2);
        TreeNode n2 = new TreeNode(-3);
        n1.right = n2;
        List<TreeNode> l = new ArrayList<>();
        l.add(n1);
        l.add(n2);
        Solution s = new Solution();
//        System.err.println(s.sumList(l));
//        System.err.println(s.minDistance("zoologicoarchaeologist","zoogeologist"));
        System.err.println(s.constructArray(5, 3));
        return;
    }

    public int firstUniqChar(String s) {

        Map<Character, Integer> m = new LinkedHashMap<>();
        for (char c : s.toCharArray()) {
            m.putIfAbsent(c, 0);
            m.put(c, m.get(c) + 1);
        }
        int i = 0;
        for (char c : m.keySet()) {
            if (m.get(c) == 1) {
                return s.indexOf(c);
            }
            i++;
        }
        return -1;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        Queue<TreeNode> working = new LinkedList<>();
        working.offer(root);

        while (!working.isEmpty()) {
            int size = working.size();
            List<Integer> thisLayer = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode tmp = working.poll();
                if (tmp != null) {
                    thisLayer.add(tmp.val);
                    working.offer(tmp.left);
                    working.offer(tmp.right);
                }
            }
            result.add(thisLayer);
        }
        result.remove(result.size() - 1);
        return result;
    }
}