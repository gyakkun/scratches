import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.partition("abb"));
    }

    public int pathSum(TreeNode root, int sum) {
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        prefixSumCount.put(0, 1);
        return recursionPathSum(root, prefixSumCount, sum, 0);
    }

    private int recursionPathSum(TreeNode node, Map<Integer, Integer> prefixSumCount, int target, int currSum) {
        if (node == null) {
            return 0;
        }
        int res = 0;
        currSum += node.val;

        res += prefixSumCount.getOrDefault(currSum - target, 0);
        prefixSumCount.put(currSum, prefixSumCount.getOrDefault(currSum, 0) + 1);

        res += recursionPathSum(node.left, prefixSumCount, target, currSum);
        res += recursionPathSum(node.right, prefixSumCount, target, currSum);

        prefixSumCount.put(currSum, prefixSumCount.get(currSum) - 1);
        return res;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }


    boolean[][] dp;
    List<List<String>> ret = new ArrayList<>();
    List<String> ans = new ArrayList<>();
    int n;

    public List<List<String>> partition(String s) {
        n = s.length();
        dp = new boolean[n][n];
        for (boolean[] b : dp) {
            Arrays.fill(b, true);
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1];
            }
        }
        backtrack(s, 0);
        return ret;
    }

    public void backtrack(String s, int idx) {
        if (idx == n) {
            ret.add(new ArrayList<>(ans));
            return;
        }
        for (int j = idx; j < n; j++) {
            if (dp[idx][j]) {
                ans.add(s.substring(idx, j + 1));
                backtrack(s, j + 1);
                ans.remove(ans.size() - 1);
            }
        }
    }

    public int minCut(String s) {

        int n = s.length();
//        boolean[][] dp = new boolean[n][n];
        boolean[][] dp2 = new boolean[n][n];


        // 取得判定数组
//        for (int l = 0; l < n; ++l) {
//            for (int i = 0; i + l < n; ++i) {
//                int j = i + l;
//                if (l == 0) {
//                    dp[i][j] = true;
//                } else if (l == 1) {
//                    dp[i][j] = (s.charAt(i) == s.charAt(j));
//                } else {
//                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
//                }
//            }
//        }

        // 取得判定数组 方法2
        for (boolean[] b : dp2)
            Arrays.fill(b, true);
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp2[i][j] = s.charAt(i) == s.charAt(j) && dp2[i + 1][j - 1];
            }
        }

        int[] f = new int[n];
        Arrays.fill(f, n - 1);
        for (int i = 0; i < n; i++) {
            if (dp2[0][i]) {
                f[i] = 0;
            } else {
                for (int j = 0; j < i; j++) {
                    if (dp2[j + 1][i]) {
                        f[i] = Math.min(f[j] + 1, f[i]);
                    }
                }
            }
        }

        return f[n - 1];


    }

    public String longestPalindrome(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        String ans = "";
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = true;
                } else if (l == 1) {
                    dp[i][j] = (s.charAt(i) == s.charAt(j));
                } else {
                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.length()) {
                    ans = s.substring(i, i + l + 1);
                }
            }
        }
        return ans;
    }


    public String removeDuplicates(String s) {
        Deque<Character> stack = new LinkedList<>();
        char[] cArr = s.toCharArray();
        for (char c : cArr) {
            if (stack.peek() == null) {
                stack.push(c);
            } else {
                if (stack.peek() != c) {
                    stack.push(c);
                } else {
                    stack.pop();
                }
            }
        }
        StringBuffer sb = new StringBuffer();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        sb.reverse();
        return sb.toString();
    }

}