import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.permute(new int[]{}));
    }

    class Solution {
        public List<String> letterCasePermutation(String S) {
            List<StringBuilder> ans = new ArrayList();
            ans.add(new StringBuilder());

            for (char c: S.toCharArray()) {
                int n = ans.size();
                if (Character.isLetter(c)) {
                    for (int i = 0; i < n; i++) {
                        ans.add(new StringBuilder(ans.get(i)));
                        ans.get(i).append(Character.toLowerCase(c));
                        ans.get(n+i).append(Character.toUpperCase(c));
                    }
                } else {
                    for (int i = 0; i < n; ++i)
                        ans.get(i).append(c);
                }
            }

            List<String> finalans = new ArrayList();
            for (StringBuilder sb: ans)
                finalans.add(sb.toString());
            return finalans;
        }
    }




    List<List<Integer>> permuteResult;

    public List<List<Integer>> permute(int[] nums) {
        permuteResult = new ArrayList<>();
        permuteBacktrack(nums, 0, nums.length);
        return permuteResult;
    }

    public void permuteBacktrack(int[] nums, int curIdx, int fin) {
        if (curIdx == fin) {
            List<Integer> result = new ArrayList<>();
            for (int i = 0; i < fin; i++) {
                result.add(nums[i]);
            }
            permuteResult.add(result);
        }
        for (int i = curIdx; i < fin; i++) {
            // 交换
            int tmp = nums[i];
            nums[i] = nums[curIdx];
            nums[curIdx] = tmp;

            // 回溯
            permuteBacktrack(nums, curIdx + 1, fin);

            // 复原
            tmp = nums[i];
            nums[i] = nums[curIdx];
            nums[curIdx] = tmp;
        }
    }


    public List<List<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> subsetsResult = new ArrayList<>();
        int n = nums.length;
        long maxBin = 1l << n;
        for (long i = 0l; i < maxBin; i++) {
            List<Integer> tmp = new ArrayList<>();
            boolean illegal = false;
            for (int j = 0; j < n; j++) {
                if (((i >> j) & 1l) == 1) {
                    if (j > 0 && nums[j] == nums[j - 1] && (((i >> (j - 1)) & 1l) == 0)) {
                        illegal = true;
                        break;
                    } else {
                        tmp.add(nums[j]);
                    }
                }
            }
            if (!illegal) {
                subsetsResult.add(tmp);
            }
        }
        return subsetsResult;
    }

    public int calculate(String s) {
        // 用1/-1来控制拆括号后的正负, 存在栈中
        Deque<Integer> ops = new LinkedList<Integer>();
        ops.push(1);
        int sign = 1;

        int result = 0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            if (s.charAt(i) == ' ') {
                i++;
            } else if (s.charAt(i) == '+') {
                sign = ops.peek();
                i++;
            } else if (s.charAt(i) == '-') {
                sign = -ops.peek();
                i++;
            } else if (s.charAt(i) == '(') {
                ops.push(sign);
                i++;
            } else if (s.charAt(i) == ')') {
                ops.pop();
                i++;
            } else {
                long num = 0;
                while (i < n && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + s.charAt(i) - '0';
                    i++;
                }
                result += sign * num;
            }
        }
        return result;
    }

    public int findMaximizedCapital(int k, int W, int[] pureProfits, int[] capital) {
        int n = pureProfits.length;
        Set<Integer> tmp = new HashSet<>();
        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return pureProfits[o2] - pureProfits[o1];
            }
        });
        for (int i = 0; i < n; i++) {
            q.add(i);
        }

        // greedy strategy: select the project with maximum profit within the feasible project ( W>=capital )
        while (k != 0) {
            while (!q.isEmpty() && capital[q.peek()] > W) {
                tmp.add(q.poll());
            }
            if (q.isEmpty()) return W;
            W += pureProfits[q.poll()];
            for (int i : tmp) {
                q.add(i);
            }
            tmp.clear();
            k--;
        }
        return W;
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

class MedianFinder {

    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;

    public MedianFinder() {
        maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        minHeap = new PriorityQueue<>((o1, o2) -> o1 - o2);
    }

    public void addNum(int num) {
        if (maxHeap.isEmpty() || num < maxHeap.peek()) {
            maxHeap.offer(num);
        } else {
            minHeap.offer(num);
        }

        if (maxHeap.size() == minHeap.size() + 2) {
            minHeap.offer(maxHeap.poll());
        }
        if (minHeap.size() == maxHeap.size() + 1) {
            maxHeap.offer(minHeap.poll());
        }

    }

    public double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            double median = (maxHeap.peek() + minHeap.peek()) / 2d;
            return median;
        }
        return maxHeap.peek();
    }
}

