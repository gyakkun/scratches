import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.reorganizeString("zrhmhyevkojpsegvwolkpystdnkyhcjrdvqtyhucxdcwm"));

        PriorityQueue<Integer> integers;
    }

    // LC767

    class Pair {
        char left;
        int right;

        public Pair(char left, int right) {
            this.left = left;
            this.right = right;
        }
    }

    char[] result;

    public String reorganizeString(String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        int[] ctr = new int[26];
        for (char c : ca) {
            ctr[c - 'a']++;
        }
        int maxIdx = 0;
        for (int i = 0; i < 26; i++) {
            if (ctr[i] > ctr[maxIdx]) {
                maxIdx = i;
            }
        }
        if ((ctr[maxIdx] - 1) * 2 + 1 > n) return "";
        int gapLength = (int) Math.ceil((double) n / (double) ctr[maxIdx]);
        int gapNum = (int) Math.ceil((double) n / (double) gapLength);
        List<Pair> pairList = new ArrayList<>(26);
        for (int i = 0; i < 26; i++) {
            pairList.add(new Pair((char) (i + 'a'), ctr[i]));
        }
        Collections.sort(pairList, Comparator.comparingInt(o -> -o.right));
        int tmpCtr = 0;
        for (Pair p : pairList) {
            for (int i = 0; i < p.right; i++) {
                ca[tmpCtr + i] = p.left;
            }
            tmpCtr += p.right;
        }
        result = new char[n];
        for (int i = 0; i < n; i++) {
            int idx = getIdx(i, n, gapLength, gapNum);
            result[idx] = ca[i];
        }
        return new String(result);
    }

    public int getIdx(int current, int length, int gapLength, int gapNum) {

        int round = current / gapNum;
        int whichGap = current % gapNum;
        int result = round + whichGap * gapLength;
        if (result >= length) {
            return getIdx(current + 1, length, gapLength, gapNum);
        }
        if (this.result[result] != 0) {
            return getIdx(current + 1, length, gapLength, gapNum);
        }
        return result;
    }

    public List<List<Integer>> minimumAbsDifference(int[] arr) {
        Set<Integer> s = new HashSet<>();
        List<List<Integer>> result = new ArrayList<>();
        for (int i : arr)
            s.add(i);
        Arrays.sort(arr);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < arr.length; i++) {
            min = Math.min(min, arr[i] - arr[i - 1]);
        }
        for (int i : arr) {
            if (i + min > arr[arr.length]) break;

            if (s.contains(i + min)) {
                List<Integer> tmp = new ArrayList<>(2);
                tmp.add(i);
                tmp.add(i + min);
                result.add(tmp);
            }
        }
        return result;
    }

    public int rangeBitwiseAnd(int left, int right) {
        int shift = 0;
        while (left < right) {
            left = left >> 1;
            right = right >> 1;
            shift++;
        }
        return left << shift;
    }

    // LC754

    public int reachNumber(int target) {
        double absTarget = Math.abs(target);
        double root = (Math.sqrt(1 + 8 * absTarget) - 1d) / 2d;
        int min = (int) Math.ceil(root);

        int sum = ((1 + min) * min) / 2;
        if (sum == absTarget) return min;

        int diff = sum - (int) absTarget;
        if (diff % 2 == 0) return min;

        if (((1 + min) * (2 + min) / 2 - (int) absTarget) % 2 == 0) return min + 1;
        return min + 2;
    }

    public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {
        int n = s.length();
        Map<String, Integer> occ = new HashMap<>();
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            Set<Character> exist = new HashSet<>();
            String cur = "";
            for (int j = i; j < Math.min(n, i + maxSize); j++) {
                exist.add(s.charAt(j));
                if (exist.size() > maxLetters) {
                    break;
                }
                cur += s.charAt(j);
                if (j - i + 1 >= minSize) {
                    occ.put(cur, occ.getOrDefault(cur, 0) + 1);
                    ans = Math.max(ans, occ.get(cur));
                }
            }
        }
        return ans;
    }

    public int maxFreqTLE(String s, int maxLetters, int minSize, int maxSize) {
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i; j < s.length(); j++) {
                int length = j - i + 1;
                if (length >= minSize && length <= maxSize) {
                    String sub = s.substring(i, j + 1);
                    if (!judgeExceedDistinctChar(sub, maxLetters)) {
                        result = Math.max(result, countSubArray(s, sub));
                    }
                }
            }
        }
        return result;
    }

    private int countSubArray(String m, String s) {
        int strPtr = 0;
        int ctr = 0;

        while (strPtr < m.length()) {
            int sPtr = 0;
            int mPtr = strPtr;
            while (sPtr < s.length() && mPtr < m.length()) {
                if (m.charAt(mPtr) == s.charAt(sPtr)) {
                    mPtr++;
                    sPtr++;
                } else {
                    break;
                }
            }
            if (sPtr == s.length()) {
                ctr++;
            }
            strPtr++;
        }
        return ctr;
    }

    private boolean judgeExceedDistinctChar(String s, int maxLetters) {
        int[] ctr = new int[26];
        int num = 0;
        for (char c : s.toCharArray()) {
            ctr[c - 'a']++;
        }
        for (int i : ctr) {
            if (i > 0) num++;
        }
        return num > maxLetters;
    }


    private void generateAbbreviationsDfs(char[] wa, int from, int last, String abbr, List<String> result) {
        if (from == wa.length) {
            result.add(abbr);
            return;
        }
        for (int i = 0; i <= wa.length - from; i++) {
            generateAbbreviationsDfs(wa, from + Math.max(1, i), i, (i == 0 ? abbr + wa[from] : abbr + i), result);
            if (last > 0) break;
        }
    }

    public List<String> generateAbbreviations(String word) {
        List<String> result = new ArrayList<>();
        generateAbbreviationsDfs(word.toCharArray(), 0, 0, "", result);
        return result;
    }

    // LC320
    // Given word = "word", return the following list (order does not matter):
    // ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

    public List<String> generateAbbreviationsBinaryMask(String word) {
        List<String> result = new ArrayList<>();
        int n = word.length();
        long maxMask = 1 << n;
        for (long mask = 0; mask < maxMask; mask++) {
            StringBuffer sb = new StringBuffer();
            int tmpOneCtr = 0;
            // 掩码为1表示当前待缩写的字符 (用0也一样)
            for (int i = 0; i < n; i++) {
                if (((mask >> i) & 1) == 1) {
                    tmpOneCtr++;
                    continue;
                } else {
                    if (tmpOneCtr != 0) {
                        sb.append(tmpOneCtr);
                    }
                    sb.append(word.charAt(i));
                    tmpOneCtr = 0;
                }
            }
            if (tmpOneCtr != 0) {
                sb.append(tmpOneCtr);
            }
            result.add(sb.toString());
        }
        return result;
    }


    public int minNumberOfSemesters(int n, int[][] dependencies, int k) {

        // 前置课程约束: 不能在同一个学期, 既选择一门课程C有选择C的前置课程(如C有前置课程)

        int[] prereq = new int[n];

        for (int[] dep : dependencies) {
            // 每门课程编号的前置课程掩码
            prereq[dep[1] - 1] |= (1 << (dep[0] - 1));
        }

        int[] setPrereq = new int[1 << n];

        boolean[] valid = new boolean[1 << n];

        // 用掩码表示一个学期的一种选课组合, setPrereq[A]表示当前掩码表示的课程组合A的前置课程组合preA。
        // 只有当A与preA没有交集, 即A不会同时包含A中任意课程及其前置课程
        // 即不违反前置课程约束, valid[mask]才为真
        for (int mask = 0; mask < (1 << n); mask++) {
            // 另一个约束: A不能超过k门
            if (Integer.bitCount(mask) <= k) {
                for (int i = 0; i < n; i++) {
//                    if ((mask & (1 << i)) != 0) {
                    if (((mask >> i) & 1) == 1) {
                        setPrereq[mask] |= prereq[i];
                    }
                }
                valid[mask] = ((setPrereq[mask] & mask) == 0);
            }
        }
        // dp[mask] 表示选课组合为mask时候的最少上课学期
        // 初始化为dp[0]=0, 表示什么课都不选的时候只需要上0学期, 为边界条件
        int[] dp = new int[1 << n];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            // 构造mask的二进制位子集
            for (int subset = mask; subset > 0; subset = (subset - 1) & mask) {
                // 如果当前的子集不违反前置课程约束
                // 且当前选课A 与 A子集a的前置课程集 的交集 等于 子集a的前置课程集
                // <=> 子集a前置课程集是当前选课A的一个子集, 即可以在某个学期上完了(A-a), 再在下一个学期上a
                // 则有转移方程 dp[A] = Min(dp[A], dp[A-a]+1), A-a为集合减法, 实际运算中用掩码的亦或运算表达
                // (注意仅当a为A的子集才能用亦或表达集合减法), 否则应该是 (mask ^ subset) & mask
                if (valid[subset] && ((mask & setPrereq[subset]) == setPrereq[subset])) {
                    dp[mask] = Math.min(dp[mask], dp[mask ^ subset] + 1);
                }
            }
        }
        return dp[(1 << n) - 1];
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

