import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.err.println(s.canPartitionKSubsetsTLE(new int[]{3522, 181, 521, 515, 304, 123, 2512, 312, 922, 407, 146, 1932, 4037, 2646, 3871, 269}, 5));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC13
    public int romanToInt(String s) {
        Map<Character, Integer> m = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};
        int pre = m.get(s.charAt(0));
        int cur = 1;
        int sum = 0;
        while (cur < s.length()) {
            int tmp = m.get(s.charAt(cur++));
            if (pre < tmp) {
                sum -= pre;
            } else {
                sum += pre;
            }
            pre = tmp;
        }
        sum += pre;
        return sum;
    }

    // LC698
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (sum % k != 0) return false;
        int subsetSum = sum / k;
        // 从空集的补集(全集)开始(11111) > 产生子集(10101) > 如果和为k > (选择)产生补集(01010) > 对补集使用backtrack函数 > 取消选择
        // 开一个数组存和值, 初始化为-1, 按需求和
        int[] sums = new int[1 << nums.length];
        Arrays.fill(sums, -1);

        return lc698Helper(nums, (1 << nums.length) - 1, new ArrayList<>(k), k, sums, subsetSum);
    }

    private boolean lc698Helper(int[] nums, int fullSetMask, List<Integer> selectedSet, int k, int[] sums, int subsetSum) {
        if (selectedSet.size() == k) {
            return true;
        }
        if (fullSetMask == 0) {
            return false;
        }
        // 产生二进制子集算法
        for (int subset = fullSetMask; subset != 0; subset = (subset - 1) & fullSetMask) {
            if (sums[subset] == -1) {
                sums[subset] = 0;
                for (int i = 0; i < nums.length; i++) {
                    if (((subset >> i) & 1) == 1) {
                        sums[subset] += nums[i];
                    }
                }
            }
            if (sums[subset] == subsetSum) {
                selectedSet.add(subset);
                // 求当前已经选的集合的并集
                int all = 0;
                for (int i : selectedSet) {
                    all |= i;
                }
                // 求当前并集的补集
                int sup = (1 << nums.length) - 1;
                sup ^= all;
                if (lc698Helper(nums, sup, selectedSet, k, sums, subsetSum)) {
                    return true;
                }
                selectedSet.remove(selectedSet.size() - 1);
            }
        }
        return false;
    }

    // LC698 TLE 剪枝后还是TLE
    public boolean canPartitionKSubsetsTLE(int[] nums, int k) {
        int sum = 0;
        List<Integer> option = new LinkedList<>();
        for (int i : nums) {
            option.add(i);
            sum += i;
        }
        if (sum % k != 0) return false;
        int subsetSum = sum / k;

        return lc698helperTLE(nums, subsetSum, k, 0, option);
    }

    private boolean lc698helperTLE(int[] nums, int subsetSum, int leftSetNum, int currentSum, List<Integer> option) {
        if (leftSetNum == 0 && currentSum == 0) {
            return true;
        }
        for (int i = 0; i < option.size(); i++) {
            int tmpOption = option.get(i);
            int tmpSum = currentSum + tmpOption;
            option.remove(i);

            if (tmpSum == subsetSum) {
                if (lc698helperTLE(nums, subsetSum, leftSetNum - 1, 0, option)) {
                    return true;
                }
            } else if(tmpSum<subsetSum) { // 注意这里的剪枝
                if (lc698helperTLE(nums, subsetSum, leftSetNum, tmpSum, option)) {
                    return true;
                }
            }

            option.add(i, tmpOption);
        }
        return false;
    }

    // LC494 DP
    public int findTargetSumWays(int[] nums, int target) {
        // array[i] >=0
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        if (Math.abs(target) > sum) return 0;
        int t = sum * 2 + 1;
        int[][] dp = new int[nums.length + 1][t];
        // dp[i][j]: 加入前i个元素, 到达j的方案个数
        // dp[i][j] = dp[i-1][j-nums[i-1]] + dp[i-1][j+nums[i+1]], 注意使用sum做负下标修正偏移量
        dp[0][sum] = 1;

        for (int i = 1; i <= nums.length; i++) {
            for (int j = 0; j < t; j++) {
                int tmp = 0;
                if (j - nums[i - 1] >= 0) {
                    tmp += dp[i - 1][j - nums[i - 1]];
                }
                if (j + nums[i - 1] < t) {
                    tmp += dp[i - 1][j + nums[i - 1]];
                }
                dp[i][j] = tmp;
            }
        }

        return dp[nums.length][sum + target];
    }

    // LC416 DP 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }
        int halfSum = sum / 2;
        int[] dp = new int[halfSum + 1];
        // dp[i][j] 表示添加前i个元素 在背包大小限制为j的情况下能达到的最大值
        for (int i = 1; i <= nums.length; i++) {
            for (int j = halfSum; j >= 0; j--) {
//                dp[j] = dp[j];
                if (j - nums[i - 1] >= 0 && dp[j - nums[i - 1]] + nums[i - 1] <= halfSum) {
                    dp[j] = Math.max(dp[j], dp[j - nums[i - 1]] + nums[i - 1]);
                }
            }
            if (dp[halfSum] == halfSum) return true;
        }

        return false;
    }

    // LC494
    int lc494Result = 0;

    public int findTargetSumWaysDFS(int[] array, int target) {
        lc494Helper(array, target, 0, 0);
        return lc494Result;
    }

    private void lc494Helper(int[] array, int target, int currentIdx, int currentSum) {
        if (currentIdx == array.length) {
            if (currentSum == target) {
                lc494Result++;
            }
        } else {
            lc494Helper(array, target, currentIdx + 1, currentSum + array[currentIdx]);
            lc494Helper(array, target, currentIdx + 1, currentSum - array[currentIdx]);
        }
    }

    // LC503
    public int[] nextGreaterElements(int[] nums) {
        int[] snge = simpleNGE(nums);
        int[] doubleArray = new int[nums.length * 2];
        System.arraycopy(nums, 0, doubleArray, 0, nums.length);
        System.arraycopy(nums, 0, doubleArray, nums.length, nums.length);
        int[] dnge = simpleNGE(doubleArray);
        for (int i = 0; i < nums.length; i++) {
            if (snge[i] != -1) {
                continue;
            } else {
                snge[i] = dnge[i];
            }
        }
        return snge;
    }

    public int[] simpleNGE(int[] nums) {
        int n = nums.length;
        Deque<Integer> stack = new LinkedList<>();
        int[] result = new int[n];
        Arrays.fill(result, -1);
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                result[stack.pop()] = nums[i];
            }
            stack.push(i);
        }
        return result;
    }

    // LC993
    public boolean isCousins(TreeNode root, int x, int y) {
        if (root == null || root.left == null || root.right == null) return false;
        Deque<TreeNode> q = new LinkedList<>();
        Map<TreeNode, TreeNode> father = new HashMap<>();
        Map<TreeNode, Integer> layer = new HashMap<>();
        q.offer(root);
        int layerCtr = -1;
        boolean xFlag = false, yFlag = false;
        TreeNode xTN = null, yTN = null;
        while (!q.isEmpty()) {
            layerCtr++;
            int qLen = q.size();
            for (int i = 0; i < qLen; i++) {
                TreeNode tmp = q.poll();
                layer.put(tmp, layerCtr);
                if (tmp.left != null) {
                    father.put(tmp.left, tmp);
                    q.offer(tmp.left);
                }
                if (tmp.right != null) {
                    father.put(tmp.right, tmp);
                    q.offer(tmp.right);
                }
                if (tmp.val == x) {
                    xTN = tmp;
                    xFlag = true;
                }
                if (tmp.val == y) {
                    yTN = tmp;
                    yFlag = true;
                }
                if (xFlag && yFlag) break;
            }
        }
        return father.get(xTN) != father.get(yTN) && layer.get(xTN) == layer.get(yTN);
    }

    // LC451
    public String frequencySort(String s) {
        StringBuffer sb = new StringBuffer(s.length());

        Map<Character, Integer> freq = new HashMap<>();
        for (char c : s.toCharArray()) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }
        List<Pair<Character, Integer>> freqList = new ArrayList<>(freq.keySet().size());
        for (Map.Entry<Character, Integer> entry : freq.entrySet()) {
            freqList.add(new Pair<>(entry.getKey(), entry.getValue()));
        }
        Collections.sort(freqList, new Comparator<Pair<Character, Integer>>() {
            @Override
            public int compare(Pair<Character, Integer> o1, Pair<Character, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }
        });
        for (Pair<Character, Integer> pair : freqList) {
            for (int i = 0; i < pair.getValue(); i++) {
                sb.append(pair.getKey());
            }
        }

        return sb.toString();
    }

    // LC784
    public List<String> letterCasePermutation(String S) {
        List<String> result = new ArrayList<>();
        List<Integer> letterIdx = new ArrayList<>();
        for (int i = 0; i < S.length(); i++) {
            if (Character.isLetter(S.charAt(i))) {
                letterIdx.add(i);
            }
        }
        char[] lowerCase = new char[]{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
        char[] upperCase = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
        int maxMask = 1 << letterIdx.size();
        for (int mask = 0; mask < maxMask; mask++) {
            StringBuffer sb = new StringBuffer(S);
            for (int i = 0; i < letterIdx.size(); i++) {
                // 1 变 0 不变
                if (((mask >> i) & 1) == 1) {
                    int idx = letterIdx.get(i);
                    char c = S.charAt(idx);
                    sb.setCharAt(idx, Character.isLowerCase(c) ? upperCase[c - 'a'] : lowerCase[c - 'A']);
                }
            }
            result.add(sb.toString());
        }

        return result;
    }

    // LC480
    PriorityQueue<Long> maxHeap;
    PriorityQueue<Long> minHeap;

    public double[] medianSlidingWindow(int[] nums, int k) {
        minHeap = new PriorityQueue<>(Comparator.comparingLong(o -> o));
        maxHeap = new PriorityQueue<>(Comparator.comparingLong(o -> -o));

        int len = nums.length;
        double[] result = new double[len - k + 1];
        for (int i = 0; i < k; i++) {
            addNum(nums[i]);
        }
        for (int i = 0; i < len - k + 1; i++) {
            if (minHeap.size() == maxHeap.size()) {
                long sum = (minHeap.peek() + maxHeap.peek());
                result[i] = ((double) (sum)) / 2;
            } else {
                result[i] = (double) maxHeap.peek();
            }
            removeNum(nums[i]);
            if (i + k < len) {
                addNum(nums[i + k]);
            }
        }
        return result;

    }

    private void addNum(long i) {
        if (maxHeap.isEmpty() || i < maxHeap.peek()) {
            maxHeap.offer(i);
        } else {
            minHeap.offer(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    private void removeNum(long i) {
        if (maxHeap.contains(i)) {
            maxHeap.remove(i);
        } else {
            minHeap.remove(i);
        }

        while (maxHeap.size() > minHeap.size()) {
            minHeap.offer(maxHeap.poll());
        }
        while (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
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
