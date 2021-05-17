import com.sun.javafx.image.IntPixelGetter;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.nextGreaterElement(new int[]{4, 1, 2}, new int[]{1, 3, 4, 2}));
    }

    // LC437
    int pathSumIiiResult;

    public int pathSumIiiHelper(TreeNode root, int sum) {
        if (root == null) return 0;
        sum -= root.val;
        return (sum == 0 ? 1 : 0) + pathSumIiiHelper(root.left, sum) + pathSumIiiHelper(root.right, sum);
    }

    public int pathSumIii(TreeNode root, int sum) {
        if (root == null) return 0;
        return pathSumIiiHelper(root, sum) + pathSumIii(root.left, sum) + pathSumIii(root.right, sum);
    }

    List<List<Integer>> pathSumResult = new ArrayList<>();

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        pathSumHelper(root, targetSum, new ArrayList<>());
        return pathSumResult;
    }

    public void pathSumHelper(TreeNode root, int sum, List<Integer> l) {
        if (root == null) return;
        l.add(root.val);
        if (root.val - sum == 0 && root.left == null && root.right == null) {
            pathSumResult.add(l);
            return;
        }
        pathSumHelper(root.left, sum - root.val, new ArrayList<>(l));
        pathSumHelper(root.right, sum - root.val, new ArrayList<>(l));
    }


    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.val - sum == 0 && root.left == null && root.right == null) return true;
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    // LC496
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] m2 = simpleNGE(nums2);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < m2.length; i++) {
            map.put(nums2[i], m2[i]);
        }
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = map.getOrDefault(nums1[i], -1);
        }
        return result;
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


    // LC503 
    public int[] nextGreaterElements(int[] nums) {
        int[] doubleNums = new int[nums.length * 2 + 1];
        int[] result = new int[nums.length];
        Arrays.fill(result, -1);
        for (int i = 0; i < nums.length * 2 + 1; i++) {
            doubleNums[i] = nums[i % nums.length];
        }
        doubleNums[nums.length * 2] = Integer.MAX_VALUE;
        Deque<Integer> stack = new ArrayDeque<>(); // 效率优于LinkedList

        for (int i = 0; i < nums.length * 2 + 1; i++) {
            if (!stack.isEmpty()) {
                // 队首 == 栈底
                // 队尾 == 栈顶
                if (nums[i % nums.length] < stack.peekLast()) {
                    stack.offer(nums[i % nums.length]);
                } else {
                    int ctr = 1;
                    // 注意此时i指向较大的元素
                    while (!stack.isEmpty() && nums[i % nums.length] > stack.peekLast()) {

                        result[(i - ctr) % nums.length] = nums[i % nums.length];
                        stack.pollLast();
                        ctr++;
                    }
//                    if (stack.isEmpty() || nums[i % nums.length] != stack.peekLast()){
//                    if (stack.isEmpty()) {
                    stack.offer(nums[i % nums.length]);
//                    }
                }
            } else {
                stack.offer(nums[i % nums.length]);
            }
        }

        return result;

    }


    // Definition for a binary tree node.
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    int maxSum = Integer.MIN_VALUE;

    public int maxSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }

    public int maxGain(TreeNode root) {
        if (root == null) return 0;
        int leftGain = Math.max(maxGain(root.left), 0);
        int rightGain = Math.max(maxGain(root.right), 0);

        int currentSum = leftGain + rightGain + root.val;
        maxSum = Math.max(maxSum, currentSum);

        return root.val + Math.max(leftGain, rightGain);
    }

    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int layerCtr = 0;
        Queue<TreeNode> working = new LinkedList<>();

        working.offer(root);
        while (working != null) {
            int size = working.size();
            layerCtr++;
            for (int i = 0; i < size; i++) {
                TreeNode tmp = working.poll();
                if (tmp.left == null && tmp.right == null) return layerCtr++;
                if (tmp.left != null) working.add(tmp.left);
                if (tmp.right != null) working.add(tmp.right);
            }
        }
        return layerCtr;
    }

    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> result = new LinkedList<>();
        Queue<TreeNode> working = new LinkedList<>();
        working.offer(root);
        while (!working.isEmpty()) {
            int size = working.size();
            double tmpAverage = 0;
            for (int i = 0; i < size; i++) {
                TreeNode tmp = working.poll();
                tmpAverage += tmp.val;
                if (tmp.left != null) working.offer(tmp.left);
                if (tmp.right != null) working.offer(tmp.right);
            }
            tmpAverage /= size;
            result.add(tmpAverage);
        }
        return result;
    }

    public List<List<Integer>> levelOrderFromBottom(TreeNode root) {
        List<List<Integer>> levelOrder = new LinkedList<List<Integer>>();
        if (root == null) {
            return levelOrder;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                TreeNode left = node.left, right = node.right;
                if (left != null) {
                    queue.offer(left);
                }
                if (right != null) {
                    queue.offer(right);
                }
            }
            levelOrder.add(0, level);
        }
        return levelOrder;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }

        Queue<TreeNode> nodeQueue = new LinkedList<>();
        nodeQueue.offer(root);
        boolean isOrderLeft = true;

        while (!nodeQueue.isEmpty()) {
            Deque<Integer> levelList = new LinkedList<>();
            int size = nodeQueue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode curNode = nodeQueue.poll();
                if (isOrderLeft) {
                    levelList.offerLast(curNode.val);
                } else {
                    levelList.offerFirst(curNode.val);
                }
                if (curNode.left != null) {
                    nodeQueue.offer(curNode.left);
                }
                if (curNode.right != null) {
                    nodeQueue.offer(curNode.right);
                }
            }
            ans.add(new LinkedList<Integer>(levelList));
            isOrderLeft = !isOrderLeft;
        }

        return ans;
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

    class ListNode {
        int val;
        ListNode next;

        public ListNode(int n) {
            this.val = n;
        }
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;

        ListNode traversal = head;
        int ctr = 0;
        while (traversal != null) {
            traversal = traversal.next;
            ctr++;
        }

        ctr -= ctr % k;
        while (ctr != 0) {
            head = reverseBetween(head, ctr - k + 1, ctr);
            ctr -= k;
        }
        return head;

    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null) return null;

        ListNode cur = head, prev = null;
        while (left > 1) {
            prev = cur;
            cur = cur.next;
            left--;
            right--;
        }

        ListNode tail = cur, concierge = prev;

        ListNode origNext = null;
        while (right > 0) {
            origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
            right--;
        }

        if (concierge != null) {
            concierge.next = prev;
        } else {
            head = prev;
        }

        tail.next = cur;
        return head;
    }


    public ListNode reverseBetweenLC(ListNode head, int left, int right) {

        // Empty list
        if (head == null) {
            return null;
        }

        // original:
        // 1->2->3->4->5->6
        // left = 1, right = 3
        // target:
        // 3->2->1->4->5->6


        // Move the two pointers until they reach the proper starting point
        // in the list.
        ListNode cur = head, prev = null;
        while (left > 1) {
            prev = cur;
            cur = cur.next;
            left--;
            right--;
        }


        // The two pointers that will fix the final connections.
        ListNode con = prev, tail = cur;  // prev = null, cur = 1, con=null, tail=1

        // Iteratively reverse the nodes until n becomes 0.
        ListNode third = null;
        while (right > 0) {
            third = cur.next;
            cur.next = prev;
            prev = cur;
            cur = third;
            right--;
        }

        // Adjust the final connections as explained in the algorithm
        if (con != null) {
            con.next = prev;
        } else {
            head = prev;
        }

        tail.next = cur;
        return head;
    }

    public ListNode reverseLinkedTreeNodes(ListNode head) {
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

    public int maxEnvelopes(int[][] env) {
        if (env.length <= 1) return env.length;
        // (长, 宽), 先按长排序
        Arrays.sort(env, (a, b) -> {
            if (a[0] == b[0]) return b[1] - a[1];
            return a[0] - b[0];
        });
        int dp[] = new int[env.length];
        Arrays.fill(dp, 1);
        // dp[i] 存储当前坐标的信封最多可以装下多少个信封
        // dp[i] = 0<=k<=i-1中, 宽小于env[i]且dp[k]最大的 +1


        for (int i = 1; i < env.length; i++) {
//            int m = env[i][1];
            int tmpMax = 0;
            for (int j = i; j >= 0; j--) {
                if (env[j][1] < env[i][1]) {
                    tmpMax = Math.max(tmpMax, dp[j]);
                }
            }
            dp[i] = tmpMax + 1;
        }
        return Arrays.stream(dp).max().getAsInt();
    }

    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> result = new ArrayList<>(nums.length / 2);
        int n = nums.length;
        for (int i : nums) {
            int absI = Math.abs(i);
            if (nums[absI - 1] < 0) {
                result.add(absI);
            } else {
                nums[absI - 1] *= -1;
            }
        }
        return result;

    }

    public int findDuplicate(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i : nums) {
            if (s.contains(i)) return i;
            s.add(i);
        }
        return -1;
    }

    // LC41
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    // LC268
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int missing = n;
        for (int i = 0; i < n; i++) {
            missing ^= i ^ nums[i];
        }
        return missing;
    }

    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int n = intervals.length;
        Arrays.sort(intervals, Comparator.comparingInt((a) -> a[1]));
        int right = intervals[0][1];
        int selected = 1;
        for (int i = 1; i < n; i++) {
            if (intervals[i][0] >= right) {
                selected++;
                right = intervals[i][1];
            }
        }
        return n - selected;
    }

    public int[][] intervalIntersection(int[][] A, int[][] B) {
        List<int[]> ans = new ArrayList();
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            // Let's check if A[i] intersects B[j].
            // lo - the startpoint of the intersection
            // hi - the endpoint of the intersection
            int lo = Math.max(A[i][0], B[j][0]);
            int hi = Math.min(A[i][1], B[j][1]);
            // 如果存在交集
            if (lo <= hi)
                ans.add(new int[]{lo, hi});
            // Remove the interval with the smallest endpoint
            // 如果上面已经根据当前最小右端点的位置删除了某个区间, 则向右移
            // 动态更新A,B中的最小右端点所在的坐标, 确保A[i]和B[j]中存在当前最小右端点。
            if (A[i][1] < B[j][1])
                i++;
            else
                j++;
        }
        return ans.toArray(new int[ans.size()][]);
    }

    public int[][] insertInterval(int[][] intervals, int[] newInterval) {
        boolean isPlaced = false;
        int left = newInterval[0];
        int right = newInterval[1];
        List<int[]> result = new ArrayList<>();
        for (int[] i : intervals) {
            if (i[0] > right) {
                if (!isPlaced) {
                    result.add(new int[]{left, right});
                    isPlaced = true;
                }
                result.add(i);
            } else if (i[1] < left) {
                result.add(i);
            } else {
                left = Math.min(left, i[0]);
                right = Math.min(right, i[1]);
            }
        }
        if (!isPlaced) {
            result.add(new int[]{left, right});
        }
        return result.toArray(new int[result.size()][]);
    }

    public int[] countBits(int num) {
        int[] result = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            result[i] = Integer.bitCount(i);
        }
        return result;
    }
}