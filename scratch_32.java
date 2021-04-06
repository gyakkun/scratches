import javax.swing.*;
import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        TreeNode root = s.recoverBinaryTreeFromInOrderAndPostOrder(new int[]{4, 2, 1, 5, 3}, new int[]{4, 2, 5, 3, 1});
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC36 依次遍历 注意boxIdx的求法
    public boolean isValidSudoku(char[][] board) {
        HashMap<Integer, Integer>[] row = new HashMap[9];
        HashMap<Integer, Integer>[] col = new HashMap[9];
        HashMap<Integer, Integer>[] box = new HashMap[9];
        for (int i = 0; i < 9; i++) {
            row[i] = new HashMap<>();
            col[i] = new HashMap<>();
            box[i] = new HashMap<>();
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char num = board[i][j];
                if (num != '.') {
                    int n = num - '0';
                    int boxIdx = (i / 3) * 3 + j / 3;

                    row[i].put(n, row[i].getOrDefault(n, 0) + 1);
                    col[j].put(n, col[j].getOrDefault(n, 0) + 1);
                    box[boxIdx].put(n, box[boxIdx].getOrDefault(n, 0) + 1);

                    if (row[i].get(n) > 1 || col[j].get(n) > 1 || box[boxIdx].get(n) > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // recover binary tree 根据中序+后序恢复二叉树, 并输出前序遍历结果, 确保数组值唯一
    public TreeNode recoverBinaryTreeFromInOrderAndPostOrder(int[] inOrder, int[] postOrder) {
        return recoverBinaryTreeFromInOrderAndPostOrderRecursive(inOrder, postOrder, inOrder.length, 0, 0);
    }

    private TreeNode recoverBinaryTreeFromInOrderAndPostOrderRecursive(int[] inOrder, int[] postOrder, int subTreeLength, int inOrderBeginIdx, int postOrderBeginIdx) {
        if (subTreeLength == 0) return null;
        if (subTreeLength == 1) return new TreeNode(postOrder[postOrderBeginIdx]);
        TreeNode root = new TreeNode(postOrder[postOrderBeginIdx + subTreeLength - 1]);
        int i;
        for (i = 0; i < subTreeLength; i++) {
            if (inOrder[i + inOrderBeginIdx] == postOrder[postOrderBeginIdx + subTreeLength - 1]) {
                break;
            }
        }
        int nextLeftSubTreeLength = i;
        int nextRightSubTreeLength = subTreeLength - i - 1;
        root.left = recoverBinaryTreeFromInOrderAndPostOrderRecursive(inOrder, postOrder, nextLeftSubTreeLength, inOrderBeginIdx, postOrderBeginIdx);
        root.right = recoverBinaryTreeFromInOrderAndPostOrderRecursive(inOrder, postOrder, nextRightSubTreeLength, i + 1, i);
        return root;
    }

    // recover binary tree 根据中序+前序恢复二叉树, 并输出后序遍历结果, 确保数组值唯一
    public TreeNode recoverBinaryTreeFromInOrderAndPreOrder(int[] inOrder, int[] preOrder) {
        return recoverBinaryTreeFromInOrderAndPreOrderRecursive(inOrder, preOrder, inOrder.length, 0, 0);
    }

    private TreeNode recoverBinaryTreeFromInOrderAndPreOrderRecursive(int[] inOrder, int[] preOrder,
                                                                      int subTreeLength, int inOrderBeginIdx, int preOrderBeginIdx) {
        if (subTreeLength == 0) return null;
        if (subTreeLength == 1) return new TreeNode(preOrder[preOrderBeginIdx]);
        TreeNode root = new TreeNode(preOrder[preOrderBeginIdx]);
        int i;
        for (i = 0; i < subTreeLength; i++) {
            if (inOrder[i + inOrderBeginIdx] == preOrder[preOrderBeginIdx]) {
                break;
            }
        }
        int nextLeftSubTreeLength = i;
        int nextRightSubTreeLength = subTreeLength - i - 1;
        root.left = recoverBinaryTreeFromInOrderAndPreOrderRecursive(inOrder, preOrder, nextLeftSubTreeLength, inOrderBeginIdx, preOrderBeginIdx + 1);
        root.right = recoverBinaryTreeFromInOrderAndPreOrderRecursive(inOrder, preOrder, nextRightSubTreeLength, i + 1, i + 1);
        return root;
    }


    // LC1143
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }

    // LC781
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : answers) {
            m.put(i + 1, m.getOrDefault(i + 1, 0) + 1);
        }
        int result = 0;
        for (Map.Entry<Integer, Integer> e : m.entrySet()) {
            result += (int) Math.ceil((double) e.getValue() / (double) e.getKey()) * e.getKey();
        }
        return Math.max(result, answers.length);
    }

    // LC88 二路归并
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = 0, j = 0;
        int k = 0;
        int[] sorted = new int[m + n];
        while (i != m && j != n) {
            if (nums1[i] < nums2[j]) {
                sorted[k++] = nums1[i++];
            } else {
                sorted[k++] = nums2[j++];
            }
        }
        if (i == m) {
            while (k != m + n) {
                sorted[k++] = nums2[j++];
            }
        } else if (j == n) {
            while (k != m + n) {
                sorted[k++] = nums1[i++];
            }
        }
        for (int t = 0; t < m + n; t++) {
            nums1[t] = sorted[t];
        }
    }

    // LC80
    public int removeDuplicates(int[] nums) {
        int i = 1;
        int ctr = 1;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] == nums[j - 1]) {
                ctr++;
            } else {
                ctr = 1;
            }
            if (ctr <= 2) {
                nums[i++] = nums[j];
            }
        }
        return i;
    }

    // LC50 快速幂 迭代
    public double myPow(double x, int n) {
        if (n == 0) return 1d;
        return n > 0 ? quickMul(x, n) : 1d / quickMul(x, -n);
    }

    private double quickMul(double x, int n) {
        double result = 1d;
        double xCon = x;
        for (int i = 0; i < Integer.SIZE; i++) {
            if (((n >> i) & 1) == 1) {
                result *= xCon;
            }
            xCon *= xCon;
        }
        return result;
    }

    // LC50 快速幂 递归
    public double myPowRecursive(double x, int n) {
        long target = n;
        return n >= 0 ? quickMulReverse(x, target) : 1d / quickMulReverse(x, -target);
    }

    public double quickMulReverse(double x, long target) {
        if (target == 0) return 1d;
        double result = quickMulReverse(x, target / 2);
        return target % 2 == 1 ? result * result * x : result * result;

    }

    // LC48 ROTATE 90 DEGREE CLOCKWISE, O(1) SPACE
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int rounds = (n + 1) / 2;
        for (int i = 0; i < rounds; i++) {
            int sideLen = n - i * 2;
            if (sideLen == 1) break;
            int sideLenMinusOne = sideLen - 1;
            reverseAround(matrix, sideLen, 0, 3 * sideLenMinusOne - 1);
            reverseAround(matrix, sideLen, 3 * sideLenMinusOne, 4 * sideLenMinusOne - 1);
            reverseAround(matrix, sideLen, 0, 4 * sideLenMinusOne - 1);
        }
        return;
    }

    private void reverseAround(int[][] matrix, int sideLen, int begin, int end) {
        int mid = (end - begin) / 2;
        for (int i = 0; i <= mid; i++) {
            int aFlatIdx = begin + i;
            int bFlatIdx = end - i;
            Pair a = idxFlatten(matrix.length, sideLen, aFlatIdx);
            Pair b = idxFlatten(matrix.length, sideLen, bFlatIdx);
            int tmp = matrix[b.row][b.col];
            matrix[b.row][b.col] = matrix[a.row][a.col];
            matrix[a.row][a.col] = tmp;
        }
        return;
    }

    // 上右下左, idx从0计
    private Pair idxFlatten(int n, int sideLen, int idx) {
        int sideLenMinusOne = sideLen - 1;
        int offset = (n - sideLen) / 2;
        if (idx < sideLenMinusOne) {
            return new Pair(offset, offset + idx);
        } else if (idx < sideLenMinusOne * 2) {
            return new Pair(offset + idx - sideLenMinusOne, offset + sideLenMinusOne);
        } else if (idx < sideLenMinusOne * 3) {
            return new Pair(offset + sideLenMinusOne, offset + sideLenMinusOne - (idx - 2 * sideLenMinusOne));
        } else {
            return new Pair(offset + sideLenMinusOne - (idx - 3 * sideLenMinusOne), offset);
        }
    }


    class Pair {
        int col, row;

        public Pair() {
        }

        public Pair(int row, int col) {
            this.col = col;
            this.row = row;
        }
    }

    private void reverseFlat(int[] arr, int begin, int end) {
        int mid = (end - begin) / 2;
        for (int i = 0; i <= mid; i++) {
            int tmp = arr[end - i];
            arr[end - i] = arr[begin + i];
            arr[begin + i] = tmp;
        }
        return;
    }

    // LC1 TWO SUM BINARY SEARCH
    public int[] twoSum(int[] nums, int target) {
        int[][] idxPair = new int[nums.length][2];
        for (int i = 0; i < nums.length; i++) {
            idxPair[i] = new int[]{nums[i], i};
        }
        Arrays.sort(idxPair, Comparator.comparingInt(o -> o[0]));
        for (int i = 0; i < nums.length; i++) {
            int l = 0, h = nums.length - 1;
            while (l <= h) {
                int mid = l + (h - l) / 2;
                if (idxPair[mid][0] == target - nums[i] && idxPair[mid][1] != i) {
                    return new int[]{i, idxPair[mid][1]};
                } else if (idxPair[mid][0] < target - nums[i]) {
                    l = mid + 1;
                } else {
                    h = mid - 1;
                }
            }
        }
        return new int[]{-1, -1};
    }

    // LC1 TWO SUM hashmap
    public int[] twoSumHashMap(int[] nums, int target) {
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (idxMap.containsKey(target - nums[i])) {
                return new int[]{i, idxMap.get(target - nums[i])};
            }
            idxMap.put(nums[i], i);
        }
        return new int[]{-1, -1};
    }

    // LC99 恢复二叉搜索树(左小右大) Hard
    public void recoverTree(TreeNode root) {
        List<Integer> l = new LinkedList<>();
        inOrder(root, l);
        int[] toRecover = new int[2];
        Arrays.fill(toRecover, -1);
        for (int i = 0; i < l.size() - 1; i++) {
            if (l.get(i) > l.get(i + 1)) {
                toRecover[1] = l.get(i + 1);
                if (toRecover[0] == -1) {
                    toRecover[0] = l.get(i);
                } else {
                    break;
                }
            }
        }
        doRecoverTree(root, 2, toRecover);
        return;

    }

    private void doRecoverTree(TreeNode root, int count, int[] vals) {
        if (root != null) {
            if (root.val == vals[0] || root.val == vals[1]) {
                root.val = root.val == vals[0] ? vals[1] : vals[0];
                count--;
                if (count == 0) {
                    return;
                }
            }
            doRecoverTree(root.left, count, vals);
            doRecoverTree(root.right, count, vals);
        }
    }

    private void inOrder(TreeNode root, List<Integer> list) {
        if (root == null) return;
        inOrder(root.left, list);
        list.add(root.val);
        inOrder(root.right, list);
    }

    // LC269 火星字典 , 拓扑排序 (LOCKED题目)
    public String alienDict(String[] words) {
        Set<Character> charSet = new HashSet<>();
        Set<String> pairSet = new HashSet<>();
        Deque<Character> q = new LinkedList<>();
        Map<Character, Integer> inDeg = new HashMap<>();
        StringBuffer result = new StringBuffer();
        for (String word : words) {
            for (char c : word.toCharArray()) {
                charSet.add(c);
            }
        }
        for (int i = 1; i < words.length; i++) {
            String wordA = words[i - 1];
            String wordB = words[i];
            int minLen = Math.min(wordA.length(), wordB.length());
            for (int j = 0; j < minLen; j++) {
                if (wordA.charAt(j) != wordB.charAt(j)) {
                    pairSet.add("" + wordA.charAt(j) + wordB.charAt(j));
                    break;
                }
            }
        }
        for (String s : pairSet) {
            inDeg.put(s.charAt(1), inDeg.getOrDefault(s.charAt(1), 0) + 1);
        }
        // 入度为0的
        for (char c : charSet) {
            if (!inDeg.containsKey(c)) {
                q.offer(c);
            }
        }

        while (!q.isEmpty()) {
            char c = q.poll();
            result.append(c);
            Iterator<String> it = pairSet.iterator();
            while (it.hasNext()) {
                String s = it.next();
                if (s.charAt(0) == c) {
                    inDeg.put(s.charAt(1), inDeg.get(s.charAt(1)) - 1);
                    if (inDeg.get(s.charAt(1)) == 0) {
                        inDeg.remove(s.charAt(1));
                        q.offer(s.charAt(1));
                    }
                    it.remove();
                }
            }
        }
        return result.length() == charSet.size() ? result.toString() : "";

    }


    // LC34
    public int[] searchRange(int[] nums, int target) {
        int[] result = new int[]{-1, -1};
        if (nums.length == 0) return result;
        if (nums.length == 1) return nums[0] == target ? new int[]{0, 0} : result;
        // 二分-1 找是否存在
        int l = 0, h = nums.length - 1;
        int mid = -1;
        boolean exist = false;
        while (l <= h) {
            mid = l + (h - l) / 2;
            if (nums[mid] == target) {
                exist = true;
                break;
            } else if (nums[mid] > target) {
                h = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if (!exist) return result;
        int possibleIdx = mid;

        // 二分-2 找第一个比target小的坐标
        int smallerIdx = -1;
        l = 0;
        h = nums.length - 1;
        while (l <= h) {
            mid = l + (h - l) / 2;
            if (nums[mid] < target) {
                smallerIdx = mid;
                l = mid + 1;
            } else {
                h = mid - 1;
            }
        }
        result[0] = smallerIdx + 1;

        // 二分-3 找第一个比target大的坐标
        int biggerIdx = nums.length;
        l = 0;
        h = nums.length - 1;
        while (l <= h) {
            mid = l + (h - l) / 2;
            if (nums[mid] > target) {
                biggerIdx = mid;
                h = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        result[1] = biggerIdx - 1;

        return result;
    }


    // LC74 搜索升序矩阵, 考虑扁平化+二分
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int totalEle = m * n;
        return binarySearchInSortedMatrix(matrix, target, m, n) != -1;
    }

    private int binarySearchInSortedMatrix(int[][] matrix, int target, int m, int n) {
        int l = 0;
        int h = m * n - 1;
        while (l <= h) {
            int mid = l + (h - l) / 2;
            int midEle = flatMatrix(mid, matrix, m, n);
            if (midEle == target) {
                return mid;
            }
            if (midEle > target) {
                h = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }

    private int flatMatrix(int nth, int[][] matrix, int m, int n) {
        return matrix[nth / n][nth % n];
    }

    // LC33 旋转后的数组的二分查找
    public int binarySearchInRotatedSortedArray(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }

    // LC29
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1)
            return Integer.MAX_VALUE;

        boolean posFlag = (dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0);
        int result = 0;
        // 转换为负数处理, 避免abs(Min_Value)还是它本身导致的越界
        dividend = -Math.abs(dividend);
        divisor = -Math.abs(divisor);
        while (dividend <= divisor) {
            int temp = divisor;
            int ctr = 1; // 移位计数器
            while (dividend - temp <= temp) {
                temp = temp << 1;
                ctr = ctr << 1;
            }
            dividend -= temp;
            result += ctr;
        }
        return posFlag ? result : -result;
    }

    // LC28, String.indexOf(), Sunday 算法
    public int strStr(String mainString, String patternString) {
        Map<Character, Integer> firstOccurIdx = new HashMap<>();
        for (int i = 0; i < patternString.length(); i++) {
            firstOccurIdx.put(patternString.charAt(i), i); // 最右出现的位置
        }
        int i = 0, j;
        // i -> 主串上的指针
        // j -> 模式串上的指针
        while (i < mainString.length() - patternString.length() + 1) {
            for (j = 0; j < patternString.length(); j++) {
                if (mainString.charAt(i + j) != patternString.charAt(j)) {
                    // main:   i s p d f g
                    // patt:     s d f

                    // 对主串: j==1, i==3, i现在是d, 回归s( idx(s)==1, 减几? ), 跳到f(+3), 找f有没有在patt中出现过
                    int tmpIdx = i + patternString.length();

                    // 如果主串在匹配串后一位的f在模式串中出现过
                    if (tmpIdx < mainString.length() && firstOccurIdx.containsKey(mainString.charAt(tmpIdx))) {
                        // 移动主串指针i
                        i += (patternString.length() - firstOccurIdx.get(mainString.charAt(tmpIdx)));
                    } else {
                        i += patternString.length();
                    }
                    break;
                }
            }
            if (j == patternString.length()) return i;
        }
        return -1;
    }

    // LC22
    public List<String> generateParenthesis(int n) {
        List<String> result = new LinkedList<>();

        generateParenthesisBacktrack(result, n, new StringBuffer(), 0, 0);

        return result;
    }

    private void generateParenthesisBacktrack(List<String> result, int max, StringBuffer sb, int open, int close) {
        if (sb.length() == max * 2) {
            result.add(sb.toString());
            return;
        }
        if (open < max) {
            sb.append('(');
            generateParenthesisBacktrack(result, max, sb, open + 1, close);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (close < open) {
            sb.append(')');
            generateParenthesisBacktrack(result, max, sb, open, close + 1);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    // LC20
    public boolean isValidPar(String s) {
        HashMap<Character, Character> matchPair = new HashMap<Character, Character>() {{
            put('(', ')');
            put('[', ']');
            put('{', '}');
        }};
        Deque<Character> stack = new LinkedList<>();

        for (char c : s.toCharArray()) {
            if (matchPair.containsKey(c)) {
                stack.push(c);
            } else {
                if (!stack.isEmpty() && c == matchPair.get(stack.peek())) {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    // LC19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (n <= 0) return head;

        ListNode fast = head;
        ListNode dummy = new ListNode();
        dummy.val = 0;
        dummy.next = head;
        ListNode slowPre = dummy;

        for (int i = 0; i < n; i++) {
            fast = fast.next;

        }

        while (fast != null) {
            slowPre = slowPre.next;
            fast = fast.next;
        }

        slowPre.next = slowPre.next.next;

        return dummy.next;
    }

    // LC61
    public ListNode rotateRight(ListNode head, int k) {
        int len = 0;
        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = head;

        while (cur != null) {
            cur = cur.next;
            len++;
        }
        if (len == 0) return null;
        if (len == 1) return head;
        k = k % len;
        if (k == 0) return head;

        cur = head;
        ListNode pre = dummy;

        for (int i = 0; i < len - k; i++) {
            cur = cur.next;
            pre = pre.next;
        }
        pre.next = null;
        dummy.next = cur;
        while (cur.next != null) {
            cur = cur.next;
        }
        cur.next = head;

        return dummy.next;
    }

    // LC190
    public int reverseBits(int n) {
        int result = 0;
        int i = 0;
        while (i != 31) {
            result += (n >> i) & 1;
            result = result << 1;
            i++;
        }
        result += (n >> 31) & 1;
        return result;
    }

    // LC17
    public List<String> letterCombinations(String digits) {
        Map<Character, Character[]> m = new HashMap<>();
        m.put('2', new Character[]{'a', 'b', 'c'});
        m.put('3', new Character[]{'d', 'e', 'f'});
        m.put('4', new Character[]{'g', 'h', 'i'});
        m.put('5', new Character[]{'j', 'k', 'l'});
        m.put('6', new Character[]{'m', 'n', 'o'});
        m.put('7', new Character[]{'p', 'q', 'r', 's'});
        m.put('8', new Character[]{'t', 'u', 'v'});
        m.put('9', new Character[]{'w', 'x', 'y', 'z'});
        List<String> result = new LinkedList<>();
        for (char c : digits.toCharArray()) {
            int size = result.size();
            if (size != 0) {
                for (int i = 0; i < size; i++) {
                    for (char innerC : m.get(c)) {
                        result.add(result.get(i) + innerC);
                    }
                }
                for (int i = 0; i < size; i++) {
                    result.remove(0);
                }
            } else {
                for (char innerC : m.get(c)) {
                    result.add(String.valueOf(innerC));
                }
            }
        }
        return result;
    }

    // LC13
    public int romanToInt(String s) {

        // 字符          数值
        //  I             1
        //  V             5
        //  X             10
        //  L             50
        //  C             100
        //  D             500
        //  M             1000

        // I可以放在V(5) 和X(10) 的左边，来表示 4 和 9。
        // X可以放在L(50) 和C(100) 的左边，来表示 40 和90。
        // C可以放在D(500) 和M(1000) 的左边，来表示400 和900。

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
        int sum = 0;
        int i = 1;
        while (i < s.length()) {
            int cur = m.get(s.charAt(i++));
            if (pre < cur) {
                sum -= pre;
            } else {
                sum += pre;
            }
            pre = cur;
        }
        sum += pre;
        return sum;
    }

    // LC11
    public int maxArea(int[] height) {
        int result = Integer.MIN_VALUE;
        int left = 0, right = height.length - 1;
        while (left < right) {
            result = Math.max(result, (right - left) * Math.min(height[right], height[left]));
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
            // 假设两侧高度x<=y, 宽为t 面积min(x,y)*t = xt
            // 移动y -> y1
            // 面积min(x,y1)*(t-1)
            //  1) if y1<=y, min(x,y1) <= min(x,y) -> 面积必定更小
            //  2) if y1>y , min(x,y1) = x, 又因为t-1<t, 面积必定更小
            // 因此无论如何, 小的一端(x) 已经无法再作为一端边界取得更大的面积, 只能相向移动
        }
        return result;
    }

    // My Eval
    public double myEval(String expression) {
        return evalRPN(toRPN(decodeExpression(expression)));
    }

    public List<String> toRPN(List<String> express) {
        List<String> rpn = new LinkedList<>();
        Deque<String> stack = new LinkedList<>();
        Set<String> notNumber = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
            add("(");
            add(")");
        }};
        String tmp;
        for (String token : express) {
            if (!notNumber.contains(token)) {
                rpn.add(token);
            } else if (token.equals("(")) {
                stack.push(token);
            } else if (token.equals(")")) {
//                while (!(tmp = stack.pop()).equals("(")) {
//                    rpn.add(tmp);
//                }
                while (!stack.isEmpty()) {
                    tmp = stack.pop();
                    if (tmp.equals("(")) {
                        break;
                    } else {
                        rpn.add(tmp);
                    }
                }

            } else {
                while (!stack.isEmpty() && getOperPriority(stack.peek()) >= getOperPriority(token)) {
                    rpn.add(stack.pop());
                }
                stack.push(token);
            }
        }
        while (!stack.isEmpty()) {
            rpn.add(stack.pop());
        }
        return rpn;
    }

    private List<String> decodeExpression(String express) {
        express = express.replaceAll(" ", "")
                .replaceAll("\\(\\+", "(0+")
                .replaceAll("\\(-", "(0-")
                .replaceAll("\\((\\d+(\\.\\d+)?)\\)", "$1");
        List<String> result = new LinkedList<>();
        int i = 0;
        StringBuffer sb;
        do {
            if ((express.charAt(i) < '0' || express.charAt(i) > '9') && express.charAt(i) != '.') {
                result.add(express.charAt(i) + "");
                i++;
            } else {
                sb = new StringBuffer();
                while (i < express.length() && ((express.charAt(i) >= '0' && express.charAt(i) <= '9') || express.charAt(i) == '.')) {
                    sb.append(express.charAt(i));
                    i++;
                }
                result.add(sb.toString());
            }
        } while (i < express.length());
        return result;
    }

    private int getOperPriority(String oper) {
        switch (oper) {
            case "+":
            case "-":
                return 1;
            case "*":
            case "/":
                return 2;
            default:
                return -1;
        }
    }

    // LC150 逆波兰表达式
    public double evalRPN(List<String> tokens) {
        Deque<String> stack = new LinkedList<>();
        stack.push("0");
        Set<String> oper = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
        }};
        for (String token : tokens) {
            if (oper.contains(token)) {
                double a = Double.parseDouble(stack.pop());
                double b = Double.parseDouble(stack.pop());
                double tmp;
                switch (token) {
                    case "+":
                        tmp = a + b;
                        break;
                    case "-":
                        tmp = b - a;
                        break;
                    case "/":
                        tmp = b / a;
                        break;
                    case "*":
                        tmp = a * b;
                        break;
                    default:
                        tmp = 0;
                }
                stack.push(String.valueOf(tmp));
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0d : Double.parseDouble(stack.pop());
    }

    // LC14
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        StringBuffer sb = new StringBuffer();
        sb.append(strs[0]);
        for (int i = 1; i < strs.length; i++) {
            if (sb.length() == 0) return "";
            if (sb.length() > strs[i].length()) sb.delete(strs[i].length(), sb.length());
            for (int j = 0; j < strs[i].length(); j++) {
                if (j + 1 > sb.length()) break;
                if (strs[i].charAt(j) != sb.charAt(j)) {
                    sb.delete(j, sb.length());
                    break;
                }
            }
        }
        return sb.toString();
    }

    // LC7, 不能使用long, 注意溢出判断
    public int reverse(int x) {
        if (x == 0) return 0;
        boolean negFlag = x < 0;
        if (x < 0) x = -x;
        int result = 0;
        while (x != 0) {
            // 溢出判断
            if (result > Integer.MAX_VALUE / 10) {
                return 0;
            }
            if (result * 10 > Integer.MAX_VALUE - x % 10) {
                return 0;
            }

            result = result * 10 + x % 10;
            x /= 10;
        }
        return negFlag ? -result : result;
    }

    // LC4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int totalLen = nums1.length + nums2.length;
        boolean oddFlag = totalLen % 2 == 1;
        int finalLen = totalLen / 2 + 1;
        List<Integer> l = new ArrayList<>(finalLen);
        int[] longer = nums1.length > nums2.length ? nums1 : nums2;
        int[] shorter = longer == nums1 ? nums2 : nums1;
        int shorterPtr = 0, longerPtr = 0;
        while (l.size() < finalLen) {
            if (shorterPtr != shorter.length && longerPtr != longer.length) {
                if (shorter[shorterPtr] < longer[longerPtr]) {
                    l.add(shorter[shorterPtr++]);
                } else {
                    l.add(longer[longerPtr++]);
                }
            } else if (shorterPtr == shorter.length) {
                l.add(longer[longerPtr++]);
            } else {
                l.add(shorter[shorterPtr++]);
            }
        }
        if (oddFlag) return l.get(l.size() - 1);
        return ((double) l.get(l.size() - 1) + (double) l.get(l.size() - 2)) / 2d;
    }

    // LC83
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                cur = cur.next;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }


    // LC82
    public ListNode deleteDuplicatesLC82(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static class TreeNode {
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

}