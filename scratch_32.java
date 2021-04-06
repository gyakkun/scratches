import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        s.rotate(new int[][]{{5, 1, 9, 11}, {2, 4, 8, 10}, {13, 3, 6, 7}, {15, 14, 12, 16}});
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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