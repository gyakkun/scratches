import javafx.util.*;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        int[][] towers = {{2, 1, 9}, {0, 1, 9}};


        System.err.println(s.parseBoolExpr("!(t)"));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1106 Hard
    public boolean parseBoolExpr(String expression) {
        char[] oper = {'!', '|', '&'};
        char[] token = {'!', '|', '&', ',', '(', ')'};

        Set<Character> operSet = new HashSet<>();
        for (char c : oper) {
            operSet.add(c);
        }
        Deque<Character> stack = new LinkedList<>();
        char[] exp = expression.toCharArray();
        int n = exp.length;
        Boolean last = null;
        for (int i = 0; i < n; i++) {
            if (operSet.contains(exp[i])) {
                stack.push(exp[i]);
            } else if(exp[i]=='t'||exp[i]=='f'){
                stack.push(exp[i]);
            } else if (exp[i] == ')') {
                List<Boolean> tmpList = new ArrayList<>();
                while(stack.peek()=='t'||stack.peek()=='f'){
                    tmpList.add(stack.peek() == 't' ? true : false);
                    stack.pop();
                }
                char tmpOper = stack.pop();
                switch (tmpOper) {
                    case '&':
                        last = true;
                        for (boolean tf : tmpList) {
                            last &= tf;
                        }
                        break;
                    case '|':
                        last = false;
                        for (boolean tf : tmpList) {
                            last |= tf;
                        }
                        break;
                    case '!':
                        if(tmpList.size()!=1){
                            last = false;
                            break;
                        }
                        last = !tmpList.get(0);
                        break;
                    default:
                        last = false;
                }
                stack.push(last ? 't' : 'f');
            }
        }
        return stack.peek() == 't' ? true : false;
    }

    // JZOF66 不使用除法
    public int[] constructArr(int[] a) {
        if (a.length == 0) return new int[0];
        int n = a.length;
        int[] left = new int[n], right = new int[n];
        left[0] = right[n - 1] = 1;
        for (int i = 1; i < n; i++) {
            left[i] = left[i - 1] * a[i - 1];
            right[n - 1 - i] = right[n - i] * a[n - i];
        }
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = left[i] * right[i];
        }
        return result;
    }

    // LC1209
    public String removeDuplicates(String s, int k) {
        int origLen = s.length();
        int nextLen = s.length();

        // 正则解法
        //        do  {
        //            origLen = s.length();
        //            s = s.replaceAll("(\\w)\\1{" + (k - 1) + "}", "");
        //            nextLen = s.length();
        //        } while(origLen != nextLen);
        //        return s;

        Deque<Pair<Character, Integer>> origStack = new LinkedList<>();
        boolean evenOdd = false;
        for (char c : s.toCharArray()) {
            if (origStack.isEmpty()) {
                origStack.push(new Pair<>(c, 1));
            } else {
                if (c == origStack.peek().getKey()) {
                    origStack.push(new Pair<>(c, origStack.pop().getValue() + 1));
                } else {
                    origStack.push(new Pair<>(c, 1));
                }
            }
        }
        do {
            evenOdd = !evenOdd;
            origLen = nextLen;
            Deque<Pair<Character, Integer>> newStack = new LinkedList<>();
            while (!origStack.isEmpty()) {
                if (!newStack.isEmpty() && origStack.peek().getKey() == newStack.peek().getKey()) {
                    char tmpChar = origStack.peek().getKey();
                    int tmpCount = origStack.pop().getValue() + newStack.peek().getValue();
                    nextLen -= (tmpCount / k) * k;
                    newStack.pop();
                    if (tmpCount % k != 0) {
                        newStack.push(new Pair<>(tmpChar, tmpCount % k));
                    }
                } else {
                    Pair<Character, Integer> tmpPop = origStack.pop();
                    int tmpCount = tmpPop.getValue();
                    nextLen -= (tmpCount / k) * k;
                    if (tmpCount % k != 0) {
                        tmpPop = new Pair<>(tmpPop.getKey(), tmpPop.getValue() % k);
                        newStack.push(tmpPop);
                    }
                }
            }
            origStack = newStack;
        } while (origLen != nextLen);

        StringBuilder sb = new StringBuilder();
        if (evenOdd) {
            while (!origStack.isEmpty()) {
                Pair<Character, Integer> tmpPop = origStack.pop();
                for (int i = 0; i < tmpPop.getValue(); i++) {
                    sb.append(tmpPop.getKey());
                }
            }
        } else {
            while (!origStack.isEmpty()) {
                Pair<Character, Integer> tmpPoll = origStack.pollLast();
                for (int i = 0; i < tmpPoll.getValue(); i++) {
                    sb.append(tmpPoll.getKey());
                }
            }
        }
        return sb.toString();
    }

    // LC820
    public int minimumLengthEncoding(String[] words) {
        int n = words.length;
        Trie trie = new Trie();
        for (int i = 0; i < n; i++) {
            words[i] = new StringBuilder(words[i]).reverse().toString();
            trie.insert(words[i]);
        }
        Arrays.sort(words);
        Map<String, String> m = new HashMap<>();
        for (String reverseWord : words) {
            for (int i = 1; i <= reverseWord.length(); i++) {
                String cut = reverseWord.substring(0, i);
                if (trie.search(cut)) {
                    if (m.containsKey(cut)) {
                        if (m.get(cut).length() < reverseWord.length()) {
                            m.put(cut, reverseWord);
                        }
                    } else {
                        m.put(cut, reverseWord);
                    }
                }
            }
        }
        Set<String> set = new HashSet<>();
        int result = 0;
        for (Map.Entry<String, String> entry : m.entrySet()) {
            if (set.add(entry.getValue())) {
                result += entry.getValue().length() + 1;
            }
        }
        return result;
    }

    // LC1620 模拟
    public int[] bestCoordinate(int[][] towers, int radius) {
        int[] result = new int[2];
        int max = Integer.MIN_VALUE;
        int maxDis = radius * radius;
        for (int x = 0; x <= 50; x++) {
            for (int y = 0; y <= 50; y++) {
                int qualitySum = 0;
                for (int[] ot : towers) {
                    int distance = (ot[0] - x) * (ot[0] - x) + (ot[1] - y) * (ot[1] - y);
                    if (distance <= maxDis) {
                        int quality = (int) Math.floor(ot[2] / (1 + Math.sqrt(distance)));
                        qualitySum += quality;
                    }
                }
                if (qualitySum > max) {
                    result = new int[]{x, y};
                    max = qualitySum;
                } else if (qualitySum == max) {
                    int origX = result[0];
                    if (origX < x) {
                        continue;
                    } else if (origX == x) {
                        int origY = result[1];
                        if (origY < y) {
                            continue;
                        } else {
                            result = new int[]{x, y};
                        }
                    } else {
                        result = new int[]{x, y};
                    }
                }
            }
        }
        return result;
    }

    // LC523 0是任何数的倍数
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;

        // 目标 prefix[i] - prefix[j] == 0 MOD k , i-j >=2
        // <=> prefix[i] == prefix[j] MOD k , i-j>=2
        Map<Integer, Integer> m = new HashMap<>(); // key: 余数 value: 最早出现的下标
        m.put(0, 0);
        int prefix = 0;
        for (int i = 1; i <= n; i++) {
            prefix = prefix + nums[i - 1];
            Integer floor = m.get(prefix % k);
            if (floor != null) {
                if (i - floor >= 2) {
                    return true;
                }
            } else {
                m.put(prefix % k, i);
            }
        }
        return false;
    }

    // LC563
    int lc563Result = 0;

    public int findTilt(TreeNode root) {
        sumTree(root);
        return lc563Result;
    }

    private int sumTree(TreeNode root) {
        if (root == null) return 0;
        int left = 0, right = 0;
        if (root.left != null) left = sumTree(root.left);
        if (root.right != null) right = sumTree(root.right);
        lc563Result += Math.abs(left - right);
        return left + right + root.val;
    }


    // LC962 ** 单调栈
    public int maxWidthRamp(int[] nums) {
        int n = nums.length;
        int result = 0;
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            while (stack.isEmpty() || nums[stack.peek()] > nums[i]) { // 先找递减坡 (下坡)
                stack.push(i);
            }
        }
        for (int i = nums.length - 1; i >= 0; i--) { // 反向遍历
            while (!stack.isEmpty() && nums[stack.peek()] <= nums[i]) {
                result = Math.max(result, i - stack.pop());
            }
        }
        return result;
    }

    // LC1711 因为下标有先后顺序, 不需要一开始就全部添加, 只需迭代的时候递增即可
    public int countPairs(int[] deliciousness) {
        Arrays.sort(deliciousness);
        int min = deliciousness[0];
        int max = deliciousness[deliciousness.length - 1];
        int mod = 1000000007;
        int maxPotLog = (int) Math.ceil((Math.log(2 * max) / Math.log(2)));
        long result = 0;
        int[] map = new int[max - min + 1];

        for (int i : deliciousness) {
            int pot = 1;
            for (int j = 0; j <= maxPotLog; j++) {
                if (pot >= i && (pot - i) <= max && (pot - i) >= min && map[(pot - i) - min] > 0) {
                    result += map[(pot - i) - min];
                }
                pot *= 2;
            }
            map[i - min]++;
        }
        return (int) (result % mod);
    }

    private boolean isPowerOfTwo(int i) {
        return i > 0 && (i & (i - 1)) == 0;
    }

    // LC1124 ** 注意单调栈的使用
    public int longestWPI(int[] hours) {
        int[] prefix = new int[hours.length + 1];
        for (int i = 1; i <= hours.length; i++) {
            prefix[i] = prefix[i - 1] + (hours[i - 1] > 8 ? 1 : -1);
        }
        int result = 0;
        // 在 prefix 中找 i,j, j>i 且 prefix[j] > prefix[i], 使得 j - i 最大
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i <= hours.length; i++) {
            while (stack.isEmpty() || prefix[stack.peek()] > prefix[i]) {
                stack.push(i);
            }
        }
        for (int i = hours.length; i >= 0; i--) {
            while (!stack.isEmpty() && prefix[stack.peek()] < prefix[i]) {
                result = Math.max(result, i - stack.pop());
            }
        }
        return result;
    }

    // LC1744 注意int long类型转换
    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        boolean[] result = new boolean[queries.length];
        long[] prefix = new long[candiesCount.length + 1];
        for (int i = 1; i <= candiesCount.length; i++) {
            prefix[i] = prefix[i - 1] + candiesCount[i - 1];
        }

        for (int i = 0; i < queries.length; i++) {
            int favoriteType = queries[i][0];
            int favoriteDay = queries[i][1];
            int dailyCap = queries[i][2];

            // 当天可吃到的糖果的区间
            long minEating = favoriteDay + 1;
            long maxEating = (long) (favoriteDay + 1) * dailyCap;

            // 需要的当天需要吃到的糖果区间
            long minCondition = prefix[favoriteType] + 1;
            long maxCondition = prefix[favoriteType + 1];

            if (minEating <= maxCondition && minCondition <= maxEating) {
                result[i] = true;
            }
        }
        return result;
    }

    // LC668 乘法表m*n 中第k小的数
    public int findKthNumber(int m, int n, int k) {
        int low = 1;
        int high = m * n;
        while (low < high) {
            int mid = low + (high - low) / 2;

            // 小于等于mid的数有几个?
            int count = 0;
            count += n * (mid / n);
            int upper = Math.min(mid, m);
            for (int i = (mid / n) + 1; i <= upper; i++) {
                count += mid / i;
            }
            if (count >= k) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    //  LC373
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<Pair<Integer, List<Integer>>> pq = new PriorityQueue<>(new Comparator<Pair<Integer, List<Integer>>>() {
            @Override
            public int compare(Pair<Integer, List<Integer>> o1, Pair<Integer, List<Integer>> o2) {
                return o2.getKey() - o1.getKey();
            }
        });
        for (int i : nums1) {
            for (int j : nums2) {
                if (pq.size() < k) {
                    pq.offer(new Pair<>(i + j, Arrays.asList(new Integer[]{i, j})));
                } else {
                    if (i + j < pq.peek().getKey()) {
                        pq.poll();
                        pq.offer(new Pair<>(i + j, Arrays.asList(new Integer[]{i, j})));
                    }
                }
            }
        }
        List<List<Integer>> result = new ArrayList<>(k);
        while (!pq.isEmpty()) {
            result.add(pq.poll().getValue());
        }
        result.sort((o1, o2) -> (o1.get(0) == o2.get(0) ? o1.get(1) - o2.get(1) : o1.get(0) - o2.get(0)));
        return result;
    }


    // LC373
    public List<List<Integer>> kSmallestPairsMy(int[] nums1, int[] nums2, int k) {
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                return (o2.getValue() + o2.getKey()) - (o1.getValue() - o1.getKey());
            }
        });
        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                if (pq.size() < k) {
                    pq.offer(new Pair<>(nums1[i], nums2[j]));
                } else {
                    if (nums1[i] + nums2[j] < pq.peek().getKey() + pq.peek().getValue()) {
                        pq.poll();
                        pq.offer(new Pair<>(nums1[i], nums2[j]));
                    }
                }
            }
        }
        List<List<Integer>> result = new ArrayList<>(k);
        while (!pq.isEmpty()) {
            Pair<Integer, Integer> tmp = pq.poll();
            result.add(Arrays.asList(new Integer[]{tmp.getKey(), tmp.getValue()}));
        }
        return result;
    }

    // LC373
    public List<List<Integer>> kSmallestPairsWA(int[] nums1, int[] nums2, int k) {
        int n1L = nums1.length, n2L = nums2.length;
        if (k > n1L * n2L) k = n1L * n2L;
        List<List<Integer>> result = new ArrayList<>();
//        Arrays.sort(nums1);
//        Arrays.sort(nums2);
        Map<Integer, Integer> m1 = new HashMap<>();
        Map<Integer, Integer> m2 = new HashMap<>();
        TreeSet<Integer> ts1 = new TreeSet<>();
        TreeSet<Integer> ts2 = new TreeSet<>();
        for (int i = 0; i < n1L; i++) {
            m1.put(nums1[i], i);
            ts1.add(nums1[i]);
        }
        for (int i = 0; i < n2L; i++) {
            m2.put(nums2[i], i);
            ts2.add(nums2[i]);
        }

        int low = nums1[0] + nums2[0];
        int high = nums1[n1L - 1] + nums2[n2L - 1];
        while (low < high) {
            int mid = low + (high - low) / 2;
            int count = 0, n1Ptr = 0, n2Ptr = 0;
            // 固定每一个 nums1[i] , 找到最后一个nums2[j] 使得 nums1[i] + nums2[j] <= mid -> nums2[j] <= mid - nums1[i]
            for (; n1Ptr < n1L; n1Ptr++) {
                Integer floor = ts2.floor(mid - nums1[n1Ptr]);
                if (floor != null) {
                    int idx = m2.get(floor);
                    count += idx + 1;
                }
            }
            if (count >= k) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        int count = 0, n1Ptr = 0, n2Ptr = 0;
        for (; n1Ptr < n1L; n1Ptr++) {
            Integer floor = ts2.floor(low - nums1[n1Ptr]);
            if (floor != null) {
                int idx = m2.get(floor);
                for (int i = 0; i <= idx; i++) {
                    Integer[] tmp = {nums1[n1Ptr], nums2[i]};
                    result.add(Arrays.asList(tmp));
                    if (result.size() == k) break;
                }
            }
            if (result.size() == k) break;
        }

        return result;
    } //

    // LC719 ** 第k小数对差 Hard
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int lo = 0;
        int hi = nums[n - 1];
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = 0, left = 0, right = 0;
            for (; right < n; right++) {
                while (nums[right] - nums[left] > mid) {
                    left++;
                }
                count += right - left;
            }
            if (count >= k) hi = mid; // 找大于等于k的第一个数
            else lo = mid + 1;
        }
        return lo;
    }

    // LC958
    public boolean isCompleteTree(TreeNode root) {
        if (root == null) return true;
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int maybeLastButOneLayer = -1;
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int size = q.size();

            for (int i = 0; i < size; i++) {
                TreeNode tmpTreeNode = q.poll();
                // 当疑似倒数第二层出现时候, 如果该层后续节点仍有节点拥有非空子树, 则该树不可能为完全二叉树
                if (maybeLastButOneLayer != -1 && (tmpTreeNode.left != null || tmpTreeNode.right != null)) {
                    return false;
                }

                if (tmpTreeNode.left == null || tmpTreeNode.right == null) {
                    // 第一个有空子树的节点出现时, 该节点 1) 要不两个都空 2) 要不只空右子树 , 否则该树不可能为完全二叉树
                    if (tmpTreeNode.left == null && tmpTreeNode.right != null) {
                        return false;
                    }
                    maybeLastButOneLayer = layer;
                }

                if (tmpTreeNode.left != null) {
                    q.offer(tmpTreeNode.left);
                }
                if (tmpTreeNode.right != null) {
                    q.offer(tmpTreeNode.right);
                }
            }
        }

        return true;
    }

    // LC Interview 16.06
    public int smallestDifference(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        long smallestDifference = Math.abs((long) a[0] - (long) b[0]);


        int aPtr = 0;
        int bPtr = 0;
        while (aPtr < a.length && bPtr < b.length) {
            // 移动aptr 还是移动bptr?
            // a<b, a右移 直到 a>=b
            while (aPtr < a.length && bPtr < b.length && a[aPtr] <= b[bPtr]) {
                smallestDifference = Math.min(smallestDifference, Math.abs((long) a[aPtr] - (long) b[bPtr]));
                aPtr++;
                if (smallestDifference == 0) return 0;
            }

            while (aPtr < a.length && bPtr < b.length && a[aPtr] >= b[bPtr]) {
                smallestDifference = Math.min(smallestDifference, Math.abs((long) a[aPtr] - (long) b[bPtr]));
                bPtr++;
                if (smallestDifference == 0) return 0;
            }
        }

        return (int) smallestDifference;
    }

    // LC326
    public boolean isPowerOfThree(int n) {
        int upper = (int) (Math.log(Integer.MAX_VALUE) / Math.log(3));
        int max3Power = (int) (Math.pow(3, upper));
        return n > 0 && max3Power % n == 0;
    }

    // LC342 Power of 4
    public boolean isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }

    // LC1074
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int rowNum = matrix.length;
        int colNum = matrix[0].length;
        int[][] prefix = new int[rowNum + 1][colNum + 1];
        for (int i = 1; i <= rowNum; i++) {
            for (int j = 1; j <= colNum; j++) {
                prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }

        int result = 0;

        // 枚举
        for (int i = 1; i <= rowNum; i++) {
            for (int j = 1; j <= colNum; j++) {

                // 长x宽  = i * j
                for (int row = 0; row <= rowNum - i; row++) {
                    for (int col = 0; col <= colNum - j; col++) {

                        // 四个点 upLeft upRight downLeft downRight
                        int upLeftRow = row;
                        int upLeftCol = col;
                        int upRightRow = row;
                        int upRightCol = col + j;
                        int downLeftRow = row + i;
                        int downLeftCol = col;
                        int downRightRow = row + i;
                        int downRightCol = col + j;

                        int sum = prefix[downRightRow][downRightCol] - prefix[downLeftRow][downLeftCol] - prefix[upRightRow][upRightCol] + prefix[upLeftRow][upLeftCol];
                        if (sum == target) {
                            result++;
                        }

                    }
                }
            }
        }
        return result;

    }

    // LC1629
    // LeetCode 设计了一款新式键盘，正在测试其可用性。测试人员将会点击一系列键（总计 n 个），每次一个。
    //
    // 给你一个长度为 n 的字符串 keysPressed ，其中 keysPressed[i] 表示测试序列中第 i 个被按下的键。releaseTimes 是一个升序排列的列表，其中 releaseTimes[i] 表示松开第 i 个键的时间。字符串和数组的 下标都从 0 开始 。第 0 个键在时间为 0 时被按下，接下来每个键都 恰好 在前一个键松开时被按下。
    //
    // 测试人员想要找出按键 持续时间最长 的键。第 i 次按键的持续时间为 releaseTimes[i] - releaseTimes[i - 1] ，第 0 次按键的持续时间为 releaseTimes[0] 。
    //
    // 注意，测试期间，同一个键可以在不同时刻被多次按下，而每次的持续时间都可能不同。
    //
    // 请返回按键 持续时间最长 的键，如果有多个这样的键，则返回 按字母顺序排列最大 的那个键。
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        int n = keysPressed.length();
        char[] cArr = keysPressed.toCharArray();
        long[] timeCount = new long[26];
        timeCount[cArr[0] - 'a'] += releaseTimes[0];
        for (int i = 1; i < n; i++) {
            timeCount[cArr[i] - 'a'] = Math.max((long) releaseTimes[i] - (long) releaseTimes[i - 1], timeCount[cArr[i] - 'a']);
        }
        long max = timeCount[0];
        char ans = 'a';
        for (int i = 1; i < 26; i++) {
            if (timeCount[i] >= max) {
                max = timeCount[i];
                ans = (char) ('a' + i);
            }
        }
        return ans;
    }

    // LC472 这都行???
    List<String> lc472Result = new LinkedList<>();
    Trie lc472Trie;

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        lc472Trie = new Trie();
        for (String word : words) {
            if (!word.equals("")) {
                lc472Trie.insert(word);
            }
        }
        for (String word : words) {
            if (!word.equals("")) {
                if (lc472Helper(word, 0, 1, 0)) {
                    lc472Result.add(word);
                }
            }
        }
        return lc472Result;
    }

    private boolean lc472Helper(String word, int idx, int cur, int ctr) {
        if (cur >= word.length()) {
            if (lc472Trie.search(word.substring(idx)) && ctr >= 1) {
                return true;
            }
            return false;
        }
        if (!lc472Trie.startsWith(word.substring(idx, cur))) {
            return false;
        }
        if (lc472Trie.search(word.substring(idx, cur)) && lc472Helper(word, cur, cur + 1, ctr + 1)) {
            return true;
        }
        return lc472Helper(word, idx, cur + 1, ctr);
    }

    // 标准蓄水池抽样算法, 通常nums的长度很大, 或是只是一个链表头节点, 未知总长度
    public int[] reservoirSampling(int[] nums, int m) {
        int[] result = new int[m];
        int total = 0;
        int count = 0;
        Random r = new Random();
        for (int i : nums) {
            total++;
            if (count < m) {
                result[count++] = i;
            } else {
                int ran = r.nextInt(total);
                if (ran >= 0 && ran < m) {
                    result[ran] = i;
                }
            }
        }
        return result;
    }

    // LC398
    public int pick(int target, int[] nums) {
        Random r = new Random();
        int count = 0, result = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == target) {
                count++;
                if (r.nextInt(count) == 0) { // 蓄水池大小为1, 落在[0,1)范围的数即只有0
                    result = i;
                }
            }
        }
        return result;
    }

    // LC382 流中的随机抽样问题, 蓄水池算法
    ListNode h;
    ListNode dummy = new ListNode(-1);
    Random r = new Random();
    int len = -1;

    public int getRandom() {
        // 蓄水池算法, 以1/n的概率保留第n个数, 每个数的期望概率都是1/len
        int reserve = 0;
        ListNode cur = h;
        int count = 0;
        while (cur != null) {
            count++;
            int ran = r.nextInt(count);
            if (ran == 0) {
                reserve = cur.val;
            }
            cur = cur.next;
        }
        return reserve;
    }

    public int getRandomMy() {
        int nth;
        if (len == -1) {
            nth = r.nextInt();
        } else {
            nth = r.nextInt(len);
        }

        ListNode ptr = h;
        int ctr = 1;
        while (ptr.next != null && ctr <= nth) {
            ptr = ptr.next;
            ctr++;
        }
        if (ptr.next == null) {
            len = ctr;
        }
        return ptr.val;

    }

    // LC1298 Learn from Solution
    public int maxCandiesS(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        int n = status.length;
        int ans = 0;
        boolean[] hasBox = new boolean[n];
        boolean[] canOpen = new boolean[n];
        boolean[] visited = new boolean[n];
        Deque<Integer> boxQueue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (status[i] == 1) {
                canOpen[i] = true;
            }
        }
        for (int i : initialBoxes) {
            hasBox[i] = true;
            if (canOpen[i]) {
                boxQueue.offer(i);
                visited[i] = true;
                ans += candies[i];
            }
        }
        while (!boxQueue.isEmpty()) {
            int frontBoxIdx = boxQueue.poll();
            for (int key : keys[frontBoxIdx]) {
                canOpen[key] = true;
                if (!visited[key] && hasBox[key]) {
                    boxQueue.offer(key);
                    visited[key] = true;
                    ans += candies[key];
                }
            }
            for (int box : containedBoxes[frontBoxIdx]) {
                hasBox[box] = true;
                if (!visited[box] && canOpen[box]) {
                    boxQueue.offer(box);
                    visited[box] = true;
                    ans += candies[box];
                }
            }
        }
        return ans;
    }

    // LC1298 Hard BFS
    // 给你n个盒子，每个盒子的格式为[status, candies, keys, containedBoxes]，其中：
    //
    // 状态字status[i]：整数，如果box[i]是开的，那么是 1，否则是 0。
    // 糖果数candies[i]: 整数，表示box[i] 中糖果的数目。
    // 钥匙keys[i]：数组，表示你打开box[i]后，可以得到一些盒子的钥匙，每个元素分别为该钥匙对应盒子的下标。
    // 内含的盒子containedBoxes[i]：整数，表示放在box[i]里的盒子所对应的下标。
    // 给你一个initialBoxes 数组，表示你现在得到的盒子，你可以获得里面的糖果，也可以用盒子里的钥匙打开新的盒子，还可以继续探索从这个盒子里找到的其他盒子。
    //
    // 请你按照上述规则，返回可以获得糖果的 最大数目。
    //
    // 每个盒子最多被一个盒子包含。
    public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        Set<Integer> acquiredKey = new HashSet<>();
        Deque<Integer> boxVisitQueue = new LinkedList<>();
//        Set<Integer> visited = new HashSet<>();
        int[] noKeyCounter = new int[status.length];
        int ans = 0;
        for (int i : initialBoxes) {
            boxVisitQueue.offer(i);
        }
        while (!boxVisitQueue.isEmpty()) {
            int tmpBoxIdx = boxVisitQueue.poll();
//            if (visited.contains(tmpBoxIdx)) {
//                continue;
//            }
            if (status[tmpBoxIdx] == 1 || acquiredKey.contains(tmpBoxIdx)) {
                ans += candies[tmpBoxIdx];
                for (int j : containedBoxes[tmpBoxIdx]) {
                    boxVisitQueue.offer(j);
                }
                for (int j : keys[tmpBoxIdx]) {
                    acquiredKey.add(j);
                }
//                visited.add(tmpBoxIdx);
            } else {
                noKeyCounter[tmpBoxIdx]++;
                if (noKeyCounter[tmpBoxIdx] > status.length) {
                    break;
                }
                boxVisitQueue.offer(tmpBoxIdx);
            }
        }
        return ans;
    }

    // LC477 Solution
    public int totalHammingDistance(int[] nums) {
        int ans = 0, n = nums.length;
        for (int i = 0; i < 30; i++) {
            int c = 0;
            for (int val : nums) {
                c += (val >> i) & 1;
            }
            ans += c * (n - c);
        }
        return ans;
    }

    // LC874
    public int robotSim(int[] commands, int[][] obstacles) {
        // -2 左转90度
        // -1 右转90度
        int x = 0, y = 0;
        int direct = 0; // 0 - 北, 1 - 东 , 2 - 南, 3 - 西
        int[][] step = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int ans = 0;
        Set<Pair<Integer, Integer>> obSet = new HashSet<>();
        for (int[] i : obstacles) {
            obSet.add(new Pair<>(i[0], i[1]));
        }
        for (int c : commands) {
            if (c < 0) {
                if (c == -2) {
                    direct = (direct + 4 - 1) % 4;
                } else if (c == -1) {
                    direct = (direct + 1) % 4;
                }
            } else {
                for (int j = 0; j < c; j++) {
                    if (!obSet.contains(new Pair<>(x + step[direct][0], y + step[direct][1]))) {
                        x += step[direct][0];
                        y += step[direct][1];
                        ans = Math.max(ans, x * x + y * y);
                    } else {
                        break;
                    }
                }
            }
        }
        return ans;
    }
}


class ListNode {
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

class Trie {
    Map<String, Boolean> m;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        m = new HashMap<>();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        for (int i = 0; i < word.length(); i++) {
            m.putIfAbsent(word.substring(0, i + 1), false);
        }
        m.put(word, true);
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        return m.getOrDefault(word, false);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        return m.containsKey(prefix);
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