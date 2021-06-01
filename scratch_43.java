import javafx.util.*;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        int[] arr1 = {0, 0, 0, 0, 0};
        int[] arr2 = {-3, 22, 35, 56, 76};


        System.err.println(s.canEat(new int[]{5215, 14414, 67303, 93431, 44959, 34974, 22935, 64205, 28863, 3436, 45640, 34940, 38519, 5705, 14594, 30510, 4418, 87954, 8423, 65872, 79062, 83736, 47851, 64523, 15639, 19173, 88996, 97578, 1106, 17767, 63298, 8620, 67281, 76666, 50386, 97303, 26476, 95239, 21967, 31606, 3943, 33752, 29634, 35981, 42216, 88584, 2774, 3839, 81067, 59193, 225, 8289, 9295, 9268, 4762, 2276, 7641, 3542, 3415, 1372, 5538, 878, 5051, 7631, 1394, 5372, 2384, 2050, 6766, 3616, 7181, 7605, 3718, 8498, 7065, 1369, 1967, 2781, 7598, 6562, 7150, 8132, 1276, 6656, 1868, 8584, 9442, 8762, 6210, 6963, 4068, 1605, 2780, 556, 6825, 4961, 4041, 4923, 8660, 4114},
                new int[][]{{46, 4191056, 444472063}, {75, 865431, 146060662}, {91, 244597, 840227137}, {89, 2601754, 901415801}, {69, 1777314, 444098682}, {78, 2957259, 231019870}, {19, 4350225, 516815116}, {42, 4081198, 594990005}, {59, 3176552, 508520222}, {77, 4577766, 38900694}, {92, 320256, 1362}, {44, 3992014, 7209}, {55, 1950613, 1370}, {97, 734069, 3066}, {39, 1188632, 661}, {58, 4526426, 6202}, {51, 3083812, 1767}, {46, 2563654, 9680}, {21, 4012578, 7014}, {66, 2185952, 7039}, {67, 3712445, 1239}, {0, 1840130, 185}, {35, 605159, 7105}, {94, 2269908, 416}, {68, 4117247, 2076}, {0, 4540381, 2412}, {20, 579583, 8917}, {62, 4407388, 7127}, {17, 4468545, 6287}, {50, 3462654, 1410}, {7, 1883037, 77}, {4, 4089924, 5849}, {5, 4340465, 3843}, {68, 596099, 5796}, {29, 542371, 5952}, {91, 441898, 2227}, {35, 912775, 6110}, {12, 267236, 3248}, {27, 990261, 771}, {76, 320119, 5220}, {23, 738123, 2504}, {66, 439801, 4436}, {18, 372357, 1654}, {51, 846227, 5325}, {80, 502088, 3751}, {49, 117408, 102}, {75, 837527, 8747}, {46, 984134, 7924}, {42, 463312, 7558}, {50, 214995, 1043}, {94, 981465, 6758}, {79, 892988, 1063}, {17, 985872, 2314}, {71, 870151, 2004}, {63, 793308, 7608}, {49, 873121, 2846}, {32, 453564, 3739}, {42, 890492, 6026}, {19, 278107, 2649}, {64, 792101, 2208}, {98, 577463, 526}, {41, 572006, 748}, {99, 478120, 895}, {52, 224338, 423}, {83, 532978, 600}, {67, 92281, 486}, {28, 829955, 925}, {22, 171381, 749}, {82, 986821, 603}, {57, 294692, 194}, {9, 730892, 973}, {69, 241093, 931}, {70, 646855, 27}, {45, 233480, 669}, {60, 369922, 965}, {27, 935011, 659}, {96, 667580, 837}, {7, 919344, 188}, {99, 584762, 131}, {5, 93173, 898}, {16, 736395, 184}, {57, 893061, 196}, {28, 352640, 924}, {87, 980414, 80}, {88, 432895, 129}, {23, 461032, 85}, {73, 645991, 268}, {5, 241036, 458}, {9, 422324, 785}, {28, 124913, 224}, {51, 815633, 765}, {59, 894120, 559}, {70, 459876, 192}, {80, 423125, 584}, {85, 824496, 142}, {18, 578975, 104}, {56, 477816, 303}, {6, 702127, 400}, {43, 35371, 850}, {3, 226423, 10}}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1744 注意int long类型转换
    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        boolean[] result = new boolean[queries.length];
        long[] prefix = new long[candiesCount.length + 1];
        for (int i = 1; i <= candiesCount.length; i++) {
            prefix[i] = prefix[i - 1] + candiesCount[i - 1];
        }
        // 你从第0天开始吃糖果。
        // 你在吃完 所有第 i - 1类糖果之前，不能吃任何一颗第 i类糖果。
        // 在吃完所有糖果之前，你必须每天 至少吃 一颗糖果。

        // queries[i] = [favoriteType_i, favoriteDay_i, dailyCap_i]

        // answer[i]为true的条件是：在每天吃 不超过 dailyCap_i颗糖果的前提下，你可以在第favoriteDay_i天吃到第favoriteType_i类糖果；否则 answer[i]为 false。

        // 输入：candiesCount = [7,4,5,3,8], queries = [[0,2,2],[4,2,4],[2,13,1000000000]]
        // 输出：[true,false,true]
        // prefix 0,7,11,16,19,27

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