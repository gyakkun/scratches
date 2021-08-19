import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


//        System.out.println(s.maxCrossSum(new int[]{12, 29, 38, 48, 57, 69, 10}, new int[]{34, 67, 8, 9, 10, 10, 20}));
        System.out.println(s.maxCrossSum(new int[]{4, 2, 7, 13, 9, 25}, new int[]{5, 0, 18, 21, 3, 6}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }


    // Microsoft O(n^2)
    // https://leetcode-cn.com/circle/discuss/OPC9WF/
    public int[] maxCrossSum(int[] m, int[] n) {
        // m,n 是长度为l的数列, 求区间[a,b] [c,d], 使得  sum(m, a,b) -sum(n,a,b) - (sum(m,c,d) - sum(n,c,d)) 最大
        // [a,b] [c,d] 没有交集
        // 返回[a,b,c,d]
        int l = m.length;
        int[] mMinusN = new int[l];
        for (int i = 0; i < l; i++) mMinusN[i] = m[i] - n[i];
        int[][] minDp = new int[l][l], maxDp = new int[l][l];
        int[][][] minDpRange = new int[l][l][2], maxDpRange = new int[l][l][2];
        // 求使得和式左侧最大的区间
        for (int i = 0; i < l; i++) {
            maxDp[i][i] = mMinusN[i];
            maxDpRange[i][i] = new int[]{i, i};
            for (int j = i + 1; j < l; j++) {
                if (maxDp[i][j - 1] + mMinusN[j] > mMinusN[j]) {
                    maxDp[i][j] = maxDp[i][j - 1] + mMinusN[j];
                    if (maxDp[i][j] > maxDp[i][j - 1]) {
                        maxDpRange[i][j] = new int[]{maxDpRange[i][j - 1][0], j};
                    } else {
                        maxDpRange[i][j] = maxDpRange[i][j - 1];
                    }
                } else {
                    maxDp[i][j] = mMinusN[j];
                    maxDpRange[i][j] = new int[]{j, j};
                }
            }
        }
        // 同理 求使得和式右侧最小的区间
        for (int i = 0; i < l; i++) {
            minDp[i][i] = mMinusN[i];
            minDpRange[i][i] = new int[]{i, i};
            for (int j = i + 1; j < l; j++) {
                if (minDp[i][j - 1] + mMinusN[j] < mMinusN[j]) {
                    minDp[i][j] = minDp[i][j - 1] + mMinusN[j];
                    if (minDp[i][j] < minDp[i][j - 1]) {
                        minDpRange[i][j] = new int[]{minDpRange[i][j - 1][0], j};
                    } else {
                        minDpRange[i][j] = minDpRange[i][j - 1];
                    }
                } else {
                    minDp[i][j] = mMinusN[j];
                    minDpRange[i][j] = new int[]{j, j};
                }
            }
        }
        int maxValue = Integer.MIN_VALUE;
        int[] result = new int[]{};
        for (int i = 0; i < l; i++) {
            if (i + 1 < l && maxDp[0][i] - minDp[i + 1][l - 1] > maxValue) {
                maxValue = maxDp[0][i] - minDp[i + 1][l - 1];
                result = new int[]{maxDpRange[0][i][0], maxDpRange[0][i][1], minDpRange[i + 1][l - 1][0], minDpRange[i + 1][l - 1][1]};
            }
            if (i + 1 < l && maxDp[i + 1][l - 1] - minDp[0][i] > maxValue) {
                maxValue = maxDp[i + 1][l - 1] - minDp[0][i];
                result = new int[]{maxDpRange[i + 1][l - 1][0], maxDpRange[i + 1][l - 1][1], minDpRange[0][i][0], minDpRange[0][i][1]};
            }
        }
        return result;
    }

    // LC991
    public int brokenCalc(int startValue, int target) {
        int result = 0;
        while (target > startValue) {
            result++;
            if (target % 2 == 1) {
                target++;
            } else {
                target /= 2;
            }
        }
        return result + startValue - target;
    }

    // LC1144
    public int movesToMakeZigzag(int[] nums) {
        if (nums.length <= 2) return 0;
        // 一开始下降
        int result1 = 0;
        for (int i = 1; i < nums.length; i += 2) {
            if (i - 1 >= 0 && i + 1 < nums.length) {
                result1 += Math.max(nums[i] - Math.min(nums[i - 1], nums[i + 1]) + 1, 0);
            } else if (i - 1 < 0) {
                result1 += Math.max(nums[i] - nums[i + 1] + 1, 0);
            } else if (i + 1 >= nums.length) {
                result1 += Math.max(nums[i] - nums[i - 1] + 1, 0);
            }
        }

        // 一开始上升
        int result2 = 0;
        for (int i = 0; i < nums.length; i += 2) {
            if (i - 1 >= 0 && i + 1 < nums.length) {
                result2 += Math.max(nums[i] - Math.min(nums[i - 1], nums[i + 1]) + 1, 0);
            } else if (i - 1 < 0) {
                result2 += Math.max(nums[i] - nums[i + 1] + 1, 0);
            } else if (i + 1 >= nums.length) {
                result2 += Math.max(nums[i] - nums[i - 1] + 1, 0);
            }
        }
        return Math.min(result1, result2);
    }

    // LC873
    public int lenLongestFibSubseq(int[] arr) {
        // arr 严格递增
        int n = arr.length;
        int[] result = new int[1001];
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < n; i++) m.put(arr[i], i);
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                int len = 2, last = i, lastButOne = j, next = arr[last] - arr[lastButOne];
                while (m.containsKey(next) && m.get(next) < lastButOne) {
                    len++;
                    last = lastButOne;
                    lastButOne = m.get(next);
                    next = arr[last] - arr[lastButOne];
                }
                if (len != 2) {
                    result[len]++;
                }
            }
        }
        for (int i = 1000; i >= 0; i--) {
            if (result[i] != 0) return i;
        }
        return 0;
    }

    // JZOF II 057
    public int[][] findContinuousSequence(int target) {
        // target = (a1 + an) * n /2
        // product = (a1 +  a1 + (n-1) *d) *n
        List<List<Integer>> result = new ArrayList<>();
        Set<Integer> a1Set = new HashSet<>();
        int product = target * 2;
        int sqrt = (int) (Math.sqrt(product));
        for (int i = 1; i <= sqrt; i++) {
            if (product % i == 0) {
                // 令i为n
                int n = i;
                int k = product / n;
                jzofii057Handle(target, result, a1Set, n, k);

                // 令product/i为n
                n = product - i;
                k = product / n;
                jzofii057Handle(target, result, a1Set, n, k);
            }
        }
        result.sort(Comparator.comparingInt(o -> o.get(0)));
        int[][] resultArr = new int[result.size()][];
        for (int i = 0; i < result.size(); i++) {
            resultArr[i] = result.get(i).stream().mapToInt(Integer::valueOf).toArray();
        }
        return resultArr;
    }

    private void jzofii057Handle(int target, List<List<Integer>> result, Set<Integer> a1Set, int n, int k) {
        if ((k - n + 1) % 2 == 0) {
            int a1 = (k - n + 1) / 2;
            if (a1 > 0 && !a1Set.contains(a1) && a1 != target) {
                a1Set.add(a1);
                List<Integer> tmp = new ArrayList<>(n);
                for (int j = 0; j < n; j++) {
                    tmp.add(a1++);
                }
                result.add(tmp);
            }
        }
    }

    // LC1754 **
    public String largestMerge(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        char[] ca1 = word1.toCharArray(), ca2 = word2.toCharArray();
        int ptr1 = 0, ptr2 = 0;
        while (ptr1 != ca1.length && ptr2 != ca2.length) {
            int compare = charSeqCompare(ca1, ptr1, ca2, ptr2);
            if (compare > 0) {
                sb.append(ca1[ptr1++]);
            } else {
                sb.append(ca2[ptr2++]);
            }
        }
        while (ptr1 != ca1.length) {
            sb.append(ca1[ptr1++]);
        }
        while (ptr2 != ca2.length) {
            sb.append(ca2[ptr2++]);
        }
        return sb.toString();
    }

    // ** 字典序比较算法
    private int charSeqCompare(char[] arr1, int startIdx1, char[] arr2, int startIdx2) {
        int idx1 = startIdx1, idx2 = startIdx2;
        while (idx1 < arr1.length && idx2 < arr2.length) {
            if (arr1[idx1] == arr2[idx2]) {
                idx1++;
                idx2++;
                continue;
            }
            return arr1[idx1] - arr2[idx2];
        }
        return arr1.length - startIdx1 - (arr2.length - startIdx2);
    }

    // LC542
    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] result = new int[m][n], directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) {
                    q.offer(new int[]{i, j});
                }
            }
        }
        int layer = 0;
        while (!q.isEmpty()) {
            int qSize = q.size();
            layer++;
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                for (int[] d : directions) {
                    int x = p[0] + d[0], y = p[1] + d[1];
                    if (x >= 0 && x < mat.length && y >= 0 && y < mat[0].length) {
                        if (!visited[x][y]) {
                            if (mat[x][y] == 1 && result[x][y] == 0) {
                                result[x][y] = layer;
                            }
                            q.offer(new int[]{x, y});
                        }
                    }
                }
            }
        }
        return result;
    }

    // LCP 22 **
    public int paintingPlan(int n, int k) {
        if (k == 0) return 1;
        if (k < n) return 0;
        if (k == n * n) return 1;
        // 排列数递推公式
        int[][] C = new int[n + 1][n + 1];
        int result = 0;
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) C[i][j] = 1;
                else C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
            }
        }
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                if (i * n + j * n - i * j == k) {
                    result += C[n][i] * C[n][j];
                }
            }
        }
        return result;
    }

    // LC1119
    public String removeVowels(String s) {
        Set<Character> vowelSet = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u'));
        char[] ca = s.toCharArray();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ca.length; i++) if (!vowelSet.contains(ca[i])) sb.append(ca[i]);
        return sb.toString();
    }

    // LC345
    public String reverseVowels(String s) {
        Set<Character> vowelSet = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
        char[] ca = s.toCharArray();
        List<Character> l = new ArrayList<>();
        for (int i = 0; i < ca.length; i++) {
            if (vowelSet.contains(ca[i])) l.add(ca[i]);
        }
        int ptr = 0;
        for (int i = ca.length - 1; i >= 0; i--) {
            if (vowelSet.contains(ca[i])) ca[i] = l.get(ptr++);
        }
        return new String(ca);
    }

    // LCP 03
    public boolean robot(String command, int[][] obstacles, int x, int y) {
        Set<Pair<Integer, Integer>> trackSetInOneCycle = new HashSet<>();
        char[] ca = command.toCharArray();
        int xCount = 0, yCount = 0;
        trackSetInOneCycle.add(new Pair<>(0, 0));
        for (char c : ca) {
            if (c == 'U') yCount++;
            else xCount++;
            trackSetInOneCycle.add(new Pair<>(xCount, yCount));
        }
        int cycles = Math.min(x / xCount, y / yCount);
        if (!trackSetInOneCycle.contains(new Pair<>(x - cycles * xCount, y - cycles * yCount))) return false;

        for (int[] o : obstacles) {
            if (o[0] > x || o[1] > y) continue;
            int oCycles = Math.min(o[0] / xCount, o[1] / yCount);
            if (trackSetInOneCycle.contains(new Pair<>(o[0] - oCycles * xCount, o[1] - oCycles * yCount))) return false;
        }
        return true;
    }

    // LC1742
    public int countBalls(int lowLimit, int highLimit) {
        int[] count = new int[46];
        for (int i = lowLimit; i <= highLimit; i++) {
            count[getDigitSum(i)]++;
        }

        return Arrays.stream(count).max().getAsInt();
    }

    private int getDigitSum(int num) {
        int result = 0;
        while (num != 0) {
            result += (num % 10);
            num /= 10;
        }
        return result;
    }

    // JZOF II 058 LC729
    class MyCalendar {

        TreeSet<int[]> left;
        TreeSet<int[]> right;

        public MyCalendar() {
            left = new TreeSet<>((o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o1[0] - o2[0]);
            right = new TreeSet<>((o1, o2) -> o1[1] == o2[1] ? o1[0] - o2[0] : o1[1] - o2[1]);
        }

        // 前闭后开
        public boolean book(int start, int end) {
            int[] lQuery = new int[]{start, start};
            int[] rQuery = new int[]{end, end};
            int[] lsf = left.floor(lQuery), lsc = left.ceiling(lQuery),
                    rsf = right.floor(rQuery), rsc = right.ceiling(rQuery);

            if ((lsf == null || lsf[1] <= start)
                    && (lsc == null || lsc[0] >= end)
                    && (rsf == null || rsf[1] <= start)
                    && (rsc == null || rsc[0] >= end)) {
                int[] entry = new int[]{start, end};
                left.add(entry);
                right.add(entry);
                return true;
            }
            return false;
        }
    }

    // LC589
    class Lc589 {
        public List<Integer> preorder(Node root) {
            List<Integer> result = new ArrayList<>();
            if (root == null) return result;
            Deque<Node> q = new LinkedList<>();
            q.push(root);
            while (!q.isEmpty()) {
                Node p = q.pop();
                result.add(p.val);
                for (int i = p.children.size() - 1; i >= 0; i--) {
                    q.push(p.children.get(i));
                }
            }
            return result;
        }

        class Node {
            public int val;
            public List<Node> children;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, List<Node> _children) {
                val = _val;
                children = _children;
            }
        }
    }

    // Interview 16.18 ** 写得太复杂了！
    public boolean patternMatching(String pattern, String value) {
        char[] cp = pattern.toCharArray();
        int aCount = 0, bCount = 0, valueLen = value.length();
        int firstAIdx = pattern.indexOf('a'), firstBIdx = pattern.indexOf('b');
        for (char c : cp) {
            if (c == 'a') aCount++;
            else bCount++;
        }
        if (aCount != 0 && bCount != 0 && value.equals("")) return false;
        else if (value.equals("")) return true;
        if (bCount == 0) {
            int aLen = valueLen / aCount;
            if (aLen * aCount != valueLen) return false;
            String ap = value.substring(0, aLen);
            int ptrOnV = -aLen;
            for (int i = 0; i < aCount; i++) {
                ptrOnV = value.indexOf(ap, ptrOnV + aLen);
                if (ptrOnV == -1) return false;
            }
            return true;
        } else if (aCount == 0) {
            int bLen = valueLen / bCount;
            if (bLen * bCount != valueLen) return false;
            String ap = value.substring(0, bLen);
            int ptrOnV = -bLen;
            for (int i = 0; i < bCount; i++) {
                ptrOnV = value.indexOf(ap, ptrOnV + bLen);
                if (ptrOnV == -1) return false;
            }
            return true;
        } else {
            for (int aLen = 0; aLen <= (valueLen / aCount); aLen++) {
                int bLen = (valueLen - aLen * aCount) / bCount;
                if (bLen * bCount + aLen * aCount != valueLen) continue;
                String ap = "", bp = "";
                if (aLen == 0) {
                    bp = value.substring(0, bLen);
                    int ptr = 0, count = 0;
                    while (ptr >= 0) {
                        ptr = value.indexOf(bp, ptr);
                        if (ptr >= 0) {
                            count++;
                            ptr += bLen;
                        }
                    }
                    if (count == bCount) {
                        return true;
                    }
                    continue;
                }
                if (bLen == 0) {
                    ap = value.substring(0, aLen);
                    int ptr = 0, count = 0;
                    while (ptr >= 0) {
                        ptr = value.indexOf(ap, ptr);
                        if (ptr >= 0) {
                            count++;
                            ptr += aLen;
                        }
                    }
                    if (count == aCount) {
                        return true;
                    }
                    continue;
                }
                ap = value.substring(firstAIdx * bLen, firstAIdx * bLen + aLen);
                bp = value.substring(firstBIdx * aLen, firstBIdx * aLen + bLen);
                int ptrOnP = 0, ptrOnV = 0;
                boolean success = true;
                for (; ptrOnP < cp.length; ptrOnP++) {
                    if (cp[ptrOnP] == 'a') {
                        ptrOnV = value.indexOf(ap, ptrOnV);
                        if (ptrOnV >= 0) {
                            ptrOnV += aLen;
                        }
                    } else {
                        ptrOnV = value.indexOf(bp, ptrOnV);
                        if (ptrOnV >= 0) {
                            ptrOnV += bLen;
                        }
                    }
                    if (ptrOnV < 0) {
                        success = false;
                        break;
                    }
                }
                if (!success) continue;
                return true;
            }
        }

        return false;
    }

    // LC1100
    public int numKLenSubstrNoRepeats(String s, int k) {
        int result = 0, n = s.length();
        if (k > n) return 0;
        int[] freq = new int[26];
        char[] ca = s.toCharArray();
        for (int i = 0; i < k; i++) {
            freq[ca[i] - 'a']++;
        }
        if (lc1100Check(freq)) result++;
        for (int i = k; i < n; i++) {
            freq[ca[i - k] - 'a']--;
            freq[ca[i] - 'a']++;
            if (lc1100Check(freq)) result++;
        }
        return result;
    }

    private boolean lc1100Check(int[] freq) {
        for (int i : freq) if (i > 1) return false;
        return true;
    }

    // LC117 ** O(1) 空间
    class Lc117 {
        public Node connect(Node root) {
            if (root == null) return null;
            Node start = root;
            while (start != null) { // 思想: 在本层连接下一层的next
                Node nextStart = null, last = null; // 下一层的开始节点, 遍历到的下一层的最后一个节点
                for (Node ptr = start; ptr != null; ptr = ptr.next) {
                    if (ptr.left != null) {
                        if (last != null) {
                            last.next = ptr.left;
                        }
                        if (nextStart == null) {
                            nextStart = ptr.left;
                        }
                        last = ptr.left;
                    }
                    if (ptr.right != null) {
                        if (last != null) {
                            last.next = ptr.right;
                        }
                        if (nextStart == null) {
                            nextStart = ptr.right;
                        }
                        last = ptr.right;
                    }
                }
                start = nextStart;
            }
            return root;
        }

        class Node {
            public int val;
            public Node left;
            public Node right;
            public Node next;

            public Node() {
            }

            public Node(int _val) {
                val = _val;
            }

            public Node(int _val, Node _left, Node _right, Node _next) {
                val = _val;
                left = _left;
                right = _right;
                next = _next;
            }
        }

    }

    // LC1737
    public int minCharacters(String a, String b) {
        // ONE TWO
        int result = Integer.MAX_VALUE / 2;
        int[] freqA = new int[26], freqB = new int[26], prefixA = new int[27], prefixB = new int[27];
        char[] ca = a.toCharArray(), cb = b.toCharArray();
        for (char c : ca) freqA[c - 'a']++;
        for (char c : cb) freqB[c - 'a']++;
        for (int i = 1; i <= 26; i++) prefixA[i] = freqA[i - 1] + prefixA[i - 1];
        for (int i = 1; i <= 26; i++) prefixB[i] = freqB[i - 1] + prefixB[i - 1];
        // 令A严格小于B, 使用一个指针指示B中最小的字母, 指针应该在b...z上移动, 当B最小的是a时, 首先考虑将B的所有a变为b
        for (int ptr = 'b'; ptr <= 'z'; ptr++) {
            char target = (char) ptr;
            int tmpMove = 0;
            // 将B中小于ptr的变为ptr
            tmpMove += prefixB[target - 'a'] - prefixB[0];
            // 将A中大于ptr的变为ptr-1
            tmpMove += prefixA[26] - prefixA[target - 'a'];
            result = Math.min(result, tmpMove);
        }
        // 令B严格小于A
        for (int ptr = 'b'; ptr <= 'z'; ptr++) {
            char target = (char) ptr;
            int tmpMove = 0;
            tmpMove += prefixA[target - 'a'] - prefixA[0];
            tmpMove += prefixB[26] - prefixB[target - 'a'];
            result = Math.min(result, tmpMove);
        }

        // THREE
        for (int i = 0; i < 26; i++) {
            int tmpMove = prefixA[26] + prefixB[26] - freqA[i] - freqB[i];
            result = Math.min(result, tmpMove);
        }

        return result;
    }

    // JZOF II 113
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> outEdge = new ArrayList<>(numCourses);
        List<Integer> result = new ArrayList<>(numCourses);
        Deque<Integer> q = new LinkedList<>();
        int[] indegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) outEdge.add(new ArrayList<>());
        for (int[] p : prerequisites) {
            outEdge.get(p[1]).add(p[0]);
            indegree[p[0]]++;
        }
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) q.offer(i);
        }
        while (!q.isEmpty()) {
            int p = q.poll();
            result.add(p);
            for (int next : outEdge.get(p)) {
                indegree[next]--;
                if (indegree[next] == 0) q.offer(next);
            }
        }
        if (result.size() != numCourses) return new int[]{};
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // Interview 17.01
    public int add(int a, int b) {
        int sum = a;
        while (b != 0) {
            int and = a & b;
            int xor = a ^ b;
            b = and << 1;
            sum = xor;
            a = xor;
        }
        return sum;
    }

    // Interview 01.08
    public void setZeroes(int[][] matrix) {
        boolean[] rowMark = new boolean[matrix.length], colMark = new boolean[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    rowMark[i] = true;
                    colMark[j] = true;
                }
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            if (rowMark[i]) {
                for (int j = 0; j < matrix[0].length; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < matrix[0].length; j++) {
            if (colMark[j]) {
                for (int i = 0; i < matrix.length; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    // LC1338
    public int minSetSize(int[] arr) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i : arr) freq.put(i, freq.getOrDefault(i, 0) + 1);
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -freq.getOrDefault(o, 0)));
        for (int i : freq.keySet()) {
            pq.offer(i);
        }
        int left = arr.length, half = arr.length / 2, result = 0;
        while (left > half) {
            left -= freq.get(pq.poll());
            result++;
        }
        return result;
    }

    // Interview 05.01
    public int insertBits(int N, int M, int i, int j) {
        int mask = 0;
        for (int k = i; k <= j; k++) {
            mask |= 1 << k;
        }
        mask = ~mask;
        N &= mask;
        M <<= i;
        M &= ~mask;
        N |= M;
        return N;
    }

    // LC1129 ** 可能有环 有平行边
    public int[] shortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges) {
        final int RED = 0, BLUE = 1;
        int[] result = new int[n];
        Arrays.fill(result, -1);
        result[0] = 0;
        List<List<Integer>> redOutEdge = new ArrayList<>(n), blueOutEdge = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            redOutEdge.add(new ArrayList<>());
            blueOutEdge.add(new ArrayList<>());
        }
        boolean[] redVisit = new boolean[n], blueVisit = new boolean[n];
        for (int[] re : red_edges) {
            redOutEdge.get(re[0]).add(re[1]);
        }
        for (int[] be : blue_edges) {
            blueOutEdge.get(be[0]).add(be[1]);
        }
        if (redOutEdge.get(0) == null && blueOutEdge.get(0) == null) return result;

        Deque<int[]> q = new LinkedList<>();
        for (int next : redOutEdge.get(0)) {
            q.offer(new int[]{next, RED});
        }
        for (int next : blueOutEdge.get(0)) {
            q.offer(new int[]{next, BLUE});
        }
        int layer = 0;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                // 注意剪枝时机!!!
                if (blueVisit[p[0]] && redVisit[p[0]]) continue;
                if (p[1] == RED && redVisit[p[0]]) continue;
                if (p[1] == BLUE && blueVisit[p[0]]) continue;
                if (result[p[0]] == -1) {
                    result[p[0]] = layer;
                } else {
                    result[p[0]] = Math.min(result[p[0]], layer);
                }
                if (p[1] == RED) {
                    redVisit[p[0]] = true;
                    for (int next : blueOutEdge.get(p[0])) {
                        q.offer(new int[]{next, BLUE});
                    }
                } else {
                    blueVisit[p[0]] = true;
                    for (int next : redOutEdge.get(p[0])) {
                        q.offer(new int[]{next, RED});
                    }
                }
            }
        }
        return result;
    }

    // LC1387
    Map<Integer, Integer> lc1387Memo;

    public int getKth(int lo, int hi, int k) {
        lc1387Memo = new HashMap<>();
        List<Integer> arr = new ArrayList<>(hi - lo + 1);
        for (int i = lo; i <= hi; i++) {
            arr.add(i);
        }
        arr.sort((o1, o2) -> lc1387Weight(o1) == lc1387Weight(o2) ? o1 - o2 : lc1387Weight(o1) - lc1387Weight(o2));
        return arr.get(k - 1);
    }

    private int lc1387Weight(int num) {
        if (num == 1) return 0;
        if (lc1387Memo.containsKey(num)) return lc1387Memo.get(num);
        int result = num % 2 == 1 ? lc1387Weight(3 * num + 1) + 1 : lc1387Weight(num / 2) + 1;
        lc1387Memo.put(num, result);
        return result;
    }


    // LC1513
    public int numSub(String s) {
        long oneCount = 0, mod = 1000000007;
        long result = 0;
        for (char c : s.toCharArray()) {
            if (c == '1') {
                oneCount++;
            } else {
                result += oneCount * (oneCount + 1) / 2;
                result %= mod;
                oneCount = 0;
            }
        }
        if (oneCount != 0) {
            result += oneCount * (oneCount + 1) / 2;
            result %= mod;
        }
        return (int) result;
    }

    // LC1709
    public int[] largestSubarrayBetter(int[] nums, int k) {
        int n = nums.length;
        int maxInt = 0, ptr = n - k, maxStartPoint = 0;
        while (ptr >= 0) {
            if (nums[ptr] > maxInt) {
                maxStartPoint = ptr;
                maxInt = nums[ptr];
            }
            ptr--;
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1709 没有利用数字不重复的条件
    public int[] largestSubarray(int[] nums, int k) {
        int maxStartPoint = 0, n = nums.length;
        for (int i = 1; i < n - k + 1; i++) {
            for (int j = 0; j < k; j++) {
                if (nums[maxStartPoint + j] < nums[i + j]) {
                    maxStartPoint = i;
                } else if (nums[maxStartPoint + j] == nums[i + j]) {
                    continue;
                } else {
                    break;
                }
            }
        }
        return Arrays.copyOfRange(nums, maxStartPoint, maxStartPoint + k);
    }

    // LC1039 ** 几何
    Integer[][] lc1039Memo;

    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        lc1039Memo = new Integer[n + 1][n + 1];
        return lc1039Helper(values, 0, n - 1);
    }

    private int lc1039Helper(int[] values, int start, int end) {
        if (start + 1 == end) return 0;
        if (lc1039Memo[start][end] != null) return lc1039Memo[start][end];
        int result = Integer.MAX_VALUE;
        for (int i = start + 1; i < end; i++) {
            result = Math.min(result, lc1039Helper(values, start, i) + lc1039Helper(values, i, end) + values[start] * values[end] * values[i]);
        }
        return lc1039Memo[start][end] = result;
    }

    // LC128
    // https://bbs.byr.cn/n/article/Talking/6295267
    public int longestConsecutive(int[] nums) {
        Set<Integer> s = new HashSet<>();
        int result = 0;
        for (int e : nums) {
            s.add(e);
        }
        for (int i : s) {
            if (!s.contains(i - 1)) {
                int l = 1;
                while (s.contains(i + 1)) {
                    l++;
                    i++;
                }
                result = Math.max(result, l);
            }
        }
        return result;
    }


    // LC1775
    public int minOperations(int[] nums1, int[] nums2) {
        if (nums1.length * 6 < nums2.length || nums2.length * 6 < nums1.length) return -1;
        int origSum1 = Arrays.stream(nums1).sum(), origSum2 = Arrays.stream(nums2).sum();
        if (origSum1 == origSum2) return 0;
        int[] inc = new int[6], dec = new int[6];
        for (int i : nums1) {
            inc[6 - i]++;
            dec[i - 1]++;
        }
        for (int i : nums2) {
            dec[6 - i]++;
            inc[i - 1]++;
        }
        inc[0] = dec[0] = 0;
        int result = 0;
        int delta = origSum1 - origSum2;
        if (delta > 0) { // nums1 should decrease
            for (int i = 5; i >= 1; i--) {
                while (dec[i] > 0) {
                    result++;
                    dec[i]--;
                    delta -= i;
                    if (delta <= 0) return result;
                }
            }
        } else {
            for (int i = 5; i >= 1; i--) {
                while (inc[i] > 0) {
                    result++;
                    inc[i]--;
                    delta += i;
                    if (delta >= 0) return result;
                }
            }
        }
        return -1;
    }

    // LC552 HARD
    public int checkRecord(int n) {
        final int mod = 1000000007;
        long[] dp = new long[Math.max(4, n + 1)];
        dp[0] = 1;
        dp[1] = 2;
        dp[2] = 4;
        dp[3] = 7;
        for (int i = 4; i <= n; i++) {
            dp[i] = (2 * dp[i - 1] - dp[i - 4] + mod) % mod;
        }
        long result = dp[n];
        for (int i = 1; i <= n; i++) {
            result += (dp[i - 1] * dp[n - i]) % mod;
        }
        return (int) (result % mod);
    }

    // LC551
    public boolean checkRecord(String s) {
        int lCount = 0, aCount = 0;
        for (char c : s.toCharArray()) {
            if (c == 'L') {
                lCount++;
            } else {
                lCount = 0;
            }
            if (lCount >= 3) return false;
            if (c == 'A') {
                aCount++;
            }
            if (aCount >= 2) return false;
        }
        return true;
    }

    // LC1389
    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (index[i] >= result.size()) {
                int targetSize = (index[i] + 1) - result.size();
                for (int j = 0; j < targetSize; j++) {
                    result.add(-1);
                }
                result.set(index[i], nums[i]);
            } else {
                result.add(index[i], nums[i]);
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC1390
    public int sumFourDivisors(int[] nums) {
        int result = 0;
        for (int n : nums) result += lc1390Helper(n);
        return result;
    }

    private int lc1390Helper(int n) {
        if (n <= 5) return 0;
        int sqrt = (int) Math.sqrt(n);
        Set<Integer> s = new HashSet<>();
        for (int i = 1; i <= sqrt; i++) {
            if (n % i == 0) {
                s.add(i);
                s.add(n / i);
            }
            if (s.size() > 4) return 0;
        }
        if (s.size() != 4) return 0;
        return s.stream().reduce((a, b) -> a + b).get();
    }

    // JZOF 06
    public int[] reversePrint(ListNode head) {
        // 倒置链表
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode prev = null, cur = head;
        int count = 0;
        while (cur != null) {
            count++;
            ListNode origNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = origNext;
        }
        int[] result = new int[count];
        count = 0;
        cur = prev;
        while (cur != null) {
            result[count++] = cur.val;
            cur = cur.next;
        }
        return result;
    }

    // JZOF II 090
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[1], nums[0]);
        // ROB ZERO, then n-1 can't be robbed
        dp[0] = dp[1] = nums[0];
        for (int i = 2; i < n - 1; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int noZero = dp[n - 2];
        Arrays.fill(dp, 0);
        // Not rob zero, then n-1 can be robbed
        dp[1] = nums[1];
        dp[2] = Math.max(nums[1], nums[2]);
        for (int i = 3; i < n; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int zero = dp[n - 1];
        return Math.max(zero, noZero);
    }

    public TreeNode LCA(TreeNode root, TreeNode p, TreeNode q) {
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        parent.put(root, null);
        Deque<TreeNode> queue = new LinkedList<>();
        Set<TreeNode> visited = new HashSet<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode poll = queue.poll();
            if (poll.left != null) {
                parent.put(poll.left, poll);
                queue.offer(poll.left);
            }
            if (poll.right != null) {
                parent.put(poll.right, poll);
                queue.offer(poll.right);
            }
        }
        while (p != null) {
            visited.add(p);
            p = parent.get(p);
        }
        while (q != null) {
            if (visited.contains(q)) return q;
            q = parent.get(q);
        }
        return null;
    }

    // JZOF 68
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode result = root;
        while (true) {
            if (p.val > result.val && q.val > result.val) {
                result = result.right;
            } else if (p.val < result.val && q.val < result.val) {
                result = result.left;
            } else {
                break;
            }
        }
        return result;
    }

    // LC1338
    Map<TreeNode, Long> nodeSumMap = new HashMap<>();

    public int maxProduct(TreeNode root) {
        final int mod = 1000000007;
        lc1338Helper(root);
        long result = 0;
        long total = nodeSumMap.get(root);
        for (TreeNode node : nodeSumMap.keySet()) {
            result = Math.max(result, (total - nodeSumMap.get(node)) * nodeSumMap.get(node));
        }
        return (int) (result % mod);
    }

    private long lc1338Helper(TreeNode root) {
        if (root == null) return 0;
        long result = root.val + lc1338Helper(root.left) + lc1338Helper(root.right);
        nodeSumMap.put(root, result);
        return result;
    }

    // LC733
    int[][] lc733Directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int origColor = image[sr][sc];
        boolean[][] visited = new boolean[image.length][image[0].length];
        lc733Helper(image, sr, sc, origColor, newColor, visited);
        return image;
    }

    private void lc733Helper(int[][] image, int x, int y, int origColor, int newColor, boolean[][] visited) {
        if (x < 0 || x >= image.length || y < 0 || y >= image[0].length || visited[x][y]) {
            return;
        }
        if (image[x][y] == origColor) {
            image[x][y] = newColor;
            visited[x][y] = true;
            for (int[] dir : lc733Directions) {
                lc733Helper(image, x + dir[0], y + dir[1], origColor, newColor, visited);
            }
        }
    }

    // JZOF 64 **
    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    // LC1282
    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        List<List<Integer>> result = new ArrayList<>();
        Map<Integer, List<Integer>> sizeCountMap = new HashMap<>();
        for (int i = 0; i < groupSizes.length; i++) {
            sizeCountMap.putIfAbsent(groupSizes[i], new ArrayList<>());
            sizeCountMap.get(groupSizes[i]).add(i);
        }
        for (int gs : sizeCountMap.keySet()) {
            List<Integer> users = sizeCountMap.get(gs);
            int cur = 0;
            while (cur != users.size()) {
                result.add(users.subList(cur, cur + gs));
                cur += gs;
            }
        }
        return result;
    }

    // LC1684
    public int countConsistentStrings(String allowed, String[] words) {
        int result = 0, mask = 0;
        for (char c : allowed.toCharArray()) mask |= 1 << (c - 'a');
        for (String w : words) {
            int wm = 0;
            for (char c : w.toCharArray()) wm |= 1 << (c - 'a');
            if ((wm & mask) == wm) result++;
        }
        return result;
    }

    // LC526 **
    boolean[] lc526Visited;
    int lc526Result;

    public int countArrangement(int n) {
        lc526Visited = new boolean[n + 1];
        lc526Backtrack(1, n);
        return lc526Result;
    }

    public void lc526Backtrack(int index, int n) {
        if (index == n + 1) {
            lc526Result++;
            return;
        }
        for (int i = 1; i <= n; i++) {
            if (!lc526Visited[i] && (i % index == 0 || index % i == 0)) {
                lc526Visited[i] = true;
                lc526Backtrack(index + 1, n);
                lc526Visited[i] = false;
            }
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

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

// LC622
class MyCircularQueue {
    int[] arr;
    int headIdx, count;

    public MyCircularQueue(int k) {
        arr = new int[k];
        headIdx = 0;
        count = 0;
    }

    public boolean enQueue(int value) {
        if (isFull()) return false;
        arr[(headIdx + count) % arr.length] = value;
        count++;
        return true;
    }

    public boolean deQueue() {
        if (isEmpty()) return false;
        arr[headIdx] = 0;
        headIdx = (headIdx + 1) % arr.length;
        count--;
        return true;
    }

    public int Front() {
        if (isEmpty()) return -1;
        return arr[headIdx];
    }

    public int Rear() {
        if (isEmpty()) return -1;
        return arr[(headIdx + count - 1 + arr.length) % arr.length];
    }

    public boolean isEmpty() {
        return count == 0;
    }

    public boolean isFull() {
        return count == arr.length;
    }
}
