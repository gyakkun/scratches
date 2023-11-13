package moe.nyamori.test.historical;


import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;

class scratch_49 {
    public static void main(String[] args) {
        scratch_49 s = new scratch_49();
        long timing = System.currentTimeMillis();

        int[] a = {5, 4, 3, 2, 1};

        quickSort.sort(a);

        System.out.println(a);


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1583
    public int unhappyFriends(int n, int[][] preferences, int[][] pairs) {
        int[][] m = new int[n][n];
        int[] result = new int[n], match = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - 1; j++) {
                m[i][preferences[i][j]] = j; // order 越小 越亲密
            }
        }
        for (int[] p : pairs) {
            match[p[0]] = p[1];
            match[p[1]] = p[0];
        }
        for (int x = 0; x < n; x++) {
            int y = match[x];
            int xy = m[x][y];
            for (int i = 0; i < xy; i++) {
                int u = preferences[x][i];
                int v = match[u];
                if (m[u][x] < m[u][v]) {
                    result[x] = 1;
                    break;
                }
            }
        }
        return Arrays.stream(result).sum();
    }

    // LC970
    public List<Integer> powerfulIntegers(int x, int y, int bound) {
        List<Integer> xp = new ArrayList<>(), yp = new ArrayList<>();
        int pox = 1, poy = 1;
        if (x == 1) {
            xp.add(1);
        } else {
            while (pox <= bound) {
                xp.add(pox);
                pox *= x;
            }
        }
        if (y == 1) {
            yp.add(1);
        } else {
            while (poy <= bound) {
                yp.add(poy);
                poy *= y;
            }
        }
        Set<Integer> result = new HashSet<>();
        for (int i = 0; i < xp.size(); i++) {
            for (int j = 0; j < yp.size(); j++) {
                int tmp = xp.get(i) + yp.get(j);
                if (tmp <= bound) {
                    result.add(tmp);
                }
            }
        }
        return new ArrayList<>(result);
    }

    // LC1062
    public int longestRepeatingSubstring(String s) {
        final int base = 31, mod = 1000000007;
        int n = s.length();
        char[] ca = s.toCharArray();
        for (int len = n - 1; len >= 1; len--) {
            Map<Long, Integer> m = new HashMap<>();
            // calculate init hash
            long hash = 0, mul = 1;
            for (int i = 0; i < len; i++) {
                hash = (hash * base + (ca[i] - 'a')) % mod;
                mul = (mul * base) % mod;
            }
            m.put(hash, 1);
            for (int i = len; i < n; i++) {
                hash = (hash * base - (ca[i - len] - 'a') * mul + (ca[i] - 'a')) % mod;
                m.put(hash, m.getOrDefault(hash, 0) + 1);
            }
            int count = 1;
            for (long k : m.keySet()) {
                count = Math.max(count, m.get(k));
            }
            if (count != 1) return len;
        }
        return 0;
    }

    // LC110
    public boolean isBalancedLC(TreeNode49
                                        root) {
        Function<TreeNode49
                , Integer> height = new Function<TreeNode49
                , Integer>() {
            @Override
            public Integer apply(TreeNode49
                                         root) {
                if (root == null) return 0;
                return 1 + Math.max(this.apply(root.left), this.apply(root.right));
            }
        };
        return Math.abs(height.apply(root.left) - height.apply(root.right)) <= 1
                && isBalanced(root.right) && isBalanced(root.left);
    }

    // LC1153 **
    public boolean canConvert(String str1, String str2) {
        if (str1.equals(str2)) return true;
        char[] c1 = str1.toCharArray(), c2 = str2.toCharArray();
        int[] freq2 = new int[26];
        for (char c : c2) freq2[c - 'a'] = 1;
        if (Arrays.stream(freq2).sum() == 26) return false;
        Map<Character, Character> s1s2CharMap = new HashMap<>();
        // 确保同s1中的同一个字母对应s2中的同一个字母
        for (int i = 0; i < c1.length; i++) {
            if (s1s2CharMap.containsKey(c1[i]) && c2[i] != s1s2CharMap.get(c1[i])) return false;
            s1s2CharMap.putIfAbsent(c1[i], c2[i]);
        }
        return true;
    }

    // LC1753
    public int maximumScoreMath(int a, int b, int c) {
        int[] arr = new int[]{a, b, c};
        Arrays.sort(arr);
        if (arr[0] + arr[1] <= arr[2]) return arr[0] + arr[1];
        return (a + b + c) / 2;
    }

    // LC1753 Simulation
    public int maximumScore(int a, int b, int c) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.reverseOrder());
        pq.offer(a);
        pq.offer(b);
        pq.offer(c);
        boolean end = false;
        int result = 0;
        while (!end) {
            int max = pq.poll();
            int mid = pq.poll();
            if (max == 0 || mid == 0) {
                end = true;
                break;
            }
            max--;
            mid--;
            result++;
            pq.offer(max);
            pq.offer(mid);
        }
        return result;
    }

    // LC516
    Integer[][] lc516Memo;

    public int longestPalindromeSubseq(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        lc516Memo = new Integer[n + 1][n + 1];
        return lc516Helper(0, n - 1, ca);
    }

    private int lc516Helper(int start, int end, char[] ca) {
        if (start > end) return 0;
        if (start == end) return 1;
        if (lc516Memo[start][end] != null) return lc516Memo[start][end];
        int result = 0;
        if (ca[start] == ca[end]) { // 如果两端相同 则最长长度必然在两端+内侧最长
            result = lc516Helper(start + 1, end - 1, ca) + 2;
        } else {
            result = Math.max(result, lc516Helper(start, end - 1, ca));
            result = Math.max(result, lc516Helper(start + 1, end, ca));
        }
        return lc516Memo[start][end] = result;
    }

    // LC470
    public int rand10() {
        int i = rand7() + (rand7() - 1) * 7;
        return i > 40 ? rand10() : 1 + (i - 1) % 10;
    }

    private int rand7() {
        return (int) (Math.random() * 7) + 1;
    }

    // LC887
    Map<Integer, Map<Integer, Integer>> lc887Memo;

    public int superEggDrop(int eggs, int floors) {
        lc887Memo = new HashMap<>();
        for (int i = 1; i <= floors; i++) if (lc887MaxFloor(i, eggs) >= floors) return i;
        return floors;
    }

    // return the maximum floors with the given num of oper and num of egg
    private int lc887MaxFloor(int oper, int egg) {
        if (oper == 1) return 1;
        if (egg == 1) return oper;
        if (lc887Memo.containsKey(egg) && lc887Memo.get(egg).containsKey(oper)) return lc887Memo.get(egg).get(oper);
        lc887Memo.putIfAbsent(egg, new HashMap<>());
        int result = 1 + lc887MaxFloor(oper - 1, egg - 1) + lc887MaxFloor(oper - 1, egg);
        lc887Memo.get(egg).put(oper, result);
        return result;
    }

    // LC927
    public int[] threeEqualParts(int[] arr) {
        int sum = Arrays.stream(arr).sum();
        if (sum % 3 != 0) return new int[]{-1, -1};
        if (sum == 0) return new int[]{0, arr.length - 1};
        int oneThird = sum / 3;
        int accumulate = 0;
        List<Integer> oneThirdLastIdx = new ArrayList<>(3);
        List<Integer> oneThirdFirstIdx = new ArrayList<>(3);
        boolean firstOneFlag = false;
        for (int i = 0; i < arr.length; i++) {
            if (!firstOneFlag && arr[i] == 1) {
                oneThirdFirstIdx.add(i);
                firstOneFlag = true;
            }
            accumulate += arr[i];
            if (accumulate == oneThird) {
                accumulate = 0;
                oneThirdLastIdx.add(i);
                firstOneFlag = false;
            }
        }
        if (oneThirdLastIdx.size() != 3 || oneThirdFirstIdx.size() != 3) return new int[]{-1, -1};
        List<Integer> lens = new ArrayList<>();
        Set<Integer> test = new HashSet<>();
        for (int i = 0; i < 3; i++) {
            lens.add(oneThirdLastIdx.get(i) - oneThirdFirstIdx.get(i) + 1);
            test.add(lens.get(i));
        }
        if (test.size() != 1) return new int[]{-1, -1};
        int len = lens.get(0), p0 = oneThirdLastIdx.get(0), p1 = oneThirdLastIdx.get(1), p2 = oneThirdLastIdx.get(2);
        for (int i = 0; i < len; i++) {
            if (arr[p0] == arr[p1] && arr[p1] == arr[p2]) {
                p0--;
                p1--;
                p2--;
                continue;
            }
            return new int[]{-1, -1};
        }
        // 如果后缀是1结尾的话
        if (oneThirdLastIdx.get(2) == arr.length - 1) {
            return new int[]{oneThirdLastIdx.get(0), oneThirdLastIdx.get(1) + 1};
        }
        // 否则
        int remainZero = arr.length - oneThirdLastIdx.get(2) - 1;
        int firstGapZeroCount = oneThirdFirstIdx.get(1) - oneThirdLastIdx.get(0) - 1;
        int secondGapZeroCount = oneThirdFirstIdx.get(2) - oneThirdLastIdx.get(1) - 1;
        if (firstGapZeroCount < remainZero || secondGapZeroCount < remainZero) return new int[]{-1, -1};
        return new int[]{oneThirdLastIdx.get(0) + remainZero, oneThirdLastIdx.get(1) + 1 + remainZero};
    }

    // LC1470
    public int[] shuffle(int[] nums, int n) {
        int fullMask = (1 << 10) - 1;
        for (int i = 0; i < 2 * n; i++) {
            int targetIdx;
            if (i < n) {
                targetIdx = 2 * i;
            } else {
                targetIdx = 2 * (i - n) + 1;
            }
            nums[targetIdx] |= (nums[i] & fullMask) << 10;
        }
        for (int i = 0; i < 2 * n; i++) {
            nums[i] >>= 10;
        }
        return nums;
    }

    // LC1347
    public int minSteps(String s, String t) {
        int[] freqS = new int[26], freqT = new int[26];
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        int n = s.length(), result = 0;
        for (int i = 0; i < n; i++) {
            freqS[cs[i] - 'a']++;
            freqT[ct[i] - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            result += Math.max(0, freqS[i] - freqT[i]);
        }
        return result;
    }

    // LC955
    public int minDeletionSize(String[] strs) {
        Function<String[], Boolean> isSorted = strArr -> {
            int n = strArr.length;
            if (n == 1) return true;
            for (int i = 1; i < n; i++) {
                if (strArr[i - 1].compareTo(strArr[i]) > 0) return false;
            }
            return true;
        };
        int numRow = strs.length;
        int wordLen = strs[0].length();
        String[] adopted = new String[numRow];
        Arrays.fill(adopted, "");
        for (int i = 0; i < wordLen; i++) {
            String[] working = Arrays.copyOf(adopted, numRow);
            for (int j = 0; j < numRow; j++) {
                working[j] += strs[j].charAt(i);
            }
            if (isSorted.apply(working)) {
                adopted = working;
            }
        }
        return wordLen - adopted[0].length();
    }

    // LC1514 BFS
    public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
        Map<Integer, List<Pair<Integer, Double>>> outEdge = new HashMap<>();
        for (int i = 0; i < edges.length; i++) {
            int[] e = edges[i];
            outEdge.putIfAbsent(e[0], new ArrayList<>());
            outEdge.putIfAbsent(e[1], new ArrayList<>());
            outEdge.get(e[0]).add(new Pair<>(e[1], succProb[i]));
            outEdge.get(e[1]).add(new Pair<>(e[0], succProb[i]));
        }
        if (!outEdge.containsKey(start)) return 0d;
        double[] prob = new double[n];
        // Arrays.fill(visited, 0d);
        prob[start] = 1d;
        PriorityQueue<Pair<Integer, Double>> pq = new PriorityQueue<>((o1, o2) -> {
            if (o1.getValue() == o2.getValue()) return o1.getKey() - o2.getKey();
            return o1.getValue() < o2.getValue() ? 1 : -1;
        });
        pq.offer(new Pair<>(start, 1d));
        while (!pq.isEmpty()) {
            Pair<Integer, Double> p = pq.poll();
            int cur = p.getKey();
            double curProb = p.getValue();
            if (prob[cur] > curProb) continue;
            prob[cur] = curProb;
            for (Pair<Integer, Double> e : outEdge.get(cur)) {
                int nextIdx = e.getKey();
                double nextProb = curProb * e.getValue();
                if (nextProb > prob[nextIdx]) {
                    prob[nextIdx] = nextProb;
                    Pair<Integer, Double> nextEle = new Pair<>(e.getKey(), nextProb);
                    pq.offer(nextEle);
                }
            }
        }
        return prob[end];
    }

    // LC609
    public List<List<String>> findDuplicate(String[] paths) {
        Map<String, List<String>> contentPathMap = new HashMap<>();
        List<List<String>> result = new ArrayList<>();
        for (String p : paths) {
            String[] arr = p.split(" ");
            String directory = arr[0];
            int n = arr.length;
            for (int i = 1; i < n; i++) {
                int leftParIdx = arr[i].indexOf('(');
                int rightParIdx = arr[i].indexOf(')');
                String fileName = arr[i].substring(0, leftParIdx);
                String content = arr[i].substring(leftParIdx + 1, rightParIdx);
                contentPathMap.putIfAbsent(content, new ArrayList<>());
                contentPathMap.get(content).add(directory + "/" + fileName);
            }
        }
        for (String content : contentPathMap.keySet()) {
            if (contentPathMap.get(content).size() > 1) {
                result.add(contentPathMap.get(content));
            }
        }
        return result;
    }

    // JZOF II 038
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] nge = new int[n];
//        Arrays.fill(nge, 0);
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prev = stack.pop();
                nge[prev] = i - prev;
            }
            stack.push(i);
        }
        return nge;
    }

    // LC963
    public double minAreaFreeRect(int[][] points) {
        Set<Pair<Integer, Integer>> pointSet = new HashSet<>();
        for (int[] p : points) {
            pointSet.add(new Pair<>(p[0], p[1]));
        }
        int n = points.length;
        double result = Integer.MAX_VALUE;
        double max = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            int[] a = points[i];
            for (int j = i + 1; j < n; j++) {
                int[] b = points[j];
                for (int k = j + 1; k < n; k++) {
                    int[] c = points[k];
                    Map<int[], Pair<int[], int[]>> vecPointPairMap = new HashMap<>();
                    int[] vecAB = new int[]{a[0] - b[0], a[1] - b[1]};
                    vecPointPairMap.put(vecAB, new Pair<>(a, b));
                    int[] vecBA = new int[]{b[0] - a[0], b[1] - a[1]};
                    vecPointPairMap.put(vecBA, new Pair<>(b, a));
                    int[] vecBC = new int[]{b[0] - c[0], b[1] - c[1]};
                    vecPointPairMap.put(vecBC, new Pair<>(b, c));
                    int[] vecCB = new int[]{c[0] - b[0], c[1] - b[1]};
                    vecPointPairMap.put(vecCB, new Pair<>(c, b));
                    int[] vecAC = new int[]{a[0] - c[0], a[1] - c[1]};
                    vecPointPairMap.put(vecAC, new Pair<>(a, c));
                    int[] vecCA = new int[]{c[0] - a[0], c[1] - a[1]};
                    vecPointPairMap.put(vecCA, new Pair<>(c, a));
                    int[][] allVec = new int[][]{vecAB, vecBC, vecAC, vecBA, vecCB, vecCA};
                    for (int p = 0; p < allVec.length; p++) {
                        int[] vecP = allVec[p];
                        for (int q = p + 1; q < allVec.length; q++) {
                            int[] vecQ = allVec[q];

                            int mul = vecP[0] * vecQ[0] + vecP[1] * vecQ[1];
                            if (mul != 0) continue;

                            Pair<int[], int[]> pointPairP = vecPointPairMap.get(vecP);
                            Pair<int[], int[]> pointPairQ = vecPointPairMap.get(vecQ);
                            int[][] tmpPa = new int[][]{pointPairP.getKey(), pointPairP.getValue(), pointPairQ.getKey(), pointPairQ.getValue()};
                            Set<int[]> tmpPs = new HashSet<>();
                            int[] dup = new int[0];
                            for (int[] mayDup : tmpPa) {
                                if (!tmpPs.add(mayDup)) {
                                    dup = mayDup;
                                }
                            }
                            List<int[]> vecToCompose = new ArrayList<>();
                            for (int[] point : tmpPs) {
                                if (point != dup) {
                                    vecToCompose.add(new int[]{point[0] - dup[0], point[1] - dup[1]});
                                }
                            }
                            int[] composedVec = new int[]{vecToCompose.get(0)[0] + vecToCompose.get(1)[0], vecToCompose.get(0)[1] + vecToCompose.get(1)[1]};
                            Pair<Integer, Integer> targetPoint = new Pair<>(dup[0] + composedVec[0], dup[1] + composedVec[1]);
                            if (!pointSet.contains(targetPoint)) continue;
                            double width = Math.sqrt(vecToCompose.get(0)[0] * vecToCompose.get(0)[0] + vecToCompose.get(0)[1] * vecToCompose.get(0)[1]);
                            double height = Math.sqrt(vecToCompose.get(1)[0] * vecToCompose.get(1)[0] + vecToCompose.get(1)[1] * vecToCompose.get(1)[1]);
                            double area = width * height;
                            result = Math.min(result, area);
                        }
                    }

                }
            }
        }
        return result == max ? 0d : result;
    }

    // LC1505 Hard ** 树状数组
    public String minInteger(String num, int k) {
        int[] latestIdx = new int[10];
        List<Integer>[] idxList = new List[10];
        for (int i = 0; i < 10; i++) {
            idxList[i] = new ArrayList<>();
        }
        char[] ca = num.toCharArray();
        int n = ca.length;
        for (int i = 0; i < n; i++) {
            idxList[ca[i] - '0'].add(i);
        }
        StringBuilder sb = new StringBuilder();
        BIT49 bit = new BIT49(n); // 用来标记该下标是否已被替换
        int cur = 0;
        while (cur < n) {
            if (bit.get(cur) != 0) { // 已被替换过
                cur++;
                continue;
            }
            int digit = ca[cur] - '0';
            boolean isReplaced = false;
            for (int i = 0; i < digit; i++) {
                while (latestIdx[i] < idxList[i].size() && idxList[i].get(latestIdx[i]) < cur) {
                    latestIdx[i]++;
                }
                if (latestIdx[i] == idxList[i].size()) {
                    continue;
                }
                int idxToBeReplace = idxList[i].get(latestIdx[i]);
                int distance = idxList[i].get(latestIdx[i]) - cur;
                int alreadyReplacedCount = bit.sumRange(cur, idxToBeReplace - 1);
                int steps = distance - alreadyReplacedCount;
                if (steps <= k) {
                    k -= steps;
                    latestIdx[i]++;
                    bit.update(idxToBeReplace, 1);
                    cur--;
                    sb.append(i);
                    isReplaced = true;
                    break;
                }
            }
            if (isReplaced) {
                cur = 0;
                continue;
            }
            bit.update(cur, 1);
            sb.append(digit);
            cur++;
        }
        return sb.toString();
    }

    // JZOF II 051 LC 124
    int lc124Result;

    public int maxPathSum(TreeNode49
                                  root) {
        lc124Result = Integer.MIN_VALUE;
        lc124Helper(root);
        return lc124Result;
    }

    private int lc124Helper(TreeNode49
                                    root) {
        if (root == null) return 0;
        int left = Math.max(0, lc124Helper(root.left));
        int right = Math.max(0, lc124Helper(root.right));
        lc124Result = Math.max(lc124Result, root.val + left + right);
        return root.val + Math.max(left, right);
    }

    // LC1394
    public int findLucky(int[] arr) {
        TreeMap<Integer, Integer> tm = new TreeMap<>(Comparator.comparingInt(o -> -o));
        for (int i : arr) {
            tm.put(i, tm.getOrDefault(i, 0) + 1);
        }
        for (int key : tm.keySet()) {
            if (tm.get(key) == key) return key;
        }
        return -1;
    }

    // LC1857 Hard **
    // let dp[u][c] := the maximum count of vertices with color c of any path starting from vertex u. (by JerryJin2905)
    public int largestPathValue(String colors, int[][] edges) {
        char[] ca = colors.toCharArray();
        int n = ca.length;
        int[][] dp = new int[n][26];
        List<List<Integer>> outEdge = new ArrayList<>();
        int[] indegree = new int[n];
        int result = 0;
        for (int i = 0; i < n; i++) outEdge.add(new ArrayList<>());
        for (int[] e : edges) {
            indegree[e[1]]++;
            outEdge.get(e[0]).add(e[1]);
        }
        Deque<Integer> q = new LinkedList<>();
        List<Integer> topo = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int p = q.poll();
            topo.add(p);
            dp[p][ca[p] - 'a']++;
            List<Integer> out = outEdge.get(p);
            for (int i : out) {
                indegree[i]--;
                // 注意转移的时机, 在找到下一条出边的时候, 遍历所有颜色,
                for (int j = 0; j < 26; j++) {
                    dp[i][j] = Math.max(dp[p][j], dp[i][j]);
                }
                if (indegree[i] == 0) {
                    q.offer(i);
                }
            }
        }
        if (topo.size() != n) return -1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 26; j++) {
                result = Math.max(result, dp[i][j]);
            }
        }

        return result;
    }

    // LC18 4sum
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new LinkedList<>();
        int n = nums.length;
        Arrays.sort(nums);
        for (int a = 0; a < n; a++) {
            if (a > 0 && nums[a - 1] == nums[a]) continue;
            for (int b = a + 1; b < n; b++) {
                if (b > a + 1 && nums[b - 1] == nums[b]) continue;
                int c = b + 1, d = n - 1;
                while (c < d) {
                    int tmpSum = nums[a] + nums[b] + nums[c] + nums[d];
                    if (tmpSum == target) {
                        result.add(Arrays.asList(nums[a], nums[b], nums[c], nums[d]));
                        while (c < d && nums[c] == nums[c + 1]) c++;
                        while (c < d && nums[d] == nums[d - 1]) d--;
                        c++;
                        d--;
                    } else if (tmpSum > target) {
                        d--;
                    } else {
                        c++;
                    }
                }
            }
        }
        return result;
    }

    // JZOF II 012
    public int pivotIndex(int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n + 1];
        // Arrays.fill(prefix,0);
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        for (int i = 1; i <= n; i++) {
            int left = prefix[i - 1] - prefix[0];
            int right = prefix[n] - prefix[i];
            if (left == right) return i - 1;
        }
        return -1;
    }

    // LC446 **
    public int numberOfArithmeticSlicesHard(int[] nums) {
        // 状态 dp[i][d] 到nums[i]为止公差为d的等差数列的个数, 用map离散化, 避免公差d的范围过大
        // how transfer
        // [2,4,6,8,10]
        // j<i, d=nums[i]-nums[j], dp[d][i] += (dp[d][j]+1)
        // 考虑到长度为2的序列从dp[d][i](左侧)中生成(等式右边的+1), 所以实际result+=的是右侧的(dp[d][j]), 即长度一定大于2的数列的数量
        int n = nums.length, result = 0;
        Map<Integer, Integer>[] dp = new Map[n];
        for (int i = 0; i < n; i++) {
            dp[i] = new HashMap<>();
            for (int j = 0; j < i; j++) {
                long diff = (long) (nums[i]) - (long) (nums[j]);
                if (diff < Integer.MIN_VALUE || diff > Integer.MAX_VALUE) continue;
                int d = (int) (diff);
                int sumJ = dp[j].getOrDefault(d, 0);
                int sumI = dp[i].getOrDefault(d, 0);
                result += sumJ;
                dp[i].put(d, sumI + sumJ + 1);
            }
        }
        return result;
    }

    // LC413
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n <= 2) return 0;
        int d = nums[n - 1] - nums[n - 2];
        int result = 0, len = 2, curCount = 0;
        for (int i = n - 2; i >= 1; i--) {
            int newD = nums[i] - nums[i - 1];
            if (newD == d) {
                if (++len >= 3) {
                    result += ++curCount;
                }
            } else {
                d = newD;
                len = 2;
                curCount = 0;
            }
        }
        return result;
    }

    // LC76
    public String minWindow(String s, String t) {
        int n = s.length(), m = t.length();
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        int[] freq = new int[256];
        String result = "";
        for (char c : ct) freq[c]--;
        int count = 0;
        for (int left = 0, right = 0; right < n; right++) {
            freq[cs[right]]++;
            if (freq[cs[right]] <= 0) count++;
            while (count == m && freq[cs[left]] > 0) freq[cs[left++]]--;
            if (count == m) {
                if (result.equals("") || result.length() > right - left + 1) {
                    result = s.substring(left, right + 1);
                }
            }
        }
        return result;
    }

    // LC37
    char[][] lc37result = new char[9][9];

    public void solveSudoku(char[][] board) {
        int left = 0;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    left++;
                }
            }
        }
        lc37Helper(board, left);
        for (int i = 0; i < 9; i++) {
            System.arraycopy(lc37result[i], 0, board[i], 0, 9);
        }
    }

    private boolean lc37Helper(char[][] board, int left) {
        if (left == 0) {
            for (int i = 0; i < 9; i++) {
                System.arraycopy(board[i], 0, lc37result[i], 0, 9);
            }
            return true;
        }
        int i = 0, j = 0;
        for (i = 0; i < 9; i++) {
            boolean flag = false;
            for (j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    flag = true;
                    break;
                }
            }
            if (flag) break;
        }
        if (i == 9 || j == 9) return false;
        int r = i, c = j;
        Set<Integer> notAllow = new HashSet<>();
        for (int k = 0; k < 9; k++) {
            if (Character.isDigit(board[i][k])) {
                notAllow.add(board[i][k] - '0');
            }
            if (Character.isDigit(board[k][j])) {
                notAllow.add(board[k][j] - '0');
            }
        }
        i /= 3;
        j /= 3;
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                if (Character.isDigit(board[i * 3 + m][j * 3 + n])) {
                    notAllow.add(board[i * 3 + m][j * 3 + n] - '0');
                }
            }
        }
        Set<Integer> allow = new HashSet<>();
        for (int k = 1; k <= 9; k++) {
            if (!notAllow.contains(k)) {
                allow.add(k);
            }
        }
        if (allow.size() == 0) return false;
        for (int choice : allow) {
            board[r][c] = (char) (choice + '0');
            lc37Helper(board, left - 1);
            board[r][c] = '.';
        }
        return false;
    }

    // LC93
    List<String> lc93Result = new ArrayList<>();

    public List<String> restoreIpAddresses(String s) {
        lc93Backtrack(new ArrayList<>(), 0, s);
        return lc93Result;
    }

    private void lc93Backtrack(List<String> selections, int curIdx, String s) {
        if (selections.size() == 4) {
            if (curIdx == s.length()) {
                lc93Result.add(String.join(".", selections));
            } else {
                return;
            }
        }
        for (int i = 1; i <= 3; i++) {
            if (i + curIdx > s.length()) break;
            String tmp = s.substring(curIdx, curIdx + i);
            int tmpInt = Integer.valueOf(tmp);
            if (tmpInt >= 0 && tmpInt <= 255 && String.valueOf(tmpInt).equals(tmp)) {
                selections.add(tmp);
                lc93Backtrack(selections, curIdx + i, s);
                selections.remove(selections.size() - 1);
            }
        }
    }

    public boolean isPalindrome(String s) {
        char[] ca = s.toCharArray();
        int left = 0, right = ca.length - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(ca[left])) left++;
            while (left < right && !Character.isLetterOrDigit(ca[right])) right--;
            if (left < right) {
                if (Character.toLowerCase(ca[left]) != Character.toLowerCase(ca[right])) return false;
                left++;
                right--;
            }
        }
        return true;
    }

    // LC757 **
    public int intersectionSizeTwo(int[][] intervals) {
        int n = intervals.length, result = 0;
        // 排序
        Arrays.sort(intervals, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);
        int[] todo = new int[n];
        Arrays.fill(todo, 2);
        for (int i = n - 1; i >= 0; i--) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            int m = todo[i];
            for (int j = start; j < start + m; j++) {
                for (int k = 0; k <= i; k++) {
                    if (todo[k] > 0 && j <= intervals[k][1]) {
                        todo[k]--;
                    }
                }
                result++;
            }
        }
        return result;
    }

    // LC652
    public List<TreeNode49
            > findDuplicateSubtrees(TreeNode49
                                            root) {
        List<TreeNode49
                > result = new ArrayList<>();
        if (root == null) return result;
        Map<String, TreeNode49
                > map = new HashMap<>();
        Set<String> duplicateKey = new HashSet<>();
        lc652Dfs(root, map, duplicateKey);
        for (String key : duplicateKey) {
            result.add(map.get(key));
        }
        return result;
    }

    private String lc652Dfs(TreeNode49
                                    root, Map<String, TreeNode49
            > map, Set<String> duplicateKey) {
        if (root == null) return "()";
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        sb.append(root.val);
        sb.append(lc652Dfs(root.left, map, duplicateKey));
        sb.append(lc652Dfs(root.right, map, duplicateKey));
        sb.append(')');
        String key = sb.toString();
        if (map.containsKey(key)) {
            duplicateKey.add(key);
        }
        map.put(key, root);
        return key;
    }

    // LC606
    public String tree2str(TreeNode49
                                   root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        lc606Preorder(root, sb);
        return sb.toString().substring(1, sb.length() - 1);
    }

    private void lc606Preorder(TreeNode49
                                       root, StringBuilder sb) {
        if (root == null) return;
        sb.append('(');
        sb.append(root.val);
        if (root.left == null && root.right != null) {
            sb.append("()");
            lc606Preorder(root.right, sb);
        } else if (root.left != null && root.right == null) {
            lc606Preorder(root.left, sb);
        } else if (root.left != null && root.right != null) {
            lc606Preorder(root.left, sb);
            lc606Preorder(root.right, sb);
        } else {
            ;
        }
        sb.append(')');
    }

    // JZOF II 101 01背包
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 == 1) return false;
        int half = sum / 2;
        int[] dp = new int[half + 1];
        // dp[i][j] 加入前i个数 在背包限制为j的情况下能达到的最大值
        for (int i = 1; i <= nums.length; i++) {
            for (int j = half; j >= 0; j--) {
                if (j - nums[i - 1] >= 0 && dp[j - nums[i - 1]] + nums[i - 1] <= half) {
                    dp[j] = Math.max(dp[j], dp[j - nums[i - 1]] + nums[i - 1]);
                }
            }
            if (dp[half] == half) {
                return true;
            }
        }
        return dp[half] == half;
    }

    // LC313
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] dp = new int[n + 1]; // 第一个是1
        dp[1] = 1;
        int[] ptr = new int[primes.length];
        Arrays.fill(ptr, 1);
        for (int i = 2; i <= n; i++) {
            int next = Integer.MAX_VALUE;
            List<Integer> changedIdx = new ArrayList<>();
            for (int j = 0; j < primes.length; j++) {
                int tmp = dp[ptr[j]] * primes[j];
                if (tmp < next) {
                    next = tmp;
                    changedIdx = new ArrayList<>();
                    changedIdx.add(j);
                } else if (tmp == next) {
                    changedIdx.add(j);
                }
            }
            dp[i] = next;
            for (int idx : changedIdx) ptr[idx]++;
        }
        return dp[n];
    }

    // LC1137
    public int tribonacci(int n) {
        if (n == 0) return 0;
        if (n <= 2) return 1;
        int[] dp = new int[]{0, 1, 1};
        int ctr = 2;
        for (; ctr < n; ctr++) {
            dp[(ctr + 1) % 3] = dp[ctr % 3] + dp[(ctr - 1) % 3] + dp[(ctr - 2) % 3];
        }
        return dp[ctr % 3];
    }

    // LC457 Solution
    public boolean circularArrayLoop(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) continue;
            int slow = i, fast = lc457Next(nums, i);
            while (nums[slow] * nums[fast] > 0 && nums[slow] * nums[lc457Next(nums, fast)] > 0) {
                if (slow == fast) {
                    if (slow != lc457Next(nums, slow)) {
                        return true;
                    } else {
                        break; // 循环长度为1
                    }
                }
                slow = lc457Next(nums, slow);
                fast = lc457Next(nums, lc457Next(nums, fast));
            }
            int toMark = i;
            while (nums[toMark] * nums[lc457Next(nums, toMark)] > 0) {
                int tmp = toMark;
                toMark = lc457Next(nums, toMark);
                nums[tmp] = 0;
            }
        }
        return false;
    }

    private int lc457Next(int[] nums, int idx) {
        return ((idx + nums[idx]) % nums.length + nums.length) % nums.length;
    }

    // JZOF 54
    int lc54Ctr;
    int lc54Result = -1;

    public int kthLargest(TreeNode49
                                  root, int k) {
        lc54Ctr = 0;
        lc54InOrder(root, k);
        return lc54Result;
    }

    private void lc54InOrder(TreeNode49
                                     root, int k) {
        if (root == null) return;
        lc54InOrder(root.right, k);
        lc54Ctr++;
        if (lc54Ctr == k) {
            lc54Result = root.val;
            return;
        }
        lc54InOrder(root.left, k);
    }

    // JZOF 65 **
    public int add(int a, int b) {
        int sum = a;
        while (b != 0) {
            int xor = a ^ b;
            int and = a & b;
            b = and << 1;
            sum = xor;
            a = sum;
        }
        return sum;
    }

    // JZOF 61
    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int zeroCount = 0;
        int[] freq = new int[14];
        for (int i : nums) {
            if (i == 0) zeroCount++;
            else {
                freq[i]++;
                if (freq[i] != 1) return false;
            }
        }
        if (zeroCount >= 4) return true;
        int diff = 0;
        for (int i = zeroCount; i < 4; i++) {
            diff += nums[i + 1] - nums[i] - 1;
        }
        if (diff > zeroCount) return false;
        return true;
    }

    // JZOF 62 同LC1823 约瑟夫环
    // 迭代
    public int lastRemaining(int n, int m) {
        int f = 0; // i=1
        for (int i = 2; i <= n; i++) {
            f = (m + f) % i;
        }
        return f;
    }

    // 递归
    private int f(int n, int m) {
        if (n == 1) {
            return 0;
        }
        int x = f(n - 1, m);
        return (m + x) % n;
    }

    // JZOF 55
    public boolean isBalanced(TreeNode49
                                      root) {
        if (root == null) return true;
        int leftDepth = getTreeDepth(root.left);
        int rightDepth = getTreeDepth(root.right);
        return Math.abs(leftDepth - rightDepth) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    private int getTreeDepth(TreeNode49
                                     root) {
        if (root == null) return 0;
        int left = getTreeDepth(root.left);
        int right = getTreeDepth(root.right);
        return 1 + Math.max(left, right);
    }

    // LC847 **
    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        Deque<int[]> q = new LinkedList<>();
        int[][] dist = new int[1 << n][n];
        int fullMask = (1 << n) - 1;
        for (int[] d : dist) Arrays.fill(d, 0x3f3f3f3f);
        for (int i = 0; i < n; i++) {
            q.offer(new int[]{1 << i, i}); // [bitmask, head]
            dist[1 << i][i] = 0;
        }
        while (!q.isEmpty()) {
            int[] state = q.poll();
            int d = dist[state[0]][state[1]];
            if (state[0] == fullMask) return d;
            for (int next : graph[state[1]]) {
                int nextMask = state[0] | (1 << next);
                if (d + 1 < dist[nextMask][next]) {
                    dist[nextMask][next] = d + 1;
                    q.offer(new int[]{nextMask, next});
                }
            }
        }
        return -1;
    }

    // LC1834 ** 模拟
    public int[] getOrder(int[][] tasks) {
        Map<int[], Integer> idxMap = new HashMap<>();
        for (int i = 0; i < tasks.length; i++) idxMap.put(tasks[i], i);
        List<Integer> result = new ArrayList<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> o1[1] == o2[1] ? idxMap.get(o1) - idxMap.get(o2) : o1[1] - o2[1]);
        Arrays.sort(tasks, Comparator.comparingInt(o -> o[0]));
        int nextAvail = tasks[0][0];
        int i = 0;
        while (result.size() < tasks.length) {
            while (i < tasks.length && tasks[i][0] <= nextAvail) pq.offer(tasks[i++]);
            if (pq.isEmpty()) {
                nextAvail = tasks[i][0];
            } else {
                int[] cur = pq.poll();
                result.add(idxMap.get(cur));
                nextAvail += cur[1];
            }
        }
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC497
    static class Lc497 {
        TreeMap<Integer, Integer> tm;
        int[][] rects;
        Random rand = new Random();
        int totalArea = 0;

        public Lc497(int[][] rects) {
            tm = new TreeMap<>();
            this.rects = rects;
            int ctr = 0;
            for (int[] r : rects) {
                int x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
                totalArea += (Math.abs(y1 - y2) + 1) * (Math.abs(x1 - x2) + 1);
                tm.put(totalArea, ctr++);
            }
        }

        public int[] pick() {
            int target = rand.nextInt(totalArea) + 1; // 注意+1这些细节，判题能判出来的
            int ceil = tm.ceilingKey(target);
            int rectIdx = tm.get(ceil);
            int[] r = rects[rectIdx];
            int x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
            int offset = ceil - target;
            int len = Math.abs(x1 - x2) + 1;
            int relX = offset % len, relY = offset / len;
            return new int[]{x1 + relX, y1 + relY};
        }
    }


    // LC429
    class Lc429 {

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

        class Solution {
            public List<List<Integer>> levelOrder(Node root) {
                Deque<Node> q = new LinkedList<>();
                List<List<Integer>> result = new LinkedList<>();
                if (root == null) return result;
                q.offer(root);
                while (!q.isEmpty()) {
                    int qSize = q.size();
                    List<Integer> thisLayer = new LinkedList<>();
                    for (int i = 0; i < qSize; i++) {
                        Node p = q.poll();
                        thisLayer.add(p.val);
                        for (Node c : p.children) q.offer(c);
                    }
                    result.add(thisLayer);
                }
                return result;
            }
        }
    }

    // LC940 **
    public int distinctSubseqII(String s) {
        int n = s.length();
        char[] ca = s.toCharArray();
        int[] dp = new int[n + 1];
        dp[0] = 1; // 空串
        int[] lastOccur = new int[26];
        Arrays.fill(lastOccur, -1);
        final int mod = 1000000007;
        for (int i = 0; i < n; i++) {
            dp[i + 1] = dp[i] * 2 % mod;
            if (lastOccur[ca[i] - 'a'] != -1) {
                dp[i + 1] -= dp[lastOccur[ca[i] - 'a']];
            }
            dp[i + 1] %= mod;
            lastOccur[ca[i] - 'a'] = i;
        }
        dp[n] = (dp[n] - 1 + mod) % mod; // -1 处理空串
        return dp[n];
    }

    // LC1696 单纯DP不行 求max是O(n), 加起来O(n^2)超时, 用TreeMap求max是O(log(n)), 总复杂度O(nlogn)
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[n - 1] = nums[n - 1];
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        tm.put(dp[n - 1], 1);
        int tmCounter = 1;
        for (int i = n - 2; i >= 0; i--) {
            int gain = tm.lastKey();
            if (tmCounter == k) {
                tm.put(dp[i + k], tm.get(dp[i + k]) - 1);
                if (tm.get(dp[i + k]) == 0) tm.remove(dp[i + k]);
                tmCounter--;
            }
            dp[i] = gain + nums[i];
            tm.put(dp[i], tm.getOrDefault(dp[i], 0) + 1);
            tmCounter++;
        }
        return dp[0];
    }

    // LC1696 TLE
    public int maxResultBottomUp(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            int gain = Integer.MIN_VALUE;
            for (int j = i + 1; j <= Math.min(n - 1, i + k); j++) {
                gain = Math.max(gain, dp[j]);
            }
            dp[i] = gain + nums[i];
        }
        return dp[0];
    }

    // LC1696 TLE
    Integer[] lc1696Memo;

    public int maxResultTopDown(int[] nums, int k) {
        lc1696Memo = new Integer[nums.length];
        return lc1696Helper(0, nums, k);
    }

    private int lc1696Helper(int cur, int[] nums, int k) {
        if (cur == nums.length - 1) return nums[nums.length - 1];
        if (lc1696Memo[cur] != null) return lc1696Memo[cur];
        int gain = Integer.MIN_VALUE;
        for (int i = cur + 1; i <= Math.min(cur + k, nums.length - 1); i++) {
            gain = Math.max(gain, lc1696Helper(i, nums, k));
        }
        return lc1696Memo[cur] = nums[cur] + gain;
    }

    // LC1780
    public boolean checkPowersOfThree(int n) {
        while (n != 0) {
            if (n % 3 == 2) return false;
            n /= 3;
        }
        return true;
    }

    // LC1523
    public int countOdds(int low, int high) {
        if (low % 2 == 1) low--;
        if (high % 2 == 1) high++;
        return (high - low) / 2;
    }

    // LC1782 ***
    public int[] countPairs(int n, int[][] edges, int[] queries) {
        int[] result = new int[queries.length];
        int[] deg = new int[n + 1];
        Map<Pair<Integer, Integer>, Integer> edgeCount = new HashMap<>();
        for (int[] e : edges) {
            int a = Math.min(e[0], e[1]), b = Math.max(e[0], e[1]);
            deg[a]++;
            deg[b]++;
            Pair<Integer, Integer> key = new Pair<>(a, b);
            edgeCount.put(key, edgeCount.getOrDefault(key, 0) + 1);
        }
        int[] sortedDeg = Arrays.copyOfRange(deg, 1, deg.length);
        Arrays.sort(sortedDeg);
        for (int i = 0; i < queries.length; i++) {
            // 容斥原理
            // c1: deg[a] + deg[b] - edgeCount(a,b) > q[i], ab存在边
            // c2: deg[a] + deg[b] > q[i], ab 存在边
            // c3: deg[a] + deg[b] > q[i], 对于所有点
            // result[i] = c1 + c3 - c2, c2被重复计算了
            int c1 = 0, c2 = 0, c3 = 0;
            for (Pair<Integer, Integer> edge : edgeCount.keySet()) {
                int a = edge.getKey(), b = edge.getValue();
                if (deg[a] + deg[b] - edgeCount.get(edge) > queries[i]) c1++;
                if (deg[a] + deg[b] > queries[i]) c2++;
            }
            int left = 0, right = n - 1;
            // 双指针求有序数组中和大于queries[i]的数对的个数
            while (left < n && right >= 0) {
                while (right > left && sortedDeg[left] + sortedDeg[right] <= queries[i]) {
                    left++;
                }
                if (right > left && sortedDeg[left] + sortedDeg[right] > queries[i]) {
                    c3 += right - left; // 求的是**数对**个数, 而不是两个数之间(含端点)共有多少个数, 所以不用+1
                }
                right--;
            }
            result[i] = c1 + c3 - c2;
        }
        return result;
    }

    // LC678 ** 两个栈
    public boolean checkValidString(String s) {
        char[] ca = s.toCharArray();
        Deque<Integer> left = new LinkedList<>(), star = new LinkedList<>();
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == '(') left.push(i);
            else if (ca[i] == '*') star.push(i);
            else {
                if (left.size() > 0) left.pop();
                else if (star.size() > 0) star.pop();
                else return false;
            }
        }
        if (left.size() > star.size()) return false;
        while (left.size() > 0 && star.size() > 0) {
            if (left.pop() > star.pop()) return false;
        }
        return true;
    }

    // LC761 ** 非常巧妙 看成括号对
    public String makeLargestSpecial(String s) {
        if (s.length() == 2) return s;
        Map<String, Integer> m = new HashMap<>();
        char[] ca = s.toCharArray();
        int prev = 0;
        int oneCount = 0;
        for (int i = 0; i < ca.length; i++) {
            if (ca[i] == '1') oneCount++;
            else {
                oneCount--;
                if (oneCount == 0) {
                    String magic = s.substring(prev, i + 1);
                    m.put(magic, m.getOrDefault(magic, 0) + 1);
                    prev = i + 1;
                }
            }
        }
        List<String> result = new ArrayList<>();
        for (String k : m.keySet()) {
            String kResult = k;
            if (k.length() > 2) {
                kResult = "1" + makeLargestSpecial(k.substring(1, k.length() - 1)) + "0";
            }
            for (int i = 0; i < m.get(k); i++) {
                result.add(kResult);
            }
        }
        result.sort(Comparator.reverseOrder());
        return String.join("", result);
    }

    // LC1605 ** 贪心
    public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
        int numRow = rowSum.length, numCol = colSum.length;
        int[][] result = new int[numRow][numCol];
        for (int i = 0; i < numRow; i++) {
            for (int j = 0; j < numCol; j++) {
                result[i][j] = Math.min(rowSum[i], colSum[j]);
                rowSum[i] -= result[i][j];
                colSum[j] -= result[i][j];
            }
        }
        return result;
    }

    // LC210 Topology
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        Deque<Integer> q = new LinkedList<>();
        List<List<Integer>> graph = new ArrayList<>(numCourses); // 拓扑排序算法中需要记录的出度表
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) graph.add(new ArrayList<>());
        for (int[] p : prerequisites) {
            inDegree[p[0]]++;
            graph.get(p[1]).add(p[0]);
        }
        for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) q.offer(i);
        while (!q.isEmpty()) {
            int p = q.poll();
            result.add(p);
            for (int next : graph.get(p)) {
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        for (int i = 0; i < numCourses; i++) if (inDegree[i] != 0) return new int[0];
        return result.stream().mapToInt(Integer::valueOf).toArray();
    }

    // LC802 Topology Sort
    public List<Integer> eventualSafeNodesTopologySort(int[][] graph) {
        List<Integer> result = new ArrayList<>();
        int n = graph.length;
        List<List<Integer>> reverseGraph = new ArrayList<>(n);
        int[] inDegree = new int[n];
        for (int i = 0; i < n; i++) reverseGraph.add(new LinkedList<>());
        for (int i = 0; i < n; i++) {
            int[] ithOutDegree = graph[i];
            for (int j : ithOutDegree) {
                reverseGraph.get(j).add(i);
                inDegree[i]++;
            }
        }
        Deque<Integer> zeroInDegreeQueue = new LinkedList<>();
        for (int i = 0; i < n; i++) if (inDegree[i] == 0) zeroInDegreeQueue.offer(i);
        while (!zeroInDegreeQueue.isEmpty()) {
            int i = zeroInDegreeQueue.poll();
            List<Integer> out = reverseGraph.get(i);
            for (int j : out) {
                inDegree[j]--;
                if (inDegree[j] == 0) {
                    zeroInDegreeQueue.offer(j);
                }
            }
        }
        for (int i = 0; i < n; i++) if (inDegree[i] == 0) result.add(i);
        return result;
    }

    // LC802 ** 三色算法 垃圾回收时候的判断有无依赖的一种算法
    int[] lc802Mark;
    final int UNVISITED = 0, IN_STACK = 1, SAFE = 2;

    public List<Integer> eventualSafeNodes(int[][] graph) {
        // graph[i] 为节点i的出度向量
        int n = graph.length;
        lc802Mark = new int[n];
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) { // 实际GC的三色算法中, 枚举的只是GC Root节点, 枚举完后, 如果有节点为UNVISITED, 则对其执行GC
            if (lc802Helper(i, graph)) {
                result.add(i);
            }
        }
        return result;
    }

    private boolean lc802Helper(int cur, int[][] graph) {
        if (lc802Mark[cur] != UNVISITED) {
            return lc802Mark[cur] == SAFE;
        }
        lc802Mark[cur] = IN_STACK;
        for (int next : graph[cur]) {
            if (!lc802Helper(next, graph)) {
                return false;
            }
        }
        lc802Mark[cur] = SAFE;
        return true;
    }

    // LC934
    int[][] lc934Directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int shortestBridge(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int ctr = 0;
        Set<Integer> s1 = new HashSet<>(), s2 = new HashSet<>();
        Set<Integer> curSet = s1;
        // DFS
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    lc934DfsHelper(grid, i, j, curSet);
                    ctr++;
                    if (ctr == 2) break;
                    curSet = s2;
                }
            }
            if (ctr == 2) break;
        }
        Set<Integer> smallSet = s1.size() > s2.size() ? s2 : s1;
        Set<Integer> largeSet = smallSet == s1 ? s2 : s1;
        Set<Integer> visited = new HashSet<>();
        // BFS
        int layer = -2;
        Deque<Integer> q = new LinkedList<>();
        for (int i : smallSet) {
            q.offer(i);
        }
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int p = q.poll();
                if (visited.contains(p)) continue;
                visited.add(p);
                if (largeSet.contains(p)) return layer;
                int row = p / grid[0].length, col = p % grid[0].length;
                for (int[] dir : lc934Directions) {
                    int newRow = row + dir[0], newCol = col + dir[1];
                    int newNum = newRow * grid[0].length + newCol;
                    if (newRow >= 0 && newRow < grid.length && newCol >= 0 && newCol < grid[0].length && !visited.contains(newNum)) {
                        q.offer(newNum);
                    }
                }
            }
        }
        return -1;
    }

    private void lc934DfsHelper(int[][] grid, int row, int col, Set<Integer> set) {
        grid[row][col] = -1;
        set.add(row * grid[0].length + col);
        for (int[] dir : lc934Directions) {
            int newRow = row + dir[0], newCol = col + dir[1];
            if (newRow >= 0 && newRow < grid.length && newCol >= 0 && newCol < grid[0].length && grid[newRow][newCol] == 1) {
                lc934DfsHelper(grid, newRow, newCol, set);
            }
        }
    }

    // LC1488
    public int[] avoidFlood(int[] rains) {
        int n = rains.length;
        int[] ans = new int[n];
        TreeSet<Integer> unrain = new TreeSet<>();
        Map<Integer, Integer> tbd = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (rains[i] == 0) unrain.add(i);
            else {
                ans[i] = -1;
                if (tbd.containsKey(rains[i])) {
                    if (unrain.isEmpty()) {
                        return new int[0];
                    }
                    int prevRainDay = tbd.get(rains[i]);
                    Integer ceiling = unrain.ceiling(prevRainDay);
                    if (ceiling == null) {
                        return new int[]{};
                    }
                    ans[ceiling] = rains[i];
                    unrain.remove(ceiling);
                }
                tbd.put(rains[i], i);
            }
        }
        for (int i : unrain) ans[i] = tbd.keySet().iterator().next();
        return ans;
    }

    // LC1300
    public int findBestValue(int[] arr, int target) {
        int n = arr.length;
        int[] prefix = new int[n + 1];
        Arrays.sort(arr);
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + arr[i - 1];
        int lo = 0, hi = arr[n - 1];
        while (lo < hi) { // 找value的下届
            int mid = lo + (hi - lo + 1) / 2;
            int idx = bsLargerOrEqualMin(arr, mid);
            while (idx < arr.length && arr[idx] == mid) idx++;
            if (idx > arr.length) break;
            // 此时idx及其之后的值都大于value
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * mid;
            if (curSum <= target) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int lowBound = lo;
        lo = 0;
        hi = arr[n - 1];
        while (lo < hi) { // 找value的上界
            int mid = lo + (hi - lo) / 2;
            int idx = bsLargerOrEqualMin(arr, mid);
            while (idx < arr.length && arr[idx] == mid) idx++;
            if (idx > arr.length) break;
            // 此时idx及其之后的值都大于value
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * mid;
            if (curSum < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int hiBound = lo;

        int result = lowBound, minDiff = Integer.MAX_VALUE;
        for (int i = lowBound; i <= hiBound; i++) {
            int idx = bsLargerOrEqualMin(arr, i);
            while (idx < arr.length && arr[idx] == i) idx++;
            int curSum = prefix[idx] - prefix[0] + (arr.length - idx) * i;
            if (Math.abs(curSum - target) < minDiff) {
                minDiff = Math.abs(curSum - target);
                result = i;
            }
        }
        return result;
    }

    private int bsLessOrEqualMax(int[] arr, int target) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (arr[mid] <= target) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if (arr[lo] > target) return -1;
        return lo;
    }

    private int bsLargerOrEqualMin(int[] arr, int target) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if (arr[hi] < target) return -1;
        return hi;
    }

    // LC1553 ** 类比LCP20
    Map<Integer, Integer> lc1553Memo = new HashMap<>();

    public int minDays(int n) {
        return lc1553Helper(n);
    }

    private int lc1553Helper(int cur) {
        if (cur <= 1) return 1;
        if (cur == 2 || cur == 3) return 2;
        if (lc1553Memo.get(cur) != null) return lc1553Memo.get(cur);
        int result = cur;
        result = Math.min(result, 1 + lc1553Helper(cur / 2) + cur % 2);
        result = Math.min(result, 1 + lc1553Helper(cur / 3) + cur % 3);
        lc1553Memo.put(cur, result);
        return result;
    }

    // LCP 20 ** bottom up dfs
    Map<Long, Long> lcp20Map;
    final long lcp20Mod = 1000000007L;
    int lcp20Inc, lcp20Dec;
    int[] lcp20Jump, lcp20Cost;

    public int busRapidTransit(int target, int inc, int dec, int[] jump, int[] cost) {
        lcp20Map = new HashMap<>();
        lcp20Map.put(0l, 0l);
        lcp20Map.put(1l, (long) inc);
        lcp20Inc = inc;
        lcp20Cost = cost;
        lcp20Jump = jump;
        lcp20Dec = dec;
        return (int) (lcp20Helper(target) % lcp20Mod);
    }

    private long lcp20Helper(long cur) {
        if (lcp20Map.containsKey(cur)) return lcp20Map.get(cur);
        long result = cur * lcp20Inc;
        for (int i = 0; i < lcp20Jump.length; i++) {
            long remainder = cur % lcp20Jump[i];
            if (remainder == 0l) {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i]);
            } else {
                result = Math.min(result, lcp20Helper(cur / lcp20Jump[i]) + lcp20Cost[i] + remainder * lcp20Inc);
                result = Math.min(result, lcp20Helper((cur / lcp20Jump[i]) + 1) + lcp20Cost[i] + (lcp20Jump[i] - remainder) * lcp20Dec);
            }
        }
        lcp20Map.put(cur, result);
        return result;
    }

    // LC1940 Prime Locked
    public List<Integer> longestCommomSubsequence(int[][] arrays) {
        List<Integer> result = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            int count = 0;
            for (int[] arr : arrays) {
                int bsResult = Arrays.binarySearch(arr, i);
                if (bsResult >= 0) count++;
            }
            if (count == arrays.length) result.add(i);
        }
        return result;
    }

    // LC1781
    public int beautySum(String s) {
        char[] ca = s.toCharArray();
        int[] freq = new int[26];
        int left = 0;
        int result = 0;
        while (left < s.length()) {
            freq = new int[26];
            int right = left;
            while (right < s.length()) {
                freq[ca[right++] - 'a']++;
                int[] j = lc1781Judge(freq);
                if (j[0] != -1) {
                    result += freq[j[1]] - freq[j[0]];
                }
            }
            left++;
        }
        return result;
    }

    private int[] lc1781Judge(int[] freq) {
        int min = Integer.MAX_VALUE, minIdx = -1, max = 0, maxIdx = -1;
        int notZeroCount = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] != 0) notZeroCount++;
            if (freq[i] > max) {
                max = freq[i];
                maxIdx = i;
            }
            if (freq[i] != 0 && freq[i] < min) {
                min = freq[i];
                minIdx = i;
            }
        }
        if (notZeroCount <= 1 || max == min) return new int[]{-1, -1};
        return new int[]{minIdx, maxIdx};
    }

    // LC417 **
    boolean[][] lc417P, lc417A;
    int[][] lc417Direction = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        int n = heights.length, m = heights[0].length;
        lc417A = new boolean[n][m];
        lc417P = new boolean[n][m];
        for (int i = 0; i < n; i++) {
            lc417Helper(heights, i, 0, lc417P);
            lc417Helper(heights, i, m - 1, lc417A);
        }
        for (int i = 0; i < m; i++) {
            lc417Helper(heights, 0, i, lc417P);
            lc417Helper(heights, n - 1, i, lc417A);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (lc417P[i][j] && lc417A[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }
        return result;
    }

    private boolean lc417IdxLegalJudge(int row, int col) {
        return (row >= 0 && row < lc417P.length && col >= 0 && col < lc417P[0].length);
    }

    private void lc417Helper(int[][] heights, int row, int col, boolean[][] judge) {
        if (judge[row][col]) return;
        judge[row][col] = true;
        for (int[] dir : lc417Direction) {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if (lc417IdxLegalJudge(newRow, newCol) && heights[newRow][newCol] >= heights[row][col]) {
                lc417Helper(heights, newRow, newCol, judge);
            }
        }
    }


    // LC611 **
    public int triangleNumber(int[] nums) {
        // nums.length <=1000
        // A + B > C
        int n = nums.length, result = 0;
        if (n <= 2) return 0;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int k = i;
            for (int j = i + 1; j < n; j++) {
                while (k + 1 < n && nums[k + 1] < nums[i] + nums[j]) {
                    k++;
                }
                result += Math.max(k - j, 0);
            }
        }
        return result;
    }

    // LC167
    public int[] twoSum(int[] numbers, int target) {
        int n = numbers.length;
        for (int i = 0; i < n; i++) {
            int tmp = target - numbers[i];
            int bsResult = Arrays.binarySearch(numbers, i + 1, n, tmp);
            if (bsResult >= 0) return new int[]{i + 1, bsResult + 1};
        }
        return new int[]{-1, -1};
    }

    // LC1823
    public int findTheWinner(int n, int k) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int i = 1; i <= n; i++) s.add(i);
        int cur = 1;
        while (s.size() > 1) {
            int ctr = 1;
            while (ctr < k) {
                Integer higher = s.higher(cur);
                if (higher == null) higher = s.first();
                cur = higher;
                ctr++;
            }
            Integer next = s.higher(cur);
            if (next == null) next = s.first();
            s.remove(cur);
            cur = next;
        }
        return s.first();
    }

    // LC1567 Solution DP
    public int getMaxLen(int[] nums) {
        int n = nums.length;
        int[] pos = new int[2], neg = new int[2];
        if (nums[0] > 0) pos[0] = 1;
        if (nums[0] < 0) neg[0] = 1;
        int result = pos[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                pos[i % 2] = pos[(i - 1) % 2] + 1;
                neg[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
            } else if (nums[i] < 0) {
                pos[i % 2] = neg[(i - 1) % 2] == 0 ? 0 : neg[(i - 1) % 2] + 1;
                neg[i % 2] = pos[(i - 1) % 2] + 1;
            } else {
                pos[i % 2] = neg[i % 2] = 0;
            }
            result = Math.max(result, pos[i % 2]);
        }
        return result;
    }

    // LC1567 慢
    public int getMaxLenSimple(int[] nums) {
        int n = nums.length;
        int[] nextZero = new int[n];
        int[] negCount = new int[n];
        Arrays.fill(nextZero, -1);
        int nextZeroIdx = -1;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] == 0) {
                nextZeroIdx = i;
            }
            nextZero[i] = nextZeroIdx;
        }
        negCount[0] = nums[0] < 0 ? 1 : 0;
        for (int i = 1; i < n; i++) {
            negCount[i] = negCount[i - 1] + (nums[i] < 0 ? 1 : 0);
        }
        int result = 0;
        // 在下一个0来临之前, 找到最大的偶数个负数所在IDX 求长度
        for (int i = 0; i < n; i++) {
            if (nums[i] != 0) {
                int curNegCount = negCount[i];
                if (nums[i] < 0) curNegCount--;
                int start = i, end = -1;
                if (nextZero[i] == -1) end = n - 1;
                else end = nextZero[i] - 1;
                int j;
                for (j = end; j >= start; j--) {
                    if (negCount[j] % 2 == curNegCount % 2) break;
                }
                result = Math.max(result, j - i + 1);
            }
        }
        return result;
    }

    // LC198
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[3]; // 滚数组
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[(i + 3) % 3] = Math.max(dp[(i - 1 + 3) % 3], dp[(i - 2 + 3) % 3] + nums[i]);
        }
        return dp[(n - 1 + 3) % 3];
    }

    // LC740 ** 打家劫舍
    public int deleteAndEarn(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int[] sum = new int[max + 1];
        for (int i : nums) sum[i] += i;
        if (max == 1) return sum[1];
        if (max == 2) return Math.max(sum[1], sum[2]);
        int[] dp = new int[max + 1];
        dp[1] = sum[1];
        dp[2] = Math.max(sum[1], sum[2]);
        for (int i = 3; i <= max; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + sum[i]);
        }
        return dp[max];
    }

    // LC673 **
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n;
        int[] dp = new int[n], count = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(count, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (dp[i] <= dp[j]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
            }
        }
        int max = Arrays.stream(dp).max().getAsInt();
        int result = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == max) {
                result += count[i];
            }
        }
        return result;
    }
}

// Interview 03.06
class AnimalShelf {
    int seq = 0;
    // type 0-cat 1-dog
    final int CAT = 0, DOG = 1;
    Map<Integer, Integer> idSeqMap = new HashMap<>();
    Deque<Integer> catQueue = new LinkedList<>();
    Deque<Integer> dogQueue = new LinkedList<>();

    public AnimalShelf() {

    }

    public void enqueue(int[] a) {
        // a[0] = id, a[1] = type
        int sequence = getSeq();
        idSeqMap.put(a[0], sequence);
        if (a[1] == CAT) {
            catQueue.offer(a[0]);
        } else {
            dogQueue.offer(a[0]);
        }
    }

    public int[] dequeueAny() {
        if (catQueue.isEmpty() && dogQueue.isEmpty()) {
            return new int[]{-1, -1};
        } else if (catQueue.isEmpty() && !dogQueue.isEmpty()) {
            return dequeueDog();
        } else if (!catQueue.isEmpty() && dogQueue.isEmpty()) {
            return dequeueCat();
        } else if (idSeqMap.get(catQueue.peek()) < idSeqMap.get(dogQueue.peek())) {
            return dequeueCat();
        } else {
            return dequeueDog();
        }

    }

    public int[] dequeueDog() {
        if (dogQueue.isEmpty()) return new int[]{-1, -1};
        int polledDogId = dogQueue.poll();
        idSeqMap.remove(polledDogId);
        return new int[]{polledDogId, DOG};
    }

    public int[] dequeueCat() {
        if (catQueue.isEmpty()) return new int[]{-1, -1};
        int polledCatId = catQueue.poll();
        idSeqMap.remove(polledCatId);
        return new int[]{polledCatId, CAT};
    }

    private int getSeq() {
        return seq++;
    }
}

// LC478
class Lc478 {
    double x_center;
    double y_center;
    double radius;

    public Lc478(double radius, double x_center, double y_center) {
        this.x_center = x_center;
        this.y_center = y_center;
        this.radius = radius;
    }

    public double[] randPoint() {
        double len = Math.sqrt(Math.random()) * radius; // 注意开方 , 参考solution
        double theta = Math.random() * Math.PI * 2;

        double x = len * Math.sin(theta) + x_center;
        double y = len * Math.cos(theta) + y_center;
        return new double[]{x, y};
    }
}

// JZOF 59
class KthLargest {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        for (int i : nums) {
            add(i);
        }
    }

    public int add(int val) {
        if (pq.size() < k) {
            pq.offer(val);
        } else {
            if (val > pq.peek()) {
                pq.poll();
                pq.offer(val);
            }
        }
        return pq.peek();
    }
}

class quickSort {

    static Random r = new Random();

    public static void sort(int[] arr) {
        helper(arr, 0, arr.length - 1);
    }

    private static void helper(int[] arr, int start, int end) {
        if (start >= end) return;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }

        int left = start, right = end;

        int pivotVal = arr[start];
        while (left < right) {
            while (left < right && arr[right] >= pivotVal) right--;
            while (left < right && arr[left] <= pivotVal) left++;
            if (left < right) {
                int t = arr[left];
                arr[left] = arr[right];
                arr[right] = t;
            }
        }
        arr[start] = arr[left];
        arr[left] = pivotVal;

        helper(arr, start, left - 1);
        helper(arr, right + 1, end);
    }

}

class quickSelect49 {
    static Random r = new Random();

    public static int topK(int[] arr, int topK) {
        return helper(arr, 0, arr.length - 1, topK);
    }

    private static Integer helper(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }

        int left = start, right = end;

        int pivotVal = arr[start];
        while (left < right) {
            while (left < right && arr[right] >= pivotVal) right--;
            while (left < right && arr[left] <= pivotVal) left++;
            if (left < right) {
                int t = arr[left];
                arr[left] = arr[right];
                arr[right] = t;
            }
        }
        arr[start] = arr[left];
        arr[left] = pivotVal;

        if (left == arr.length - topK) return arr[left];
        Integer leftResult = helper(arr, start, left - 1, topK);
        if (leftResult != null) return leftResult;
        Integer rightResult = helper(arr, right + 1, end, topK);
        if (rightResult != null) return rightResult;
        return null;
    }
}


// JZOF II 043 同LC919
class CBTInserter {

    Deque<TreeNode49
            > lastButOneLayer;
    Deque<TreeNode49
            > lastLayer;
    TreeNode49
            root;

    public CBTInserter(TreeNode49
                               root) {
        this.root = root;
        Deque<TreeNode49
                > q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int qSize = q.size();
            Deque<TreeNode49
                    > thisLayer = new LinkedList<>();
            Deque<TreeNode49
                    > nextLayer = new LinkedList<>();
            boolean isLastButOneLayerFlag = false;
            for (int i = 0; i < qSize; i++) {
                TreeNode49
                        p = q.poll();
                thisLayer.offer(p);
                if (p.left == null && p.right == null) {
                    isLastButOneLayerFlag = true;
                } else if (p.right == null) {
                    isLastButOneLayerFlag = true;
                    nextLayer.offer(p.left);
                    q.offer(p.left);
                } else {
                    q.offer(p.left);
                    q.offer(p.right);
                    nextLayer.offer(p.left);
                    nextLayer.offer(p.right);
                }
            }
            lastButOneLayer = thisLayer;
            lastLayer = nextLayer;
            if (isLastButOneLayerFlag) break;
        }
        while (lastButOneLayer.peek().left != null && lastButOneLayer.peek().right != null) {
            lastButOneLayer.poll();
        }
    }

    public int insert(int v) {
        TreeNode49
                next = new TreeNode49
                (v);
        TreeNode49
                p = lastButOneLayer.peek();
        if (p.left == null) {
            p.left = next;
        } else if (p.right == null) {
            p.right = next;
            lastButOneLayer.poll();
        }
        lastLayer.offer(next);
        if (lastButOneLayer.isEmpty()) {
            lastButOneLayer = lastLayer;
            lastLayer = new LinkedList<>();
        }
        return p.val;
    }

    public TreeNode49
    get_root() {
        return root;
    }
}

class TreeNode49 {
    int val;
    TreeNode49 left;
    TreeNode49 right;

    TreeNode49() {
    }

    TreeNode49(int val) {
        this.val = val;
    }

    TreeNode49(int val, TreeNode49
            left, TreeNode49
                       right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class BIT49 {
    int[] tree;
    int n;

    public BIT49(int n) {
        this.n = n;
        tree = new int[n + 1];
    }

    public BIT49(int[] arr) {
        this.n = arr.length;
        tree = new int[n + 1];
        for (int i = 0; i < n; i++) {
            updateOneBased(i + 1, arr[i]);
        }
    }

    public void update(int idx, int delta) {
        updateOneBased(idx + 1, delta);
    }

    public void set(int idx, int val) {
        update(idx, val - get(idx));
    }

    public int get(int idx) {
        return sum(idx + 1) - sum(idx);
    }

    public int sumRange(int start, int end) {
        return sum(end + 1) - sum(start);
    }

    private int sum(int oneBaseIdx) {
        int result = 0;
        while (oneBaseIdx > 0) {
            result += tree[oneBaseIdx];
            oneBaseIdx -= lowbit(oneBaseIdx);
        }
        return result;
    }

    private void updateOneBased(int oneBaseIdx, int delta) {
        while (oneBaseIdx <= n) {
            tree[oneBaseIdx] += delta;
            oneBaseIdx += lowbit(oneBaseIdx);
        }
    }

    private int lowbit(int x) {
        return x & (-x);
    }
}

// LC528 **
class Lc528 {
    int[] acc;

    public Lc528(int[] w) {
        acc = new int[w.length + 1];
        for (int i = 1; i < acc.length; i++) {
            acc[i] = acc[i - 1] + w[i - 1];
        }
    }

    public int pickIndex() {
        int target = (int) (Math.random() * acc[acc.length - 1]);
        int lo = 0, hi = acc.length - 1;
        // 大于等于 target 的最小值
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (acc[mid] >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        while (acc[lo] == target) lo++;
        return lo - 1;

        //  大于 target 的第一个值
        //  while (lo < hi) {
        //      int mid = lo + (hi - lo) / 2;
        //      if (acc[mid] > target) {
        //          hi = mid;
        //      } else {
        //          lo = mid + 1;
        //      }
        //  }
        //  return lo-1;

    }
}