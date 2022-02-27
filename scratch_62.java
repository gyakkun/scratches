import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.maximumDifference(new int[]{7, 1, 5, 4}));
        System.out.println(s.maximumDifference(new int[]{9, 4, 3, 2}));
        System.out.println(s.maximumDifference(new int[]{1, 5, 2, 10}));
        System.out.println(s.maximumDifference(new int[]{1, 5, 114, 325, 6236, 42331, 565, 34, 12, 2, 10}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC553
    public String optimalDivision(int[] nums) {
        return nums.length == 1 ? "" + nums[0] : (nums.length == 2 ? "" + nums[0] + "/" + nums[1] : "" + nums[0] + "/(" + String.join("/", Arrays.stream(Arrays.copyOfRange(nums, 1, nums.length)).boxed().map(String::valueOf).collect(Collectors.toList())) + ")");
    }

    // LC2016
    public int maximumDifference(int[] nums) {
        int n = nums.length, result = -1;
        int[] minFromLeft = new int[n], maxFromRight = new int[n];

        int mflIdx = 0, mfrIdx = n - 1;
        for (int i = 0; i < n; i++) {
            int ri = n - i - 1;
            if (nums[i] < nums[mflIdx]) {
                mflIdx = i;
            }
            minFromLeft[i] = mflIdx;

            if (nums[ri] > nums[mfrIdx]) {
                mfrIdx = ri;
            }
            maxFromRight[ri] = mfrIdx;
        }

        for (int i = 0; i < n; i++) {
            if (maxFromRight[i] > minFromLeft[i] && nums[maxFromRight[i]] > nums[minFromLeft[i]]) {
                result = Math.max(result, nums[maxFromRight[i]] - nums[minFromLeft[i]]);
            }
        }

        return result;
    }

    // LC537
    public String complexNumberMultiply(String num1, String num2) {
        Pair<Integer, Integer> first = extract(num1), second = extract(num2);
        int a = first.getKey(), b = first.getValue(), c = second.getKey(), d = second.getValue();
        int real = a * c - b * d, unreal = a * d + b * c;
        return "" + real + "+" + unreal + "i";

    }

    private Pair<Integer, Integer> extract(String cpx) {
        String[] split1 = cpx.split("\\+");
        String[] split2 = split1[1].split("i");
        int a = Integer.valueOf(split1[0]), b = Integer.valueOf(split2[0]);
        return new Pair<Integer, Integer>(a, b);
    }

    // LC1994 **
    Long[][] memo = new Long[31][1025];
    Set<Integer> banSet = Set.of(4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28);
    int[] primeUnder30 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    Map<Integer, Integer> primeIdxMap;
    Map<Integer, Integer> freq;
    int[] maskToNum = new int[1024];
    int[] numToMask = new int[31];
    long mod = 1000000007;

    public int numberOfGoodSubsets(int[] nums) {
        primeIdxMap = new HashMap<>();
        for (int i = 0; i < primeUnder30.length; i++) primeIdxMap.put(primeUnder30[i], i);
        for (int i = 2; i <= 30; i++) {
            int mask = 0;
            if (banSet.contains(i)) continue;
            for (int j = 0; j < 10; j++) {
                if (i % primeUnder30[j] == 0) {
                    mask |= 1 << j;
                }
            }
            numToMask[i] = mask;
            maskToNum[mask] = i;
        }

        freq = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(e -> 1)));

        long one = 1;
        for (int i = 0; i < freq.getOrDefault(1, 0); i++) {
            one = (one * 2) % mod;
        }
        return (int) ((helper(2, 0) * one) % mod);
    }

    private long helper(int cur, int mask) {
        if (cur == 31) {
            return mask == 0 ? 0 : 1;
        }
        if (memo[cur][mask] != null) return memo[cur][mask];
        long result = 0;
        if (!banSet.contains(cur) && (numToMask[cur] & mask) == 0) {
            result += helper(cur + 1, mask | numToMask[cur]) * freq.getOrDefault(cur, 0);
        }
        result += helper(cur + 1, mask);
        return memo[cur][mask] = (result % mod);
    }

    // LC717
    public boolean isOneBitCharacter(int[] bits) {
        if (bits.length == 1) return true; // ends with 0
        return lc717Helper(bits, bits.length - 2);
    }

    public boolean lc717Helper(int[] arr, int endIdx) { // endIdx inclusive
        if (endIdx < 0) return false;
        if (endIdx == 0 && arr[0] == 0) return true;
        if (endIdx == 0 && arr[0] == 1) return false;

        if (arr[endIdx] == 0 && arr[endIdx - 1] == 0) { // 0,0
            return lc717Helper(arr, endIdx - 1);
        }
        if (arr[endIdx] == 0 && arr[endIdx - 1] == 1) { // 1,0
            return lc717Helper(arr, endIdx - 1) || lc717Helper(arr, endIdx - 2);
        }
        if (arr[endIdx] == 1 && arr[endIdx - 1] == 0) return false; // 0,1

        return lc717Helper(arr, endIdx - 2); // 1,1
    }

    // LC969 **
    public List<Integer> pancakeSort(int[] arr) {
        int n = arr.length;
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int endIdx = n - i - 1;
            if (endIdx == 0) break;
            // 找出 [0,endIdx] 中最大数的下标, 然后放到endIdx
            int max = Integer.MIN_VALUE;
            int maxIdx = -1;
            for (int j = 0; j <= endIdx; j++) {
                if (arr[j] > max) {
                    maxIdx = j;
                    max = arr[j];
                }
            }
            if (maxIdx == endIdx) continue;
            reverse(arr, maxIdx + 1);
            reverse(arr, endIdx + 1);
            result.add(maxIdx + 1);
            result.add(endIdx + 1);
        }
        return result;
    }

    private void reverse(int[] arr, int endExclusive) {
        if (endExclusive > arr.length || endExclusive <= 1) return;
        for (int i = 0; i < endExclusive / 2; i++) {
            int tmp = arr[i];
            arr[i] = arr[endExclusive - 1 - i];
            arr[endExclusive - 1 - i] = tmp;
        }
    }

    // LC1791
    public int findCenter(int[][] edges) {
        // return Arrays.stream(edges).flatMap(intArr -> Arrays.stream(intArr).boxed()).collect(Collectors.groupingBy(i -> i, Collectors.counting())).entrySet().stream().filter(i -> i.getValue() != 1).collect(Collectors.toList()).get(0).getKey();
        Set<Integer> s = new HashSet<>();
        for (int[] i : edges) {
            for (int j : i) {
                if (!s.add(j)) return j;
            }
        }
        return -1;
    }

    // 220217 HuanFang: Rate TBD Digit DP
    public double hfRate(long l, long r) {
        // 123 32112233 {1,2,3}
        // [0,100] 10,100, {0
        long count = r - l + 1;
        double total = (double) count * (double) count;
        double pos = 0d;
        for (int i = 0; i < 1024; i++) {
            Set<Integer> digit = new HashSet<>(10);
            for (int bit = 0; bit < 10; bit++) {
                if (((i >> bit) & 1) == 1) {
                    digit.add(bit);
                }
            }
            // digit 就是数字集合, 开始构造
            double tmp = 0d;
            if (!digit.contains(0)) {
                // 从 [R,L] 构造
            } else {
                // 避开前缀零
            }

            // 3321 123 A(3,3)
            // C(3,1)

        }

        return pos / total;
    }

    // LC688
    Double[][][] lc688Memo;
    int edgeLen = 0;
    int[][] knightDirections = new int[][]{{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {-1, -2}, {-2, -1}, {1, -2}, {2, -1}};

    public double knightProbability(int n, int k, int row, int column) {
        edgeLen = n;
        lc688Memo = new Double[n][n][k + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                lc688Memo[i][j][0] = 0d;
            }
        }
        lc688Memo[row][column][0] = 1d;
        double ariRate = 0d;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ariRate += lc688Helper(i, j, k);
            }
        }
        return ariRate;
    }

    private double lc688Helper(int r, int c, int step) {
        if (step < 0) return 0d;
        if (lc688Memo[r][c][step] != null) return lc688Memo[r][c][step];
        double result = 0d;
        for (int[] d : knightDirections) {
            int pr = r - d[0], pc = c - d[1];
            if (pr >= 0 && pr < edgeLen && pc >= 0 && pc < edgeLen) {
                result += 0.125d * lc688Helper(pr, pc, step - 1);
            }
        }
        return lc688Memo[r][c][step] = result;
    }


    // LC1380
    public List<Integer> luckyNumbers(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[] rowMin = new int[m], colMax = new int[n];
        List<Integer> result = new ArrayList<>();
        Arrays.fill(rowMin, Integer.MAX_VALUE);
        Arrays.fill(colMax, Integer.MIN_VALUE);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowMin[i] = Math.min(rowMin[i], matrix[i][j]);
                colMax[j] = Math.max(colMax[j], matrix[i][j]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rowMin[i] == colMax[j]) {
                    result.add(matrix[i][j]);
                }
            }
        }
        return result;
    }

    // LC1020
    public int numEnclaves(int[][] grid) {
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int m = grid.length, n = grid[0].length, allUnit = 0;
        BitSet edgeUnit = new BitSet(m * n);
        Deque<Integer> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    allUnit++;
                    if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                        q.offer(i * n + j);
                    }
                }
            }
        }
        while (!q.isEmpty()) {
            int p = q.poll();
            if (edgeUnit.get(p)) continue;
            edgeUnit.set(p);
            int r = p / n, c = p % n;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 1 && !edgeUnit.get(nr * n + nc)) {
                    q.offer(nr * n + nc);
                }
            }
        }
        return allUnit - edgeUnit.cardinality();
    }

    // LC1984
    public int minimumDifference(int[] nums, int k) {
        int n = nums.length, result = Integer.MAX_VALUE;
        Arrays.sort(nums);
        for (int i = 0; i <= n - k; i++) {
            result = Math.min(result, nums[i + k - 1] - nums[i]);
        }
        return result;
    }

    // LC1447
    int[][] gcdCache = new int[101][101];

    public List<String> simplifiedFractions(int n) {
        List<String> result = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                if (gcd(j, i) == 1) {
                    result.add("" + j + "/" + i);
                }
            }
        }
        return result;
    }

    private int gcd(int a, int b) {
        if (gcdCache[a][b] != 0) return gcdCache[a][b];
        if (gcdCache[b][a] != 0) return gcdCache[b][a];
        return b == 0 ? (gcdCache[a][b] = (gcdCache[b][a] = a)) : gcd(b, a % b);
    }

    // LC2006
    public int countKDifference(int[] nums, int k) {
        int result = 0;
        Map<Integer, Integer> m = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            result += m.getOrDefault(nums[i] - k, 0);
            result += m.getOrDefault(nums[i] + k, 0);
            m.put(nums[i], m.getOrDefault(nums[i], 0) + 1);
        }
        return result;
    }

    // LC1405
    public String longestDiverseString(int a, int b, int c) {
        StringBuffer sb = new StringBuffer();
        char last = '\0';
        int lastLen = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(i -> -i[1]));
        pq.offer(new int[]{'a', a});
        pq.offer(new int[]{'b', b});
        pq.offer(new int[]{'c', c});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            if (last == p[0] && lastLen >= 2) {
                if (pq.isEmpty()) {
                    return "";
                }
                int[] sub = pq.poll();
                sb.append((char) sub[0]);
                lastLen = 1;
                last = (char) sub[0];
                if (sub[1] - 1 > 0) {
                    pq.offer(new int[]{sub[0], sub[1] - 1});
                }
                pq.offer(p);
            } else {
                sb.append((char) p[0]);
                last = (char) p[0];
                if (sb.length() >= 2 && (char) p[1] != sb.charAt(sb.length() - 2)) {
                    lastLen = 1;
                } else {
                    lastLen++;
                }
                if (p[1] - 1 > 0) {
                    pq.offer(new int[]{p[0], p[1] - 1});
                }
            }
        }
        return sb.toString();
    }

    // LC1748
    public int sumOfUnique(int[] nums) {
        return Arrays.stream(nums).boxed().collect(Collectors.groupingBy(i -> i, Collectors.counting())).entrySet().stream().filter(i -> i.getValue().equals(1l)).mapToInt(i -> i.getKey()).sum();
    }

    // LC1219
    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    int[][] grid;
    boolean[][] visited;

    public int getMaximumGold(int[][] grid) {
        this.grid = grid;
        int m = grid.length, n = grid[0].length, result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0) {
                    visited = new boolean[m][n];
                    result = Math.max(result, lc1219Helper(i, j));
                }
            }
        }
        return result;
    }

    private int lc1219Helper(int r, int c) {
        if (visited[r][c]) return 0;
        visited[r][c] = true;
        int cur = grid[r][c];
        int next = 0;
        for (int[] d : knightDirections) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && grid[nr][nc] != 0 && !visited[nr][nc]) {
                next = Math.max(next, lc1219Helper(nr, nc));
            }
        }
        visited[r][c] = false;
        return cur + next;
    }

    // LC1725
    public int countGoodRectangles(int[][] rectangles) {
        return Arrays.stream(rectangles).collect(Collectors.groupingBy(i -> Math.min(i[0], i[1]), Collectors.counting())).entrySet().stream().max(Comparator.comparingInt(i -> i.getKey())).get().getValue().intValue();
    }

    // LC1414
    public int findMinFibonacciNumbers(int k) {
        int result = 0;
        while (k != 0) {
            int h = lc1414Helper(k);
            if (k != -1) {
                result++;
                k -= h;
            }
        }
        return result;
    }

    // 二分 找小于等于的最大值
    public int lc1414Helper(int n) {
        int l = 0, h = 46;
        while (l < h) {
            int mid = l + (h - l + 1) / 2;
            if (fib(mid) <= n) {
                l = mid;
            } else {
                h = mid - 1;
            }
        }
        if (fib(l) > n) return -1;
        return fib(l);
    }

    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;
        return fib(n, 2, 1, 1);
    }

    public int fib(int targetIdx, int curIdx, int curVal, int prevVal) {
        if (targetIdx == curIdx) return curVal;
        return fib(targetIdx, curIdx + 1, curVal + prevVal, curVal);
    }

    // LC2000
    public String reversePrefix(String word, char ch) {
        if (!word.contains("" + ch)) return word;
        return new StringBuffer(word.substring(0, word.indexOf(ch) + 1)).reverse() + word.substring(word.indexOf(ch) + 1);
    }

    // LC1763
    public String longestNiceSubstring(String s) {
        String result = "";
        char[] ca = s.toCharArray();
        for (int left = 0; left < ca.length; left++) {
            inner:
            for (int right = left + 1; right <= ca.length; right++) {
                int[] freq = new int[128];
                for (int i = left; i < right; i++) {
                    freq[ca[i]]++;
                }
                for (int i = 0; i < 26; i++) {
                    if ((freq['a' + i] > 0 && freq['A' + i] == 0) || (freq['a' + i] == 0 && freq['A' + i] > 0)) {
                        continue inner;
                    }
                }
                if (right - left > result.length()) {
                    result = s.substring(left, right);
                }
            }
        }
        return result;
    }

    // LC884
    public String[] uncommonFromSentences(String s1, String s2) {
        List<String> result = new ArrayList<>();
        Map<String, Integer> m1 = new HashMap<>(), m2 = new HashMap<>();
        for (String w : s1.split(" ")) {
            m1.put(w, m1.getOrDefault(w, 0) + 1);
        }
        for (String w : s2.split(" ")) {
            m2.put(w, m2.getOrDefault(w, 0) + 1);
        }
        for (String w : m1.keySet()) {
            if (m1.get(w) == 1 && !m2.containsKey(w)) {
                result.add(w);
            }
        }
        for (String w : m2.keySet()) {
            if (m2.get(w) == 1 && !m1.containsKey(w)) {
                result.add(w);
            }
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1765
    public int[][] highestPeak(int[][] isWater) {
        int m = isWater.length, n = isWater[0].length;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}}, result = new int[m][n];
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isWater[i][j] == 1) {
                    q.offer(new int[]{i, j, 0});
                }
            }
        }
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int r = p[0], c = p[1], h = p[2];
            if (visited[r][c]) continue;
            visited[r][c] = true;
            if (isWater[r][c] == 1) {
                h = 0;
            }
            result[r][c] = h;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc]) {
                    q.offer(new int[]{nr, nc, h + 1});
                }
            }
        }
        return result;
    }

    // LC1996 ** Answer
    public int numberOfWeakCharacters(int[][] properties) {
        Arrays.sort(properties, (o1, o2) -> o1[0] == o2[0] ? (o1[1] - o2[1]) : (o2[0] - o1[0]));
        int maxDef = 0;
        int ans = 0;
        for (int[] p : properties) {
            if (p[1] < maxDef) {
                ans++;
            } else {
                maxDef = p[1];
            }
        }
        return ans;
    }


    // LC2047 WA
    public int countValidWords(String sentence) {
        int result = 0;
        Set<Character> punc = Set.of('!', '.', ',');
        Function<String, Boolean> isValid = new Function<String, Boolean>() {
            @Override
            public Boolean apply(String s) {
                if (s.length() == 0) return false;
                Stream<Character> cs = s.chars().mapToObj(i -> (char) i);
                if (cs.anyMatch(Character::isDigit)) return false;
                cs = s.chars().mapToObj(i -> (char) i);
                long puncCount = cs.filter(punc::contains).count();
                if (puncCount > 1l) return false;
                if (puncCount == 1l && !punc.contains(s.charAt(s.length() - 1))) return false;
                cs = s.chars().mapToObj(i -> (char) i);
                long hyphenCount = cs.filter(i -> '-' == i).count();
                if (hyphenCount > 1l) return false;

                if (hyphenCount == 1) {
                    String[] split = s.split("-");
                    if (split.length != 2) return false;
                }
                return true;
            }
        };
        for (String token : sentence.split(" ")) {
            if (isValid.apply(token)) result++;
        }
        return result;
    }

    // LC1345
    public int minJumps(int[] arr) {
        Map<Integer, Set<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            m.putIfAbsent(arr[i], new HashSet<>());
            Set<Integer> ts = m.get(arr[i]);
            ts.add(i);
        }
        Deque<Integer> q = new LinkedList<>();
        q.offer(0);
        BitSet visited = new BitSet(arr.length);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int p = q.poll();
                int val = arr[p], idx = p;
                if (visited.get(idx)) continue;
                visited.set(idx);

                if (idx == arr.length - 1) return layer;

                // i + 1
                if (idx + 1 < arr.length && !visited.get(idx + 1)) {
                    q.add(idx + 1);
                }

                // i-1
                if (idx - 1 >= 0 && !visited.get(idx - 1)) {
                    q.add(idx - 1);
                }

                // same value
                for (int smi : m.getOrDefault(val, new HashSet<>())) {
                    if (!visited.get(smi)) {
                        q.add(smi);
                    }
                }
                m.remove(val);
            }
        }
        return -1;
    }

    // LC219
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, TreeSet<Integer>> m = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            m.putIfAbsent(nums[i], new TreeSet<>());
            TreeSet<Integer> ts = m.get(nums[i]);
            int lb = i - k;
            int hb = i + k;
            Integer hr = ts.higher(lb);
            Integer lw = ts.lower(hb);
            if (hr != null && Math.abs(i - hr) <= k) {
                System.err.println(ts.ceiling(hr));
                return true;
            }
            if (lw != null && Math.abs(i - lw) <= k) {
                System.err.println(ts.floor(lw));
                return true;
            }
            ts.add(i);
        }
        return false;
    }

    // LC1220
    Long[][] lc1220Memo;

    public int countVowelPermutation(int n) {
        lc1220Memo = new Long[n + 1][6];
        long result = 0;
        for (int i = 0; i < 5; i++) {
            result = (result + lc1220Helper(n - 1, i)) % mod;
        }
        return (int) result;
    }

    private long lc1220Helper(int remainLetters, int currentLetterIdx) {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        if (remainLetters == 0) return 1;
        if (lc1220Memo[remainLetters][currentLetterIdx] != null)
            return lc1220Memo[remainLetters][currentLetterIdx] % mod;
        switch (currentLetterIdx) {
            case 0: // a
                return lc1220Memo[remainLetters][currentLetterIdx]
                        = lc1220Helper(remainLetters - 1, 1);
            case 1: // e
                return lc1220Memo[remainLetters][currentLetterIdx]
                        = (lc1220Helper(remainLetters - 1, 0)
                        + lc1220Helper(remainLetters - 1, 2)) % mod;
            case 2:
                return lc1220Memo[remainLetters][currentLetterIdx]
                        = (lc1220Helper(remainLetters - 1, 0)
                        + lc1220Helper(remainLetters - 1, 1)
                        + lc1220Helper(remainLetters - 1, 3)
                        + lc1220Helper(remainLetters - 1, 4)) % mod;
            case 3:
                return lc1220Memo[remainLetters][currentLetterIdx]
                        = (lc1220Helper(remainLetters - 1, 2)
                        + lc1220Helper(remainLetters - 1, 4));
            case 4:
                return lc1220Memo[remainLetters][currentLetterIdx]
                        = lc1220Helper(remainLetters - 1, 0) % mod;
        }
        return 0;
    }

    // LC1036
    public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
        if (blocked.length < 2) return true;
        final int bound = 1000000;
        final int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        TreeSet<Integer> rowSet = new TreeSet<>(), colSet = new TreeSet<>();
        List<int[]> allPoints = new ArrayList<int[]>(Arrays.stream(blocked).collect(Collectors.toList())) {{
            add(source);
            add(target);
        }};
        for (int[] b : allPoints) {
            rowSet.add(b[0]);
            colSet.add(b[1]);
        }

        int rid = rowSet.first() == 0 ? 0 : 1, cid = colSet.first() == 0 ? 0 : 1; // bound it from 0 to 999999
        Iterator<Integer> rit = rowSet.iterator(), cit = colSet.iterator();
        Map<Integer, Integer> rowMap = new HashMap<>(), colMap = new HashMap<>();
        int pr = -1, pc = -1;
        if (rit.hasNext()) rowMap.put((pr = rit.next()), rid);
        if (cit.hasNext()) colMap.put((pc = cit.next()), cid);
        while (rit.hasNext()) {
            int nr = rit.next();
            rid += (nr == pr + 1) ? 1 : 2;
            rowMap.put(nr, rid);
            pr = nr;
        }
        while (cit.hasNext()) {
            int nc = cit.next();
            cid += (nc == pc + 1) ? 1 : 2;
            colMap.put(nc, cid);
            pc = nc;
        }
        int rBound = (pr == bound - 1) ? rid : rid + 1;
        int cBound = (pc == bound - 1) ? cid : cid + 1;
        boolean[][] mtx = new boolean[rBound + 1][cBound + 1]; // use as visited[][] too
        for (int[] b : blocked) {
            mtx[rowMap.get(b[0])][colMap.get(b[1])] = true;
        }
        int sr = rowMap.get(source[0]), sc = colMap.get(source[1]), tr = rowMap.get(target[0]), tc = colMap.get(target[1]);
        Deque<int[]> q = new LinkedList<>();
        q.offer(new int[]{sr, sc});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            if (p[0] == tr && p[1] == tc) return true;
            if (mtx[p[0]][p[1]]) continue;
            mtx[p[0]][p[1]] = true;
            for (int[] d : dir) {
                int inr = p[0] + d[0], inc = p[1] + d[1];
                if (inr >= 0 && inr <= rBound && inc >= 0 && inc <= cBound && !mtx[inr][inc]) {
                    q.offer(new int[]{inr, inc});
                }
            }
        }
        return false;
    }

    // LC2022
    public int[][] construct2DArray(int[] original, int m, int n) {
        int len = original.length;
        if (len != m * n) return new int[][]{};
        int[][] result = new int[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(original, i * n, result[i], 0, n);
        }
        return result;
    }

    // LC472
    Trie lc472Trie = new Trie();

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Arrays.sort(words, Comparator.comparingInt(o -> o.length()));
        List<String> result = new ArrayList<>();
        for (String w : words) {
            if (w.length() == 0) continue;
            if (lc472Helper(w, 0)) {
                result.add(w);
            } else {
                lc472Trie.addWord(w);
            }
        }
        return result;
    }

    private boolean lc472Helper(String word, int startIdx) {
        if (word.length() == startIdx) return true;
        Trie.TrieNode cur = lc472Trie.root;
        for (int i = startIdx; i < word.length(); i++) {
            char c = word.charAt(i);
            if (cur.children[c] == null) return false;
            cur = cur.children[c];
            if (cur.end > 0) {
                if (lc472Helper(word, i + 1)) return true;
            }
        }
        return false;
    }

    // LC902 **
    public int atMostNGivenDigitSet(String[] digits, int n) {
        String nStr = String.valueOf(n);
        char[] ca = nStr.toCharArray();
        int k = ca.length;
        int[] dp = new int[k + 1];
        dp[k] = 1;

        for (int i = k - 1; i >= 0; i--) {
            int digit = ca[i] - '0';
            for (String dStr : digits) {
                int d = Integer.valueOf(dStr);
                if (d < digit) {
                    dp[i] += Math.pow(digits.length, /*剩下的位数可以随便选*/ k - i - 1);
                } else if (d == digit) {
                    dp[i] += dp[i + 1];
                }
            }
        }

        for (int i = 1; i < k; i++) {
            dp[0] += Math.pow(digits.length, i);
        }
        return dp[0];
    }

    // LC825
    public int numFriendRequests(int[] ages) {
        // 以下情况 X 不会向 Y 发送好友请求
        // age[y] <= 0.5 * age[x] + 7
        // age[y] > age[x]
        // age[y] > 100 && age[x] < 100
        int result = 0;
        Arrays.sort(ages);
        int idx = 0, n = ages.length;
        while (idx < n) {
            int age = ages[idx], same = 1;
            while (idx + 1 < n && ages[idx + 1] == ages[idx]) {
                same++;
                idx++;
            }

            // 找到 大于 0.5 * age + 7 的最小下标
            int lo = 0, hi = idx - 1, target = (int) (0.5 * age + 7);
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (ages[mid] > target) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            int count = 0;
            if (ages[lo] > target) {
                count = idx - lo;
            }
            result += same * count;
            idx++;
        }
        return result;
    }

    // LC1609
    public boolean isEvenOddTree(TreeNode root) {
        int layer = -1;
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            if (layer % 2 == 0) { // 偶数层
                int prev = -1;
                for (int i = 0; i < qs; i++) {
                    TreeNode p = q.poll();
                    int v = p.val;
                    if (v % 2 == 0) return false;
                    if (i > 0) {
                        if (v <= prev) return false;
                    }
                    prev = v;
                    if (p.left != null) q.offer(p.left);
                    if (p.right != null) q.offer(p.right);
                }
            } else { // 奇数层
                int prev = -1;
                for (int i = 0; i < qs; i++) {
                    TreeNode p = q.poll();
                    int v = p.val;
                    if (v % 2 == 1) return false;
                    if (i > 0) {
                        if (v >= prev) return false;
                    }
                    prev = v;
                    if (p.left != null) q.offer(p.left);
                    if (p.right != null) q.offer(p.right);
                }
            }
        }
        return true;
    }


    // LC1044
    final long lc1044Mod = 1000000007l;
    final long lc1044Base1 = 29;
    final long lc1044Base2 = 31;
    String lc1044S;

    public String longestDupSubstring(String s) {
        this.lc1044S = s;
        char[] ca = s.toCharArray();
        int lo = 0, hi = s.length() - 1;
        String result = "";
        String tmp = null;
        while (lo < hi) { // 找最大值
            int mid = lo + (hi - lo + 1) / 2;
            if ((tmp = lc1044Helper(ca, mid)) != null) {
                result = tmp;
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }

    private String lc1044Helper(char[] ca, int len) {
        Set<Integer> m1 = new HashSet<>();
        Set<Integer> m2 = new HashSet<>();
        long hash1 = 0, hash2 = 0, accu1 = 1, accu2 = 1;
        for (int i = 0; i < len; i++) {
            hash1 *= lc1044Base1;
            hash1 %= lc1044Mod;
            hash2 *= lc1044Base2;
            hash2 %= lc1044Mod;
            hash1 += ca[i];
            hash1 %= lc1044Mod;
            hash2 += ca[i];
            hash2 %= lc1044Mod;
            accu1 *= lc1044Base1;
            accu1 %= lc1044Mod;
            accu2 *= lc1044Base2;
            accu2 %= lc1044Mod;
        }
        m1.add((int) hash1);
        m2.add((int) hash2);
        for (int i = len; i < ca.length; i++) {
            String victim = lc1044S.substring(i - len + 1, i + 1);
            hash1 = (((hash1 * lc1044Base1 - accu1 * (ca[i - len])) % lc1044Mod) + lc1044Mod + ca[i]) % lc1044Mod;
            hash2 = (((hash2 * lc1044Base2 - accu2 * (ca[i - len])) % lc1044Mod) + lc1044Mod + ca[i]) % lc1044Mod;
            if (m1.contains((int) hash1) && m2.contains((int) hash2)) {
                return victim;
            }
            m1.add((int) hash1);
            m2.add((int) hash2);
        }
        return null;
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

class Trie {
    TrieNode root = new TrieNode();

    public boolean search(String query) {
        TrieNode result = getNode(query);
        return result != null && result.end > 0;
    }

    public boolean beginWith(String query) {
        return getNode(query) != null;
    }

    public void addWord(String word) {
        // if (getNode(word) != null) return;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c] == null) {
                cur.children[c] = new TrieNode();
            }
            cur = cur.children[c];
            cur.path++;
        }
        cur.end++;
    }

    public boolean removeWord(String word) {
        if (getNode(word) == null) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children[c].path-- == 1) {
                cur.children[c] = null;
                return true;
            }
            cur = cur.children[c];
        }
        cur.end--;
        return true;
    }


    private TrieNode getNode(String query) {
        TrieNode cur = root;
        for (char c : query.toCharArray()) {
            if (cur.children[c] == null) return null;
            cur = cur.children[c];
        }
        return cur;
    }


    class TrieNode {
        TrieNode[] children = new TrieNode[128];
        int end = 0;
        int path = 0;
    }
}

// LC2034
class StockPrice {
    TreeMap<Integer, Integer> timePriceMap = new TreeMap<>();
    TreeMap<Integer, Set<Integer>> priceTimeMap = new TreeMap<>();

    public StockPrice() {

    }

    public void update(int timestamp, int price) {
        if (timePriceMap.containsKey(timestamp)) {
            int priceToCorrect = timePriceMap.get(timestamp);
            priceTimeMap.get(priceToCorrect).remove(timestamp);
            if (priceTimeMap.get(priceToCorrect).size() == 0) {
                priceTimeMap.remove(priceToCorrect);
            }
        }
        timePriceMap.put(timestamp, price);
        priceTimeMap.putIfAbsent(price, new HashSet<>());
        priceTimeMap.get(price).add(timestamp);
    }

    public int current() {
        return timePriceMap.lastEntry().getValue();
    }

    public int maximum() {
        return priceTimeMap.lastKey();
    }

    public int minimum() {
        return priceTimeMap.firstKey();
    }
}

// LC2013
class DetectSquares {

    Map<Integer, Map<Integer, Integer>> xyMap = new HashMap<>();

    public DetectSquares() {

    }

    public void add(int[] point) {
        xyMap.putIfAbsent(point[0], new HashMap<>());
        Map<Integer, Integer> yPointsCount = xyMap.get(point[0]);
        yPointsCount.put(point[1], yPointsCount.getOrDefault(point[1], 0) + 1);
    }

    public int count(int[] point) {
        // 比较同一x坐标/y坐标上 [独特点(指重复位置的点算一个)] 的个数, 挑选少的集合来进行遍历
        int x = point[0], y = point[1], result = 0;
        if (!xyMap.containsKey(x)) return 0;
        Map<Integer, Integer> yPoints = xyMap.get(x);
        int c0 = 1;
        for (Map.Entry<Integer, Integer> e : yPoints.entrySet()) {
            int thisY = e.getKey();
            if (thisY == y) continue;
            int c1 = e.getValue();
            int distance = y - thisY;
            int absDistance = Math.abs(distance);

            for (int sideX : new int[]{x - absDistance, x + absDistance}) {
                if (xyMap.containsKey(sideX) && xyMap.get(sideX).containsKey(y)) {
                    int c2 = xyMap.get(sideX).get(y);
                    // 找左下角
                    if (xyMap.containsKey(sideX) && xyMap.get(sideX).containsKey(thisY)) {
                        int c3 = xyMap.get(sideX).get(thisY);
                        result += c0 * c1 * c2 * c3;
                    }
                }
            }
        }
        return result;
    }
}