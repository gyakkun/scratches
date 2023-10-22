import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Solution {
    public static void main(String[] args) {
        var s = new Solution();
        long timing = System.currentTimeMillis();
        System.err.println(s.maxSatisfaction(new int[]{-1, -8, 0, 5, -9}));
        timing = System.currentTimeMillis() - timing;
        System.err.println(timing + "ms");
    }

    // LC1402 Hard ** 题解 
    // https://leetcode.cn/u/endlesscheng/
    public int maxSatisfaction(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int f = 0; // f(0) = 0
        int s = 0;
        for (int i = satisfaction.length - 1; i >= 0; i--) {
            s += satisfaction[i];
            if (s <= 0) { // 后面不可能找到更大的 f(k)
                break;
            }
            f += s; // f(k) = f(k-1) + s
        }
        return f;
    }

    // LC1665 Hard **
    public int minimumEffort(int[][] tasks) {
        Arrays.sort(tasks, (i, j) -> i[1] - i[0] - (j[1] - j[0]));
        int res = Integer.MIN_VALUE;
        for (int[] t : tasks) res = Math.max(res + t[0], t[1]);
        return res;
    }


    // LC2136 Hard **
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        int n = plantTime.length;
        Integer[] id = new Integer[n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
        }
        Arrays.sort(id, (i, j) -> growTime[j] - growTime[i]);
        int res = Integer.MIN_VALUE, plantingDays = 0;
        for (int i : id) {
            plantingDays += plantTime[i];
            res = Math.max(res, plantingDays + growTime[i]);
        }
        return res;
    }

    // LC2438
    public int[] productQueries(int n, int[][] queries) {
        List<Integer> binaryArr = toRadix(n, 2);
        List<Integer> twoPowers = new ArrayList<>();
        for (int i = 0; i < binaryArr.size(); i++) {
            if (binaryArr.get(i) == 0) continue;
            int rank = i;
            twoPowers.add(rank);
        }
        twoPowers.sort(Comparator.naturalOrder());
        int[] prefix = new int[twoPowers.size() + 1];
        for (int i = 1; i < prefix.length; i++) {
            prefix[i] = prefix[i - 1] + twoPowers.get(i - 1);
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            // inclusive
            int left = Math.max(0, queries[i][0]);
            int right = Math.min(twoPowers.size() - 1, queries[i][1]);
            int pow = prefix[right + 1] - prefix[left];
            res[i] = (int) quickPower(2, pow);
        }
        return res;
    }

    long mod = 1000000007L;

    private long quickPower(long num, long pow) {
        if (pow == 0L) return 1;
        long res = quickPower(num, pow / 2) % mod;
        res = (pow % 2 == 1 ? res * res * num : res * res) % mod;
        return res;
    }

    // LC2412 Hard **
    public long minimumMoney(int[][] transactions) {
        List<int[]> earnTrans = new ArrayList<>(), lossTrans = new ArrayList<>();
        for (int[] i : transactions) {
            if (i[1] >= i[0]) {
                earnTrans.add(i);
            } else {
                lossTrans.add(i);
            }
        }
        // 困难模式: 先做亏钱的, 从回报最少的开始
        // 然后做赚钱的, 从投入最大的开始
        earnTrans.sort(Comparator.comparingInt(i -> -i[0]));
        lossTrans.sort(Comparator.comparingInt(i -> i[1]));
        long mostLoss = Long.MAX_VALUE;
        long curMoney = 0;
        for (int[] i : lossTrans) {
            curMoney -= i[0];
            mostLoss = Math.min(mostLoss, curMoney);
            curMoney += i[1];
            mostLoss = Math.min(mostLoss, curMoney);
        }
        for (int[] i : earnTrans) {
            curMoney -= i[0];
            mostLoss = Math.min(mostLoss, curMoney);
            curMoney += i[1];
            mostLoss = Math.min(mostLoss, curMoney);
        }
        return -mostLoss;
    }

    // LC2396
    public boolean isStrictlyPalindromic(int n) {
        for (int rad = 2; rad < n - 1; rad++) {
            if (!isPalinDromicList(toRadix(n, rad))) return false;
        }
        return true;
    }

    private boolean isPalinDromicList(List<Integer> arr) {
        int half = arr.size() / 2;
        int len = arr.size();
        for (int i = 0; i < half; i++) {
            if (arr.get(i) != arr.get(len - 1 - i)) return false;
        }
        return true;
    }

    private List<Integer> toRadix(int num, int radix) {
        int left = num;
        List<Integer> res = new ArrayList<>();
        while (left > 0) {
            int remain = left % radix;
            left = left / radix;
            res.add(remain);
        }
        return res;
    }

    // LC2410
    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        int np = players.length, nt = trainers.length;
        int res = 0;
        if (np >= nt) {
            // 运动员更多, 教练员应该尽量匹配能力值高的
            Map<Integer, Integer> freq = Arrays.stream(players).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
            TreeMap<Integer, Integer> tm = new TreeMap<>(freq);
            Arrays.sort(trainers);
            for (int i : trainers) {
                Integer candidateVal = tm.floorKey(i); // 小于等于教练能力值的第一个运动员
                if (candidateVal == null) continue;
                int origCount = tm.get(candidateVal);
                if (origCount == 1) tm.remove(candidateVal);
                else tm.put(candidateVal, origCount - 1);
                res++;
            }
        } else {
            // 教练员更多, 运动员应该尽量匹配能力值高的
            Map<Integer, Integer> freq = Arrays.stream(trainers).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
            TreeMap<Integer, Integer> tm = new TreeMap<>(freq);
            Arrays.sort(players);
            for (int i : players) {
                Integer candidateVal = tm.ceilingKey(i); // 大于等于运动员能力值的第一个教练
                if (candidateVal == null) continue;
                int origCount = tm.get(candidateVal);
                if (origCount == 1) tm.remove(candidateVal);
                else tm.put(candidateVal, origCount - 1);
                res++;
            }
        }
        return res;
    }

    // LC2367
    public int arithmeticTriplets(int[] nums, int diff) {
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            idxMap.put(nums[i], i);
        }
        int res = 0;
        for (int i : nums) {
            int nextOne = i + diff;
            int nextTwo = nextOne + diff;
            if (idxMap.containsKey(nextOne) && idxMap.containsKey(nextTwo)) {
                res++;
            }
        }
        return res;
    }

    // LC2344 ** Hard
    public int minOperations(int[] nums, int[] numsDivide) {
        int gcd = gcd(numsDivide);
        Map<Integer, Integer> freqImMap = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        TreeMap<Integer, Integer> freqTm = new TreeMap<>(freqImMap);
        int counter = 0;
        for (var i : freqTm.entrySet()) {
            int k = i.getKey(), freq = i.getValue();
            if (k == gcd) return counter;
            if (k > gcd) return -1;
            // k < gcd
            if (gcd % k == 0) return counter;
            counter += freq;
        }
        return -1;
    }

    private int gcd(int[] numsDivide) {
        List<Integer> reverseSortedUniqNumsToDivide = Arrays.stream(numsDivide).boxed()
                .distinct()
                .sorted(Comparator.reverseOrder())
                .toList();
        Optional<Integer> reduce = reverseSortedUniqNumsToDivide.stream()
                .reduce(this::gcd);
        return reduce.get();
    }

    private int gcd(int a, int b) {
        return a % b == 0 ? b : gcd(b, a % b);
    }


    // LC2337
    public boolean canChange(String start, String target) {
        if (!start.replaceAll("_", "").equals(target.replaceAll("_", ""))) return false;
        // From left to right
        char[] startCarr = start.toCharArray(), targetCarr = target.toCharArray();
        int n = start.length();
        int startSpaceCounter = 0, startLetterCounter = 0;
        int targetSpaceCounter = 0, targetLetterCounter = 0;
        char wantedChar = 'L', unwantedChar = 'R';
        for (int i = 0; i < n; i++) {
            char s = startCarr[i], t = targetCarr[i];
            if (s == unwantedChar || t == unwantedChar) {
                if (Character.isLetter(s) && Character.isLetter(t) && s != t) {
                    return false;
                }
                startSpaceCounter = 0;
                startLetterCounter = 0;
                targetSpaceCounter = 0;
                targetLetterCounter = 0;
                continue;
            }
            if (s == '_') startSpaceCounter++;
            if (t == '_') targetSpaceCounter++;
            if (s == wantedChar) startLetterCounter++;
            if (t == wantedChar) targetLetterCounter++;
            if (!judge(startSpaceCounter, startLetterCounter, targetSpaceCounter, targetLetterCounter)) {
                return false;
            }
        }
        wantedChar = 'R';
        unwantedChar = 'L';
        for (int i = n - 1; i >= 0; i--) {
            char s = startCarr[i], t = targetCarr[i];
            if (s == unwantedChar || t == unwantedChar) {
                if (Character.isLetter(s) && Character.isLetter(t) && s != t) {
                    return false;
                }
                startSpaceCounter = 0;
                startLetterCounter = 0;
                targetSpaceCounter = 0;
                targetLetterCounter = 0;
                continue;
            }
            if (s == '_') startSpaceCounter++;
            if (t == '_') targetSpaceCounter++;
            if (s == wantedChar) startLetterCounter++;
            if (t == wantedChar) targetLetterCounter++;
            if (!judge(startSpaceCounter, startLetterCounter, targetSpaceCounter, targetLetterCounter)) {
                return false;
            }
        }
        return true;
    }

    private boolean judge(int startSpaceCounter, int startLetterCounter, int targetSpaceCounter, int targetLetterCounter) {
        return startSpaceCounter >= targetSpaceCounter && startLetterCounter <= targetLetterCounter;
    }

    private static ListNode prepareNode(int[] arr) {
        ListNode root = new ListNode(arr[0]);
        ListNode pivot = root;
        for (int i = 1; i < arr.length; i++) {
            pivot.next = new ListNode(arr[i]);
            pivot = pivot.next;
        }
        return root;
    }


    // LC2326
    int[][] directions = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] res = new int[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) res[i][j] = -1;
        BitSet bs = new BitSet(m * n);
        int dir = 0;
        ListNode node = head;
        int curX = 0, curY = 0;
        while (node != null) {
            res[curX][curY] = node.val;
            bs.set(curX * n + curY);
            if (node.next == null) break;
            int[] next = nextStep(curX, curY, dir, m, n, bs);
            curX = next[0];
            curY = next[1];
            dir = next[2];
            node = node.next;
        }
        return res;
    }

    private int[] nextStep(int curX, int curY, int dir, int m, int n, BitSet bs) {
        boolean shouldTurn = false;
        int nextDir = dir;
        int nextX = curX + directions[nextDir][0], nextY = curY + directions[nextDir][1];
        int nextRank = nextX * n + nextY;
        shouldTurn = isShouldTurn(m, n, bs, nextX, nextY, nextRank);
        int loopCount = 0;
        while (shouldTurn) {
            nextDir = (dir + 1) % 4;
            nextX = curX + directions[nextDir][0];
            nextY = curY + directions[nextDir][1];
            nextRank = nextX * n + nextY;
            shouldTurn = isShouldTurn(m, n, bs, nextX, nextY, nextRank);
            if (shouldTurn && ++loopCount >= 4) {
                throw new IllegalStateException("Next step is not available! next x: %d, next y: %d. m: %d, n: %d".formatted(nextX, nextY, m, n));
            }
        }
        return new int[]{nextX, nextY, nextDir};
    }

    private static boolean isShouldTurn(int m, int n, BitSet bs, int nextX, int nextY, int nextRank) {
        return (nextX < 0 || nextX >= m || nextY < 0 || nextY >= n || bs.get(nextRank));
    }


    // LC2316
    public long countPairs(int n, int[][] edges) {
        var dsu = new DSU();
        for (int i = 0; i < n; i++) dsu.add(i);
        for (int[] i : edges) {
            dsu.merge(i[0], i[1]);
        }
        List<Integer> groups = dsu.getAllGroups().values().stream().map(Set::size).toList();
        long sum = 0L;
        for (int i : groups) sum += i;
        long len = groups.size();
        long res = 0L;
        for (int i = 0; i < len; i++) {
            long the = groups.get(i);
            res += the * (sum - the);
            sum -= the;
        }
        return res;
    }
}

class DSU {
    Map<Integer, Integer> parent = new HashMap<>();
    Map<Integer, Integer> rank = new HashMap<>();

    public boolean add(int i) {
        if (parent.containsKey(i)) return false;
        parent.put(i, i);
        rank.put(i, 1);
        return true;
    }

    public int find(int i) { // find the root parent
        int root = i;
        int tmp;
        while ((tmp = parent.get(root)) != root) {
            root = tmp;
        }
        int ptr = i;
        while ((tmp = parent.get(ptr)) != root) { // Compress route
            parent.put(ptr, root);
            rank.put(root, rank.get(root) + 1); // merge by higher ranking
            ptr = tmp;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iParent = find(i);
        int jParent = find(j);
        if (iParent == jParent) return false;
        int iPRank = rank.get(iParent);
        int jPRank = rank.get(jParent);
        if (iPRank >= jPRank) {
            parent.put(jParent, iParent);
            rank.put(iParent, rank.get(iParent) + rank.get(jParent));
        } else {
            parent.put(iParent, jParent);
            rank.put(jParent, rank.get(iParent) + rank.get(jParent));
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        // Find all roots
        Map<Integer, Set<Integer>> res = new HashMap<>();
        for (int i : parent.keySet()) {
            int p = find(i);
            Set<Integer> s = res.computeIfAbsent(p, j -> new HashSet<>());
            s.add(i);
        }
        return res;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
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
