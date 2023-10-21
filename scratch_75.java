import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Solution {
    public static void main(String[] args) {
        var s = new Solution();

        System.err.println(s.minOperations(new int[]{14}, new int[]{86}));
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
