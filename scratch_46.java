import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.err.println(s.makeConnected(5,
                new int[][]{{0, 1}, {0, 2}, {3, 4}, {2, 3}}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1319
    public int makeConnected(int n, int[][] connections) {
        int totalEdges = connections.length;
        if (totalEdges < (n - 1)) return -1;
        DisjointSetUnion dsu = new DisjointSetUnion();
        for (int i = 0; i < n; i++) {
            dsu.add(i);
        }
        for (int[] i : connections) {
            dsu.merge(i[0], i[1]);
            dsu.merge(i[1], i[0]);
        }
        Map<Integer, Set<Integer>> groups = dsu.getAllGroups();
        return groups.size() - 1;
    }

    // LC552 ** DP
    public int checkRecord(int n) {
        final long mod = 1000000007;
        long[] lc552Memo = new long[n + 1];
        lc552Memo[0] = 1;
        lc552Memo[1] = 2;
        lc552Memo[2] = 4;
        lc552Memo[3] = 7;
        for (int i = 4; i <= n; i++) {
            lc552Memo[i] = ((((2 * lc552Memo[i - 1]) % mod) - lc552Memo[i - 4]) + mod) % mod;
        }
        long sum = lc552Memo[n];
        for (int i = 1; i <= n; i++) {
            sum += (lc552Memo[i - 1] * lc552Memo[n - i]) % mod;
        }
        return (int) (sum % mod);
    }

    // LC1901
    public int[] findPeakGrid(int[][] mat) {
        int rowNum = mat.length, colNum = mat[0].length;
        if (colNum >= rowNum) {
            int left = -1, right = colNum;
            while (left >= -1 && right <= colNum) {
                int centerColIdx = (left + right) / 2;
                int leftColIdx = centerColIdx - 1;
                int rightColIdx = centerColIdx + 1;
                int max = Integer.MIN_VALUE;
                int maxColIdx = -1;
                int maxRowIdx = -1;
                for (int i = leftColIdx; i <= rightColIdx; i++) {
                    for (int j = 0; j < rowNum; j++) {
                        int curEle;
                        if (i == -1 || i == colNum) curEle = -1;
                        else curEle = mat[j][i];
                        if (curEle >= max) {
                            max = curEle;
                        }
                    }
                }
                for (int i = leftColIdx; i <= rightColIdx; i++) {
                    for (int j = 0; j < rowNum; j++) {
                        int curEle;
                        if (i == -1 || i == colNum) curEle = -1;
                        else curEle = mat[j][i];
                        if (curEle == max) {
                            if (i != -1 && i != colNum) {
                                if (i == centerColIdx) {
                                    return new int[]{j, i};
                                } else {
                                    maxRowIdx = j;
                                    maxColIdx = i;
                                }
                            }
                        }
                    }
                }
                if (maxColIdx == leftColIdx) {
                    right = centerColIdx;
                } else {
                    left = centerColIdx;
                }
            }
        } else {
            int up = -1, down = rowNum;
            while (up >= -1 && down <= rowNum) {
                int centerRowIdx = (up + down) / 2;
                int upRowIdx = centerRowIdx - 1;
                int downRowIdx = centerRowIdx + 1;
                int max = Integer.MIN_VALUE;
                int maxColIdx = -1;
                int maxRowIdx = -1;
                for (int i = upRowIdx; i <= downRowIdx; i++) {
                    for (int j = 0; j < colNum; j++) {
                        int curEle;
                        if (i == -1 || i == rowNum) curEle = -1;
                        else curEle = mat[i][j];
                        if (curEle >= max) {
                            max = curEle;
                        }
                    }
                }
                for (int i = upRowIdx; i <= downRowIdx; i++) {
                    for (int j = 0; j < colNum; j++) {
                        int curEle;
                        if (i == -1 || i == rowNum) curEle = -1;
                        else curEle = mat[i][j];
                        if (curEle == max) {
                            if (i != -1 && i != rowNum) {
                                if (i == centerRowIdx) {
                                    return new int[]{i, j};
                                } else {
                                    maxRowIdx = i;
                                    maxColIdx = j;
                                }
                            }
                        }
                    }
                }
                if (maxRowIdx == upRowIdx) {
                    down = centerRowIdx;
                } else {
                    up = centerRowIdx;
                }
            }
        }
        return new int[]{-1, -1};
    }

    // LC483
    public String smallestGoodBase(String n) {
        long num = Long.valueOf(n);
        // num = (ans^k)-1
        // (num+1) = ans ^ k
        // Math.log(num+1) = k * Math.log(ans) // k不会超过63
        int kMax = (int) Math.floor(Math.log(num + 1) / Math.log(2));
        long min = num;
        for (int i = kMax; i >= 0; i--) {
            long posAns = (long) Math.pow(num + 1, (double) (1 / (i + 0.0)));
            long mul = 1, sum = 1;
            for (int j = 0; j < i; j++) {
                mul *= posAns;
                sum += mul;
            }
            if (sum == num) {
                return String.valueOf(posAns);
            }
        }
        return String.valueOf(num - 1);

    }

    // LC1010
    public int numPairsDivisibleBy60(int[] time) {
        int[] modFreq = new int[60];
        for (int i : time) {
            modFreq[i % 60]++;
        }
        int result = 0;
        // mod == 0
        result += ((modFreq[0] - 1) * modFreq[0]) / 2;
        // mod == 30
        result += ((modFreq[30] - 1) * modFreq[30]) / 2;
        for (int i = 1; i <= 29; i++) {
            result += modFreq[i] * modFreq[60 - i];
        }
        return result;
    }

    // LC495
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        int total = 0;
        int expire = 0;
        for (int attackTime : timeSeries) {
            if (attackTime < expire) {
                total += (attackTime + duration - expire);
            } else {
                total += duration;
            }
            expire = Math.max(expire, attackTime + duration);
        }
        return total;
    }

    // LC605
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int zeroCount = 0;
        int total = 0;
        for (int i = -1; i <= flowerbed.length; i++) {
            if (i == -1) {
                zeroCount++;
            } else if (i == flowerbed.length) {
                zeroCount++;
                total += Math.max(0, (zeroCount - 1) / 2);
            } else {
                if (flowerbed[i] == 1) {
                    total += Math.max(0, (zeroCount - 1) / 2);
                    zeroCount = 0;
                } else {
                    zeroCount++;
                }
            }
        }
        return total >= n;
    }

    // LC735
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> stack = new LinkedList<>();
        for (int i : asteroids) {
            boolean collision = false;
            while (!stack.isEmpty() && i < 0 && stack.peek() > 0) {
                if (stack.peek() < -i) {
                    stack.pop();
                    continue;
                } else if (stack.peek() == -i) {
                    stack.pop();
                }
                collision = true;
                break;
            }
            if (!collision) stack.push(i);
        }
        int[] result = new int[stack.size()];
        for (int i = result.length - 1; i >= 0; i--) {
            result[i] = stack.pop();
        }
        return result;
    }

    // LC913 Minmax
    final int TIE = 0, CAT_WIN = 2, MOUSE_WIN = 1;
    Integer[][][] lc913Memo;

    public int catMouseGame(int[][] graph) {
        lc913Memo = new Integer[graph.length * 2 + 1][graph.length + 1][graph.length + 1];
        return lc913Helper(0, graph, 1, 2);
    }

    private int lc913Helper(int steps, int[][] graph, int mousePoint, int catPoint) {
        if (steps >= 2 * graph.length) return TIE;
        if (lc913Memo[steps][mousePoint][catPoint] != null) return lc913Memo[steps][mousePoint][catPoint];
        if (mousePoint == catPoint) return lc913Memo[steps][mousePoint][catPoint] = CAT_WIN;
        if (mousePoint == 0) return lc913Memo[steps][mousePoint][catPoint] = MOUSE_WIN;
        boolean isMouse = steps % 2 == 0;
        if (isMouse) {
            boolean catCanWin = true;
            for (int i : graph[mousePoint]) {
                int nextResult = lc913Helper(steps + 1, graph, i, catPoint);
                if (nextResult == MOUSE_WIN) {
                    return lc913Memo[steps][mousePoint][catPoint] = MOUSE_WIN;
                } else if (nextResult == TIE) {   // 极小化极大: 猫嬴是一个极大值, 如果nextResult == CAT_WIN, 但是此前nextWin取到较小值TIE, 则选TIE不选CAT_WIN
                    catCanWin = false;
                }
            }
            if (catCanWin) return lc913Memo[steps][mousePoint][catPoint] = CAT_WIN;
            return lc913Memo[steps][mousePoint][catPoint] = TIE;
        } else {
            boolean mouseCanWin = true;
            for (int i : graph[catPoint]) {
                if (i == 0) continue;
                int nextResult = lc913Helper(steps + 1, graph, mousePoint, i);
                if (nextResult == CAT_WIN) {
                    return lc913Memo[steps][mousePoint][catPoint] = CAT_WIN;
                } else if (nextResult == TIE) {
                    mouseCanWin = false;
                }
            }
            if (mouseCanWin) return lc913Memo[steps][mousePoint][catPoint] = MOUSE_WIN;
            return lc913Memo[steps][mousePoint][catPoint] = TIE;
        }
    }


    // LC843 Minmax **
    public void findSecretWord(String[] wordlist, Master master) {
        int[][] H;
        int n = wordlist.length;
        H = new int[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                int match = 0;
                for (int k = 0; k < 6; ++k) {
                    if (wordlist[i].charAt(k) == wordlist[j].charAt(k))
                        match++;
                }
                H[i][j] = H[j][i] = match;
            }
        }

        Set<Integer> possible = new HashSet();
        Set<Integer> selected = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            possible.add(i);
        }

        while (!possible.isEmpty()) {
            int guess = guess(possible, selected, H);
            int match = master.guess(wordlist[guess]);
            if (match == 6) return;
            Set<Integer> alterPossible = new HashSet<>();
            possible.remove(guess);
            for (int i : possible) {
                if (H[guess][i] == match) {
                    alterPossible.add(i);
                }
            }
            if (match == 0) {
                for (int i : possible) {
                    if (H[guess][i] != 0) {
                        selected.add(i);
                    }
                }
            }
            selected.add(guess);
            possible = alterPossible;
        }
    }

    private int guess(Set<Integer> possible, Set<Integer> selected, int[][] H) {
        int ansGuess = -1;
        Set<Integer> ansGroup = possible;

        for (int guess = 0; guess < H.length; guess++) {
            if (!selected.contains(guess)) {
                Set<Integer>[] groups = new Set[7];
                for (int j = 0; j < 7; j++) {
                    groups[j] = new HashSet<>();
                }

                for (int j : possible) {
                    if (j != guess) {
                        groups[H[guess][j]].add(j);
                    }
                }

                Set<Integer> maxGroup = groups[0];
                for (int i = 1; i < 7; i++) {
                    if (groups[i].size() > maxGroup.size()) { // 最大化
                        maxGroup = groups[i];
                    }
                }

                if (maxGroup.size() < ansGroup.size()) { // 最小化
                    ansGroup = maxGroup;
                    ansGuess = guess;
                }
            }
        }
        return ansGuess;
    }
}

// LC843
interface Master {
    public default int guess(String word) {
        return -1;
    }
}

// LC65 有限状态自动机
class IsNumber {
    static enum CharacterType {
        DIGIT,
        POINT,
        SIGN,
        EXP,
        INVALID
    }

    static enum INState {
        START,
        SIGNING,
        EMPTY_INT_DEC,
        INTEGERING,
        WITH_INT_DEC,
        DECIMALING,
        EXP,
        EXP_SIGN,
        EXPING
    }

    static CharacterType getCharacterType(char c) {
        if (Character.isDigit(c)) return CharacterType.DIGIT;
        if (c == '.') return CharacterType.POINT;
        if (c == 'e' || c == 'E') return CharacterType.EXP;
        if (c == '+' || c == '-') return CharacterType.SIGN;
        return CharacterType.INVALID;
    }

    static Map<INState, Map<CharacterType, INState>> DFA = new HashMap<INState, Map<CharacterType, INState>>() {{
        put(INState.START, new HashMap<CharacterType, INState>() {{
            put(CharacterType.SIGN, INState.SIGNING);
            put(CharacterType.POINT, INState.EMPTY_INT_DEC);
            put(CharacterType.DIGIT, INState.INTEGERING);
        }});
        put(INState.SIGNING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.POINT, INState.EMPTY_INT_DEC);
            put(CharacterType.DIGIT, INState.INTEGERING);
        }});
        put(INState.EMPTY_INT_DEC, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
        }});
        put(INState.INTEGERING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.INTEGERING);
            put(CharacterType.POINT, INState.WITH_INT_DEC);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.WITH_INT_DEC, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.DECIMALING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.EXP, new HashMap<CharacterType, INState>() {{
            put(CharacterType.SIGN, INState.EXP_SIGN);
            put(CharacterType.DIGIT, INState.EXPING);
        }});
        put(INState.EXP_SIGN, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.EXPING);
        }});
        put(INState.EXPING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.EXPING);
        }});
    }};

    public static boolean isNumber(String s) {
        char[] cArr = s.toCharArray();
        INState state = INState.START;
        for (int i = 0; i < cArr.length; i++) {
            CharacterType t = getCharacterType(cArr[i]);
            if (!DFA.get(state).containsKey(t)) {
                return false;
            } else {
                state = DFA.get(state).get(t);
            }
        }
        return state == INState.INTEGERING || state == INState.WITH_INT_DEC || state == INState.DECIMALING || state == INState.EXPING;
    }
}

class DisjointSetUnion {
    Map<Integer, Integer> parent;

    DisjointSetUnion() {
        parent = new HashMap<>();
    }

    public boolean add(int i) {
        if (parent.containsKey(i)) {
            return false;
        }
        parent.put(i, i);
        return true;
    }

    public void merge(int i, int j) {
        int jParent = find(j);
        int iParent = find(i);
        if (iParent == jParent) return;
        parent.put(iParent, jParent);
    }

    public int find(int i) {
        int root = i;
        while (parent.get(root) != root) {
            root = parent.get(root);
        }
        int ptr = i;
        while (parent.get(ptr) != root) { // 路径压缩
            int tmp = parent.get(ptr);
            parent.put(ptr, root);
            ptr = tmp;
        }
        return root;
    }

    public boolean isConnect(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        for (int i : parent.keySet()) {
            result.putIfAbsent(find(i), new HashSet<>());
            result.get(find(i)).add(i);
        }
        return result;
    }

    public int getGroupCount() {
        return getAllGroups().size();
    }

}

