import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.err.println(s.countLargestGroup(10000));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1399
    public int countLargestGroup(int n) {
        int[] count = new int[37];

        for (int i = 1; i <= n; i++) {
            int num = i;
            int sum = 0;
            while (num != 0) {
                sum += num % 10;
                num /= 10;
            }
            count[sum]++;
        }
        Arrays.sort(count);
        int result = 1;
        for (int i = 35; i >= 0; i--) {
            if (count[i] == count[i + 1]) result++;
            else break;
        }
        return result;
    }

    // LC1418
    public List<List<String>> displayTable(List<List<String>> orders) {
        List<List<String>> result = new LinkedList<>();
        // orders: customer, table num, dish
        Map<String, Map<String, Integer>> m = new HashMap<>();
        for (List<String> order : orders) {
            m.putIfAbsent(order.get(1), new HashMap<>());
            m.get(order.get(1)).put(order.get(2), m.get(order.get(1)).getOrDefault(order.get(2), 0) + 1);
        }
        TreeSet<String> dishes = new TreeSet<>();
        for (String table : m.keySet()) {
            for (String dish : m.get(table).keySet()) {
                dishes.add(dish);
            }
        }
        List<String> tableHead = new ArrayList<>(dishes.size() + 1);
        tableHead.add("Table");
        for (String dish : dishes) {
            tableHead.add(dish);
        }
        for (String table : m.keySet()) {
            List<String> thisTable = new ArrayList<>(dishes.size() + 1);
            thisTable.add(table);
            for (String dish : dishes) {
                thisTable.add(String.valueOf(m.get(table).getOrDefault(dish, 0)));
            }
            result.add(thisTable);
        }
        result.sort(new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                return Integer.valueOf(o1.get(0)) - Integer.valueOf(o2.get(0));
            }
        });
        result.add(0, tableHead);

        return result;
    }

    // LC1922 快速幂
    final long lc1922Mod = 1000000007l;

    public int countGoodNumbers(long n) {
        return (int) ((quickPow(5, ((n + 1) / 2)) % lc1922Mod) * (quickPow(4, (n / 2)) % lc1922Mod) % lc1922Mod);
    }

    public long quickPow(long x, long pow) {
        if (pow == 0) return 1L;
        long xCon = x;
        long result = 1l;
        while (pow != 0) {
            if (pow % 2 == 1) {
                result = (result * xCon) % lc1922Mod;
            }
            xCon = (xCon * xCon) % lc1922Mod;
            pow >>= 1;
        }
        return result;
    }

    // LC1921
    public int eliminateMaximum(int[] dist, int[] speed) {
        int n = dist.length;
        double[] time = new double[n];
        for (int i = 0; i < n; i++) {
            time[i] = (0.0d + dist[i]) / (0.0d + speed[i]);
        }
        Arrays.sort(time);
        int min = 0;
        for (double i : time) {
            if (min < i) {
                min++;
                continue;
            }
            return min;
        }
        return n;
    }

    // LC672 **
    public int flipLights(int n, int presses) {
        int origN = n;
        n = Math.min(n, 6);
        int zeroMask = 0;
        for (int i = 0; i < n; i++) zeroMask |= 1 << i;
        int[] ops = new int[]{0b101010, 0b010101, 0b001001, 0b111111};
        Set<Integer> visited = new HashSet<>();
        // 最多有2^4=16种操作法, 即一个操作执行1次或0次, 用1表示执行了1次, 0表示执行了偶数次(0次)
        // 整体操作次数的奇偶性应当与presses的奇偶性相同, 且不超过presses次
        for (int i = 0; i < (1 << 4); i++) {
            if (Integer.bitCount(i) % 2 == presses % 2 && Integer.bitCount(i) <= presses) {
                int status = zeroMask;
                for (int j = 0; j < ops.length; j++) {
                    if ((((i >> j) & 1)) == 1) {
                        status ^= ops[j];
                    }
                }
                visited.add(status & zeroMask);
            }
        }
        return visited.size();
    }

    // LC726
    public String countOfAtoms(String formula) {
        formula = '(' + formula + ')';
        char[] cArr = formula.toCharArray();
        Deque<Character> stack = new LinkedList<>();
        Deque<Map<String, Integer>> mStack = new LinkedList<>();
        mStack.push(new HashMap<>()); // root counter map
        for (int i = 0; i < cArr.length; i++) {
            char c = cArr[i];
            if (c == '(') {
                mStack.push(new HashMap<>());
                stack.push(c);
            } else if (c == ')') {
                int parenthesesCtr = 0;
                while (i + 1 < cArr.length && Character.isDigit(cArr[i + 1])) {
                    i++;
                    parenthesesCtr *= 10;
                    parenthesesCtr += cArr[i] - '0';
                }
                if (parenthesesCtr == 0) parenthesesCtr = 1;

                StringBuilder atomCtrSb = new StringBuilder();
                StringBuilder atomString = new StringBuilder(2);
                // 开始出栈, 并逐个处理
                while (!stack.isEmpty() && stack.peek() != '(') {
                    if (Character.isDigit(stack.peek())) {
                        atomCtrSb.insert(0, stack.pop());
                    } else if (Character.isLowerCase(stack.peek())) {
                        atomString.insert(0, stack.pop());
                    } else if (Character.isUpperCase(stack.peek())) {
                        atomString.insert(0, stack.pop());
                        String atom = atomString.toString();
                        int atomCtr = 1;
                        if (!atomCtrSb.toString().equals("")) {
                            atomCtr = Integer.valueOf(atomCtrSb.toString());
                        }
                        mStack.peek().put(atom, mStack.peek().getOrDefault(atom, 0) + atomCtr);
                        atomCtrSb = new StringBuilder();
                        atomString = new StringBuilder(2);
                    }
                }
                if (!stack.isEmpty() && stack.peek() == '(') {
                    stack.pop();
                    Map<String, Integer> mTop = mStack.pop();
                    for (String atom : mTop.keySet()) {
                        mStack.peek().put(atom, mStack.peek().getOrDefault(atom, 0) + parenthesesCtr * mTop.get(atom));
                    }
                }
            } else {
                stack.push(c);
            }
        }
        if (mStack.size() != 1) return "";
        TreeMap<String, Integer> tm = new TreeMap<>(mStack.pop());
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> entry : tm.entrySet()) {
            sb.append(entry.getKey());
            if (entry.getValue() != 1) {
                sb.append(entry.getValue());
            }
        }
        return sb.toString();
    }


    // LC645
    public int[] findErrorNums(int[] nums) {
        int len = nums.length;
        int sum = 0;
        for (int i : nums) {
            sum += Math.abs(i);
            nums[Math.abs(i) - 1] *= -1;
        }
        int[] suspect = new int[2];
        int sctr = 0;
        for (int i = 1; i <= len; i++) {
            if (nums[i - 1] > 0) {
                suspect[sctr++] = i;
                if (sctr == 2) break;
            }
        }
        int expectSum = (1 + len) * len / 2;
        if (sum - suspect[0] + suspect[1] == expectSum) {
            return new int[]{suspect[0], suspect[1]};
        }
        return new int[]{suspect[1], suspect[0]};

    }

    // LC1833
    public int maxIceCream(int[] costs, int coins) {
        Arrays.sort(costs);
        for (int i = 1; coins >= 0 && i <= costs.length; i++) {
            coins -= costs[i - 1];
            if (coins < 0) return i - 1;
        }
        if (coins >= 0) return costs.length;
        return 0;
    }

    // LC1529
    public int minFlips(String target) {
        int flag = 0;
        int result = 0;
        char[] cArr = target.toCharArray();
        for (char c : cArr) {
            int status = c - '0';
            if (status == flag) continue;
            result++;
            flag = 1 - flag;
        }
        return result;
    }

    // Interview 16.21
    public int[] findSwapValues(int[] array1, int[] array2) {
        int[] min = array1.length < array2.length ? array1 : array2;
        int[] max = min == array1 ? array2 : array1;
        Set<Integer> maxSet = new HashSet<>();
        boolean flag = min == array1;
        int sumMin = 0, sumMax = 0;
        for (int i : min) sumMin += i;
        for (int i : max) {
            sumMax += i;
            maxSet.add(i);
        }
        if ((sumMax + sumMin) % 2 == 1) return new int[0];
        int diff = (sumMin - sumMax) / 2;
        for (int i : min) {
            if (maxSet.contains(i - diff)) {
                if (flag) return new int[]{i, i - diff};
                return new int[]{i - diff, i};
            }
        }
        return new int[0];
    }

    // LC1027
    private TreeMap<Integer, Integer> lc1027Ts;
    private int lc1027Result = 0;

    public int maxAncestorDiff(TreeNode root) {
        lc1027Ts = new TreeMap<>();
        lc1027Helper(root);
        return lc1027Result;
    }

    private void lc1027Helper(TreeNode root) {
        if (root == null) {
            if (lc1027Ts.size() != 0) {
                lc1027Result = Math.max(lc1027Result, Math.abs(lc1027Ts.firstKey() - lc1027Ts.lastKey()));
            }
            return;
        }
        lc1027Ts.put(root.val, lc1027Ts.getOrDefault(root.val, 0) + 1);
        lc1027Helper(root.left);
        lc1027Helper(root.right);
        lc1027Ts.put(root.val, lc1027Ts.get(root.val) - 1);
        if (lc1027Ts.get(root.val) == 0) {
            lc1027Ts.remove(root.val);
        }
    }

    // LCP07
    public int numWays(int n, int[][] relation, int k) {
        Map<Integer, Set<Integer>> rm = new HashMap<>();
        for (int[] r : relation) {
            rm.putIfAbsent(r[0], new HashSet<>());
            rm.get(r[0]).add(r[1]);
        }
        int layer = -1;
        Deque<Integer> q = new LinkedList<>();
        q.offer(0);
        int result = 0;
        while (!q.isEmpty()) {
            layer++;
            if (layer == k) {
                while (!q.isEmpty()) {
                    if (q.poll() == n - 1) result++;
                }
                return result;
            }
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int p = q.poll();
                if (rm.containsKey(p)) {
                    for (int j : rm.get(p)) {
                        q.offer(j);
                    }
                }
            }
        }
        return 0;
    }

    // LC670
    public int maximumSwap(int num) {
        int orig = num;
        int numDigit = getNumDigit(num);
        if (numDigit == 1) return num;

        int[] arr = new int[numDigit];
        int[] max = new int[numDigit];
        for (int i = numDigit - 1; i >= 0; i--) {
            arr[i] = num % 10;
            num /= 10;
        }
        System.arraycopy(arr, 0, max, 0, numDigit);

        for (int i = 0; i < numDigit; i++) {
            for (int j = i + 1; j < numDigit; j++) {
                if (arr[j] > arr[i]) {
                    int[] tmp = new int[numDigit];
                    System.arraycopy(arr, 0, tmp, 0, numDigit);
                    int t = tmp[i];
                    tmp[i] = tmp[j];
                    tmp[j] = t;
                    if (sequenceCompare(max, 0, tmp, 0) < 0) {
                        System.arraycopy(tmp, 0, max, 0, numDigit);
                    }
                }
            }
        }

        int result = 0;
        for (int i = 0; i < numDigit; i++) {
            result *= 10;
            result += max[i];
        }

        // nums <=10^8
        return result;
    }

    private int swapInteger(int i, int idx1, int idx2) {
        int numDigit = getNumDigit(i);
        if (idx1 < 0 || idx1 >= numDigit || idx2 < 0 || idx2 >= numDigit) return i;
        int dig1 = getDigit(i, idx1, numDigit);
        int dig2 = getDigit(i, idx2, numDigit);
        int pow1 = numDigit - idx1 - 1;
        int pow2 = numDigit - idx2 - 1;
        i -= (int) (Math.pow(10, pow1)) * dig1;
        i += (int) (Math.pow(10, pow1)) * dig2;
        i -= (int) (Math.pow(10, pow2)) * dig2;
        i += (int) (Math.pow(10, pow2)) * dig1;
        return i;
    }

    private int getDigit(int i, int idx, int numDigit) {
        int ctr = numDigit - idx;
        while (ctr != 1) {
            ctr--;
            i /= 10;
        }
        return i % 10;
    }

    private int getDigit(int i, int idx) {
        return getDigit(i, idx, getNumDigit(i));
    }

    private int getNumDigit(int i) {
        if (i == 0) return 1;
        int result = 0;
        while (i != 0) {
            result++;
            i /= 10;
        }
        return result;
    }

    // LC321 **
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        // 总共出k个数字
        int n1l = nums1.length, n2l = nums2.length;
        int start, end;
        if (n2l >= k) start = 0; // 如果n2l都比k大, 则n1可以一个都不出
        else start = k - n2l; // 否则, n1最少要出k-n2l个数
        if (n1l >= k) end = k; // 如果n1l比k大, 则最后n1可以出k个数, n2一个都不出
        else end = n1l; // 否则n1最后至少要出所有数, 剩下的数从n2出
        int[] maxSequence = new int[k];
        Arrays.fill(maxSequence, -1);
        for (int i = start; i <= end; i++) {
            int[] n1s = maxSequence(nums1, i);
            int[] n2s = maxSequence(nums2, k - i);
            int[] mergeSequence = mergeSequence(n1s, n2s);
            if (sequenceCompare(maxSequence, 0, mergeSequence, 0) < 0) {
                maxSequence = mergeSequence;
            }
        }
        return maxSequence;
    }

    private int[] maxSequence(int[] arr, int size) {
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < arr.length; i++) {
            int remain = arr.length - i; // 还剩下多少数可以加到栈里
            // int lack = size - stack.size(); // 栈里还缺多少个数, 这个要动态算!
            // 如果lack>=remain, 就不能pop()了, 只能push, 否则栈里的数的数量不够
            while (!stack.isEmpty() && stack.peek() < arr[i] && remain > (size - stack.size())) {
                stack.pop();
            }
            stack.push(arr[i]);
        }
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            result[i] = stack.pollLast();
        }
        return result;
    }

    private int[] mergeSequence(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length + arr2.length];
        int idx1 = 0, idx2 = 0;
        for (int i = 0; i < result.length; i++) {
            if (sequenceCompare(arr1, idx1, arr2, idx2) > 0) {
                result[i] = arr1[idx1++];
            } else {
                result[i] = arr2[idx2++];
            }
        }
        return result;
    }

    private int sequenceCompare(int[] arr1, int startIdx1, int[] arr2, int startIdx2) {
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

    // LC532
    public int findPairs(int[] nums, int k) {
        Set<Integer> iterSet = new HashSet<>();
        Set<Integer> visited = new HashSet<>();
        for (int i : nums) {
            if (iterSet.contains(i + k)) {
                visited.add(i);
            }
            if (iterSet.contains(i - k)) {
                visited.add(i - k);
            }
            iterSet.add(i);
        }
        return visited.size();
    }

    // LC168
    public String convertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();
        while (columnNumber != 0) {
            columnNumber--;
            sb.insert(0, (char) (columnNumber % 26 + 'A'));
            columnNumber /= 26;
        }
        return sb.toString();
    }

    // LC815
    public int numBusesToDestination(int[][] routes, int source, int target) {
        // routes.length <=500
        if (source == target) return 0;
        Map<Integer, Set<Integer>> stationRouteMap = new HashMap<>();
        for (int i = 0; i < routes.length; i++) {
            for (int j : routes[i]) {
                stationRouteMap.putIfAbsent(j, new HashSet<>());
                stationRouteMap.get(j).add(i);
            }
        }

        Map<Integer, Set<Integer>> routeEdge = new HashMap<>();
        for (int i : stationRouteMap.keySet()) {
            Set<Integer> routeSet = stationRouteMap.get(i);
            for (int j : routeSet) {
                for (int k : routeSet) {
                    if (j != k) {
                        routeEdge.putIfAbsent(j, new HashSet<>());
                        routeEdge.get(j).add(k);
                    }
                }
            }
        }

        Set<Integer> targetRouteSet = stationRouteMap.get(target);
        if (targetRouteSet == null) return -1;
        Set<Integer> sourceRouteSet = stationRouteMap.get(source);
        Set<Integer> visited = new HashSet<>();

        int layer = 0;
        Deque<Integer> q = new LinkedList<>();
        for (int i : sourceRouteSet) {
            q.offer(i);
        }
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int p = q.poll();
                if (targetRouteSet.contains(p)) return layer;
                if (visited.contains(p)) continue;
                visited.add(p);
                if (!routeEdge.containsKey(p)) continue;
                for (int j : routeEdge.get(p)) {
                    if (!visited.contains(j)) q.offer(j);
                }
            }
        }

        return -1;
    }

    // LC909
    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        boolean[][] visited = new boolean[n][n];
        Deque<int[]> q = new LinkedList<>();
        q.offer(lc909IdxConverter(1, n));
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] t = q.poll();
                // 在下一层返回
                // if (lc909IdxConverter(t, n) == n * n) return layer;
                if (visited[t[0]][t[1]]) continue;
                visited[t[0]][t[1]] = true;
                int num = lc909IdxConverter(t, n);
                for (int j = num + 1; j <= Math.min(num + 6, n * n); j++) {
                    int[] next = lc909IdxConverter(j, n);
                    if (board[next[0]][next[1]] != -1) {
                        // 注意不能提前标记已访问, 因为每次都最多只能访问下一格, 所以这一格仍有可能被访问到
                        // visited[next[0]][next[1]] = true; WRONG!!!
                        next = lc909IdxConverter(board[next[0]][next[1]], n);
                    }
                    // 提前返回, 不用等到下一层
                    if (lc909IdxConverter(next, n) == n * n) return layer + 1;

                    if (!visited[next[0]][next[1]]) {
                        q.offer(next);
                    }
                }
            }
        }
        return -1;
    }

    private int[] lc909IdxConverter(int idx, int n) {
        int row = (n - 1) - (idx - 1) / n;
        int col = (n + row) % 2 == 1 ? ((idx - 1) % n) : ((n - 1) - ((idx - 1) % n));
        return new int[]{row, col};
    }

    private int lc909IdxConverter(int[] idx, int n) {
        int row = idx[0], col = idx[1];
        int numRow = (n - 1 - row) * n;
        int numCol = (n + row) % 2 == 1 ? col : n - 1 - col;
        return numRow + numCol + 1;
    }

    // LC738
    public int monotoneIncreasingDigits(int n) {
        Deque<Integer> stack = new LinkedList<>();
        while (n != 0) {
            stack.add(n % 10);
            n /= 10;
        }
        LinkedList<Integer> result = new LinkedList<>();
        while (!stack.isEmpty()) {
            int last = stack.pop();
            if (!stack.isEmpty() && last < stack.peek()) {
                last = 9;
                int rs = result.size();
                result.clear();
                for (int i = 0; i < rs; i++) {
                    result.add(9);
                }
                int lastButOne = stack.pop();
                lastButOne--;
                stack.push(lastButOne);
            }
            result.push(last);
        }

        int ret = 0;
        while (!result.isEmpty()) {
            ret = ret * 10 + result.pop();
        }
        return ret;
    }

    // LC402 Solution **
    public String removeKdigits(String num, int k) {
        Deque<Character> deque = new LinkedList<>();
        int length = num.length();
        for (int i = 0; i < length; ++i) {
            char digit = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && deque.peek() > digit) {
                deque.pop();
                k--;
            }
            deque.push(digit);
        }
        for (int i = 0; i < k; ++i) {
            deque.pop();
        }

        StringBuilder ret = new StringBuilder();
        while (!deque.isEmpty() && deque.peekLast() == '0') {
            deque.pollLast();
        }
        while (!deque.isEmpty()) {
            ret.append(deque.pollLast());
        }
        return ret.length() == 0 ? "0" : ret.toString();
    }

    // LC773
    public int slidingPuzzle(int[][] board) {
        boolean[][][][][] visited = new boolean[6][6][6][6][6]; // 第六个格子可以由前5个推出
        final int[][] directs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Deque<int[][]> q = new LinkedList<>();
        q.offer(board);
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[][] b = q.poll();

                if (b[0][0] == 1 && b[0][1] == 2 && b[0][2] == 3 && b[1][0] == 4 && b[1][1] == 5) return layer;

                if (visited[b[0][0]][b[0][1]][b[0][2]][b[1][0]][b[1][1]]) {
                    continue;
                }
                visited[b[0][0]][b[0][1]][b[0][2]][b[1][0]][b[1][1]] = true;
                int zX = -1, zY = -1;
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 3; k++) {
                        if (b[j][k] == 0) {
                            zX = j;
                            zY = k;
                            break;
                        }
                    }
                }
                if (zX != -1 && zY != -1) {
                    for (int[] d : directs) {
                        int x = zX + d[0];
                        int y = zY + d[1];
                        if (x >= 0 && x < 2 && y >= 0 && y < 3) {
                            int[][] e = {{b[0][0], b[0][1], b[0][2]}, {b[1][0], b[1][1], b[1][2]}};
                            int tmp = e[x][y];
                            e[x][y] = 0;
                            e[zX][zY] = tmp;
                            if (!visited[e[0][0]][e[0][1]][e[0][2]][e[1][0]][e[1][1]]) {
                                q.offer(e);
                            }
                        }
                    }
                }
            }
        }
        return -1;
    }

    // LC1342
    public int numberOfSteps(int num) {
//        int step = 0;
//        while (num != 0) {
//            if((num&1)==1) num--;
//            else num >>= 1;
//            step++;
//        }
//        return step;
        if (num == 0) return 0;
        return Integer.SIZE - Integer.numberOfLeadingZeros(num) + Integer.bitCount(num) - 1;
    }

    // LC752 BFS
    public int openLock(String[] deadends, String target) {
        boolean[][][][] visited = new boolean[10][10][10][10];
        int[] t = new int[]{(target.charAt(0) - '0'), (target.charAt(1) - '0'), (target.charAt(2) - '0'), (target.charAt(3) - '0')};

        for (String d : deadends) {
            int[] idxes = new int[]{d.charAt(0) - '0', d.charAt(1) - '0', d.charAt(2) - '0', d.charAt(3) - '0'};
            visited[idxes[0]][idxes[1]][idxes[2]][idxes[3]] = true;
        }
        int[] init = new int[]{0, 0, 0, 0};
        Deque<int[]> q = new LinkedList<>();
        int layer = -1;
        q.offer(init);
        while (!q.isEmpty()) {
            layer++;
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                int[] p = q.poll();
                if (p[0] == t[0] && p[1] == t[1] && p[2] == t[2] && p[3] == t[3]) {
                    return layer;
                }
                if (visited[p[0]][p[1]][p[2]][p[3]]) continue; // 注意这里剪枝时机
                visited[p[0]][p[1]][p[2]][p[3]] = true;
                for (int j = 0; j < 4; j++) {
                    int[] n = new int[]{p[0], p[1], p[2], p[3]};
                    n[j] = (n[j] + 1) % 10;
                    if (!visited[n[0]][n[1]][n[2]][n[3]]) {
                        q.offer(n);
                    }
                }
                for (int j = 0; j < 4; j++) {
                    int[] n = new int[]{p[0], p[1], p[2], p[3]};
                    n[j] = (n[j] + 10 - 1) % 10;
                    if (!visited[n[0]][n[1]][n[2]][n[3]]) {
                        q.offer(n);
                    }
                }
            }
        }
        return -1;
    }

    // LC1461 Bitset
    public boolean hasAllCodesBitset(String s, int k) {
        if (k >= 17) return false;
        if (s.length() - k + 1 < (1 << k)) return false;
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        BitSet bs = new BitSet((1 << k));
        int cur = 0;
        for (int i = 0; i < k; i++) {
            cur = (cur << 1) | (cArr[i] - '0');
        }
        bs.set(cur);
        int allBitMask = (1 << k) - 1;
        for (int i = k; i < n; i++) {
            cur = ((cur << 1) & allBitMask) | (cArr[i] - '0');
            bs.set(cur);
        }
        return bs.cardinality() == (1 << k);
    }

    // LC1461
    public boolean hasAllCodes(String s, int k) {
        if (k >= 17) return false;
        if (s.length() - k + 1 < (1 << k)) return false;
        if (s.length() <= k) return false;
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        if (k <= 5) {
            int correct;
            if (k != 5) {
                correct = (1 << (1 << k)) - 1;
            } else {
                correct = -1;
            }
            int count = 0;
            int allBitMask = (1 << k) - 1;
            int cur = 0;
            for (int i = 0; i < k; i++) {
                cur = (cur << 1) | (cArr[i] - '0');
            }
            count |= (1 << cur);
            for (int i = k; i < n; i++) {
                cur = ((cur << 1) & allBitMask) | (cArr[i] - '0');
                count |= (1 << cur);
                if (count == correct) return true;
            }
            return false;
        } else {
            int[] bitmap = new int[1 << (k - 5)];
            int allBitMask = (1 << k) - 1;
            int cur = 0;
            int modAnd = ~((1 << 32) - 1);  // 神秘加速, 理论上应该是 modAnd=31的
            for (int i = 0; i < k; i++) {
                cur = (cur << 1) | (cArr[i] - '0');
            }
            bitmap[cur >> 5] |= 1 << (cur & modAnd);
            for (int i = k; i < n; i++) {
                cur = ((cur << 1) & allBitMask) | (cArr[i] - '0');
                bitmap[cur >> 5] |= 1 << (cur & modAnd);
            }
            for (int i : bitmap) {
                if (i != -1) return false;
            }
            return true;
        }
    }

    // LC1222
    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
        Set<Pair<Integer, Integer>> s = new HashSet<>();
        List<List<Integer>> result = new ArrayList<>(8);
        for (int[] q : queens) {
            s.add(new Pair<>(q[0], q[1]));
        }
        int kingLeft = king[1], kingRight = 8 - king[1] - 1;
        int kingUp = king[0], kingDown = 8 - king[0] - 1;
        int kingX = king[1], kingY = king[0];
        for (int i = 1; i <= kingUp; i++) {
            if (s.contains(new Pair<>(kingY - i, kingX))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY - finalI);
                    add(kingX);
                }});
                break;
            }
        }
        for (int i = 1; i <= kingDown; i++) {
            if (s.contains(new Pair<>(kingY + i, kingX))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY + finalI);
                    add(kingX);
                }});
                break;
            }
        }
        for (int i = 1; i <= kingLeft; i++) {
            if (s.contains(new Pair<>(kingY, kingX - i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY);
                    add(kingX - finalI);
                }});
                break;
            }
        }
        for (int i = 1; i <= kingRight; i++) {
            if (s.contains(new Pair<>(kingY, kingX + i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY);
                    add(kingX + finalI);
                }});
                break;
            }
        }

        // 左上
        for (int i = 1; i <= Math.min(kingLeft, kingUp); i++) {
            if (s.contains(new Pair<>(kingY - i, kingX - i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY - finalI);
                    add(kingX - finalI);
                }});
                break;
            }
        }

        for (int i = 1; i <= Math.min(kingRight, kingUp); i++) {
            if (s.contains(new Pair<>(kingY - i, kingX + i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY - finalI);
                    add(kingX + finalI);
                }});
                break;
            }
        }

        for (int i = 1; i <= Math.min(kingLeft, kingDown); i++) {
            if (s.contains(new Pair<>(kingY + i, kingX - i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY + finalI);
                    add(kingX - finalI);
                }});
                break;
            }
        }

        for (int i = 1; i <= Math.min(kingRight, kingDown); i++) {
            if (s.contains(new Pair<>(kingY + i, kingX + i))) {
                int finalI = i;
                result.add(new ArrayList<Integer>(2) {{
                    add(kingY + finalI);
                    add(kingX + finalI);
                }});
                break;
            }
        }
        return result;
    }

    // LC840
    public int numMagicSquaresInside(int[][] grid) {
        final int LINE_SUM = 15;
        int m = grid.length;
        int n = grid[0].length;
        if (m < 3 || n < 3) return 0;
        int result = 0;
        for (int i = 0; i < m - 3 + 1; i++) {
            for (int j = 0; j < n - 3 + 1; j++) {
                boolean legal = true;
                boolean[] visited = new boolean[10];
                // legal matrix
                for (int k = 0; k < 3; k++) {
                    for (int o = 0; o < 3; o++) {
                        if ((grid[i + k][j + o] > 9 || grid[i + k][j + o] < 1) || visited[grid[i + k][j + o]]) {
                            legal = false;
                            break;
                        }
                        visited[grid[i + k][j + o]] = true;
                    }
                    if (!legal) break;
                }
                if (!legal) continue;
                for (int k = 1; k <= 9; k++) {
                    if (!visited[k]) {
                        legal = false;
                    }
                }
                if (!legal) continue;

                // 行和
                for (int k = 0; k < 3; k++) {
                    if (grid[i + k][j] + grid[i + k][j + 1] + grid[i + k][j + 2] != LINE_SUM) {
                        legal = false;
                    }
                }
                if (!legal) continue;

                // 列和
                for (int k = 0; k < 3; k++) {
                    if (grid[i][j + k] + grid[i + 1][j + k] + grid[i + 2][j + k] != LINE_SUM) {
                        legal = false;
                    }
                }
                if (!legal) continue;

                // 对角线和
                if (grid[i][j] + grid[i + 1][j + 1] + grid[i + 2][j + 2] != LINE_SUM) legal = false;
                if (grid[i + 2][j] + grid[i + 1][j + 1] + grid[i][j + 2] != LINE_SUM) legal = false;
                if (!legal) continue;

                if (legal) result++;
            }
        }

        return result;
    }

    // LC942
    public int[] diStringMatch(String s) {
        int n = s.length();
        int[] result = new int[n + 1];
        char[] cArr = s.toCharArray();
        int ptr1 = n, ptr2 = 0;
        for (int i = 0; i < n; i++) {
            if (cArr[i] == 'D') {
                result[i] = ptr1--;
            } else {
                result[i] = ptr2++;
            }
        }
        if (cArr[n - 1] == 'I') {
            result[n] = ptr2;
        } else {
            result[n] = ptr1;
        }
        return result;
    }

    // JZ 57
    public int[] twoSum(int[] nums, int target) {
        Arrays.sort(nums);
        for (int i : nums) {
            if (Arrays.binarySearch(nums, target - i) >= 0) {
                return new int[]{i, target - i};
            }
        }
        return new int[]{-1, -1};
    }

    // LC24
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            ListNode close = cur.next;
            ListNode far = cur.next.next;

            ListNode farNext = far.next;
            far.next = close;
            close.next = farNext;
            cur.next = far;

            cur = cur.next.next;
        }
        return dummy.next;
    }

    // IPv4 Parser
    public int ipv4Parser(String ip) {
        final int err = 0xffffffff;
        String[] groups = ip.split("\\.", -1);
        if (groups.length != 4) return err;
        int result = 0;
        int ctr = 0;
        for (String part : groups) {
            if (part.length() > 3 || part.length() == 0) return err;
            if (part.charAt(0) == '0' && part.length() != 1) return err;
            if (Integer.valueOf(part) > 255 || Integer.valueOf(part) < 0) return err;
            result |= Integer.valueOf(part) << ((3 - ctr) * 8);
            ctr++;
        }
        return result;
    }

    // LC1582
    public int numSpecial(int[][] mat) {
        int[] posRowColIdx = new int[mat.length];
        Set<Integer> posRowSet = new HashSet<>();
        int result = 0;
        for (int i = 0; i < mat.length; i++) {
            boolean flag = true;
            for (int j = 0; j < mat[0].length; j++) {
                if (mat[i][j] == 1) {
                    if (flag) {
                        posRowColIdx[i] = j;
                        posRowSet.add(i);
                        flag = false;
                    } else {
                        posRowColIdx[i] = -1;
                        posRowSet.remove(i);
                        break;
                    }
                }
            }
        }
        for (int i : posRowSet) {
            int colNum = posRowColIdx[i];
            int sum = 0;
            for (int j = 0; j < mat.length; j++) {
                sum += mat[j][colNum];
                if (sum == 2) break;
            }
            if (sum == 1) result++;
        }
        return result;
    }

    // Interview 01.09
    public boolean isFlipedString(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        return (s1 + s1).indexOf(s2) != -1;
    }

    // LC1382
    List<Integer> lc1382List;

    public TreeNode balanceBST(TreeNode root) {
        lc1382List = new ArrayList<>();
        inOrder(root);
        return buildBalanceBst(0, lc1382List.size() - 1);
    }

    private TreeNode buildBalanceBst(int start, int end) {
        if (start > end) return null;
        if (start == end) {
            return new TreeNode(lc1382List.get(start));
        }

        int mid = (start + end) / 2;

        TreeNode root = new TreeNode(lc1382List.get(mid));
        root.left = buildBalanceBst(start, mid - 1);
        root.right = buildBalanceBst(mid + 1, end);
        return root;
    }

    private void inOrder(TreeNode root) {
        if (root == null) return;
        inOrder(root.left);
        lc1382List.add(root.val);
        inOrder(root.right);
    }

    // Interview 04.12 Double DFS **
    public int pathSumDfs(TreeNode root, int sum) {
        if (root == null) return 0;
        int r = pathSumDfsHelper(root, sum);
        int left = pathSumDfs(root.left, sum);
        int right = pathSumDfs(root.right, sum);
        return left + r + right;
    }

    public int pathSumDfsHelper(TreeNode root, int sum) {
        if (root == null) return 0;
        int r = 0;
        if (root.val == sum) r = 1;
        int left = pathSumDfsHelper(root.left, sum - root.val);
        int right = pathSumDfsHelper(root.right, sum - root.val);
        return left + r + right;
    }

    // Interview 04.12 **
    Map<Integer, Integer> itv0412PrefixMap;

    public int pathSum(TreeNode root, int sum) {
        itv0412PrefixMap = new HashMap<>();
        itv0412PrefixMap.put(0, 1);
        return itv0412Helper(root, sum, 0);
    }

    private int itv0412Helper(TreeNode root, int target, int curSum) {
        if (root == null) return 0;
        int result = 0;
        curSum += root.val;

        result += itv0412PrefixMap.getOrDefault(curSum - target, 0);
        itv0412PrefixMap.put(curSum, itv0412PrefixMap.getOrDefault(curSum, 0) + 1);

        result += itv0412Helper(root.left, target, curSum);
        result += itv0412Helper(root.right, target, curSum);

        itv0412PrefixMap.put(curSum, itv0412PrefixMap.get(curSum) - 1);
        return result;
    }

    // LC1029 Sort
    public int twoCitySchedCostSort(int[][] costs) {
        int sum = 0;
        for (int[] c : costs) {
            sum += c[1];
        }
        Arrays.sort(costs, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return (o1[0] - o1[1]) - (o2[0] - o2[1]);
            }
        });
        for (int i = 0; i < costs.length / 2; i++) {
            sum += costs[i][0] - costs[i][1];
        }
        return sum;
    }

    // LC1029
    Integer[][] lc1029Memo;

    public int twoCitySchedCost(int[][] costs) {
        int n = costs.length / 2;
        lc1029Memo = new Integer[n * 2 + 1][n + 1];
        return lc1029Helper(0, 0, 0, n, costs);
    }

    private int lc1029Helper(int i, int aCtr, int bCtr, int n, int[][] costs) {
        if (i == 2 * n) return 0;
        if (lc1029Memo[i][aCtr] != null) return lc1029Memo[i][aCtr];
        int a = Integer.MAX_VALUE;
        int b = Integer.MAX_VALUE;
        if (aCtr < n) {
            a = costs[i][0] + lc1029Helper(i + 1, aCtr + 1, bCtr, n, costs);
        }
        if (bCtr < n) {
            b = costs[i][1] + lc1029Helper(i + 1, aCtr, bCtr + 1, n, costs);
        }
        return lc1029Memo[i][aCtr] = Math.min(a, b);
    }

    // LC891
    public int sumSubseqWidths(int[] nums) {
        final long mod = 1000000007;
        int n = nums.length;
        Arrays.sort(nums);
        // nums[i]的贡献 = nums[i] * (2^(i)-2^(n-i-1))
        long[] powTwo = new long[n];
        powTwo[0] = 1;
        for (int i = 1; i < n; i++) {
            powTwo[i] = 2 * powTwo[i - 1] % mod;
        }
        long result = 0;
        for (int i = 0; i < n; i++) {
            result = (result + (nums[i] * (powTwo[i] - powTwo[n - i - 1])) % mod) % mod;
        }
        return (int) (result % mod);

    }

    // LC685
    public int[] findRedundantDirectedConnection(int[][] edges) {
        // Case 1: 所有节点都有父节点, 此时需要删去成环的最后一条边
        // Case 2: 有一个节点入度为2, 此时需要删去其中一条边
        // 如果有节点入度为2 且图成环 需要删去造成入度为2且成环的边
        DisjointSetUnion dsu = new DisjointSetUnion();
        Map<Integer, Integer> parent = new HashMap<>();
        boolean isConflict = false, isCircle = false;
        int[] circle = null, conflict = null;
        for (int[] e : edges) {
            dsu.add(e[0]);
            dsu.add(e[1]);

            if (parent.containsKey(e[1])) { // 标记冲突边, 最多有一条
                isConflict = true;
                conflict = e;
            } else {
                parent.put(e[1], e[0]);
                if (dsu.isConnect(e[1], e[0])) {  // 成环就不加这条边了, 标记为最后一条可能导致成环的边
                    isCircle = true;
                    circle = e;
                } else {
                    dsu.merge(e[0], e[1]);
                }
            }
        }
        if (!isConflict) {
            return circle;
        } else {
            if (isCircle) { // 如果有冲突边且成环, 那必然是在环上导致入度为2的边要被删除, 也就是第一条标记入度为2的点的父节点构成的边
                return new int[]{parent.get(conflict[1]), conflict[1]};
            } else {
                return conflict; // 否则直接返回冲突边
            }
        }
    }

    // JZ Offer 38
    public String[] permutation(String s) {
        int size = 1;
        for (int i = 1; i <= s.length(); i++) size *= i;
        List<String> result = new ArrayList<>(size);
        char[] cArr = s.toCharArray();
        permutation(cArr, 0, result);
        return result.toArray(new String[result.size()]);
    }

    private void permutation(char[] s, int start, List<String> result) {
        if (start == s.length - 1) {
            result.add(new String(s));
            return;
        }
        Set<Character> set = new HashSet<>();
        for (int i = start; i < s.length; i++) {
            if (set.contains(s[i])) continue;
            set.add(s[i]);
            char tmp = s[start];
            s[start] = s[i];
            s[i] = tmp;
            permutation(s, start + 1, result);
            tmp = s[start];
            s[start] = s[i];
            s[i] = tmp;
        }
    }

    // Interview 16.17 非空子序列
    public int maxSubArray(int[] nums) {
        int dp = 0;
        int max = nums[0];
        for (int i : nums) {
            dp = Math.max(dp + i, i);
            max = Math.max(dp, max);
        }
        return max;
    }

    // Interview 05.08
    public int[] drawLine(int length, int w, int x1, int x2, int y) {
        final int full = 0xffffffff;
        int height = length * 32 / w;
        int width = w / 32;
        int[] result = new int[length];
        int startIdxOfLine = y * width;
        if (x2 < x1) {
            int tmp = x1;
            x1 = x2;
            x2 = tmp;
        }
        int first = x1 / 32;
        int last = x2 / 32;
        for (int i = first + 1; i <= last; i++) {
            result[startIdxOfLine + i] = full;
        }
        for (int i = x1 % 32; i < 32; i++) {
            result[startIdxOfLine + first] |= 1 << (32 - i - 1); // 置1
        }
        for (int i = (x2 % 32) + 1; i < 32; i++) {
            result[startIdxOfLine + last] &= ~(1 << (32 - i - 1)); // 置0
        }

        return result;
    }

    // LC514
    Integer[][] lc514Memo;
    Map<Character, List<Integer>> lc514Pos;

    public int findRotateSteps(String ring, String key) {
        lc514Memo = new Integer[key.length() + 1][ring.length() + 1];
        lc514Pos = new HashMap<>();
        char[] ringArr = ring.toCharArray();
        char[] keyArr = key.toCharArray();
        for (char c : keyArr) {
            lc514Pos.putIfAbsent(c, new ArrayList<>());
        }
        for (int i = 0; i < ringArr.length; i++) {
            if (lc514Pos.containsKey(ringArr[i])) {
                lc514Pos.get(ringArr[i]).add(i);
            }
        }
        return lc514Helper(0, 0, keyArr, ringArr) + key.length();
    }

    private int lc514Helper(int keyIdx, int ringIdx, char[] key, char[] ring) {
        if (keyIdx == key.length) return 0;
        if (lc514Memo[keyIdx][ringIdx] != null) return lc514Memo[keyIdx][ringIdx];
        int result = Integer.MAX_VALUE / 2;
        // keyIdx: 目标字符位置, ringIdx: 当前位置
        for (int i : lc514Pos.get(key[keyIdx])) {
            result = Math.min(result, Math.min(Math.abs(ringIdx - i), ring.length - Math.abs(ringIdx - i)) + lc514Helper(keyIdx + 1, i, key, ring));
        }
        return lc514Memo[keyIdx][ringIdx] = result;
    }

    // LC1493
    public int longestSubarray(int[] nums) {
        int left = 0, right = 0;
        int max = 0;
        int result = 0x3f3f3f3f;
        int[] count = new int[2];
        while (left < nums.length && right < nums.length) {
            count[nums[right++]]++;
            while (count[0] > 1) {
                count[nums[left++]]--;
            }
            max = Math.max(max, count[1]);
        }
        if (max == nums.length) return max - 1;
        return max;
    }

    // LC459
    public boolean repeatedSubstringPattern(String s) {
        for (int i = 1; i <= (s.length() / 2); i++) {
            if (s.length() % i != 0) continue;
            int ctr = 0;
            boolean flag = true;
            while (ctr * i < s.length()) {
                if (s.indexOf(s.substring(0, i), i * ctr) != i * ctr) {
                    flag = false;
                    break;
                }
                ctr++;
            }
            if (flag) {
                return true;
            }
        }
        return false;
    }

    // LC401
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < (1 << 10); i++) {
            if (Integer.bitCount(i) == turnedOn) {
                int hour = (i >> 6);
                int minute = i - (hour << 6);
                if (hour >= 0 && hour < 12 && minute >= 0 && minute < 60) {
                    result.add(String.format("%d:%02d", hour, minute));
                }
            }
        }
        return result;
    }

    // LC1239
    int lc1239Result;
    List<Integer> lc1239MaskArr;

    public int maxLength(List<String> arr) {
        lc1239Result = 0;
        lc1239MaskArr = new ArrayList<>(arr.size());
        for (int i = 0; i < arr.size(); i++) {
            int mask = 0;
            for (char c : arr.get(i).toCharArray()) {
                int idx = c - 'a';
                if (((mask >> idx) & 1) == 1) {
                    mask = Integer.MAX_VALUE;
                    break;
                }
                mask |= 1 << idx;
            }
            if (mask != Integer.MAX_VALUE) {
                lc1239MaskArr.add(mask);
            }
        }
        if (lc1239MaskArr.size() == 0) return 0;
        lc1239Backtrack(0, 0);
        return lc1239Result;
    }

    private void lc1239Backtrack(int mask, int curIdx) {
        lc1239Result = Math.max(lc1239Result, Integer.bitCount(mask));
        for (int i = curIdx; i < lc1239MaskArr.size(); i++) {
            if ((mask & lc1239MaskArr.get(i)) == 0) {
                mask ^= lc1239MaskArr.get(i);
                lc1239Backtrack(mask, i + 1);
                mask ^= lc1239MaskArr.get(i);
            }
        }
    }

    // LC818
    Integer[] lc818Memo;

    public int racecar(int target) {
        lc818Memo = new Integer[2 * target];
        return lc818Helper(target);
    }

    private int lc818Helper(int target) {
        if (lc818Memo[target] != null) return lc818Memo[target];
        int twoPowCeil = Integer.SIZE - Integer.numberOfLeadingZeros(target);
        if (target == (1 << twoPowCeil) - 1) {
            return twoPowCeil;
        }
        // Case 1 超越然后再返回
        int min = twoPowCeil + 1 + lc818Helper(((1 << twoPowCeil) - 1) - target);
        // Case 2 不超越 先掉头(R) 然后加速若干次(-1/-3/-7...)(A*back) 再掉头(R) 再加速
        for (int back = 0; back < twoPowCeil - 1; back++) {
            int distance = target - (1 << (twoPowCeil - 1)) + (1 << back);
            min = Math.min(min, (twoPowCeil - 1) + 2 + back + lc818Helper(distance));
        }
        return lc818Memo[target] = min;
    }


    // LC878
    public int nthMagicalNumber(int n, int a, int b) {
        final long mod = 1000000007;
        int gcd = gcd(a, b);
        int lcm = a * b / gcd;
        TreeSet<Integer> ts = new TreeSet<>();
        int aTimes = a;
        while (aTimes <= lcm) {
            ts.add(aTimes);
            aTimes += a;
        }
        int bTimes = b;
        while (bTimes <= lcm) {
            ts.add(bTimes);
            bTimes += b;
        }
        List<Integer> l = new ArrayList<>(ts);
        n--;
        int nthCycle = n / l.size();
        int m = n % l.size();
        long result = (((lcm % mod) * (nthCycle % mod)) % mod + l.get(m) % mod) % mod;
        return (int) result;
    }


    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC1563
    Integer[][] lc1563Memo;

    public int stoneGameV(int[] stoneValue) {
        int n = stoneValue.length;
        lc1563Memo = new Integer[n][n];
        int[] prefix = new int[stoneValue.length + 1];
        for (int i = 1; i <= stoneValue.length; i++) {
            prefix[i] = prefix[i - 1] + stoneValue[i - 1];
        }
        int left = 0, right = stoneValue.length - 1;
        return lc1563Helper(stoneValue, prefix, left, right);
    }

    private int lc1563Helper(int[] stoneValue, int[] prefix, int left, int right) {
        if (left >= right) return 0;
        if (lc1563Memo[left][right] != null) return lc1563Memo[left][right];
        int max = 0;
        if (left >= 0 && left < stoneValue.length && right >= 0 && right < stoneValue.length && left < right) {
            for (int i = left; i < right; i++) {
                int midLeft = i;
                int midRight = i + 1;
                int sumLeft = prefix[midLeft + 1] - prefix[left];
                int sumRight = prefix[right + 1] - prefix[midRight];
                int tmpResult;
                if (sumLeft < sumRight) {
                    tmpResult = sumLeft + lc1563Helper(stoneValue, prefix, left, midLeft);
                } else if (sumLeft > sumRight) {
                    tmpResult = sumRight + lc1563Helper(stoneValue, prefix, midRight, right);
                } else {
                    tmpResult = sumLeft + Math.max(lc1563Helper(stoneValue, prefix, left, midLeft), lc1563Helper(stoneValue, prefix, midRight, right));
                }
                max = Math.max(tmpResult, max);
            }
        }
        return lc1563Memo[left][right] = max;
    }


    // LC1563 理解错了题目, 以为是任意选择分成两堆, 使得绝对值的差最小; 实际是分成左右两半, 使得绝对值的差最小
    public int stoneGameVWA(int[] stoneValue) {
        int gain = 0;
        while (stoneValue.length > 1) {
            int n = stoneValue.length;
            int sum = Arrays.stream(stoneValue).sum();
            int bound = sum / 2;
            int[][] dpFrom = new int[n + 1][bound + 1];
            boolean[][] dp = new boolean[n + 1][bound + 1];
            dp[0][0] = true;
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j <= bound; j++) {
                    if (dp[i - 1][j]) {
                        dp[i][j] = true;
                        dpFrom[i][j] = dpFrom[i - 1][j];
                    } else {
                        if (j - stoneValue[i - 1] >= 0) {
                            dp[i][j] = dp[i - 1][j - stoneValue[i - 1]];
                        }
                        if (dp[i][j]) {
                            dpFrom[i][j] = i;
                        }
                    }
                }
            }
            int maxReachable;
            for (maxReachable = bound; maxReachable >= 0; maxReachable--) {
                if (dp[n][maxReachable]) break;
            }
            gain += maxReachable;
            List<Integer> selectedIdx = new ArrayList<>();
            int ptrIdx = maxReachable;
            while (dpFrom[n][ptrIdx] != 0) {
                selectedIdx.add(dpFrom[n][ptrIdx]);
                ptrIdx = ptrIdx - stoneValue[dpFrom[n][ptrIdx] - 1];
            }
            int[] alterSV = new int[selectedIdx.size()];
            int ctr = 0;
            for (int i = 0; i < selectedIdx.size(); i++) {
                alterSV[ctr++] = stoneValue[selectedIdx.get(i) - 1];
            }
            stoneValue = alterSV;
        }
        return gain;
    }

    // LC1191 这都可以???
    public int kConcatenationMaxSum(int[] arr, int k) {
        final long mod = 1000000007;
        long dp = 0;
        long max = 0;
        for (int i = 1; i <= arr.length; i++) {
            dp = Math.max(dp + arr[i - 1], arr[i - 1]);
            max = Math.max(dp, max);
        }
        if (k == 1) return (int) (max % mod);
        Pair<Long, Pair<Integer, Integer>> doubleDp = new Pair<>(0l, new Pair<>(0, 0));
        Pair<Long, Pair<Integer, Integer>> doubleMax = new Pair<>(0l, new Pair<>(0, 0));
        for (int i = 1; i <= 2 * arr.length; i++) {
            if (doubleDp.getKey() + arr[(i - 1) % arr.length] > arr[(i - 1) % arr.length]) {
                doubleDp = new Pair<>(doubleDp.getKey() + +arr[(i - 1) % arr.length], new Pair<>(doubleDp.getValue().getKey(), i));
            } else {
                doubleDp = new Pair<>((long) arr[(i - 1) % arr.length], new Pair<>(i, i));
            }
//            doubleDp = Math.max(doubleDp + arr[(i - 1 % arr.length)], arr[(i - 1) % arr.length]);
            if (doubleDp.getKey() > doubleMax.getKey()) {
                doubleMax = new Pair<>(doubleDp.getKey(), new Pair<>(doubleDp.getValue().getKey(), doubleDp.getValue().getValue()));
            }
        }
        if (doubleMax.getKey() == max) return (int) (max % mod);
        int left = doubleMax.getValue().getKey(), right = doubleMax.getValue().getValue();
        if (left == 0) left = 1;
        left--;
        right--;
        if (right - left + 1 < arr.length) {
            return (int) (doubleMax.getKey() % mod);
        } else if (right - left + 1 >= arr.length && right - left + 1 < 2 * arr.length) {
            // doubleMax = max*2-k -> k=max*2-doubleMax
            long gap = max * 2 - doubleMax.getKey();
            return (int) ((max * k - gap * (k - 1)) % mod);
        } else {
            return (int) ((k * max) % mod);
        }
    }

    // LC1614
    public int maxDepth(String s) {
        Deque<Character> stack = new LinkedList<>();
        char[] cArr = s.toCharArray();
        int maxDepth = 0;
        for (char c : cArr) {
            if (c == '(') {
                stack.push(c);
                maxDepth = Math.max(maxDepth, stack.size());
            } else if (c == ')') {
                stack.pop();
            }
        }
        return maxDepth;
    }

    // LC990 并查集
    public boolean equationsPossible(String[] equations) {
        DisjointSetUnion dsu = new DisjointSetUnion();
        Set<Pair<Integer, Integer>> notEqual = new HashSet<>();
        for (String e : equations) {
            dsu.add(e.charAt(0));
            dsu.add(e.charAt(3));
            if (e.charAt(1) == '=') {
                dsu.merge(e.charAt(0), e.charAt(3));
            } else {
                notEqual.add(new Pair<>((int) e.charAt(0), (int) e.charAt(3)));
            }
        }
        for (Pair<Integer, Integer> p : notEqual) {
            if (dsu.find(p.getKey()) == dsu.find(p.getValue())) return false;
        }
        return true;
    }

    // LC1319 并查集
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

// LC1600 lc判题有问题
class ThroneInheritance {
    String king;
    Map<String, List<String>> children;
    Set<String> death;

    public ThroneInheritance(String kingName) {
        king = kingName;
        children = new HashMap<>();
        death = new HashSet<>();
        children.put(kingName, new LinkedList<>());
    }

    public void birth(String parentName, String childName) {
        children.put(childName, new LinkedList<>());
        children.get(parentName).add(0, childName); // 倒序插入孩子
    }

    public void death(String name) {
        death.add(name);
    }

    public List<String> getInheritanceOrder() {
        return preorder();
    }

    private List<String> preorder() {
        List<String> result = new LinkedList<>();
        Deque<String> stack = new LinkedList<>();
        stack.add(king);
        while (!stack.isEmpty()) {
            String top = stack.pop();
            if (!death.contains(top)) {
                result.add(top);
            }
            for (String c : children.get(top)) {
                stack.push(c);
            }
        }
        return result;
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

// LC1845
class SeatManager {
    PriorityQueue<Integer> pq;

    public SeatManager(int n) {
        pq = new PriorityQueue<>();
        for (int i = 1; i <= n; i++) {
            pq.offer(i);
        }
    }

    public int reserve() {
        return pq.poll();
    }

    public void unreserve(int seatNumber) {
        pq.offer(seatNumber);
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

// LC427
// Definition for a QuadTree node.
class Node {
    public boolean val;
    public boolean isLeaf;
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;


    public Node() {
        this.val = false;
        this.isLeaf = false;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }

    public Node(boolean val, boolean isLeaf) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }

    public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomLeft = bottomLeft;
        this.bottomRight = bottomRight;
    }
};


class Solution {
    public Node construct(int[][] grid) {
        return helper(grid, 0, grid.length - 1, 0, grid[0].length - 1);
    }

    private Node helper(int[][] grid, int up, int down, int left, int right) {
        Node n = new Node();
        if (isSame(grid, up, down, left, right)) {
            n.isLeaf = true;
            n.val = grid[up][left] == 1 ? true : false;
            return n;
        }
        n.isLeaf = false;
        n.topLeft = helper(grid, up, (up + down) / 2, left, (left + right) / 2);
        n.topRight = helper(grid, up, (up + down) / 2, ((left + right) / 2) + 1, right);
        n.bottomLeft = helper(grid, ((up + down) / 2) + 1, down, left, (left + right) / 2);
        n.bottomRight = helper(grid, ((up + down) / 2) + 1, down, ((left + right) / 2) + 1, right);
        return n;
    }

    private boolean isSame(int[][] grid, int up, int down, int left, int right) {
        int val = grid[up][left];
        for (int i = up; i <= down; i++) {
            for (int j = left; j <= right; j++) {
                if (grid[i][j] != val) return false;
            }
        }
        return true;
    }
}
