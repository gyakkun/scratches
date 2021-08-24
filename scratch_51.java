import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println();

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC205
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> m = new HashMap<>();
        Map<Character, Character> reverseM = new HashMap<>();
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        for (int i = 0; i < cs.length; i++) {
            if (!m.containsKey(cs[i])) {
                m.put(cs[i], ct[i]);
                if (!reverseM.containsKey(ct[i])) {
                    reverseM.put(ct[i], cs[i]);
                } else {
                    return false;
                }
            } else {
                if (m.get(cs[i]) != ct[i]) return false;
                if (reverseM.get(ct[i]) != cs[i]) return false;
            }
        }
        return true;
    }

    // LC787
    int lc787Result = Integer.MAX_VALUE;

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        // k站中转, 即最多可以坐k+1次航班
        // flights[i] = [fromi, toi, pricei]
        int[][] reach = new int[n][n];
        int[] maxStop = new int[n]; // 经停i 的最大跳数
        int[] minCost = new int[n];
        Arrays.fill(minCost, Integer.MAX_VALUE / 2);
        for (int[] r : reach) Arrays.fill(r, -1);
        for (int[] f : flights) reach[f[0]][f[1]] = f[2];
        lc787Dfs(src, dst, reach, maxStop, minCost, k + 1, 0);
        return lc787Result == Integer.MAX_VALUE ? -1 : lc787Result;
    }

    private void lc787Dfs(int cur, int dst, int[][] reach, int[] maxStop, int[] minCost, int limit, int price) {
        if (cur == dst) {
            lc787Result = Math.min(lc787Result, price);
            return;
        }
        if (limit > 0) {
            for (int i = 0; i < reach.length; i++) {
                if (reach[cur][i] != -1) {
                    int minCostToI = minCost[i], costFromCurToI = price + reach[cur][i];
                    if (costFromCurToI > lc787Result) continue; // 剪枝, 如果在这一站的中转都要比最小结果大, 就没必要DFS下去了
                    if (minCostToI > costFromCurToI) {
                        lc787Dfs(i, dst, reach, maxStop, minCost, limit - 1, costFromCurToI);
                        minCost[i] = costFromCurToI;
                        maxStop[i] = limit - 1;
                    } else if (maxStop[i] < limit - 1) {
                        lc787Dfs(i, dst, reach, maxStop, minCost, limit - 1, costFromCurToI);
                    }
                }
            }
        }
    }

    // LC282 ** Hard
    List<String> lc282Result = new ArrayList<>();

    public List<String> addOperators(String num, int target) {
        lc282Dfs(0, num, target, new StringBuilder(), 0, 0, 0);
        return lc282Result;
    }

    private void lc282Dfs(int idx, String num, int target, StringBuilder sb, int cur, int pre, int accumulate) {
        if (idx == num.length()) {
            if (accumulate == target && cur == 0) {
                lc282Result.add(sb.substring(1));
            }
            return;
        }
        // 溢出判断
        if ((cur * 10 + num.charAt(idx) - '0') / 10 != cur) return;

        cur = cur * 10 + num.charAt(idx) - '0';
        String curStr = String.valueOf(cur);

        // 空操作
        if (cur > 0) {
            lc282Dfs(idx + 1, num, target, sb, cur, pre, accumulate);
        }

        // +
        sb.append("+");
        sb.append(cur);
        lc282Dfs(idx + 1, num, target, sb, 0, cur, accumulate + cur);
        sb.delete(sb.length() - 1 - curStr.length(), sb.length());


        if (sb.length() != 0) {

            // -
            sb.append("-");
            sb.append(cur);
            lc282Dfs(idx + 1, num, target, sb, 0, -cur, accumulate - cur);
            sb.delete(sb.length() - 1 - curStr.length(), sb.length());

            // *
            sb.append("*");
            sb.append(cur);
            lc282Dfs(idx + 1, num, target, sb, 0, cur * pre, accumulate - pre + cur * pre);
            sb.delete(sb.length() - 1 - curStr.length(), sb.length());

        }
    }

    // LC1896 ** Hard
    public int minOperationsToFlip(String expression) {
        Deque<int[]> stack = new LinkedList<>(); // [p,q] 表示变为0要p步, 变为1要q步
        Deque<Character> ops = new LinkedList<>();
        Set<Character> rightSideOp = new HashSet<Character>() {{
            add('0');
            add('1');
            add(')');
        }};
        for (char c : expression.toCharArray()) {
            if (rightSideOp.contains(c)) {
                if (c == '0') stack.push(new int[]{0, 1});
                else if (c == '1') stack.push(new int[]{1, 0});
                else if (c == ')') ops.pop();

                if (ops.size() != 0 && ops.peek() != '(') {
                    char op = ops.pop();
                    int[] pair2 = stack.pop();
                    int[] pair1 = stack.pop();
                    int[] newEntry;
                    if (op == '&') {
                        newEntry = new int[]{
                                Math.min(pair1[0], pair2[0]),
                                Math.min(pair1[1] + pair2[1], Math.min(pair1[1], pair2[1]) + 1)
                        };
                    } else {
                        newEntry = new int[]{
                                Math.min(pair1[0] + pair2[0], Math.min(pair1[0], pair2[0]) + 1),
                                Math.min(pair1[1], pair2[1])
                        };
                    }
                    stack.push(newEntry);
                }
            } else {
                ops.push(c);
            }
        }
        int[] last = stack.pop();
        return Math.max(last[0], last[1]);
    }

    // LC1223
    public List<String> removeSubfolders(String[] folder) {
        Set<String> prefix = new HashSet<>();
        Arrays.sort(folder, Comparator.comparingInt(o -> o.length()));
        for (String f : folder) {
            int ptr = 0;
            while (ptr < f.length()) {
                int last = ptr + 1;
                while (last != f.length() && f.charAt(last) != '/') last++;
                if (prefix.contains(f.substring(0, last))) break;
                ptr = last;
            }
            if (ptr == f.length()) prefix.add(f);
        }
        return new ArrayList<>(prefix);
    }
}

// LC 1226 哲学家进餐
class DiningPhilosophers {

    ReentrantLock[] locks = new ReentrantLock[]{
            new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock(), new ReentrantLock()
    };

    ReentrantLock pickBoth = new ReentrantLock();


    public DiningPhilosophers() {

    }

    // call the run() method of any runnable to execute its code
    public void wantsToEat(int philosopher,
                           Runnable pickLeftFork,
                           Runnable pickRightFork,
                           Runnable eat,
                           Runnable putLeftFork,
                           Runnable putRightFork) throws InterruptedException {
        int left = (philosopher + 1) % 5, right = philosopher;

        pickBoth.lock();

        locks[left].lock();
        locks[right].lock();
        pickLeftFork.run();
        pickRightFork.run();

        pickBoth.unlock();

        eat.run();

        putLeftFork.run();
        putRightFork.run();

        locks[left].unlock();
        locks[right].unlock();
    }
}