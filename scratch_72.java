import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.checkIfExist(new int[]{-2, 0, 10, -19, 4, 6, -8}));
    }

    // LC1346
    public boolean checkIfExist(int[] arr) {
        Map<Integer, Set<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            m.putIfAbsent(arr[i], new HashSet<>());
            m.get(arr[i]).add(i);
        }
        for (int i : arr) {
            if (i == 0 && m.get(i).size() > 1) {
                return true;
            }
            if (i != 0 && i % 2 == 0 && m.containsKey(i / 2)) {
                return true;
            }
        }
        return false;
    }

    // LCP50
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] o : operations) {
            int x = o[0], y = o[1];
            int origX = gem[x], origY = gem[y];
            gem[y] = origY + origX / 2;
            gem[x] = origX - origX / 2;
        }
        Arrays.sort(gem);

        return gem[gem.length - 1] - gem[0];
    }

    // LC779
    public int kthGrammar(int n, int k) {
        int actualN = n - 1, actualK = k - 1;
        List<Integer> remain = new ArrayList<>();
        while (actualN >= 0) {
            remain.add(actualK % 2);
            actualK /= 2;
            actualN--;
        }
        int s = remain.size(), cur = 0;
        int[] zeroOne = {0, 1}, oneZero = {1, 0};
        for (int i = s - 1; i >= 0; i--) {
            int r = remain.get(i);
            int next = -1;
            if (cur == 0) {
                next = zeroOne[r];
            } else {
                next = oneZero[r];
            }
            cur = next;
        }
        return cur;
    }

    // LCP49 Hard ** 楼教主解法
    public long ringGame(long[] challenge) {
        int n = challenge.length;
        List<Pair<Integer, Long>> idxScoreList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            idxScoreList.add(new Pair<>(i, challenge[i]));
        }
        idxScoreList.sort(Comparator.comparingLong(i -> -i.getValue()));
        Function<Integer, Integer> getNext = i -> (i + 1) % n;
        Function<Integer, Integer> getPrev = i -> (i - 1 + n) % n;
        Function<Long, Boolean> check = initVal -> {
            long ongoingVal = initVal;
            BitSet visited = new BitSet(n);
            for (Pair<Integer, Long> p : idxScoreList) { // 从最大的开始遍历所有起点
                int idx = p.getKey();
                long necessaryScore = p.getValue();
                if (ongoingVal < necessaryScore) {
                    continue;
                }
                if (visited.get(idx)) {
                    continue;
                }
                long mergedScore = ongoingVal | necessaryScore;
                visited.set(idx);
                int leftPtr = idx, rightPtr = idx;
                while (true) {
                    if (getNext.apply(rightPtr) == leftPtr) return true;
                    if (mergedScore >= challenge[getPrev.apply(leftPtr)]) {
                        leftPtr = getPrev.apply(leftPtr);
                        mergedScore |= challenge[leftPtr];
                        visited.set(leftPtr);
                    } else if (mergedScore >= challenge[getNext.apply(rightPtr)]) {
                        rightPtr = getNext.apply(rightPtr);
                        mergedScore |= challenge[rightPtr];
                        visited.set(rightPtr);
                    } else {
                        break;
                    }
                }
            }
            return false;
        };

        long result = 0;
        for (int i = 63; i >= 0; i--) {
            long initVal = (result | (1L << i)) - 1;
            if (!check.apply(initVal)) {
                result |= (1L << i);
            }
        }
        return result;
    }


    // LC2311 **
    public int longestSubsequence(String s, int k) {
        int sLen = s.length(), kLen = Integer.SIZE - Integer.numberOfLeadingZeros(k);
        if (sLen < kLen) return sLen;
        int alignWithK = Integer.parseInt(s.substring(sLen - kLen), 2);
        int result = alignWithK > k ? kLen - 1 : kLen;
        int leadingZeros = (int) s.substring(0, sLen - kLen).chars().filter(i -> i == '0').count();
        return leadingZeros + result;
    }

    // LC2341
    public int[] numberOfPairs(int[] nums) {
        Map<Integer, List<Integer>> m = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity()));
        int a0 = m.values().stream().map(i -> i.size() / 2).reduce((a, b) -> a + b).get();
        int a1 = m.values().stream().map(i -> i.size() % 2).reduce((a, b) -> a + b).get();
        return new int[]{a0, a1};
    }

    // LC1262
    Integer[][] lc1262Memo;
    int[] lc1262Nums;

    public int maxSumDivThree(int[] nums) {
        lc1262Nums = nums;
        int n = nums.length;
        lc1262Memo = new Integer[n + 1][3];
        return helper(n - 1, 0);
    }

    private int helper(int idx, int targetRemain) {
        if (idx == 0) {
            if (lc1262Nums[idx] % 3 != targetRemain) return 0;
            return lc1262Nums[idx];
        }
        if (lc1262Memo[idx][targetRemain] != null) return lc1262Memo[idx][targetRemain];
        int result = 0;
        int currentRemain = lc1262Nums[idx] % 3, currentValue = lc1262Nums[idx];
        // Choose current
        int nextTargetRemain = (targetRemain - currentRemain + 3) % 3;
        int tmpChooseRightPart = helper(idx - 1, nextTargetRemain);
        int tmpResult = currentValue + tmpChooseRightPart;
        if (tmpResult % 3 == targetRemain) {
            result = Math.max(result, tmpResult);
        }
        // Don't choose current
        tmpResult = helper(idx - 1, targetRemain);
        if (tmpResult % 3 == targetRemain) {
            result = Math.max(result, tmpResult);
        }
        return lc1262Memo[idx][targetRemain] = result;
    }

    // LC2089
    public List<Integer> targetIndices(int[] nums, int target) {
        Arrays.sort(nums);
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) result.add(i);
        }
        return result;
    }

    // LC1383 ** Hard
    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        // Pair: <speed, efficiency>
        List<Pair<Integer, Integer>> employeeList = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            employeeList.add(new Pair<>(speed[i], efficiency[i]));
        }
        employeeList.sort(Comparator.comparingInt(i -> -i.getValue()));
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i.getKey()));
        long sumSpeed = 0L, result = 0L;
        for (int i = 0; i < n; i++) {
            Pair<Integer, Integer> minEfficiencyStaff = employeeList.get(i);
            int staffSpeed = minEfficiencyStaff.getKey(), staffEfficiency = minEfficiencyStaff.getValue();
            sumSpeed += staffSpeed;
            result = Math.max(result, sumSpeed * (long) staffEfficiency);
            pq.offer(minEfficiencyStaff);
            if (pq.size() == k) {
                Pair<Integer, Integer> p = pq.poll();
                sumSpeed -= p.getKey();
            }
        }
        return (int) (result % 1000000007L);
    }

    // LC1700
    public int countStudents(int[] students, int[] sandwiches) {
        int n = students.length;
        int count = 0;
        int[] remain = new int[2];
        for (int i : sandwiches) remain[i]++;
        while (count < n && remain[sandwiches[count]] > 0) {
            remain[sandwiches[count]]--;
            count++;
        }
        return n - count;
    }

    final static class LCS03 {// LCS 03
        Set<Integer> visited = new HashSet<>();
        char[][] matrix;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int finalResult = 0, row = 0, col = 0;
        boolean isTouchBoundary;

        public int largestArea(String[] grid) {
            matrix = new char[grid.length][];
            for (int i = 0; i < grid.length; i++) {
                matrix[i] = grid[i].toCharArray();
            }
            row = matrix.length;
            col = matrix[0].length;
            for (int i = 0; i < (row * col); i++) {
                isTouchBoundary = false;
                int result = lcs03Helper(i);
                if (!isTouchBoundary) finalResult = Math.max(result, finalResult);
            }
            return finalResult;
        }

        private int lcs03Helper(int i) {
            if (visited.contains(i)) return 0;
            visited.add(i);
            int r = i / col, c = i % col;
            if (r == 0 || r == row - 1 || c == 0 || c == col - 1 || matrix[r][c] == '0') isTouchBoundary = true;
            int result = 1;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (!(nr >= 0 && nr < row && nc >= 0 && nc < col)) continue;
                if (matrix[nr][nc] == '0') {
                    isTouchBoundary = true;
                    continue;
                }
                if (matrix[nr][nc] != matrix[r][c]) continue;
                int next = lcs03Helper(nr * col + nc);
                result += next;
            }
            return result;
        }
    }
}

class DSUArray {
    int[] father;
    int[] rank;
    int size;

    public DSUArray(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray() {
        this.size = 1 << 16;
        father = new int[1 << 16];
        rank = new int[1 << 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public boolean contains(int i) {
        if (i >= this.size || i < 0) return false;
        return father[i] != -1;
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (int i = 0; i < size; i++) {
            if (father[i] != -1) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
        }
        return result;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }

    public int getSelfGroupSize(int x) {
        return rank[find(x)];
    }

}
