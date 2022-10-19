import javafx.util.Pair;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.maxSumDivThree(new int[]{1, 2, 3, 4, 4}));
    }

    // LC1262
    Integer[][] memo;
    int[] lc1262Nums;

    public int maxSumDivThree(int[] nums) {
        lc1262Nums = nums;
        int n = nums.length;
        memo = new Integer[n + 1][3];
        int result = 0;
        for (int i = n - 1; i >= 0; i--) {
            result = Math.max(result, helper(i, 0));
        }
        return result;
    }

    private int helper(int idx, int targetRemain) {
        if (idx == 0) {
            if (lc1262Nums[idx] % 3 != targetRemain) return 0;
            return lc1262Nums[idx];
        }
        if (memo[idx][targetRemain] != null) return memo[idx][targetRemain];
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
        return memo[idx][targetRemain] = result;
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