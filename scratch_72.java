import javafx.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.largestArea(new String[]{"02520253", "51551213", "03512513", "34312132", "21051025", "52005131", "34235150", "22154013"}));
    }

    // LC1262 WA
    public int maxSumDivThree(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 3 == 0) return sum;
        Map<Integer, List<Integer>> m = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(i -> i % 3));
        TreeSet<Integer> remain1 = new TreeSet<>(m.getOrDefault(1, new ArrayList<>()));
        TreeSet<Integer> remain2 = new TreeSet<>(m.getOrDefault(2, new ArrayList<>()));
        int minReduce = sum;
        int targetRemain = sum % 3;
        if (targetRemain == 1) {
            if (remain1.size() >= 1) {
                minReduce = Math.min(minReduce, remain1.first());
            }
            if (remain2.size() >= 2) {
                Iterator<Integer> it = remain2.iterator();
                int remain2First2Sum = 0;
                for (int i = 0; i < 2; i++) {
                    int tmp = it.next();
                    remain2First2Sum += tmp;
                }
                minReduce = Math.min(minReduce, remain2First2Sum);
            }
        } else if (targetRemain == 2) {
            if (remain2.size() >= 1) {
                minReduce = Math.min(minReduce, remain2.first());
            }
            if (remain1.size() >= 2) {
                Iterator<Integer> it = remain1.iterator();
                int remain1First2Sum = 0;
                for (int i = 0; i < 2; i++) {
                    int tmp = it.next();
                    remain1First2Sum += tmp;
                }
                minReduce = Math.min(minReduce, remain1First2Sum);
            }
        }
        return sum - minReduce;
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

    // LCS 03
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
            int result = helper(i);
            if (!isTouchBoundary) finalResult = Math.max(result, finalResult);
        }
        return finalResult;
    }

    private int helper(int i) {
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
            int next = helper(nr * col + nc);
            result += next;
        }
        return result;
    }
}