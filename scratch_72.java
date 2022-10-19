import java.util.HashSet;
import java.util.Set;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.largestArea(new String[]{"02520253", "51551213", "03512513", "34312132", "21051025", "52005131", "34235150", "22154013"}));
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