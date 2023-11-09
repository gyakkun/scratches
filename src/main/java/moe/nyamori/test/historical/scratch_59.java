package moe.nyamori.test.historical;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

class scratch_59 {
    public static void main(String[] args) {
        scratch_59 s = new scratch_59();
        Instant before = Instant.now();
        System.out.println(s.visiblePoints(List.of(List.of(2, 1), List.of(2, 2), List.of(3, 3)), 90, List.of(1, 1)));
        Instant after = Instant.now();
        System.err.println("TIMING: " + Duration.between(before, after).toMillis() + "ms");
    }

    // LC419
    class Lc419 {
        public int countBattleships(char[][] board) {
            int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
            boolean[][] visited;
            int result = 0;
            int m = board.length, n = board[0].length;
            visited = new boolean[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (board[i][j] == 'X' && !visited[i][j]) {
                        helper(i, j, board, direction, visited);
                        result++;
                    }
                }
            }
            return result;
        }

        private void helper(int r, int c, char[][] board, int[][] direction, boolean[][] visited) {
            if (visited[r][c]) return;
            visited[r][c] = true;
            for (int[] d : direction) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < board.length && nc >= 0 && nc < board[0].length && board[nr][nc] == 'X') {
                    helper(nr, nc, board, direction, visited);
                }
            }
        }
    }

    // LC1610 Two Pass Sliding Window
    public int visiblePoints(List<List<Integer>> points, int angle, List<Integer> location) {
        int x = location.get(0), y = location.get(1);
        double rAngle = ((double) angle / 360d) * 2;
        List<Double> radians = new ArrayList<>();
        int count = 0, result = 0;
        for (List<Integer> p : points) {
            int dx = p.get(0) - x, dy = p.get(1) - y;
            if (dx == 0 && dy == 0) count++;
            else {
                radians.add(Math.atan2(dy, dx) / Math.PI);
                radians.add(Math.atan2(dy, dx) / Math.PI + 2);
            }
        }
        if (radians.size() == 0) return count;
        Collections.sort(radians);
        double left = radians.get(0);
        double right = left + rAngle;
        int leftIdx = 0, rightIdx = 0;
        for (int i = 0; i < radians.size(); i++) {
            if (radians.get(i) > right) break;
            count++;
            rightIdx++;
        }
        result = Math.max(count, result);
        while (rightIdx < radians.size()) {
            int sameLeftCount = 1;
            while (leftIdx + 1 < radians.size() && radians.get(leftIdx) == radians.get(leftIdx + 1)) {
                leftIdx++;
                sameLeftCount++;
            }
            count -= sameLeftCount;
            leftIdx++;
            left = radians.get(leftIdx);
            right = left + rAngle;
            while (rightIdx < radians.size()) {
                if (radians.get(rightIdx) > right) break;
                count++;
                rightIdx++;
            }
            result = Math.max(result, count);
        }
        return result;
    }

    // LC851 **
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        int[] result = new int[n], indegree = new int[n];
        List<List<Integer>> mtx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) mtx.add(new ArrayList<>());
        for (int i = 0; i < n; i++) result[i] = i;
        for (int[] r : richer) { // r: [a,b] a is richer than b
            mtx.get(r[0]).add(r[1]); // a->b, first a (richer) then b
            indegree[r[1]]++;
        }
        Deque<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i++) if (indegree[i] == 0) q.offer(i);
        // result[x] = y, means that among people who are equal or richer than x, y is the quietest one
        while (!q.isEmpty()) {
            int p = q.poll();
            for (int next : mtx.get(p)) {
                if (quiet[result[p]] < quiet[result[next]]) {
                    result[next] = result[p];
                }
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.offer(next);
                }
            }
        }
        return result;
    }

    // LC630 ** Hard
    public int scheduleCourse(int[][] courses) {
        // [duration, lastDay]
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o[0]));
        Arrays.sort(courses, Comparator.comparingInt(o -> o[1]));
        int totalTime = 0;
        for (int[] i : courses) {
            if (totalTime + i[0] <= i[1]) {
                pq.offer(i);
                totalTime += i[0];
            } else {
                if (pq.isEmpty()) continue;
                int[] p = pq.peek();
                if (p[0] > i[0]) {
                    int delta = p[0] - i[0];
                    pq.poll();
                    pq.offer(i);
                    totalTime -= delta;
                }
            }
        }
        return pq.size();
    }
}

// LC911
class TopVotedCandidate59 {
    TreeMap<Integer, Integer> timeCanMap = new TreeMap<>();

    public TopVotedCandidate59(int[] persons, int[] times) {
        int n = persons.length;
        int[] canFreqMap = new int[n + 1];
        TreeMap<Integer, Integer> freqCanMap = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            int oldFreq = canFreqMap[persons[i]];
            int newFreq = oldFreq + 1;
            canFreqMap[persons[i]] = newFreq;
            freqCanMap.put(newFreq, persons[i]);
            timeCanMap.put(times[i], freqCanMap.lastEntry().getValue());
        }
    }

    public int q(int t) {
        return timeCanMap.floorEntry(t).getValue();
    }
}