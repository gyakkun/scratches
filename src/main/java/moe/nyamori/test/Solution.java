package moe.nyamori.test;

import moe.nyamori.test.ordered._1300.LC1334;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Solution {

    public static void main(String[] args) {
        long timing = System.currentTimeMillis();
        Solution s = new Solution();
        System.err.println(LC1334.INSTANCE.findTheCity(4, new int[][]{{0, 1, 3}, {1, 2, 1}, {1, 3, 4}, {2, 3, 1}}, 4));
        timing = System.currentTimeMillis() - timing;
        System.err.println("Timing: " + timing + "ms.");
    }


    public int[] handOutSnacks(int[] snacks) {
        int rows = snacks.length / 3;
        int[] res = new int[rows + 1];
        Map<Integer, Integer> freq = Arrays.stream(snacks).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(i -> 1)));
        int maxFreq = freq.keySet().stream().max(Comparator.naturalOrder()).get();
        int maxThree = (maxFreq / 3) * 3;
        TreeMap<Integer, Integer> tm = new TreeMap<>(freq);
        outer:
        for (int i = 0; i <= rows; i++) {
            TreeMap<Integer, Integer> copy = new TreeMap<>(tm);
            int target = i * 3;
            for (int j = maxThree; j >= 0; j -= 3) {
                if (!copy.containsKey(j)) continue;
                int f = copy.get(j);
                int minus = Math.min(f, target);
                copy.put(j, f - minus);
                target -= minus;
                if (target == 0) break;
            }
            if (target != 0) {
                res[i] = 0;
                continue;
            }
            int left = rows - i;
            for (Integer k : copy.descendingKeySet()) {
                int f = copy.get(k);
                if (f == 0) continue;
                if (f > left) {
                    res[i] = 0;
                    continue outer;
                }
            }
            res[i] = 1;
        }
        return res;
    }

    StringBuilder sb = new StringBuilder();
    boolean res = false;
    String resStr = null;

    public String gridTravel(int x, int y) {
        helper(0, 0, x, y, 0);
        if (!res) return "";
        return resStr;
    }

    public void helper(int curx, int cury, int targetx, int targety, int step) {
        if (step > 10) return;
        if (res) return;
        if (curx == targetx && cury == targety) {
            res = true;
            resStr = sb.toString();
            return;
        }
        int stepLen = 1 << step;
        int nextStepLen = 1 << (step + 1);
        // W
        sb.append("W");

        helper(curx - stepLen, cury, targetx, targety, step + 1);

        sb.deleteCharAt(sb.length() - 1);


        // E
        sb.append("E");

        helper(curx + stepLen, cury, targetx, targety, step + 1);

        sb.deleteCharAt(sb.length() - 1);


        // S
        sb.append("S");

        helper(curx, cury - stepLen, targetx, targety, step + 1);

        sb.deleteCharAt(sb.length() - 1);


        // N
        sb.append("N");

        helper(curx, cury + stepLen, targetx, targety, step + 1);

        sb.deleteCharAt(sb.length() - 1);

    }

    //
    public int numberOfWays(int[] people) {
        int len = people.length;
        int[] prefixZero = new int[len + 1];
        int[] prefixTwo = new int[len + 1];
        for (int i = 1; i <= len; i++) {
            if (people[i - 1] == 0) {
                prefixZero[i] = prefixZero[i - 1] + 1;
            } else {
                prefixZero[i] = prefixZero[i - 1];
            }
            if (people[i - 1] == 2) {
                prefixTwo[i] = prefixTwo[i - 1] + 1;
            } else {
                prefixTwo[i] = prefixTwo[i - 1];
            }
        }
        long res = 0;
        for (int i = 0; i < len; i++) {
            if (people[i] != 1) continue;
            int zero = prefixZero[i + 1] - prefixZero[0];
            int two = prefixTwo[len] - prefixTwo[i];
            res += (long) zero * (long) two;
            res %= 1000000007;
        }
        return (int) res;
    }

    public int squareShopping(int[][] grid, int price) {
        int len = grid.length;
        long[][] prefix = new long[len + 1][len + 1];
        for (int i = 1; i <= len; i++) {
            for (int j = 1; j <= len; j++) {
                prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + grid[i - 1][j - 1];
            }
        }
        int lo = 0, hi = len;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (judge(grid, prefix, price, mid)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if (!judge(grid, prefix, price, lo)) return 0;
        return lo;
    }

    public boolean judge(int[][] grid, long[][] prefix, int price, int side) {
        if (side == 0) return true;
        int m = grid.length;
        int n = grid.length;

        for (int i = side; i <= m; i++) { // 右侧端点
            int left = i - side;
            long area = prefix[i][i] - prefix[left][i] - prefix[i][left] + prefix[left][left];
            if (area > price) return false;
        }
        return true;
    }

    public int twoSum(int a, int b) {
        return a + b;
    }

    public int findTheLongestBalancedSubstring(String s) {
        int len = s.length();
        char[] ca = s.toCharArray();
        int[] count = new int[256];
        int res = 0;
        for (int i = 0; i < len; i++) {
            if (ca[i] == '1') {
                count['1']++;
                res = Math.max(res, 2 * Math.min(count['0'], count['1']));
            } else if (i == 0 || ca[i - 1] == '1') {
                count['0'] = 1; // ca[i] != '1'
                count['1'] = 0;
            } else {
                count['0']++;
            }
        }
        return res;
    }

}