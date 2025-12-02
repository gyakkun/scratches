package moe.nyamori.test.ordered._3600;

// Medium

import java.util.HashMap;

/**
 * You are given a 2D integer array points, where points[i] = [xi, yi] represents the coordinates of the ith point on the Cartesian plane.
 *
 * A horizontal trapezoid is a convex quadrilateral with at least one pair of horizontal sides (i.e. parallel to the x-axis). Two lines are parallel if and only if they have the same slope.
 *
 * Return the number of unique horizontal trapezoids that can be formed by choosing any four distinct points from points.
 *
 * Since the answer may be very large, return it modulo 109 + 7.
 *
 * Constraints:
 *
 * 4 <= points.length <= 105
 * –108 <= xi, yi <= 108
 * All points are pairwise distinct.
 */
public class LC3623 {
    public static void main(String[] args) {
        var sol = new LC3623();
        // var input = new int[][]{{0, 0}, {1, 0}, {0, 1}, {2, 1}};
        // var input = new int[][]{{1, 0}, {2, 0}, {3, 0}, {2, 2}, {3, 2}};
        var input = new int[][]{{-17, -75}, {49, -75}, {-81, -75}, {-75, -65}, {-63, -65}, {68, -75}};
        System.err.println(sol.countTrapezoids(input));
    }

    public int countTrapezoids(int[][] points) {
        var mod = 1_000_000_007L;
        var map = new HashMap<Integer, Integer>();
        for (var p : points) {
            // by y-axis coordinate
            var x = p[0];
            var y = p[1];
            // as points are pairwise distinct,
            // no need to consider if there are
            // points on the same x coordinate
            map.compute(y, (key, val) -> val == null ? 1 : val + 1);
        }
        var sum = 0L;
        // one cycle match? 单循环赛
        var ys = map.keySet().stream().sorted().toList();
        for (var i = 0; i < ys.size(); i++) {
            var n1 = (long) map.get(ys.get(i));
            if (n1 <= 1) continue;
            var c1 = n1 * (n1 - 1) / 2L;
            c1 %= mod;
            if (n1 == 2L) c1 = 1L;
            for (var j = i + 1; j < ys.size(); j++) {
                var n2 = (long) map.get(ys.get(j));
                if (n2 <= 1L) continue;
                var c2 = n2 * (n2 - 1) / 2L;
                c2 %= mod;
                if (n2 == 2L) c2 = 1L;
                // Combination formula: C(m,n) = n!/m!(n-m!)
                // since m is always 2
                // then it will be C(2,n) = n!/2*(n-2)! = n*(n-1)*(n-2)/2
                // n is not that big (10^5), we can store it in long (10^18)
                // but c1*c2 is too big then
                sum += c1 * c2;
                sum %= mod;
            }
        }
        return (int) sum;
    }
}
