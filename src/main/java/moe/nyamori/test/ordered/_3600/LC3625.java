package moe.nyamori.test.ordered._3600;

// Hard

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 * LC3625 Count Number of Trapezoids II
 *
 * You are given a 2D integer array points where points[i] = [xi, yi] represents the coordinates of the ith point on the Cartesian plane.
 *
 * Return the number of unique trapezoids that can be formed by choosing any four distinct points from points.
 *
 * A trapezoid is a convex quadrilateral with at least one pair of parallel sides. Two lines are parallel if and only if they have the same slope.
 *
 *
 * Constraints:
 *
 * 4 <= points.length <= 500
 * –1000 <= xi, yi <= 1000
 * All points are pairwise distinct.
 */
public class LC3625 {
    public int countTrapezoids(int[][] points) {
        // k 斜率 slope
        // b 截距 intercept
        var inf = 1_000_000_007d; // parallel to y axis
        var kToB = new TreeMap<Double, List<Double>>(); // to count without de-dup
        var midToK = new TreeMap<Integer, List<Double>>(); // to de-duplicate
        var res = 0L;
        for (var i = 0; i < points.length; i++) {
            var x1 = points[i][0];
            var y1 = points[i][1];
            for (var j = i + 1; j < points.length; j++) {
                var x2 = points[j][0];
                var y2 = points[j][1];

                var dx = x1 - x2;
                var dy = y1 - y2;

                var k = 0d;
                var b = 0d;
                if (dx == 0) {
                    k = inf;
                    b = x1; // we use the x to represent b
                } else {
                    k = ((double) dy) / ((double) dx);
                    // y = kx+b
                    // b = y1 - kx1
                    // FIXME: Here is tricky about the  precision
                    b = 1.0 * (y1 * dx - x1 * dy) / dx;
                }
                if (k == -0.0d) k = 0.0d; // float point handling
                if (b == -0.0d) b = 0.0d;
                // we use the hash technique here to represent the middle point of the line
                // abs(x1,x2) <=1000, hence times 10,000
                var mid = (x1 + x2) * 10000 + (y1 + y2);
                kToB.computeIfAbsent(k, (key) -> new ArrayList<>()).add(b);
                midToK.computeIfAbsent(mid, (key) -> new ArrayList<>()).add(k);
            }
        }

        for (var ktb : kToB.values()) { // k from small to large in order
            // only with same k can two lines form a trapezoid
            if (ktb.size() <= 1) continue; // need at least 2 points
            var count = new TreeMap<Double, Long>(); // count lines at the same b ("height")
            for (var b : ktb) count.compute(b, (key, v) -> v == null ? 1L : v + 1L);
            var numEdge = 0L;
            for (var i : count.values()) {
                res += numEdge * i;
                numEdge += i;
            }
        }

        // Subtract the parallelograms
        // 平行四边形重复计算了一遍
        for (var mtk : midToK.values()) {
            // if two line segments share a same middle point and k, then they are on the same line
            if (mtk.size() <= 1) continue;
            var count = new TreeMap<Double, Long>();
            for (var k : mtk) count.compute(k, (key, v) -> v == null ? 1L : v + 1L);
            var numLines = 0L;
            // any two line segments share a same middle point can form a parallelogram
            for (var i : count.values()) {
                res -= numLines * i; // for every
                numLines += i;
            }
        }
        return (int) res;
    }


}
