package moe.nyamori.test.ordered._3600;

// Hard

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
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
        var slopeMap = new HashMap<Pair<Integer>, List<Pair<Pair<Integer>>>>();
        for (var i = 0; i < points.length; i++) {
            for (var j = i + 1; j < points.length; j++) {
                var dx = points[i][0] - points[j][0];
                var dy = points[i][1] - points[j][1];
                if (dx == 0) {
                    dy = 1;
                } else if (dy == 0) {
                    dx = 1;
                } else {
                    var gcd = gcd(dx, dy);
                    dx /= gcd;
                    dy /= gcd;
                }
                slopeMap.computeIfAbsent(new Pair<>(dx, dy), (p) -> new ArrayList<>())
                        .add(new Pair<>(new Pair<>(points[i][0], points[i][1]), new Pair<>(points[j][0], points[j][1])));
            }
        }
    }

    private static int gcd(int a, int b) {
        return a % b == 0 ? gcd(b, a % b) : b;
    }

    record Pair<T>(
            T first,
            T second
    ) {
    }


}
