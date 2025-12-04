package moe.nyamori.test.ordered._2200;


// Medium

import java.util.LinkedList;

/**
 * 2211. Count Collisions on a Road
 *
 * There are n cars on an infinitely long road. The cars are numbered from 0 to n - 1 from left to right and each car is present at a unique point.
 *
 * You are given a 0-indexed string directions of length n. directions[i] can be either 'L', 'R', or 'S' denoting whether the ith car is moving towards the left, towards the right, or staying at its current point respectively. Each moving car has the same speed.
 *
 * The number of collisions can be calculated as follows:
 *
 * When two cars moving in opposite directions collide with each other, the number of collisions increases by 2.
 * When a moving car collides with a stationary car, the number of collisions increases by 1.
 * After a collision, the cars involved can no longer move and will stay at the point where they collided. Other than that, cars cannot change their state or direction of motion.
 *
 * Return the total number of collisions that will happen on the road.
 *
 * Constraints:
 *
 * 1 <= directions.length <= 105
 * directions[i] is either 'L', 'R', or 'S'.
 */
public class LC2211 {
    public int countCollisions(String directions) {
        var res = 0;
        var stack = new LinkedList<Integer>();
        var carr = directions.toCharArray();
        // from left to right
        for (var i = 0; i < carr.length; i++) {
            var d = carr[i];
            if (d == 'R') {
                stack.push(i);
            } else if (d == 'S') {
                while (!stack.isEmpty()) {
                    var idx = stack.pop();
                    carr[idx] = 'S';
                    res++;
                }
            } else if (d == 'L') {
                if (stack.isEmpty()) continue;
                var idx = stack.pop();
                carr[idx] = 'S';
                carr[i] = 'S';
                res += 2;
                while (!stack.isEmpty()) {
                    var furtherIdx = stack.pop();
                    carr[furtherIdx] = 'S';
                    res++;
                }
            }
        }
        stack.clear();
        // from right to left
        for (var i = carr.length - 1; i >= 0; i--) {
            var d = carr[i];
            if (d == 'L') {
                stack.push(i);
            } else if (d == 'S') {
                while (!stack.isEmpty()) {
                    var idx = stack.pop();
                    carr[idx] = 'S';
                    res++;
                }
            } else if (d == 'R') {
                if (stack.isEmpty()) continue;
                var idx = stack.pop();
                carr[idx] = 'S';
                carr[i] = 'S';
                res += 2;
                while (!stack.isEmpty()) {
                    var furtherIdx = stack.pop();
                    carr[furtherIdx] = 'S';
                    res++;
                }
            }
        }
        return res;
    }
}
