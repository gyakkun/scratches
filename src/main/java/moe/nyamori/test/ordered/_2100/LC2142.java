package moe.nyamori.test.ordered._2100;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

// Hard
class LC2142 {
    public static void main(String[] args) {
        var sol = new LC2142();
        var n = 11;
        var batteries = new int[]{9752,5199,5688,7944,611,5411,6097,5839,4207,6357,7317,6450,7450,5861,8565,4018,8770,9772,4011,5446,4499,5522,505,4536,7443,2287,5327,4999,8536,6377,1732,3939,7997,7713,2620,26,2499,4233,7797,9896,5816,736,8782,8600,1516,6505,706,1056,5403,2155,9679,4694,2286,674,7378,5549,9491,3334,7808,3817,4890,9605,8309,6838,2273,4709,1206,9228,5697,317,9869,412,9084,6852,3446,6937,8364,4774,2487,5291,3944,6420,9712,8383,4953,4455,8121,6047,3037,7501,7587,6637,5379,4655,6282,1736,3636,2649,8458,9094,2232,9609,2932,3901,9865,9870,1789,230,6678,7336,1636,1103,5879,8553,3971,3873,2485,5074,938,3292,4398,9850,9965,7982,7608,7635,1349,7496,120,9305,119,1283,6933,4494,5588,1996,507,1144,6808,9175,7510,5483,8888,5227,7234,5544,5391,302,9884,6546,8065,2683,219,9379,9557,4315,8380,873,6559,4895,8110,4965,6573,2357,4423,1283,3212,7374,7995,1946,5258,4208,6867,9802,9692,4274,4580,870,4650,7715,6861,1737,7066,1521,3172,4459,3407,2076,8657,3094,5318,7542,8964,823,1073,3352,6168,3980,9922,1526,3375,872,2370,291};
//        n = 2;
//        batteries = new int[]{1, 1, 1, 1};
        var timing = System.currentTimeMillis();
        System.err.println(sol.maxRunTime(n, batteries));
        System.err.println("Timing: " + (System.currentTimeMillis()-timing) + "ms");
    }

    public long maxRunTime(int n, int[] batteries) {
        if (n > batteries.length) return 0L;
        var lo = 1;
        var hi = Arrays.stream(batteries).sum();
        while (lo < hi) {
            var mid = lo + (hi - lo + 1) / 2;
            if (determine(mid, n, batteries)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if (!determine(lo, n, batteries)) return 0L;
        return lo;
    }

    /**
     * Determine if the batteries can run these hours
     * @param hours how many hours
     * @param n number of computers
     * @param batteries battery life array
     * @return true if it can, otherwise false
     */
    private boolean determine(int hours, int n, int[] batteries) {
        // Greedy strategy: we need to use those long-lived batteries first
        // otherwise we may run out of batteries first, while the big batteries
        // still can supply power
        var pq = new PriorityQueue<Integer>(batteries.length, Comparator.reverseOrder()); // larger first
        for (var i : batteries) pq.offer(i);
        outer:
        while (pq.size() >= n && hours > 0) {
            var list = new ArrayList<Integer>(n);

            inner:
            for (var i = 0; i < n; i++) {
                var batteryLife = pq.poll();
                if (batteryLife == null) {
                    return false;
                }
                if (batteryLife >= 2L) {
                    list.add(batteryLife - 1); // take one hour of each
                } else {
                    // continue
                }
            }
            list.forEach(pq::offer);
            hours--;
        }
        return hours == 0;
    }
}