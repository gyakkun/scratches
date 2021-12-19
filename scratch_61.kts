import java.time.Duration
import java.time.Instant
import kotlin.math.PI
import kotlin.math.atan2

class Solution {

    // LC997
    fun findJudge(n: Int, trust: Array<IntArray>): Int {
        val trustFrom = IntArray(n + 1)
        val trustTo = IntArray(n + 1)
        for (i in trust) {
            trustFrom[i[0]]++
            trustTo[i[1]]++
        }
        for (i in 1..n) {
            if (trustFrom[i] == 0 && trustTo[i] == n-1) return i
        }
        return -1
    }

    // LC1610
    fun visiblePoints(points: List<List<Int>>, angle: Int, location: List<Int>): Int {
        var count = 0
        var result = 0
        val rAngle: Double = (angle.toDouble() / 360) * 2
        val x = location[0]
        val y = location[1]
        var radians = ArrayList<Double>()
        for (point in points) {
            val dx = point[0] - x
            val dy = point[1] - y
            if (dx == 0 && dy == 0) count++
            else {
                val r = atan2(dy.toDouble(), dx.toDouble()) / PI
                radians.add(r)
                radians.add(r + 2.0)
            }
        }
        if (radians.size == 0) return count
        radians.sort()
        var left: Double = radians.get(0)
        var right: Double = left + rAngle
        var leftIdx = 0
        var rightIdx = 0
        for (i in 0 until radians.size) {
            if (radians.get(i) > right) break;
            rightIdx++
            count++
        }
        result = Math.max(result, count)
        while (rightIdx < radians.size) {
            var sameLeftCount = 1
            while (leftIdx + 1 < radians.size && radians[leftIdx + 1] == radians[leftIdx]) {
                leftIdx++
                sameLeftCount++
            }
            count -= sameLeftCount
            leftIdx++
            left = radians[leftIdx]
            right = left + rAngle
            while (rightIdx < radians.size) {
                if (radians[rightIdx] > right) break;
                count++
                rightIdx++
            }
            result = Math.max(result, count)
        }
        return result
    }

    // LC1518
    fun numWaterBottles(numBottles: Int, numExchange: Int): Int {
        var count = numBottles
        var empty = numBottles
        while (empty >= numExchange) {
            val ex = empty / numExchange
            empty %= numExchange
            count += ex
            empty += ex
        }
        return count
    }
}

var before = Instant.now()
var s = Solution()
println(s.visiblePoints(listOf(listOf(2, 1), listOf(2, 2), listOf(3, 3)), 90, listOf(1, 1)))
var after = Instant.now()
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")