import java.time.Duration
import java.time.Instant
import kotlin.math.PI
import kotlin.math.atan2
import java.util.Arrays


class Solution {

    // LC1154
    fun dayOfYear(date: String): Int {
        val monthDays = arrayOf(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        val sarr = date.split("-")
        val yyyy = sarr[0].toInt()
        val mm = sarr[1].toInt()
        val dd = sarr[2].toInt()
        var result = 0
        if (mm > 2) {
            if (yyyy % 4 == 4) {
                if (yyyy == 2000) result++
                else {
                    if (yyyy % 100 != 0) result++
                }
            }
        }
        for (i in 0 until mm - 1) result += monthDays[i]
        return result + dd
    }

    // LC475
    fun findRadius(houses: IntArray, heaters: IntArray): Int {
        Arrays.sort(houses)
        Arrays.sort(heaters)
        var lo: Long = 0
        var hi = (Int.MAX_VALUE / 2).toLong()
        while (lo < hi) {
            val mid = lo + (hi - lo) / 2
            if (check(houses, heaters, mid)) {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        return lo.toInt()
    }

    fun check(houses: IntArray, heaters: IntArray, radius: Long): Boolean {
        var maxRight = -1
        var prevEndIdx = -1
        for (heaterIdx in heaters) {
            var left = -1
            var right = -1
            left = if (radius > heaterIdx) 0 else (heaterIdx - radius).toInt()
            if (left > houses[houses.size - 1]) {
                return maxRight >= houses[houses.size - 1]
            }
            right =
                if (radius + heaterIdx.toLong() > houses[houses.size - 1]) houses[houses.size - 1] else (heaterIdx + radius).toInt()
            maxRight = Math.max(maxRight, right)
            // 然后找覆盖范围内的两个端点坐标的下标, 如果这个区间的左侧坐标的下标比原来的极右侧坐标下标大超过1, 则中间有房屋没有被覆盖, 直接返回false
            var lo = 0
            var hi = houses.size - 1
            while (lo < hi) { // 找houses里坐标大于等于left的最小值的下标
                val mid = lo + (hi - lo) / 2
                if (houses[mid] >= left) {
                    hi = mid
                } else {
                    lo = mid + 1
                }
            }
            if (houses[lo] < left) {
                // 这种情况就是说, 如果数轴上最右侧的房子都不在加热范围内, 也就是加热站太右了, 且半径太小, 覆盖不到数轴上的任意房屋
                // 那也没有继续往右遍历加热站的必要了, 直接返回当前能加热到的最右侧坐标下标是否大于等于最右侧房屋坐标
                return maxRight >= houses[houses.size - 1]
            }
            // 然后找覆盖范围内的两个端点坐标的下标, 如果这个区间的左侧坐标的下标比原来的极右侧坐标下标大超过1, 则中间有房屋没有被覆盖, 直接返回false
            val leftMostHouseIdx = lo
            if (leftMostHouseIdx - prevEndIdx > 1) return false

            // 找能覆盖到的最右侧的下标
            lo = 0
            hi = houses.size - 1
            while (lo < hi) { // 找到houses里面坐标小于等于right的最大值的下标
                val mid = lo + (hi - lo + 1) / 2
                if (houses[mid] <= right) {
                    lo = mid
                } else {
                    hi = mid - 1
                }
            }
            if (houses[lo] > right) {
                // 这种情况就是说, 你这个加热器太靠左了, 数轴上最左的房子的点都在加热范围的右侧, 完全覆盖不到。
                // 此时应该直接遍历下一个加热器
                continue
            }
            prevEndIdx = lo
            if (prevEndIdx == houses.size - 1) return true
        }
        return false
    }


    // LC997
    fun findJudge(n: Int, trust: Array<IntArray>): Int {
        val trustFrom = IntArray(n + 1)
        val trustTo = IntArray(n + 1)
        for (i in trust) {
            trustFrom[i[0]]++
            trustTo[i[1]]++
        }
        for (i in 1..n) {
            if (trustFrom[i] == 0 && trustTo[i] == n - 1) return i
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