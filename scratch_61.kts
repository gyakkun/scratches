import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.math.PI
import kotlin.math.atan2


class Solution {

    // LC1078
    fun findOcurrences(text: String, first: String, second: String): Array<String> {
        val s = text.split(" ");
        var i = 0
        val result = ArrayList<String>()
        while (i + 2 < s.size) {
            if (s[i] == first && s[i + 1] == second) {
                result.add(s[i + 2])
            }
            i++
        }
        return result.toTypedArray()
    }

    // LC1705
    fun eatenApples(apples: IntArray, days: IntArray): Int {
        val pq = PriorityQueue<IntArray>(compareBy { i -> i[1] })
        val n: Int = apples.size
        var result = 0
        for (whichDay in apples.indices) {
            if (apples[whichDay] == 0 && days[whichDay] == 0) {
                // skip
            } else if (apples[whichDay] != 0) {
                pq.offer(intArrayOf(apples[whichDay], days[whichDay] + whichDay))
            }
            if (!pq.isEmpty()) {
                var entry: IntArray? = null
                do {
                    val p = pq.poll()
                    if (whichDay >= p[1]) continue
                    entry = p
                    break
                } while (!pq.isEmpty())
                if (entry != null) {
                    entry[0]--
                    result++
                    if (entry[0] > 0) pq.offer(entry)
                }
            }
        }
        var whichDay = n
        while (!pq.isEmpty()) {
            var entry: IntArray? = null
            do {
                val p = pq.poll()
                if (whichDay >= p[1]) continue
                entry = p
                break
            } while (!pq.isEmpty())
            if (entry != null) {
                entry[0]--
                result++
                if (entry[0] > 0) pq.offer(entry)
            }
            whichDay++
        }
        return result
    }

    // LC1044
    val mod = 1000000007L
    val base1: Long = 29
    val base2: Long = 31
    var lc1044Str: String = ""

    fun longestDupSubstring(s: String): String {
        this.lc1044Str = s
        var lo = 0
        var hi = s.length - 1
        var result = ""
        var tmp: String
        while (lo < hi) { // 找最大值
            val mid = lo + (hi - lo + 1) / 2
            if (helper(mid).also { tmp = it } != "") {
                result = tmp
                lo = mid
            } else {
                hi = mid - 1
            }
        }
        return result
    }

    private fun helper(len: Int): String {
        val m1: MutableSet<Int> = HashSet()
        val m2: MutableSet<Int> = HashSet()
        var hash1: Long = 0
        var hash2: Long = 0
        var accu1: Long = 1
        var accu2: Long = 1
        for (i in 0 until len) {
            hash1 *= base1
            hash1 %= mod
            hash2 *= base2
            hash2 %= mod
            hash1 += (lc1044Str[i] - 'a').toLong()
            hash1 %= mod
            hash2 += (lc1044Str[i] - 'a').toLong()
            hash2 %= mod
            accu1 *= base1
            accu1 %= mod
            accu2 *= base2
            accu2 %= mod
        }
        m1.add(hash1.toInt())
        m2.add(hash2.toInt())
        for (i in len until lc1044Str.length) {
            val victim = lc1044Str.substring(i - len + 1, i + 1)
            hash1 =
                ((hash1 * base1 - accu1 * (lc1044Str[i - len] - 'a')) % mod + mod + lc1044Str[i].code.toLong() - 'a'.code.toLong()) % mod
            hash2 =
                ((hash2 * base2 - accu2 * (lc1044Str[i - len] - 'a')) % mod + mod + lc1044Str[i].code.toLong() - 'a'.code.toLong()) % mod
            if (m1.contains(hash1.toInt())
                && m2.contains(hash2.toInt())
            ) {
                return victim
            }
            m1.add(hash1.toInt())
            m2.add(hash2.toInt())
        }
        return ""
    }

    // LC686
    fun repeatedStringMatch(a: String, b: String): Int {
        if (a == b) return 1

        // 1 查频
        val aBool = BooleanArray(26)
        val bBool = BooleanArray(26)
        for (c in a) {
            aBool[c - 'a'] = true
        }
        for (c in b) {
            bBool[c - 'a'] = true
        }
        for (i in 0..25) {
            if (bBool[i] && !aBool[i]) {
                return -1
            }
        }

        val sb: StringBuilder = StringBuilder(b.length * 2)
        var ctr = 0
        // 2.补长
        do {
            sb.append(a)
            ctr++
        } while (sb.length < b.length)
        if (sb.indexOf(b) != -1) return ctr

        while (sb.length < b.length * 2 || ctr <= 2) {
            ctr++
            sb.append(a)
            if (sb.indexOf(b) != -1) return ctr
        }

        return -1
    }

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
            var left: Int = if (radius > heaterIdx) 0 else (heaterIdx - radius).toInt()
            if (left > houses[houses.size - 1]) {
                return maxRight >= houses[houses.size - 1]
            }
            var right: Int =
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
println(
    s.eatenApples(
        intArrayOf(3, 0, 0, 0, 0, 2),
        intArrayOf(3, 0, 0, 0, 0, 2)
    )
)
var after = Instant.now()
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")