import java.time.Duration
import java.time.Instant
import java.util.*


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

class Solution {

    // LC2022
    fun construct2DArray(original: IntArray, m: Int, n: Int): Array<IntArray> {
        val len: Int = original.size
        if (len != m * n) return arrayOf()
        val result = Array(m) { IntArray(n) }
        for (i in 0 until m) {
            System.arraycopy(original, i * n, result[i], 0, n)
        }
        return result
    }

    // LC507
    fun checkPerfectNumber(num: Int): Boolean {
        if (num === 1) return false
        val sqrt = Math.sqrt(num.toDouble()).toInt()
        var sum = 1
        for (i in 2..sqrt) {
            if (num % i === 0) {
                sum += i
                if (num / i !== i) {
                    sum += num / i
                }
            }
            if (sum > num) return false
        }
        return sum == num
    }

    // LC846
    fun isNStraightHand(hand: IntArray, groupSize: Int): Boolean {
        if (hand.size % groupSize != 0) return false
        val m = TreeMap<Int, Int>()
        for (i in hand) {
            m[i] = if (m[i] == null) 1 else m[i]!! + 1
            // m[i] = m.getOrDefault(i, 0) + 1
        }
        while (!m.isEmpty()) {
            val firstKey = m.firstKey()
            for (i in 0 until groupSize) {
                val targetKey = firstKey + i
                m[targetKey]?.let {
                    m[targetKey] = it - 1
                    if (m[targetKey] == 0) m.remove(targetKey)
                } ?: return false;

                // if (!m.containsKey(targetKey)) return false
                // m[targetKey] = m[targetKey]!! - 1
                // if (m[targetKey] == 0) m.remove(targetKey)
            }
        }
        return true
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
}


