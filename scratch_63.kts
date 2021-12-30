import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.HashMap

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

    // LC846
    fun isNStraightHand(hand: IntArray, groupSize: Int): Boolean {
        if (hand.size % groupSize != 0) return false
        val m = TreeMap<Int, Int>()
        for (i in hand) {
            m[i] = m.getOrDefault(i, 0) + 1
        }
        while(!m.isEmpty()){
            val firstKey = m.firstKey()
            for (i in 0 until groupSize) {
                if(!m.containsKey(firstKey+i)) return false
                m[firstKey + i] = m[firstKey + i]!! - 1
                if(m[firstKey+i]==0) m.remove(firstKey+i)
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


