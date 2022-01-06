import java.lang.StringBuilder
import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.HashSet


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

    // LC71
    fun simplifyPath(path: String): String {
        val escape = setOf(".", "..", "")
        val list = path.split("/")
        val stack = LinkedList<String>()
        val result = StringBuilder("/")
        for (i in list) {
            if (!escape.contains(i)) {
                stack.push(i)
            } else if (i == "..") {
                if (!stack.isEmpty()) {
                    stack.pop()
                }
            }
        }
        while (!stack.isEmpty()) {
            result.append(stack.pollLast() + "/")
        }
        result.deleteCharAt(result.length - 1)
        return if (result.toString() == "") "/" else result.toString()
    }

    // LC1756
    fun modifyString(s: String): String {
        val sb = StringBuilder(s.length)
        outer@ for (i in s.indices) {
            val c = s[i]
            if (c.isLetter()) {
                sb.append(c)
                continue@outer
            }
            val hs = HashSet<Char>(3)
            if (i > 0) {
                hs.add(sb[sb.length - 1])
            }
            if (i + 1 < s.length && s[i + 1].isLetter()) {
                hs.add(s[i + 1])
            }
            for (j in 0 until 26) {
                val alt: Char = 'a' + j
                if (!hs.contains(alt)) {
                    sb.append(alt)
                    continue@outer
                }
            }
        }
        return sb.toString()
    }

    // LC913 Minmax
    val TIE = 0
    val CAT_WIN = 2
    val MOUSE_WIN = 1
    lateinit var lc913Memo: Array<Array<Array<Int?>>>

    fun catMouseGame(graph: Array<IntArray>): Int? {
        lc913Memo = Array(graph.size * 2 + 1) {
            Array<Array<Int?>>(graph.size + 1) {
                arrayOfNulls(
                    graph.size + 1
                )
            }
        }
        return lc913Helper(0, graph, 1, 2)
    }

    private fun lc913Helper(steps: Int, graph: Array<IntArray>, mousePoint: Int, catPoint: Int): Int? {
        if (steps >= 2 * graph.size) return TIE
        if (lc913Memo[steps][mousePoint][catPoint] != null) return lc913Memo[steps][mousePoint][catPoint]
        if (mousePoint == catPoint) return CAT_WIN.also { lc913Memo[steps][mousePoint][catPoint] = it }
        if (mousePoint == 0) return MOUSE_WIN.also { lc913Memo[steps][mousePoint][catPoint] = it }
        val isMouse = steps % 2 == 0
        return if (isMouse) {
            var catCanWin = true
            for (i in graph[mousePoint]) {
                val nextResult = lc913Helper(steps + 1, graph, i, catPoint)
                if (nextResult == MOUSE_WIN) {
                    return MOUSE_WIN.also { lc913Memo[steps][mousePoint][catPoint] = it }
                } else if (nextResult == TIE) {   /// 极小化极大: 猫嬴是一个极大值, 如果nextResult == CAT_WIN, 但是nextWin存在极小值TIE, 则选TIE不选CAT_WIN
                    catCanWin = false
                }
            }
            if (catCanWin) CAT_WIN.also {
                lc913Memo[steps][mousePoint][catPoint] = it
            } else TIE.also { lc913Memo[steps][mousePoint][catPoint] = it }
        } else {
            var mouseCanWin = true
            for (i in graph[catPoint]) {
                if (i == 0) continue
                val nextResult = lc913Helper(steps + 1, graph, mousePoint, i)
                if (nextResult == CAT_WIN) {
                    return CAT_WIN.also { lc913Memo[steps][mousePoint][catPoint] = it }
                } else if (nextResult == TIE) {
                    mouseCanWin = false
                }
            }
            if (mouseCanWin) MOUSE_WIN.also {
                lc913Memo[steps][mousePoint][catPoint] = it
            } else TIE.also { lc913Memo[steps][mousePoint][catPoint] = it }
        }
    }

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


