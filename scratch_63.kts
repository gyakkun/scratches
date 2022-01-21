import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.HashSet


var before = Instant.now()
var s = Solution()
println(
    s.eatenApples(
        intArrayOf(3, 0, 0, 0, 0, 2), intArrayOf(3, 0, 0, 0, 0, 2)
    )
)
var after = Instant.now()
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")

class Solution {

    // LC1345
    fun minJumps(arr: IntArray): Int {
        val m: MutableMap<Int, MutableSet<Int>> = HashMap()
        for (i in arr.indices) {
            m[arr[i]]?.let { it.add(i) } ?: let { m[arr[i]] = HashSet<Int>().also { it.add(i) } }
        }
        val q: Deque<Int> = LinkedList<Int>().also { it.offer(0) }
        val visited = BitSet(arr.size)
        var layer = -1
        while (!q.isEmpty()) {
            layer++
            val qs = q.size
            for (i in 0 until qs) {
                val p = q.poll()
                val v = arr[p]
                if (visited[p]) continue
                visited.set(p)
                if (p == arr.size - 1) return layer

                // i + 1
                if (p + 1 < arr.size && !visited[p + 1]) {
                    q.add(p + 1)
                }

                // i-1
                if (p - 1 >= 0 && !visited[p - 1]) {
                    q.add(p - 1)
                }

                // same value
                m[v]?.let {
                    for (smi in it) {
                        if (!visited[smi]) {
                            q.add(smi)
                        }
                    }
                }
                m.remove(v)
            }
        }
        return -1
    }

    // LC219
    fun containsNearbyDuplicate(nums: IntArray, k: Int): Boolean {
        val m: MutableMap<Int, TreeSet<Int>> = HashMap()
        for (i in nums.indices) {
            m.putIfAbsent(nums[i], TreeSet())
            val ts = m[nums[i]]!!
            val lb = i - k
            val hb = i + k
            val hr = ts.higher(lb)
            val lw = ts.lower(hb)
            if (hr != null && Math.abs(i - hr) <= k) {
                System.err.println(ts.ceiling(hr))
                return true
            }
            if (lw != null && Math.abs(i - lw) <= k) {
                System.err.println(ts.floor(lw))
                return true
            }
            ts.add(i)
        }
        return false
    }

    // LC1220
    lateinit var memo: Array<Array<Long?>>
    val mod: Long = 1000000007

    fun countVowelPermutation(n: Int): Int {
        memo = Array(n + 1) { arrayOfNulls<Long?>(6) }
        var result: Long = 0
        for (i in 0..4) {
            result = (result + helper(n - 1, i)) % mod
        }
        return result.toInt()
    }

    private fun helper(remainLetters: Int, currentLetterIdx: Int): Long {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        if (remainLetters == 0) return 1
        if (memo[remainLetters][currentLetterIdx] != null) return memo[remainLetters][currentLetterIdx]!! % mod
        when (currentLetterIdx) {
            0 -> return (helper(remainLetters - 1, 1) % mod).also {
                memo[remainLetters][currentLetterIdx] = it
            }
            1 -> return ((helper(remainLetters - 1, 0)
                    + helper(remainLetters - 1, 2)) % mod).also { memo[remainLetters][currentLetterIdx] = it }
            2 -> return ((helper(remainLetters - 1, 0)
                    + helper(remainLetters - 1, 1)
                    + helper(remainLetters - 1, 3)
                    + helper(remainLetters - 1, 4)) % mod).also { memo[remainLetters][currentLetterIdx] = it }
            3 -> return ((helper(remainLetters - 1, 2)
                    + helper(remainLetters - 1, 4)) % mod).also { memo[remainLetters][currentLetterIdx] = it }
            4 -> return (helper(remainLetters - 1, 0) % mod).also { memo[remainLetters][currentLetterIdx] = it }
        }
        return 0
    }

    // LC1716
    fun totalMoney(n: Int): Int {
        val week: Int = n / 7
        val remain = n % 7
        return (week * 28) + ((7 * (week - 1) * week) / 2) + (((1 + remain) * remain) / 2) + (remain * week)
    }

    // LC373
    fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> {
        val pq = PriorityQueue<Pair<Int, List<Int>>>(compareByDescending { it.first })
        outer@ for (i in nums1) {
            for (j in nums2) {
                if (pq.size < k) {
                    pq.offer(Pair(i + j, listOf(i, j)))
                } else {
                    if (i + j < pq.peek().first) {
                        pq.poll()
                        pq.offer(Pair(i + j, listOf(i, j)))
                    } else {
                        continue@outer
                    }
                }
            }
        }
        val result = ArrayList<List<Int>>(k)
        while (!pq.isEmpty()) result.add(pq.poll().second)
        return result
    }

    // LC747
    fun dominantIndex(nums: IntArray): Int {
        val n = nums.size
        if (n == 1) return 0
        val idxMap = HashMap<Int, Int>()
        for (idx in nums.indices) {
            idxMap[nums[idx]] = idx
        }
        nums.sort()
        if (nums[n - 1] >= nums[n - 2] * 2) return idxMap[nums[n - 1]]!!
        return -1
    }

    // LC1036
    fun isEscapePossible(blocked: Array<IntArray>, source: IntArray, target: IntArray): Boolean {
        if (blocked.size < 2) return true
        val bound = 1000000
        val dir = arrayOf(intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0))
        val rowSet = TreeSet<Int>()
        val colSet = TreeSet<Int>()
        val allPoints = ArrayList<IntArray>(blocked.toList())
        allPoints.add(source)
        allPoints.add(target)
        for (b in allPoints) {
            rowSet.add(b[0])
            colSet.add(b[1])
        }

        var rid = if (rowSet.first() == 0) 0 else 1
        var cid = if (colSet.first() == 0) 0 else 1 // bound it from 0 to 999999

        val rit: Iterator<Int> = rowSet.iterator()
        val cit: Iterator<Int> = colSet.iterator()
        val rowMap: MutableMap<Int, Int> = HashMap()
        val colMap: MutableMap<Int, Int> = HashMap()
        var pr = -1
        var pc = -1
        if (rit.hasNext()) rowMap[rit.next().also { pr = it }] = rid
        if (cit.hasNext()) colMap[cit.next().also { pc = it }] = cid
        while (rit.hasNext()) {
            val nr = rit.next()
            rid += if (nr == pr + 1) 1 else 2
            rowMap[nr] = rid
            pr = nr
        }
        while (cit.hasNext()) {
            val nc = cit.next()
            cid += if (nc == pc + 1) 1 else 2
            colMap[nc] = cid
            pc = nc
        }
        val rBound = if (pr == bound - 1) rid else rid + 1
        val cBound = if (pc == bound - 1) cid else cid + 1
        val mtx = Array(rBound + 1) {
            BooleanArray(
                cBound + 1
            )
        } // use as visited[][] too

        for (b in blocked) {
            mtx[rowMap[b[0]]!!][colMap[b[1]]!!] = true
        }
        val sr = rowMap[source[0]]!!
        val sc = colMap[source[1]]!!
        val tr = rowMap[target[0]]!!
        val tc = colMap[target[1]]!!
        val q: Deque<IntArray> = LinkedList()
        q.offer(intArrayOf(sr, sc))
        while (!q.isEmpty()) {
            val p = q.poll()
            if (p[0] == tr && p[1] == tc) return true
            if (mtx[p[0]][p[1]]) continue
            mtx[p[0]][p[1]] = true
            for (d in dir) {
                val inr = p[0] + d[0]
                val inc = p[1] + d[1]
                if (inr in 0..rBound && inc in 0..cBound && !mtx[inr][inc]) {
                    q.offer(intArrayOf(inr, inc))
                }
            }
        }
        return false
    }

    // LC306
    fun isAdditiveNumber(num: String): Boolean {
        val n = num.length
        for (i in 1..n / 2) {
            val first = num.substring(0, i).toLong()
            if (first.toString().length != i) continue
            for (j in i + 1 until n) {
                val second = num.substring(i, j).toLong()
                if (second.toString().length != j - i) continue
                if (judge(first, second, j, num)) return true
            }
        }
        return false
    }

    private fun judge(first: Long, second: Long, idx: Int, num: String): Boolean {
        if (idx == num.length) return true
        val sum = first + second
        if (num.indexOf(sum.toString()) != idx) return false
        return judge(second, sum, idx + sum.toString().length, num)
    }

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


