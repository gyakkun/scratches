package moe.nyamori.test.historical

import java.time.Duration
import java.time.Instant
import java.util.*


object Main63 {
    @JvmStatic
    fun main(argv: Array<String>) {
        var before = Instant.now()!!
        var s = SolutionKt63()
        println(s)
        var after = Instant.now()!!
        System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")
    }
}

class SolutionKt63 {

    // LC1748
    fun sumOfUnique(nums: IntArray): Int {
        return nums.groupBy { it }.filter { it.value.size == 1 }.keys.sum()
    }

    // LC1342
    fun numberOfSteps(num: Int): Int {
        return if (num == 0) 0 else Integer.SIZE - Integer.numberOfLeadingZeros(num) + Integer.bitCount(num) - 1
    }

    // LC1765
    fun highestPeak(isWater: Array<IntArray>): Array<IntArray>? {
        val m = isWater.size
        val n = isWater[0].size
        val directions = arrayOf(intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0))
        val result = Array(m) { IntArray(n) }
        val visited = Array(m) { BooleanArray(n) }
        val q: Deque<IntArray> = LinkedList()
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (isWater[i][j] == 1) {
                    q.offer(intArrayOf(i, j, 0))
                }
            }
        }
        while (!q.isEmpty()) {
            val p = q.poll()
            val r = p[0]
            val c = p[1]
            var h = p[2]
            if (visited[r][c]) continue
            visited[r][c] = true
            if (isWater[r][c] == 1) {
                h = 0
            }
            result[r][c] = h
            for (d in directions) {
                val nr = r + d[0]
                val nc = c + d[1]
                if (nr in 0 until m && nc in 0 until n && !visited[nr][nc]) {
                    q.offer(intArrayOf(nr, nc, h + 1))
                }
            }
        }
        return result
    }

    // LC1996
    fun numberOfWeakCharacters(properties: Array<IntArray>): Int {
        Arrays.sort(properties) { o1: IntArray, o2: IntArray -> if (o1[0] == o2[0]) o1[1] - o2[1] else o2[0] - o1[0] }
        var maxDef = 0
        var ans = 0
        for (p in properties) {
            if (p[1] < maxDef) {
                ans++
            } else {
                maxDef = p[1]
            }
        }
        return ans
    }

    // LC1606
    fun busiestServers(k: Int, arrival: IntArray, load: IntArray): List<Int?>? {
        if (arrival.size <= k) {
            return IntRange(0, k - 1).toList()
        }
        val pq: PriorityQueue<Pair<Int, Int>> =
            PriorityQueue<Pair<Int, Int>> { o1, o2 -> o1.first - o2.first } // <在什么时刻重新空闲, 是第几个服务器>
        val ts = TreeSet<Int>() // 空闲服务器列表
        val count = IntArray(k)
        val max = arrayOf(Int.MIN_VALUE / 2)
        val result: MutableList<Int> = ArrayList()
        IntRange(0, k - 1).forEach { ts.add(it) }
        arrival.indices.forEach {
            while (!pq.isEmpty() && pq.peek().first <= arrival[it]) {
                val p = pq.poll()
                ts.add(p.second)
            }
            if (ts.isEmpty()) {
                return@forEach
            }
            var nextServer = ts.ceiling(it % k)
            if (nextServer == null) nextServer =
                ts.first() // 如果没有比这个i%k大的编号的服务器, 则说明已经只能从头开始找了, 而ts不为空, 所以总能找到编号最小的服务器响应请求
            ts.remove(nextServer)
            pq.offer(Pair(arrival[it] + load[it], nextServer))
            count[nextServer!!]++
            if (count[nextServer] > max[0]) {
                max[0] = count[nextServer]
                result.clear()
                result.add(nextServer)
            } else if (count[nextServer] == max[0]) {
                result.add(nextServer)
            }
        }
        return result
    }

    // LC1791
    fun findCenter(edges: Array<IntArray>): Int =
        edges.flatMap { it.toList() }.groupingBy { it }.eachCount().filter { it.value != 1 }.entries.first().key


    // LC1219
    var directions = arrayOf(intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0))
    lateinit var grid: Array<IntArray>
    lateinit var visited: Array<BooleanArray>

    fun getMaximumGold(grid: Array<IntArray>): Int {
        this.grid = grid
        val m = grid.size
        val n = grid[0].size
        var result = 0
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (grid[i][j] != 0) {
                    visited = Array(m) { BooleanArray(n) }
                    result = Math.max(result, lc1219Helper(i, j))
                }
            }
        }
        return result
    }

    private fun lc1219Helper(r: Int, c: Int): Int {
        if (visited[r][c]) return 0
        visited[r][c] = true
        val cur = grid[r][c]
        var next = 0
        for (d in directions) {
            val nr = r + d[0]
            val nc = c + d[1]
            if (nr >= 0 && nr < grid.size && nc >= 0 && nc < grid[0].size && grid[nr][nc] != 0 && !visited[nr][nc]) {
                next = Math.max(next, lc1219Helper(nr, nc))
            }
        }
        visited[r][c] = false
        return cur + next
    }

    // LC1725
    fun countGoodRectangles(rectangles: Array<IntArray>): Int {
        return rectangles.groupingBy { Math.min(it[0], it[1]) }.eachCount().entries.maxByOrNull { it.key }!!.value
    }

    // LC1414
    fun findMinFibonacciNumbers(k: Int): Int {
        var k = k
        var result = 0
        while (k != 0) {
            val h = lc1414Helper(k)
            if (k != -1) {
                result++
                k -= h
            }
        }
        return result
    }

    // 二分 找小于等于的最大值
    fun lc1414Helper(n: Int): Int {
        var l = 0
        var h = 46
        while (l < h) {
            val mid = l + (h - l + 1) / 2
            if (fib(mid) <= n) {
                l = mid
            } else {
                h = mid - 1
            }
        }
        return if (fib(l) > n) -1 else fib(l)
    }

    fun fib(n: Int): Int {
        if (n == 0) return 0
        return if (n == 1 || n == 2) 1 else fib(n, 2, 1, 1)
    }

    fun fib(targetIdx: Int, curIdx: Int, curVal: Int, prevVal: Int): Int {
        return if (targetIdx == curIdx) curVal else fib(targetIdx, curIdx + 1, curVal + prevVal, curVal)
    }


    fun add(a: Int, b: Int) {
        e[idx] = b
        ne[idx] = he[a]
        he[a] = idx
        idx++
    }

    // LC2045 **
    fun secondMinimum(n: Int, edges: Array<IntArray>, time: Int, change: Int): Int {
        Arrays.fill(dist1, INF)
        Arrays.fill(dist2, INF)
        Arrays.fill(he, -1)
        idx = 0
        for (e in edges) {
            val u = e[0]
            val v = e[1]
            add(u, v)
            add(v, u)
        }
        val q = PriorityQueue<IntArray>(compareBy { i -> i[1] })
        q.add(intArrayOf(1, 0))
        dist1[1] = 0
        while (!q.isEmpty()) {
            val poll = q.poll()
            val u = poll[0]
            val step = poll[1]
            var i = he[u]
            while (i != -1) {
                val j = e[i]
                val a = step / change
                val b = step % change
                val wait = if (a % 2 == 0) 0 else change - b
                val dist = step + time + wait
                if (dist1[j] > dist) {
                    dist2[j] = dist1[j]
                    dist1[j] = dist
                    q.add(intArrayOf(j, dist1[j]))
                    q.add(intArrayOf(j, dist2[j]))
                } else if (dist1[j] < dist && dist < dist2[j]) {
                    dist2[j] = dist
                    q.add(intArrayOf(j, dist2[j]))
                }
                i = ne[i]
            }
        }
        return dist2[n]
    }

    companion object {
        var N = 10010
        var M = 4 * N
        var INF = 0x3f3f3f3f
        var idx = 0
        var he = IntArray(N)
        var e = IntArray(M)
        var ne = IntArray(M)
        var dist1 = IntArray(N)
        var dist2 = IntArray(N)
    }


    // LC1332
    fun removePalindromeSub(s: String): Int {
        var flag = true
        for (i in 0 until s.length / 2) {
            if (s[i] != s[s.length - 1 - i]) {
                flag = false
                break
            }
        }
        if (flag) return 1
        return 2
    }

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
            result = (result + lc1220Helper(n - 1, i)) % mod
        }
        return result.toInt()
    }

    private fun lc1220Helper(remainLetters: Int, currentLetterIdx: Int): Long {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        if (remainLetters == 0) return 1
        if (memo[remainLetters][currentLetterIdx] != null) return memo[remainLetters][currentLetterIdx]!! % mod
        when (currentLetterIdx) {
            0 -> return (lc1220Helper(remainLetters - 1, 1) % mod).also {
                memo[remainLetters][currentLetterIdx] = it
            }

            1 -> return ((lc1220Helper(remainLetters - 1, 0)
                    + lc1220Helper(remainLetters - 1, 2)) % mod).also { memo[remainLetters][currentLetterIdx] = it }

            2 -> return ((lc1220Helper(remainLetters - 1, 0)
                    + lc1220Helper(remainLetters - 1, 1)
                    + lc1220Helper(remainLetters - 1, 3)
                    + lc1220Helper(remainLetters - 1, 4)) % mod).also { memo[remainLetters][currentLetterIdx] = it }

            3 -> return ((lc1220Helper(remainLetters - 1, 2)
                    + lc1220Helper(remainLetters - 1, 4)) % mod).also { memo[remainLetters][currentLetterIdx] = it }

            4 -> return (lc1220Helper(remainLetters - 1, 0) % mod).also { memo[remainLetters][currentLetterIdx] = it }
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

// LC2013
internal class DetectSquares {
    var xyMap: MutableMap<Int, MutableMap<Int, Int>> = HashMap()
    fun add(point: IntArray) {
        xyMap.putIfAbsent(point[0], HashMap())
        val yPointsCount = xyMap[point[0]]!!
        yPointsCount[point[1]] = yPointsCount.getOrDefault(point[1], 0) + 1
    }

    fun count(point: IntArray): Int {
        // 比较同一x坐标/y坐标上 [独特点(指重复位置的点算一个)] 的个数, 挑选少的集合来进行遍历
        val x = point[0]
        val y = point[1]
        var result = 0
        if (!xyMap.containsKey(x)) return 0
        val yPoints: Map<Int, Int> = xyMap[x]!!
        val c0 = 1
        for ((thisY, c1) in yPoints) {
            if (thisY == y) continue
            val distance = y - thisY
            val absDistance = Math.abs(distance)
            for (sideX in intArrayOf(x - absDistance, x + absDistance)) {
                if (xyMap.containsKey(sideX) && xyMap[sideX]!!.containsKey(y)) {
                    val c2 = xyMap[sideX]!![y]!!
                    // 找左下角
                    if (xyMap.containsKey(sideX) && xyMap[sideX]!!.containsKey(thisY)) {
                        val c3 = xyMap[sideX]!![thisY]!!
                        result += c0 * c1 * c2 * c3
                    }
                }
            }
        }
        return result
    }
}

// LC2034
internal class StockPrice {
    var timePriceMap = TreeMap<Int, Int>()
    var priceTimeMap = TreeMap<Int, MutableSet<Int>>()
    fun update(timestamp: Int, price: Int) {
        timePriceMap[timestamp]?.let { priceToCorrect ->
            priceTimeMap[priceToCorrect]?.let { timeSet ->
                timeSet.remove(timestamp)
                if (timeSet.size == 0) {
                    priceTimeMap.remove(priceToCorrect)
                }
            }
        }
        timePriceMap[timestamp] = price
        priceTimeMap.putIfAbsent(price, HashSet())
        priceTimeMap[price]!!.add(timestamp)
    }

    fun current(): Int {
        return timePriceMap.lastEntry().value
    }

    fun maximum(): Int {
        return priceTimeMap.lastKey()
    }

    fun minimum(): Int {
        return priceTimeMap.firstKey()
    }
}