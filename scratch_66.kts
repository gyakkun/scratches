import java.time.Duration
import java.time.Instant

//class Main {
//    companion object {
//        @JvmStatic
//        fun main(args: Array<String>) {
var before = Instant.now()!!
var s = Solution()
println(s.largestPalindrome(3))
var after = Instant.now()!!
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")
//        }
//    }
//}

class Solution {

    // LC479 **
    fun largestPalindrome(n: Int): Long {
        if (n == 1) {
            return 9
        }
        val upper = Math.pow(10.0, n.toDouble()).toInt() - 1
        var ans: Long = 0
        var left = upper
        while (ans == 0L) {
            // 枚举回文数的左半部分
            var p = left.toLong()
            run {
                var x = left
                while (x > 0) {
                    p = p * 10 + x % 10 // 翻转左半部分到其自身末尾，构造回文数 p
                    x /= 10
                }
            }
            var x = upper.toLong()
            while (x * x >= p) {
                if (p % x == 0L) { // x 是 p 的因子
                    ans = p
                    break
                }
                --x
            }
            --left
        }
        return ans % 1337L
    }

    // LC804
    private val morse = arrayOf(
        ".-",
        "-...",
        "-.-.",
        "-..",
        ".",
        "..-.",
        "--.",
        "....",
        "..",
        ".---",
        "-.-",
        ".-..",
        "--",
        "-.",
        "---",
        ".--.",
        "--.-",
        ".-.",
        "...",
        "-",
        "..-",
        "...-",
        ".--",
        "-..-",
        "-.--",
        "--.."
    )

    fun uniqueMorseRepresentations(words: Array<String>) = words.map { word ->
        word.toCharArray().joinToString(separator = "") { morse[it - 'a'] }
    }.distinct().count()

    // LC710
    fun reachingPoints(ssx: Int, ssy: Int, ttx: Int, tty: Int): Boolean {
        var sx = ssx
        var sy = ssy
        var tx = ttx
        var ty = tty
        while (tx > sx && ty > sy && tx != ty) {
            if (tx > ty) tx %= ty else ty %= tx
        }
        return if (tx == sx && ty == sy) true
        else if (tx == sx && ty != sy) ty > sy && (ty - sy) % sx == 0
        else if (ty == sy && tx != sx) tx > sx && (tx - sx) % sy == 0
        else false
    }

    // LC310
    fun findMinHeightTrees(n: Int, edges: Array<IntArray>): List<Int> {
        if (n == 1) return listOf(0)
        val edgeMtx = ArrayList<MutableList<Int>>(n).apply {
            repeat(n) {
                this.add(ArrayList())
            }
        }
        edges.forEach { pointPair ->
            edgeMtx[pointPair[0]].add(pointPair[1])
            edgeMtx[pointPair[1]].add(pointPair[0])
        }
        val startPoint = edgeMtx.withIndex().first { it.value.size == 1 }.index
        val depthArr = IntArray(n).apply { fill(-1) }
        val parent = IntArray(n).apply { fill(-1) }
        helper(startPoint, 0, edgeMtx, depthArr, parent)
        val furthestPoint = depthArr.withIndex().maxByOrNull { it.value }!!.index
        depthArr.fill(-1)
        helper(furthestPoint, 0, edgeMtx, depthArr, parent)
        var endPoint: Int
        var longestDistance: Int
        depthArr.withIndex().maxByOrNull { it.value }!!.let {
            endPoint = it.index
            longestDistance = it.value
        }
        var tmpParent = endPoint
        var pathPointSet = HashSet<Int>()
        while (tmpParent != -1) {
            pathPointSet.add(tmpParent)
            tmpParent = parent[tmpParent]
        }

        return if (longestDistance % 2 == 0) {
            depthArr.withIndex().filter { it.value == longestDistance / 2 && it.index in pathPointSet }.map { it.index }
                .toList()
        } else {
            depthArr.withIndex()
                .filter { (it.value == longestDistance / 2 || it.value == (longestDistance + 1) / 2) && it.index in pathPointSet }
                .map { it.index }.toList()
        }
    }

    private fun helper(
        cur: Int,
        depth: Int,
        edgeMtx: MutableList<MutableList<Int>>,
        depthArr: IntArray,
        parent: IntArray
    ): Unit {
        if (depthArr[cur] != -1) return
        depthArr[cur] = depth
        for (next in edgeMtx[cur]) {
            if (depthArr[next] != -1) continue
            parent[next] = cur
            helper(next, depth + 1, edgeMtx, depthArr, parent)
        }
    }


    private val prime = setOf(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)

    fun countPrimeSetBits(left: Int, right: Int) = IntRange(left, right).count { it.countOneBits() in prime }


    // LC744
    fun nextGreatestLetter(letters: CharArray, target: Char): Char {
        return if (target >= letters.last()) letters.first() else {
            var l = 0
            var r = letters.size - 1
            while (l < r) {
                (l + (r - l) / 2).let { mid ->
                    when {
                        letters[mid] > target -> r = mid
                        else -> l = mid + 1
                    }
                }
            }
            letters[l]
        }
    }
}
