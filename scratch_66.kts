import java.time.Duration
import java.time.Instant
import java.util.*
import java.util.stream.IntStream
import kotlin.collections.ArrayList
import kotlin.collections.HashSet
import kotlin.math.abs
import kotlin.math.pow

//class Main {
//    companion object {
//        @JvmStatic
//        fun main(args: Array<String>) {
var before = Instant.now()!!
var s = Solution()
println(
    s.lengthLongestPath(
        "dir\n" +
                "\tsubdir1\n" +
                "\tsubdir2\n" +
                "\t\tfile.ext"
    )
)
var after = Instant.now()!!
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")
//        }
//    }
//}

class Solution {
    // LC587 ** 凸包
    fun outerTrees(trees: Array<IntArray>): Array<IntArray> {
        val n = trees.size
        if (n < 4) {
            return trees
        }
        var leftMost = 0
        for (i in 0 until n) {
            if (trees[i][0] < trees[leftMost][0]) {
                leftMost = i
            }
        }
        val res: MutableList<IntArray> = ArrayList()
        val visit = BooleanArray(n)
        var p = leftMost
        do {
            var q = (p + 1) % n
            for (r in 0 until n) {
                /* 如果 r 在 pq 的右侧，则 q = r */
                if (cross(trees[p], trees[q], trees[r]) < 0) {
                    q = r
                }
            }
            /* 是否存在点 i, 使得 p 、q 、i 在同一条直线上 */
            for (i in 0 until n) {
                if (visit[i] || i == p || i == q) {
                    continue
                }
                if (cross(trees[p], trees[q], trees[i]) == 0) {
                    res.add(trees[i])
                    visit[i] = true
                }
            }
            if (!visit[q]) {
                res.add(trees[q])
                visit[q] = true
            }
            p = q
        } while (p != leftMost)
        return res.toTypedArray()
    }

    private fun cross(p: IntArray, q: IntArray, r: IntArray): Int {
        return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
    }

    // LC824
    private val vowel = arrayOf('a', 'e', 'i', 'o', 'u')
    fun toGoatLatin(sentence: String): String {
        return sentence.split(" ")
            .mapIndexed { idx, str ->
                var result = str
                if (str[0].lowercaseChar() !in vowel) {
                    result = str.substring(1) + str[0]
                }
                result += "ma"
                repeat(idx + 1) { result += "a" }
                result
            }.joinToString(" ")
    }

    // LC388
    var result: String = ""
    fun lengthLongestPath(input: String): Int {
        val root = Hierarchy("zwb", false)
        var prevStartToken = 0
        val pattern = Regex(".+\\..+")
        var curTabCount = 0
        var idx = 0;
        val stack: Deque<Hierarchy> = LinkedList<Hierarchy>().apply { add(root) }
        while (idx <= input.length) {
            if (idx == input.length) {
                val text = input.substring(prevStartToken, idx)
                val curNode = Hierarchy(text, pattern.matches(text))
                stack.peek().children.add(curNode)
                break;
            }
            val c = input[idx]
            if (c != '\n') {
                idx++
                continue
            }
            val text = input.substring(prevStartToken, idx)
            // if(pattern.matches(text))
            val curNode = Hierarchy(text, pattern.matches(text))
            stack.peek().children.add(curNode)
            var nextTabCount = 0
            while (idx + 1 < input.length && input[idx + 1] == '\t') {
                nextTabCount++
                idx++
            }
            if (nextTabCount > curTabCount) {
                assert(nextTabCount - curTabCount > 1)
                //  throw java.lang.IllegalStateException("Should be only one more layer")
                stack.push(curNode)
            } else if (nextTabCount < curTabCount) {
                repeat(curTabCount - nextTabCount) {
                    stack.pop()
                }
            }
            curTabCount = nextTabCount
            prevStartToken = ++idx
        }
        helper(root, "")
        return if (result.isEmpty()) 0 else {
            result.length - 2 - root.name.length
        }
    }

    fun helper(root: Hierarchy, curPrefix: String) {
        val curPath = curPrefix + "/" + root.name
        if (root.isFile) {
            if (result.length < curPath.length) {
                result = curPath
            }
        }
        root.children.forEach {
            helper(it, curPath)
        }
    }

    data class Hierarchy(
        val name: String,
        val isFile: Boolean,
        val children: MutableList<Hierarchy> = arrayListOf()
    )


    // LC821
    fun shortestToChar(s: String, c: Char): IntArray {
        var prev: Int = Integer.MAX_VALUE
        var result = IntArray(s.length)
        s.forEachIndexed { idx, ch ->
            if (ch == c) {
                result[idx] = 0
                prev = idx
            } else {
                result[idx] = abs(idx - prev)
            }
        }
        prev = Integer.MAX_VALUE
        s.reversed().forEachIndexed { idx, ch ->
            val actualIdx = s.length - 1 - idx
            if (ch == c) {
                result[actualIdx] = 0
                prev = actualIdx // 实际下标
            } else {
                result[actualIdx] = result[actualIdx].coerceAtMost(abs(actualIdx - prev))
            }
        }
        return result
    }

    // LC819
    fun mostCommonWord(paragraph: String, banned: Array<String>) = paragraph
        .lowercase(Locale.getDefault())
        .split(Regex("[!?',;.]+"))
        .groupBy { it }
        .filter { it.key !in banned }
        .maxByOrNull { it.value.size }!!
        .key

    // LC479 **
    fun largestPalindrome(n: Int): Long {
        if (n == 1) return 9
        val upper = 10.0.pow(n.toDouble()).toInt() - 1
        var ans = 0L
        var left = upper
        while (ans == 0L) {
            // 枚举回文数的左半部分
            var p = left.toLong()
            var i = left.toLong()
            while (i > 0) {
                p = p * 10 + i % 10 // 翻转左半部分到其自身末尾，构造回文数 p
                i /= 10
            }
            i = upper.toLong()
            while (i * i >= p) {
                if (p % i == 0L) { // x 是 p 的因子
                    ans = p
                    break
                }
                i--
            }
            left--
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
