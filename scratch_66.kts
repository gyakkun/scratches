import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.math.abs
import kotlin.math.pow


//class Main {
//    companion object {
//        @JvmStatic
//        fun main(args: Array<String>) {
var before = Instant.now()!!
var s = Solution()
println(
    s.possibleBipartition(
        10,
        arrayOf(
            intArrayOf(1, 2),
            intArrayOf(3, 4),
            intArrayOf(5, 6),
            intArrayOf(6, 7),
            intArrayOf(8, 9),
            intArrayOf(7, 8)
        )
    )
)
var after = Instant.now()!!
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")
//        }
//    }
//}

class Solution {

    // LC15 3-Sum. Try to generalize.
    class KSum {
        fun threeSum(nums: IntArray): List<List<Int>> {
            return kSum(nums, 0, 3)
        }

        fun kSum(nums: IntArray, target: Int, k: Int): MutableList<MutableList<Int>> {
            if (k == 1) {
                val result = java.util.ArrayList<MutableList<Int>>()
                for (i in nums) if (i == target) result.add(mutableListOf(i))
                return result
            }
            Arrays.sort(nums)
            return kSum(nums, target, k, 0)
        }

        fun kSum(nums: IntArray, target: Int, k: Int, startIdx: Int): MutableList<MutableList<Int>> {
            val len = nums.size - startIdx
            if (k > len) return mutableListOf()
            if (k == 2) {
                var left = startIdx
                var right = nums.size - 1
                val result = java.util.ArrayList<MutableList<Int>>()
                while (left < right) {
                    if (nums[left] + nums[right] == target) {
                        result.add(mutableListOf(nums[left], nums[right]))
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++
                        }
                        left++
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--
                        }
                        right--
                    } else if (nums[left] + nums[right] < target) {
                        left++
                    } else if (nums[left] + nums[right] > target) {
                        right--
                    }
                }
                return result
            }
            val result = java.util.ArrayList<MutableList<Int>>()
            for (i in startIdx..(nums.size - k)) {
                // If can include ele in different idx with same value, then allow the first half of OR
                if ((i > startIdx && nums[i] == nums[i - 1]) || (nums[i] + (k - 1) * nums[nums.size - 1] < target)) {
                    continue
                }
                if (nums[i] + (k - 1) * nums[i + 1] > target) {
                    break
                }
                val subResult = kSum(nums, target - nums[i], k - 1, i + 1)
                for (j in subResult) {
                    j.add(nums[i])
                    result.add(j)
                }
            }
            return result
        }
    }

    // LC784
    fun letterCasePermutation(s: String): List<String> {
        val result = ArrayList<String>()
        fun helper(sb: StringBuilder, idx: Int) {
            if (idx == s.length) {
                result.add(sb.toString())
                return
            }
            if (!Character.isLetter(s[idx])) {
                sb.append(s[idx])
                helper(sb, idx + 1)
                sb.deleteCharAt(idx)
                return
            }
            sb.append(s[idx].uppercaseChar())
            helper(sb, idx + 1)
            sb.deleteCharAt(idx)
            sb.append(s[idx].lowercaseChar())
            helper(sb, idx + 1)
            sb.deleteCharAt(idx)
        }
        helper(StringBuilder(), 0)
        return result
    }

    // LC1773
    fun countMatches(items: List<List<String>>, ruleKey: String, ruleValue: String): Int {
        return when (ruleKey) {
            "type" -> items.count { it[0] == ruleValue }
            "color" -> items.count { it[1] == ruleValue }
            "name" -> items.count { it[2] == ruleValue }
            else -> 0
        }
    }

    // LC1235 Hard
    fun jobScheduling(startTime: IntArray, endTime: IntArray, profit: IntArray): Int {
        val l = ArrayList<Triple<Int, Int, Int>>()
        val n = startTime.size
        val memo = Array<Int?>(n + 1) { null }
        for (i in 0 until n) {
            l.add(Triple(startTime[i], endTime[i], profit[i]))
        }
        l.sortBy { it.first } // Order by start time ascending
        // Return max profit starting by mission in this idx (inclusive)
        fun helper(idx: Int): Int {
            if (idx < 0) return 0
            if (idx == n - 1) return l[idx].third
            if (memo[idx] != null) return memo[idx]!!
            val start = l[idx].first
            val end = l[idx].second
            val profit = l[idx].third

            var result = helper(idx + 1) // not choose this mission
            // If choose this mission, then the next mission should be in far behind.
            // Use binary search to locate the closest mission the start time of which is equal or larger than end time of current
            var lo = idx + 1
            var hi = n - 1
            while (lo < hi) {
                val mid = lo + (hi - lo) / 2
                if (l[mid].first >= end) {
                    hi = mid
                } else {
                    lo = mid + 1
                }
            }
            var victimIdx = hi
            if (l[hi].first < end) victimIdx = -1
            result = Math.max(result, profit + helper(victimIdx))
            return result.also { memo[idx] = it }
        }
        return helper(0)
    }

    // LC886
    fun possibleBipartition(n: Int, dislikes: Array<IntArray>): Boolean {
        var dsu = DSUArray(2 * n)
        var dedicatedParent = HashMap<Int, Int>()
        for (d in dislikes) {
            if (dedicatedParent[d[0]] == null) {
                dedicatedParent[d[0]] = d[1]
            }
            if (dedicatedParent[d[1]] == null) {
                dedicatedParent[d[1]] = d[0]
            }
            dsu.add(d[0])
            dsu.add(d[1])
            dsu.merge(dedicatedParent[d[0]]!!, d[1])
            dsu.merge(dedicatedParent[d[1]]!!, d[0])
        }
        val allGroups = dsu.allGroups
        for ((disee, leader) in dedicatedParent) {
            var finalParent = dsu.father[leader]
            if (allGroups[finalParent]!!.contains(disee)) return false
        }
        return true
    }

    // LC1987 Hard **
    fun numberOfUniqueGoodSubsequences(binary: String): Int {
        val ca = binary.toCharArray()
        val mod = 1000000007L
        val n = ca.size
        val dp = Array(n + 1) { LongArray(2) }
        var hasZero = false
        for (i in 0 until n) {
            if (ca[i] == '1') {
                // end with 0
                dp[i][0] = if (i > 0) {
                    dp[i - 1][0]
                } else {
                    0
                }
                // end with 1
                dp[i][1] = if (i > 0) {
                    dp[i - 1][0] + dp[i - 1][1] + 1 /* sel */
                } else {
                    1
                }
            } else if (ca[i] == '0') {
                hasZero = true
                // end with 0
                dp[i][0] = if (i > 0) {
                    dp[i - 1][0] + dp[i - 1][1] /* when self is zero, not take into consideration. Handle with hasZero */
                } else {
                    0
                }
                dp[i][1] = if (i > 0) {
                    dp[i - 1][1]
                } else {
                    0
                }
            }
            dp[i][0] %= mod
            dp[i][1] %= mod
        }
        var result = (dp[n - 1][0] + dp[n - 1][1] + mod) % mod
        if (hasZero) result = (result + 1 + mod) % mod
        return result.toInt()
    }

    // LC941 Hard **
    fun distinctSubseqII(s: String): Int {
        val n = s.length
        val ca = s.toCharArray()
        val dp = IntArray(n + 1)
        dp[0] = 1 // 空串
        val lastOccur = IntArray(128)
        Arrays.fill(lastOccur, -1)
        val mod = 1000000007
        for (i in 0 until n) {
            dp[i + 1] = dp[i] * 2 % mod
            if (lastOccur[ca[i].toInt()] != -1) {
                dp[i + 1] -= dp[lastOccur[ca[i].toInt()]]
            }
            dp[i + 1] %= mod
            lastOccur[ca[i].toInt()] = i
        }
        dp[n] = (dp[n] - 1 + mod) % mod // -1 处理空串
        return dp[n]
    }

    // LC801 ** Hard
    fun minSwap(nums1: IntArray, nums2: IntArray): Int {
        val n = nums1.size
        var memo = Array(n) { Array<Int?>(2) { null } }
        fun helper(idx: Int, operateOrNot: Int): Int {
            if (idx == 0) {
                return if (operateOrNot == 0) {
                    0
                } else {
                    1
                }
            }
            if (memo[idx][operateOrNot] != null) {
                return memo[idx][operateOrNot]!!
            }
            val judge1 = nums1[idx] > nums1[idx - 1] && nums2[idx] > nums2[idx - 1];
            val judge2 = nums1[idx] > nums2[idx - 1] && nums2[idx] > nums1[idx - 1];
            var result: Int = if (judge1 && judge2) {
                operateOrNot + helper(idx - 1, operateOrNot).coerceAtMost(helper(idx - 1, 1 - operateOrNot))
            } else if (judge1) {
                operateOrNot + helper(idx - 1, operateOrNot)
            } else {
                operateOrNot + helper(idx - 1, 1 - operateOrNot)
            }
            return result!!.also { memo[idx][operateOrNot] = it }
        }
        return helper(n - 1, 1).coerceAtMost(helper(n - 1, 0))
    }


    // LC811
    fun subdomainVisits(cpdomains: Array<String>): List<String> {
        val freqMap = HashMap<String, Int>()
        for (cp in cpdomains) {
            val splitBySpace = cp.split(" ")
            val count = Integer.parseInt(splitBySpace[0])
            val fullDomain = splitBySpace[1]
            val domainSplit = fullDomain.split(".")
            val sb = StringBuilder()
            for (piece in domainSplit.reversed()) {
                sb.insert(0, piece)
                val fragmentedDomain = sb.toString()
                freqMap[fragmentedDomain] = freqMap.getOrDefault(fragmentedDomain, 0) + count
                sb.insert(0, ".")
            }
        }
        return freqMap.map { e ->
            "${e.value} ${e.key}"
        }.toList()
    }

    // LC1640
    fun canFormArray(arr: IntArray, pieces: Array<IntArray>): Boolean {
        val m: MutableMap<Int, Int> = HashMap()
        for (i in pieces.indices) {
            val ar = pieces[i]
            for (j in ar) {
                m[j] = i
            }
        }
        var i = 0
        while (i < arr.size) {
            val v = arr[i]
            if (!m.containsKey(v)) return false

            val thePiece = pieces[m[v]!!]
            var j = 0
            while (j < thePiece.size) {
                if (arr[i] != thePiece[j]) return false

                // defer
                j++
                i++
            }
        }
        return true
    }

    // LC1636
    fun frequencySort(nums: IntArray): IntArray {
        val freqMap = nums.toTypedArray().groupingBy { it }.eachCount()
        return nums.toTypedArray().apply {
            sortWith(kotlin.Comparator { o1, o2 ->
                if (freqMap[o1]!! == freqMap[o2]!!) {
                    return@Comparator o2 - o1
                }
                return@Comparator freqMap[o1]!! - freqMap[o2]!!
            })
        }.toIntArray()
    }

    // LC1592
    fun reorderSpaces(text: String): String {
        val split: List<String> = text.split(regex = Regex("\\s+")).filter { it.isNotBlank() }.toList()
        val charLenSum = split.sumOf { it.length }
        val spaceCount = text.length - charLenSum
        val avgSpace = if (split.size == 1) 0 else spaceCount / (split.size - 1)
        val remainSpace = spaceCount - avgSpace * (split.size - 1)
        val infix = StringBuffer()
        val suffix = StringBuffer()
        repeat(avgSpace) {
            infix.append(" ")
        }
        repeat(remainSpace) {
            suffix.append(" ")
        }
        return split.joinToString(separator = infix.toString(), postfix = suffix.toString())
    }

    // LC828 ** Hard
    fun uniqueLetterString(s: String): Int {
        val ca = s.toCharArray()
        val indexes = HashMap<Char, MutableList<Int>>()
        var result = 0
        ca.forEachIndexed { idx, c ->
            indexes.putIfAbsent(c, ArrayList<Int>().apply { this.add(-1) })
            indexes[c]!!.add(idx)
        }
        indexes.forEach outer@{ _, l ->
            l.add(s.length)
            val ls = l.size
            IntRange(1, ls - 1).forEach inner@{ i ->
                result += (l[i] - l[i - 1]) * (l[i + 1] - l[i])
            }
        }
        return result
    }

    // LC1624
    fun maxLengthBetweenEqualCharacters(s: String): Int {
        val ca = s.toCharArray()
        val idxMap = IntArray(128).apply { fill(-1) }
        var result = -1
        for (i in ca.indices) {
            val c = ca[i]
            var prevIdx: Int
            if (idxMap[c.toInt()].also { prevIdx = it } >= 0) {
                result = result.coerceAtLeast(i - prevIdx - 1)
            } else {
                idxMap[c.toInt()] = i
            }
        }
        return result
    }

    // LC691
    var finResult = Integer.MAX_VALUE / 2
    fun minStickers(stickers: Array<String>, target: String): Int {
        val maskArr = stickers.map { helperMask(it) }.toTypedArray()
        val totalMask = maskArr.reduce { acc, i -> acc or i }
        val targetMask = helperMask(target)
        if (targetMask and totalMask != totalMask) return -1
        val stickSet = stickers.toMutableSet()
        val msitr = stickSet.iterator()

        // Sieve all valid stickers
        outer@ while (msitr.hasNext()) {
            val next = msitr.next()
            val nextCountArr = helperCountArr(next)
            for (st in stickers) {
                if (helperAllLargerThanOrEqualToZero(helperCountArrDiff(helperCountArr(st), nextCountArr))
                    && st != next
                ) {
                    msitr.remove()
                    continue@outer
                }
            }
        }
        lc691Helper(0, helperCountArr(target), stickSet)
        return finResult
    }

    fun lc691Helper(layer: Int, countArr: IntArray, stickSet: MutableSet<String>): Unit {
        if (helperCountArrToMask(countArr) == 0) {
            finResult = finResult.coerceAtMost(layer)
            return
        }
        val nextLayer = layer + 1
        val waitingMask = helperCountArrToMask(countArr)
        stickSet.forEach {
            val curMask = helperMask(it)
            if (waitingMask and curMask == 0) return@forEach
            val curCountArr = helperCountArr(it)
            val nextCountArr = countArr.copyOf()
            for (i in nextCountArr.indices) {
                nextCountArr[i] = (nextCountArr[i] - curCountArr[i]).coerceAtLeast(0)
            }
            lc691Helper(nextLayer, nextCountArr, stickSet)
        }
    }

    fun helperAllLargerThanOrEqualToZero(diffCountArr: IntArray): Boolean = diffCountArr.all { it >= 0 }

    fun helperCountArrToMask(countArr: IntArray): Int {
        var result = 0
        countArr.forEachIndexed { index, i ->
            if (i > 0) result = result and (1 shl index)
        }
        return result
    }

    fun helperCountArrDiff(c1: IntArray, c2: IntArray): IntArray {
        val result = IntArray(26)
        IntRange(0, 25).forEach { result[it] = c1[it] - c2[it] }
        return result
    }

    fun helperMask(word: String): Int {
        var mask: Int = 0
        word.toCharArray().forEach {
            mask = mask or (it.code - 'a'.code)
        }
        return mask
    }

    fun helperCountArr(word: String): IntArray {
        val result = IntArray(26)
        word.chars().forEach { result[it - 'a'.code]++ }
        return result
    }

    // Interview 01.05
    fun oneEditAway(first: String, second: String): Boolean {
        if (abs(first.length - second.length) > 1) return false
        if (first == second) return true

        if (first.length == second.length) {
            var fp = 0
            var sp = 0
            var diffCtr = 0
            while (fp != first.length && sp != second.length) {
                if (first[fp] != second[fp]) diffCtr++
                if (diffCtr > 1) return false
                fp++
                sp++
            }
        } else {
            var lp = 0
            var sp = 0
            var longOne = if (first.length > second.length) first else second
            var shortOne = if (first.length > second.length) second else first
            while (lp != longOne.length && sp != shortOne.length && longOne[lp] == shortOne[sp]) {
                lp++
                sp++
            }
            lp++
            while (lp != longOne.length && sp != shortOne.length) {
                if (longOne[lp] != shortOne[sp]) return false
                lp++
                sp++
            }
        }
        return true
    }

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
        lc388Helper(root, "")
        return if (result.isEmpty()) 0 else {
            result.length - 2 - root.name.length
        }
    }

    fun lc388Helper(root: Hierarchy, curPrefix: String) {
        val curPath = curPrefix + "/" + root.name
        if (root.isFile) {
            if (result.length < curPath.length) {
                result = curPath
            }
        }
        root.children.forEach {
            lc388Helper(it, curPath)
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
        lc310Helper(startPoint, 0, edgeMtx, depthArr, parent)
        val furthestPoint = depthArr.withIndex().maxByOrNull { it.value }!!.index
        depthArr.fill(-1)
        lc310Helper(furthestPoint, 0, edgeMtx, depthArr, parent)
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

    private fun lc310Helper(
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
            lc310Helper(next, depth + 1, edgeMtx, depthArr, parent)
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


// LC1728 Kotlin again
class lc1728 {
    fun canMouseWin(grid: Array<String>, catJump: Int, mouseJump: Int): Boolean {
        //System.out.println(grid[1].charAt(1));
        val dp = Array(510) {
            Array(65) {
                IntArray(
                    65
                )
            }
        }
        for (i in 0..509) for (j in 0..64) for (k in 0..64) dp[i][j][k] = -1
        var x1 = -1
        var x2 = -1
        var y1 = -1
        var y2 = -1
        for (i in grid.indices) {
            for (j in 0 until grid[0].length) {
                if (grid[i][j] == 'C') {
                    y1 = i
                    y2 = j
                }
                if (grid[i][j] == 'M') {
                    x1 = i
                    x2 = j
                }
            }
        }
        val t = help(grid, dp, 0, matrixFlattenIdx(x1, x2), matrixFlattenIdx(y1, y2), catJump, mouseJump)
        return if (t == 1) true else false
    }

    fun help(
        grid: Array<String>,
        dp: Array<Array<IntArray>>,
        t: Int,
        x: Int,
        y: Int,
        catJump: Int,
        mouseJump: Int
    ): Int {
        if (t == 500) return 0.also { dp[t][x][y] = it }
        if (dp[t][x][y] != -1) return dp[t][x][y]
        val x1 = x / 8
        val x2 = x % 8
        val y1 = y / 8
        val y2 = y % 8
        if (t % 2 == 0) {
            if (grid[x1][x2] == 'F') return 1.also { dp[t][x][y] = it }
        }
        if (t % 2 == 1) {
            if (grid[y1][y2] == 'F') return 2.also { dp[t][x][y] = it }
        }
        if (x1 == y1 && x2 == y2) return 2.also { dp[t][x][y] = it }
        return if (t % 2 == 0) { // step for mouse
            var cat_win = true
            for (i in 0..mouseJump) {
                val x1_new = x1 + i
                if (x1_new >= 0 && x1_new < grid.size) {
                    if (grid[x1_new][x2] == '#') break
                    val next = help(grid, dp, t + 1, matrixFlattenIdx(x1_new, x2), y, catJump, mouseJump)
                    if (next == 1) return 1.also { dp[t][x][y] = it } else if (next != 2) cat_win = false
                }
            }
            /////////////////////
            for (i in 0..mouseJump) {
                val x1_new = x1 - i
                if (x1_new >= 0 && x1_new < grid.size) {
                    if (grid[x1_new][x2] == '#') break
                    val next = help(grid, dp, t + 1, matrixFlattenIdx(x1_new, x2), y, catJump, mouseJump)
                    if (next == 1) return 1.also { dp[t][x][y] = it } else if (next != 2) cat_win = false
                }
            }
            /////////////////////////////////////////////
            for (i in 0..mouseJump) {
                val x2_new = x2 + i
                if (x2_new >= 0 && x2_new < grid[0].length) {
                    if (grid[x1][x2_new] == '#') break
                    val next = help(grid, dp, t + 1, matrixFlattenIdx(x1, x2_new), y, catJump, mouseJump)
                    if (next == 1) return 1.also { dp[t][x][y] = it } else if (next != 2) cat_win = false
                }
            }
            /////////////////////
            for (i in 0..mouseJump) {
                val x2_new = x2 - i
                if (x2_new >= 0 && x2_new < grid[0].length) {
                    if (grid[x1][x2_new] == '#') break
                    val next = help(grid, dp, t + 1, matrixFlattenIdx(x1, x2_new), y, catJump, mouseJump)
                    if (next == 1) return 1.also { dp[t][x][y] = it } else if (next != 2) cat_win = false
                }
            }
            if (cat_win) 2.also { dp[t][x][y] = it } else 0.also {
                dp[t][x][y] = it
            }
        } else {
            var mouse_win = true
            for (i in 0..catJump) {
                val y1_new = y1 + i
                if (y1_new >= 0 && y1_new < grid.size) {
                    if (grid[y1_new][y2] == '#') break
                    val next = help(grid, dp, t + 1, x, matrixFlattenIdx(y1_new, y2), catJump, mouseJump)
                    if (next == 2) return 2.also { dp[t][x][y] = it } else if (next != 1) mouse_win = false
                }
            }
            /////////////////////
            for (i in 0..catJump) {
                val y1_new = y1 - i
                if (y1_new >= 0 && y1_new < grid.size) {
                    if (grid[y1_new][y2] == '#') break
                    val next = help(grid, dp, t + 1, x, matrixFlattenIdx(y1_new, y2), catJump, mouseJump)
                    if (next == 2) return 2.also { dp[t][x][y] = it } else if (next != 1) mouse_win = false
                }
            }
            /////////////////////////////////////////////
            for (i in 0..catJump) {
                val y2_new = y2 + i
                if (y2_new >= 0 && y2_new < grid[0].length) {
                    if (grid[y1][y2_new] == '#') break
                    val next = help(grid, dp, t + 1, x, matrixFlattenIdx(y1, y2_new), catJump, mouseJump)
                    if (next == 2) return 2.also { dp[t][x][y] = it } else if (next != 1) mouse_win = false
                }
            }
            /////////////////////
            for (i in 0..catJump) {
                val y2_new = y2 - i
                if (y2_new >= 0 && y2_new < grid[0].length) {
                    if (grid[y1][y2_new] == '#') break
                    val next = help(grid, dp, t + 1, x, matrixFlattenIdx(y1, y2_new), catJump, mouseJump)
                    if (next == 2) return 2.also { dp[t][x][y] = it } else if (next != 1) mouse_win = false
                }
            }
            if (mouse_win) 1.also { dp[t][x][y] = it } else 0.also {
                dp[t][x][y] = it
            }
        }
    }

    fun matrixFlattenIdx(a: Int, b: Int): Int {
        return a * 8 + b
    }
}

class DSUArray {
    var father: IntArray
    var rank: IntArray
    var size: Int

    constructor(size: Int) {
        this.size = size
        father = IntArray(size)
        rank = IntArray(size)
        Arrays.fill(father, -1)
        Arrays.fill(rank, -1)
    }

    constructor() {
        size = 1 shl 16
        father = IntArray(1 shl 16)
        rank = IntArray(1 shl 16)
        Arrays.fill(father, -1)
        Arrays.fill(rank, -1)
    }

    fun add(i: Int) {
        if (i >= size || i < 0) return
        if (father[i] == -1) {
            father[i] = i
        }
        if (rank[i] == -1) {
            rank[i] = 1
        }
    }

    operator fun contains(i: Int): Boolean {
        return if (i >= size || i < 0) false else father[i] != -1
    }

    fun find(i: Int): Int {
        var i = i
        if (i >= size || i < 0) return -1
        var root = i
        while ((root < size) && (root >= 0) && (father[root] != root)) {
            root = father[root]
        }
        if (root == -1) return -1
        while (father[i] != root) {
            val origFather = father[i]
            father[i] = root
            i = origFather
        }
        return root
    }

    fun merge(i: Int, j: Int): Boolean {
        if (i >= size || i < 0) return false
        if (j >= size || j < 0) return false
        val iFather = find(i)
        val jFather = find(j)
        if (iFather == -1 || jFather == -1) return false
        if (iFather == jFather) return false
        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather
            rank[iFather] += rank[jFather]
        } else {
            father[iFather] = jFather
            rank[jFather] += rank[iFather]
        }
        return true
    }

    fun isConnected(i: Int, j: Int): Boolean {
        if (i >= size || i < 0) return false
        return if (i >= size || i < 0) false else find(i) == find(j)
    }

    // 找出所有根
    val allGroups: Map<Int, MutableSet<Int>>
        get() {
            val result: MutableMap<Int, MutableSet<Int>> = HashMap()
            // 找出所有根
            for (i in 0 until size) {
                if (father[i] != -1) {
                    val f = find(i)
                    result.putIfAbsent(f, HashSet())
                    result[f]!!.add(i)
                }
            }
            return result
        }

    fun getNumOfGroups(): Int {
        return allGroups.size
    }

    fun getSelfGroupSize(x: Int): Int {
        return rank[find(x)]
    }
}