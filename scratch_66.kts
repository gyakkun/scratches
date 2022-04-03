import java.time.Duration
import java.time.Instant

var before = Instant.now()!!
var s = Solution()
println(s)
var after = Instant.now()!!
System.err.println("TIMING: ${Duration.between(before, after).toMillis()}ms")

class Solution {
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