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
        letters.iterator().forEach { c ->
            if (c.code > target.code) return c
        }
        return letters[0]
    }
}