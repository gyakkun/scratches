package moe.nyamori.test.ordered._2700

import kotlin.math.max

object LC2760 {
    fun longestAlternatingSubarray(nums: IntArray, threshold: Int): Int {
        var pivot = 0
        var res = 0
        while (pivot < nums.size) {
            if (nums[pivot] % 2 == 1) {
                pivot++
                continue
            }
            var shouldEven = true
            var i = pivot
            while (i < nums.size) {
                if (nums[i] > threshold) break
                if (shouldEven != (nums[i] % 2 == 0)) break
                shouldEven = !shouldEven
                i++
            }
            res = max(res, i - pivot)
            pivot = i.coerceAtLeast(pivot + 1)
        }
        return res
    }
}