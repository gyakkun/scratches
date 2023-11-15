package moe.nyamori.test.ordered._2600

object LC2656 {
    fun maximizeSum(nums: IntArray, k: Int) = nums.max().let { it * k + (k - 1) * k / 2 }
}