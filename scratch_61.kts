class Solution {
    fun numWaterBottles(numBottles: Int, numExchange: Int): Int {
        var count = numBottles
        var empty = numBottles
        while (empty >= numExchange) {
            val ex = empty / numExchange
            empty %= numExchange
            count += ex
            empty += ex
        }
        return count
    }
}

var s = Solution()
println(s.numWaterBottles(100, 100))

