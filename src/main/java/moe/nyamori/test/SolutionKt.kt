package moe.nyamori.test

object SolutionKt {

    @JvmStatic
    fun main(arg: Array<String>) {
        var timing = System.currentTimeMillis()
        System.err.println(twoSum(1, 2))
        timing = System.currentTimeMillis() - timing
        System.err.println("Timing: ${timing}ms.")
    }

    fun twoSum(a: Int, b: Int) = a + b
}