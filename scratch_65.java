import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.largestPalindrome(7));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC479 **
    public long largestPalindrome(int n) {
        if (n == 1) {
            return 9;
        }
        int upper = (int) Math.pow(10, n) - 1;
        long ans = 0;
        for (int left = upper; ans == 0; --left) { // 枚举回文数的左半部分
            long p = left;
            for (int x = left; x > 0; x /= 10) {
                p = p * 10 + x % 10; // 翻转左半部分到其自身末尾，构造回文数 p
            }
            for (long x = upper; x * x >= p; --x) {
                if (p % x == 0) { // x 是 p 的因子
                    ans = p;
                    break;
                }
            }
        }
        return ans % 1337L;
    }

    // 220327 LYJJ
    int backwardOne;
    int[] op;
    byte[][][] memo;

    public int maxOnes(int[] arr, int m) { // op: 0 - &, 1 - | ,2 - ^
        int n = arr.length + 1;
        backwardOne = m;
        op = arr;
        memo = new byte[2][n][1 << m];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < 1 << m; k++) {
                    memo[i][j][k] = -1;
                }
            }
        }
        return helper(1, n - 1, 0);
    }

    private int helper(int target, int numIdx, int mask) {
        boolean isBackwardMOne = ((mask >> (backwardOne - 1)) & 1) == 1;
        boolean isBackwardMMinusOneOne = ((mask >> (backwardOne - 2)) & 1) == 1;
        if (numIdx == 1) { // 边界条件, 轮到正数第二个数(numIdx==1), 此时只剩下一个运算符
            switch (op[0]) {
                case 0:
                    if (target == 1) { // 此时只能两侧各填1, 所以前m个数和前m-1个数都不能是1, 否则返回极大值
                        if (isBackwardMOne || isBackwardMMinusOneOne) {
                            return Integer.MIN_VALUE / 2;
                        }
                        return 2;
                    } else if (target == 0) {
                        return 0;
                    }
                case 1: // 这里或运算和异或运算所要判断的情形是一致的
                case 2:
                    if (target == 1) { // 前m和前m-1不能同时是1
                        if (isBackwardMOne && isBackwardMMinusOneOne) {
                            return Integer.MIN_VALUE / 2;
                        }
                        return 1;
                    } else if (target == 0) {
                        return 0;
                    }
            }
        }
        if (memo[target][numIdx][mask] != -1) {
            return memo[target][numIdx][mask];
        }
        int result = -1;
        int newMaskWithOne = ((mask << 1) | 1) & ((1 << backwardOne) - 1);
        int newMaskWithZero = ((mask << 1) | 0) & ((1 << backwardOne) - 1);
        switch (op[numIdx - 1]) {
            case 0:
                if (target == 1) { // 此时两侧都要填1, 只要第前m个数是1, 就返回极大值
                    if (isBackwardMOne) {
                        return Integer.MIN_VALUE / 2;
                    }
                    result = 1 + helper(1, numIdx - 1, newMaskWithOne);
                } else if (target == 0) { // (0,0),(0,1),(1,0) 中最大的
                    result = Math.max(
                            Math.max(
                                    0 + helper(0, numIdx - 1, newMaskWithZero),
                                    0 + helper(1, numIdx - 1, newMaskWithZero)
                            ),
                            1 + helper(0, numIdx - 1, newMaskWithOne)
                    );
                }
                break;
            case 1:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0),(1,1) 中最大的
                        result = Math.max(
                                Math.max(
                                        0 + helper(1, numIdx - 1, newMaskWithZero),
                                        1 + helper(0, numIdx - 1, newMaskWithOne)),
                                1 + helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                } else if (target == 0) { // 意味着两侧都要填0
                    result = 0 + helper(0, numIdx - 1, newMaskWithZero);
                }
                break;
            case 2:
                if (target == 1) {
                    if (isBackwardMOne) { // 意味着该位不能填1
                        result = 0 + helper(1, numIdx - 1, newMaskWithZero);
                    } else { // (0,1),(1,0)中最小的
                        result = Math.max(
                                0 + helper(1, numIdx - 1, newMaskWithZero),
                                1 + helper(0, numIdx - 1, newMaskWithOne)
                        );

                    }
                } else if (target == 0) { // (0,0), (1,1)
                    if (isBackwardMOne) {
                        result = 0 + helper(0, numIdx - 1, newMaskWithZero);
                    } else {
                        result = Math.max(
                                0 + helper(0, numIdx - 1, newMaskWithZero),
                                1 + helper(1, numIdx - 1, newMaskWithOne)
                        );
                    }
                }
        }
        return memo[target][numIdx][mask] = (byte) result;
    }

}