import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.gcdSort(new int[]{1239, 1652, 531, 4012, 4484, 2773, 4838, 649, 3245, 4248, 2537, 413, 2773, 4189, 2891, 59, 4543, 1711, 3835, 4425, 2478, 1888, 3776, 649, 2242, 649, 944, 885, 767, 3186, 2596, 4661, 3304, 2714, 4071, 4602, 2360, 3540, 1180, 2596, 590, 4720, 2183, 1947, 413, 3953, 3127, 2242, 826, 4779, 59, 2124, 4130, 2419, 2360, 3481, 3009, 4307, 4307, 1947, 3481, 1062, 3717, 3127, 2478, 4071, 2183, 3540, 118, 1534, 1829, 3304, 1947, 1121, 2773, 2537, 2478, 1416, 4366, 4366, 1357, 2419, 236, 3953, 2183, 118, 3422, 1062, 1888, 3068, 3245, 590, 3540, 4248, 4779, 1062, 2478, 4248, 1003, 4956, 1593, 4425, 1298, 4248, 2537, 3481, 1121, 2891, 1180, 3009, 2006, 1416, 4130, 2537, 3245, 1357, 2832, 1416, 4602, 4956, 1416, 2714, 177, 3186, 4366, 4307, 2301, 3953, 3304, 4248, 4720, 944, 3776, 531, 2655, 3068, 3186, 4661, 3835, 2537, 1239, 3599, 3363, 1121, 413, 4071, 2065, 3127, 2419, 3186, 4484, 3422, 3186, 4897, 1298, 2242, 4543, 3422, 531, 649, 1829, 3599, 4248, 4189, 4071, 4543, 2655, 2537, 295, 1534, 4248, 1357, 767, 3068, 4956, 4130, 3422, 1062, 3363, 4779, 2537, 1416, 3540, 2950, 3009, 354, 413, 4071, 1357, 2714, 1947, 59, 118, 1593, 1003, 4366, 472, 236, 3481, 2891, 2773, 295, 708, 4779, 4130, 2301, 3304, 2655, 3245, 2478, 4189, 1416, 3127, 3363, 2065, 3776, 1121, 2065, 2360, 1652, 236, 2419, 2360, 826, 354, 1475, 295, 2065, 1770, 1239, 1062, 2714, 3835, 708, 2950, 1888, 413, 1711, 2242, 1829, 649, 118, 1534, 2006, 3127, 4248, 4071, 1416, 1593, 4484, 3422, 2950, 4779, 2537, 4130, 1180, 4366, 767, 1475, 1416, 3422, 236, 4189, 1652, 826, 826, 4897, 3599, 3363, 118, 1003, 4779, 3894, 2891, 4307, 1298, 2596, 2242, 1829, 295, 1003, 2773, 4897, 3599, 2950, 590, 4071, 3186, 1416, 1888, 2183, 1888, 3186, 1770, 590, 3068, 2065, 2419, 944, 354, 59, 4307, 1829, 826, 3658, 354, 118, 826, 4543, 177, 177, 3717, 944, 4130, 826, 4602, 2655, 3304, 4071, 1121, 3186, 3127, 2478, 2006, 295, 118, 3658, 2124, 3835, 59, 1180, 3304, 1239, 2419, 4956, 1770, 2655, 590, 3245, 1711, 3363, 4779, 3835, 3186, 295, 767, 826, 354, 4897, 1475, 2773, 3186, 3481, 3776, 3599, 885, 1416, 2065, 2537, 4071, 885, 1003, 1180, 2360, 4307, 2596, 177, 1121, 3127, 1239, 1062, 1770, 177, 2596, 767, 4425, 3009, 2773, 4130, 3717, 2537, 1770, 1770, 1475, 4602, 59, 3540, 2714, 4602, 3658, 4012, 2891, 3599, 1829, 2006, 3363, 1829, 4012, 1239, 118, 1593, 4425, 2950, 1829, 531, 2832, 1357, 2065, 1593, 1180, 1711, 2891, 2537, 3422, 1003, 59, 177, 2419, 3540, 767, 3422, 4720, 1357, 3599, 2773, 2714, 1593, 3009, 1003, 649, 2773, 3953, 1829, 1121, 4779, 2360, 1652, 4956, 3658, 1947, 2537, 3894, 3186, 3422, 944, 1239, 3363, 295, 708, 4897, 3658, 4012, 472, 2832, 1711, 590, 2655, 1829, 3245, 1711, 3245, 885, 2419, 3304, 413, 4071, 3304, 590, 2124, 2714, 4012, 3009, 4661, 3009, 3304, 1475, 2183, 472, 4897, 1770, 2124, 649, 4012, 236, 3717, 4071, 1298, 1298, 531, 1947, 2832, 1947, 2537, 1829, 2655, 2124, 3009, 4956, 354, 3894, 4189, 1770, 4307, 1593, 3776, 4602, 767, 2950, 1062, 1062, 1534, 708, 295, 2714, 413, 4779, 4307, 2537, 59, 3717, 4484, 4543, 1534, 4425, 1711, 354, 413, 4897, 3776, 4484, 4484, 3599, 3717, 3245, 4307, 4425, 4248, 3953, 4484, 2242, 4484, 2596, 1239, 2655, 4189, 4130, 4779, 2006, 4543, 2301, 2006, 1711, 4130, 236, 1475, 2478, 4425, 1239, 1711, 3599, 1534, 2891, 4130, 59, 3363, 3304, 4602, 885, 4012, 2596, 2242, 531, 4661, 3599, 4661, 2832, 1062, 1593, 4189, 4425, 767, 2242, 2124, 236, 3186, 3422, 4897, 4602, 4956, 767, 4012, 885, 1652, 3599, 3776, 4720, 4543, 2596, 3894, 3304, 1888, 295, 4307, 4189, 3363, 885, 4543, 4071, 4661, 4189, 236, 1888, 2124, 2773, 4661, 4012, 4189, 2950, 4956, 1947, 4307, 1121, 1475, 2655, 59, 1829, 826, 3068, 295, 3245, 4189, 3127, 2183, 3304, 2773, 885, 944, 1711, 1416, 1239, 2301, 4130, 4071, 1357, 4366, 3717, 3599, 1475, 472, 2773, 1711, 1003, 826, 2242, 3894, 1770, 472, 1003, 649, 3363, 2950, 3717, 4189, 413, 4366, 3658, 3717, 3009, 531, 1534, 1239, 2655, 3009, 3068, 4130, 3540, 4956, 3245, 295, 4956, 3245, 3776, 4484, 295, 531, 4307, 2360, 3245, 295, 4838, 2124, 4720, 2006, 1711, 1298, 1829, 4130, 1829, 590, 2596, 4012, 1180, 1298, 649, 1652, 177, 177, 2832, 2714, 1888, 531, 1062, 3363, 4484, 3245, 3127, 118, 2832, 2065, 1770, 3953, 2360, 4484, 1829, 3363, 4425, 885, 295, 1829, 4189, 1239, 826, 2832, 3245, 3363, 3540, 59, 1239, 1180, 2950, 1888, 413, 708, 2950, 531, 236, 2419, 3068, 3127, 1593, 2655, 4248, 531, 4661, 2006, 354, 2773, 4189, 2537, 2183, 3835, 4602, 3186, 3009, 3009, 2478, 826, 3186, 2773, 1298, 1180, 4130, 4012, 4307, 3009, 4602, 2183, 3776, 2183, 2950, 1829, 236, 118, 4956, 236, 4956, 3186, 1180, 1770, 3422, 590, 2773, 885, 3422, 1829, 3540, 1239, 1357, 4661, 2360, 4661, 236, 708, 4307, 531, 472, 3481, 1416, 1829, 4012, 4307, 2891, 885, 354, 1711, 649, 4661, 472, 1888, 4130, 2832, 1829, 1416, 1534, 2360, 590, 4897, 4366, 2301, 2478, 708, 3540, 4956, 472, 1475, 295, 3894, 885, 3953, 236, 4307, 3658, 1888, 2950, 472, 3186, 2242, 118, 3776, 2537, 590, 1239, 4130, 2124, 1475, 2950, 2773, 177, 3068, 2891, 4720, 2537, 4012, 2183, 3776, 3599, 3127, 3068, 3658, 4484, 2065, 1239, 3422, 4425, 4484, 4189, 2596, 3422, 4189, 708, 1593, 3717, 708, 1947, 2006, 3009, 4366, 4366, 4543, 1416, 4366, 2478, 1770, 885, 3127, 2950, 3599, 4071, 1534, 4484, 826, 3776, 708, 4189, 2301, 1180, 3540, 1711, 3481, 4956, 4956, 2714, 4012, 3186, 3186, 472, 2773, 3540, 3304, 1475, 2419, 708, 3658, 4543, 2183, 4956, 944, 2832, 4897, 2006, 3776, 3835, 4307, 2065, 4012, 4897, 649, 295, 1947, 1593, 1770, 3540, 4366, 3658, 472, 1534, 177, 4838, 3304, 2773, 708, 3068, 3304, 4956, 1711, 4602, 2124, 2242, 3481, 2006, 3658, 4779, 472, 4189, 3363, 4720, 4543, 2124, 2242, 944, 59, 3304, 3776, 649, 1003, 4956, 1534, 4484, 3717, 4071, 4543, 2773, 1416, 2419, 3717, 826, 708, 2124, 3245, 1121, 1888, 2832, 1475, 4956, 2478, 1711, 1003, 177, 4543, 1475, 2183, 3894, 3304, 2596, 236, 4012, 1416, 1770, 1711, 3717, 4956, 4543, 3599, 2065, 1357, 3422, 4012, 3894, 1062, 3776, 531, 4720, 4720, 3599, 4602, 4012, 3658, 1180, 1711, 4897, 1416, 1829, 59, 2360, 4956, 1652, 1947, 4012, 531, 1416, 4897, 3894, 2006, 4425, 826, 590, 3422, 3658, 2832, 236, 3953, 4602, 4130, 4366, 118, 2124, 4838, 531, 3245, 2832, 4130, 295, 472, 3481, 767, 1475, 2006, 4602, 3009, 885, 4425, 3009, 3186, 177, 3009, 1534, 2124, 1593, 531, 4248, 4012, 2950, 3599, 1357, 1534, 2537, 3245, 4366, 4248, 118, 3422, 2183, 2419, 354, 2242, 2006, 531, 1416, 708, 4543, 4012, 4602, 295, 3127, 2360, 2124, 1298, 3540, 4484, 1239, 4661, 531, 3186, 2301, 826, 4779, 236, 472, 3717, 2006, 4602, 826, 708, 826, 4779, 1475, 3363, 531, 1180, 4484, 2478, 3894, 4602, 2360, 1888, 4366, 4779, 1357, 767, 1416, 3245, 2478, 295, 1416, 3186, 1416, 3599, 4956, 3658, 2714, 3127, 4956, 4720, 2183, 944, 767, 413, 3363, 3658, 4425, 4602, 2714, 118, 2478, 2183, 1534, 472, 2360, 3127, 944, 2124, 2478, 2124, 2596, 1475, 3835, 3127, 4602, 1357, 4779, 3127, 2242, 295, 4543, 3363, 4366, 944, 3245, 708, 4307, 4661, 708, 1593, 2891, 885, 2773, 3658, 1770, 3717, 2773, 3835, 1829, 3304, 3304, 3835, 3894, 3363, 3127, 3481, 3068, 4779, 1121, 1298, 2065, 3599, 826, 1888, 295, 4661, 1357, 4779, 1121, 4071, 590, 4425, 2360, 4602, 2242, 4779, 3009, 3245, 472, 3717, 2832, 1475, 2714, 3009, 2950, 1416, 3422, 1593, 2537, 2714, 3304, 1062, 1180, 2478, 236, 1121, 4956, 3717, 3717, 2537, 1180, 4838, 1652, 708, 2242, 2950, 1593, 3776, 3776, 4602, 1298, 3953, 2006, 2537, 4897, 1770, 2596, 4838, 2773, 2360, 4779, 3068, 1947, 3422, 3599, 4897, 4661, 4248, 885, 2714, 4071, 1239, 3363, 4484, 1062, 4248, 4130, 3599, 4189, 2124, 944, 3009, 3304, 2065, 1829, 3068, 4425, 1121, 2124, 2360, 3540, 2891, 4189, 1003, 590, 1416, 944, 118, 1062, 2301, 3068, 3481, 4071, 354, 3717, 3481, 826, 3835, 2655, 531, 472, 4484, 531, 2301, 3776, 4543, 4779, 3245, 4484, 1121, 1062, 2950, 1357, 2891, 767, 4602, 1593, 3717, 4602, 3658, 767, 1003, 3304, 1947, 3717, 1180, 2360, 4366, 3186, 4956, 2065, 3776, 1239, 4661, 2124, 3894, 4543, 2183, 4897, 4248, 2124, 3540, 4897, 2360, 1711, 944, 3245, 2773, 3422, 354, 4189, 2773, 413, 2242, 295, 472, 4189, 2478, 4956, 2006, 2124, 1003, 4897, 2655, 3953, 2419, 531, 531, 472, 4130, 4012, 295, 4779, 2124, 1947, 3599, 2478, 1652, 944, 2065, 3304, 295, 4897, 3540, 4071, 3894, 1416, 4071, 1180, 4602, 2065, 3481, 1003, 3363, 2655, 4012, 4366, 4543, 4720, 177, 767, 4720, 3481, 2655, 2832, 649, 2596, 1003, 1888, 4307, 3245, 3599, 4071, 1947, 3658, 3658, 1416, 4779, 236, 1475, 4720, 3422, 4366, 3953, 3186, 2832, 1416, 4838, 4189, 2537, 3835, 1121, 1652, 4661, 1003, 2596, 3894, 3953, 1416, 1711, 4720, 59, 2124, 590, 3658, 4425, 4720, 2773, 1829, 177, 3304, 1416, 1652, 2714, 4779, 3068, 1888, 1298, 4897, 3776, 1180, 1534, 4779, 1121, 2832, 2773, 1062, 2478, 1475, 590, 2242, 1593, 767, 4071, 3127, 3835, 236, 2183, 3363, 59, 3481, 3304, 649, 3304, 4897, 4189, 708, 3304, 2537, 767, 4897, 2419, 2537, 2478, 1770, 944, 1003, 4897, 2242, 4838, 4602, 2419, 2242, 2891, 3894, 2891, 2360, 4779, 531, 2596, 1711, 1239, 1947, 2832, 1062, 2773, 2655, 2301, 1298, 3953, 2773, 3422, 236, 3481, 1416, 2714, 236, 3422, 708, 3835, 1711, 2183, 4248, 2065, 2419, 4425, 1239, 3009, 1770, 2301, 3481, 1947, 4425, 1593, 767, 4543, 3658, 3363, 3422, 2655, 1180, 1711, 1947, 1593, 2891, 826, 4189, 1829, 3363, 4838, 3363, 1475, 4130, 2006, 2655, 3481, 1770, 2596, 1711, 3658, 4484, 3540, 4130, 1829, 59, 531, 2360, 3186, 4543, 2714, 2478, 708, 2183, 2950, 3540, 177, 4484, 4012, 2714, 1416, 1534, 4366, 3481, 4425, 59, 3776, 3068, 3658, 4248, 354, 1534, 4071, 2478, 2419, 4189, 2419, 1298, 3245, 3127, 59, 1947, 3953, 1711, 3776, 1770, 354, 1711, 177, 3422, 1829, 649, 3245, 1593, 2124, 4425, 2360, 3127, 413, 3717, 2242, 4897, 1888, 708, 2714, 1652, 2950, 295, 4130, 3894, 1003, 4366, 1711, 354, 2124, 3481, 649, 3658, 4130, 4366, 4661, 2242, 2183, 2832, 4956, 1888, 4130, 1121, 4130, 4956, 885, 1180, 472, 2301, 2950, 4779, 1888, 4779, 944, 2360, 3068, 4720, 3009, 4838, 2537, 4071, 1534, 1121, 1475, 1475, 2891, 1180, 4779, 2419, 2891, 3009, 3953, 1180, 3717, 177, 4012, 826, 2301, 3304, 4602, 2596, 2183, 4130, 4012, 3894, 826, 1357, 3835, 4130, 354, 4425, 4366, 3658, 944, 2714, 4956, 3717, 3127, 118, 590, 2596, 2478, 1593, 3422, 3599, 4189, 3304, 4779, 3009, 1475, 2065, 472, 4720, 531, 3540, 2065, 472, 177, 2773, 4779, 531, 1593, 2242, 3894, 2183, 3717, 295, 3363, 2301, 944, 3540, 2242, 1593, 2183, 2537, 3245, 1888, 1652, 3009, 3835, 2478, 2065, 4366, 236, 1947, 3658, 4543, 1239, 2124, 4189, 1770, 3953, 4956, 4602, 1475, 767, 1298, 4602, 354, 4071, 649, 2301, 3540, 4189, 1003, 3540, 2478, 4012, 2714, 1534, 2183, 4071, 2360, 2773, 1947, 2065, 236, 2301, 708, 177, 4602, 3422, 4543, 4425, 1947, 3953, 3068, 2242, 4484, 708, 4543, 2832, 2360, 118, 649, 2065, 4602, 1534, 2832, 4012, 708, 1180, 4425, 2183, 3953, 1003, 2649, 2649, 3532, 4415, 2649, 4415, 1766, 2649, 4415, 4415, 1766, 3532, 1766, 2649, 1766, 1766, 1766, 4415, 2649, 2649, 3532, 1766, 2649, 4415, 2649, 1766, 883, 883, 1766, 3532, 3532, 883, 4415, 883, 4415, 2649, 2649, 1766, 1766, 4415, 2649, 1766, 3532, 883, 2649, 1766, 3532, 2649, 2649, 1766, 2649, 4415, 2649, 3532, 2649, 4415, 4415, 883, 4415, 1766, 1766, 883, 1766, 4415, 2649, 883, 4415, 1766, 883, 2649, 2649, 1766, 2649, 3532, 883, 4415, 1766, 4415, 4415, 2649, 1766, 2649, 3532, 2649, 3532, 1766, 3532, 3532, 2649, 3532, 883, 1766, 4415, 3532, 3532, 883, 4425, 1180, 4012, 3776, 1416, 3422, 1298, 2655, 708, 1357, 4189, 1416, 4425, 767, 590, 2006, 3658, 413, 767, 826, 4838, 3481, 3186, 3894, 4779, 59, 2301, 3481, 3658, 4012, 2773, 531, 3186, 2360, 4543, 3953, 3717, 3068, 4130, 1770, 1239, 1534, 1357, 1652, 4838, 3894, 531, 177, 3776, 4012, 2478, 1888, 3894, 3599, 4362, 3635, 2181, 2181, 3635, 727, 710}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1998 **
    public boolean gcdSort(int[] nums) {
        DSUArray dsu = new DSUArray((int) 1e5);
        for (int i : nums) {
            if (dsu.contains(i)) continue;
            dsu.add(i);
            int factor = 2;
            int victim = i;
            // 分解质因数
            while (victim != 0) {
                while (victim % factor == 0) {
                    dsu.add(factor);
                    dsu.merge(factor, i);
                    victim /= factor;
                }
                factor++;
                if (factor > victim) break;
            }
        }

        int[] sorted = Arrays.copyOf(nums, nums.length);
        Arrays.sort(sorted);

        for (int i = 0; i < nums.length; i++) {
            if (sorted[i] != nums[i]) {
                if (!dsu.isConnected(sorted[i], nums[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    // LC575
    public int distributeCandies(int[] candyType) {
        int n = candyType.length, result = 0;
        Arrays.sort(candyType);
        for (int i = 0; i < n; ) {
            while (i + 1 < n && candyType[i] == candyType[i + 1]) i++;
            result++;
            if (result >= n / 2) return n / 2;
            i++;
        }
        return result;
    }

    // LC260 **
    public int[] singleNumber(int[] nums) {
        long xor = 0;
        for (long i : nums) xor ^= i;
        long lsb = xor & (-xor);
        long a = 0, b = 0;
        for (int i : nums) {
            if (((long) (i) & lsb) != 0) {
                a ^= i;
            } else {
                b ^= i;
            }
        }
        return new int[]{(int) a, (int) b};
    }

    // LC1911 **
    // from https://codeforces.com/contest/1420/submission/93658399
    public long maxAlternatingSum(int[] nums) {
        // 偶数下标之和减奇数下标之和
        int n = nums.length;
        long[][] dp = new long[n + 2][2];
        // dp[n][0/1] 表示选前n个数做子序列时候, 第n个数字下标作为偶/奇数时候的最大值
        for (int i = 0; i < n; i++) {
            dp[i + 1][0] = Math.max(dp[i][0]/*不选这个数*/, dp[i][1] + nums[i]/*选这个数, 从上一个奇数结尾的状态转移过来*/);
            dp[i + 1][1] = Math.max(dp[i][1], dp[i][0] - nums[i]);
        }
        return Math.max(dp[n][0], dp[n][1]);
    }

    // LC1102 **
    public int maximumMinimumPathDSU(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int start = 0, end = m * n - 1;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        DSUArray dsu = new DSUArray(m * n + 2);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o[2]));
        for (int i = 1; i < end; i++) {
            pq.offer(new int[]{i / n, i % n, grid[i / n][i % n]});
        }
        dsu.add(start);
        dsu.add(end);
        while (!pq.isEmpty() && !dsu.isConnected(start, end)) {
            int[] p = pq.poll();
            int r = p[0], c = p[1], val = p[2];
            int id = r * n + c;
            dsu.add(id);
            for (int[] d : direction) {
                int nr = r + d[0], nc = c + d[1];
                int nid = nr * n + nc;
                if (nr >= 0 && nr < m && nc >= 0 && nc < n && dsu.contains(nid)) {
                    dsu.merge(id, nid);
                    min = Math.min(min, val);
                }
            }
        }
        return min;
    }

    public int maximumMinimumPath(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int min = Math.min(grid[0][0], grid[m - 1][n - 1]);
        Set<Integer> possibleSet = new HashSet<>();
        possibleSet.add(min);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] <= min) {
                    possibleSet.add(grid[i][j]);
                }
            }
        }
        List<Integer> possibleList = new ArrayList<>(possibleSet);
        Collections.sort(possibleList);
        int lo = 0, hi = possibleList.size() - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            int victim = possibleList.get(mid);
            boolean[][] visited = new boolean[m][n];
            if (lc1102Helper(0, 0, grid, visited, victim)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return possibleList.get(lo);
    }

    private boolean lc1102Helper(int r, int c, int[][] grid, boolean[][] visited, int bound) {
        if (r == grid.length - 1 && c == grid[0].length - 1) return true;
        visited[r][c] = true;
        int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] d : direction) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < grid.length && nc >= 0 && nc < grid[0].length && !visited[nr][nc] && grid[nr][nc] >= bound) {
                if (lc1102Helper(nr, nc, grid, visited, bound)) return true;
            }
        }
        return false;
    }


    // LC565
    public int arrayNesting(int[] nums) {
        int n = nums.length, max = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int count = 1, cur = i;
                visited[i] = true;
                while (!visited[nums[cur]]) {
                    visited[nums[cur]] = true;
                    count++;
                    cur = nums[cur];
                }
                max = Math.max(count, max);
            }
        }
        return max;
    }

    // LC1145 ** 贪心策略: 选x周围的三个节点, 统计两个子图节点数量
    Map<Integer, TreeNode> valNodeMap = new HashMap<>();
    Map<TreeNode, TreeNode> fatherMap = new HashMap<>();

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            valNodeMap.put(p.val, p);
            n = Math.max(n, p.val);
            if (p.left != null) {
                fatherMap.put(p.left, p);
                q.offer(p.left);
            }
            if (p.right != null) {
                fatherMap.put(p.right, p);
                q.offer(p.right);
            }
        }

        TreeNode xNode = valNodeMap.get(x);
        TreeNode[] choices = new TreeNode[]{getFather(xNode), getLeft(xNode), getRight(xNode)};
        for (TreeNode y : choices) {
            if (y != null) {
                if (lc1145Helper(n, x, xNode, y)) return true;
            }
        }
        return false;
    }

    private boolean lc1145Helper(int n, int x, TreeNode rivalFirstChoice, TreeNode y) {
        Deque<TreeNode> q = new LinkedList<>();
        // BFS, 统计邻接节点数量
        boolean[] visited = new boolean[n + 1];
        visited[x] = true;
        q.offer(y);
        int myCount = getTreeNodeCount(q, visited);

        Arrays.fill(visited, false);
        visited[y.val] = true;
        q.clear();
        q.offer(rivalFirstChoice);
        int rivalCount = getTreeNodeCount(q, visited);

        return rivalCount < myCount;
    }

    private int getTreeNodeCount(Deque<TreeNode> q, boolean[] visited) {
        int count = 0;
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            if (visited[p.val]) continue;
            visited[p.val] = true;
            count++;
            TreeNode f = getFather(p), l = getLeft(p), r = getRight(p);
            if (f != null && !visited[f.val]) q.offer(f);
            if (l != null && !visited[l.val]) q.offer(l);
            if (r != null && !visited[r.val]) q.offer(r);
        }
        return count;
    }

    private TreeNode getFather(TreeNode root) {
        return fatherMap.get(root);
    }

    private TreeNode getLeft(TreeNode root) {
        return root.left;
    }

    private TreeNode getRight(TreeNode root) {
        return root.right;
    }


    // LC1983
    public int widestPairOfIndices(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int prefix1 = 0, prefix2 = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix1 += nums1[i];
            prefix2 += nums2[i];
            int diff = prefix1 - prefix2;
            if (map.containsKey(diff)) {
                result = Math.max(result, i - map.get(diff));
            } else {
                map.put(diff, i);
            }
        }
        return result;
    }

    // LC335 **
    public boolean isSelfCrossing(int[] distance) {
        for (int i = 3; i < distance.length; i++) {
            if (i >= 3
                    && distance[i] >= distance[i - 2]
                    && distance[i - 1] <= distance[i - 3])
                return true;
            else if (i >= 4
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 1] == distance[i - 3])
                return true;
            else if (i >= 5
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 5] + distance[i - 1] >= distance[i - 3]
                    && distance[i - 2] > distance[i - 4]
                    && distance[i - 3] > distance[i - 1])
                return true;
        }
        return false;
    }

    // LC869
    public boolean reorderedPowerOf2(int n) {
        if (n == 0) return false;
        List<Integer> power2List = new ArrayList<>(31);
        for (int i = 0; i < 31; i++) power2List.add(1 << i);
        int[][] freqList = new int[31][10];
        for (int i = 0; i < 31; i++) {
            int power2 = power2List.get(i);
            int[] freq = new int[10];
            while (power2 != 0) {
                freq[power2 % 10]++;
                power2 /= 10;
            }
            freqList[i] = freq;
        }
        int[] thisFreq = new int[10];
        int dummy = n;
        while (dummy != 0) {
            thisFreq[dummy % 10]++;
            dummy /= 10;
        }
        outer:
        for (int i = 0; i < 31; i++) {
            for (int j = 0; j < 10; j++) {
                if (freqList[i][j] != thisFreq[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    // JZOF II 086 LC131
    List<List<String>> lc131Result;
    List<String> lc131Tmp;

    public String[][] partition(String s) {
        lc131Result = new ArrayList<>();
        lc131Tmp = new ArrayList<>();
        int n = s.length();
        boolean[][] judge = new boolean[n][n];
        char[] ca = s.toCharArray();
        for (int i = 0; i < n; i++) judge[i][i] = true;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left + len - 1 < n; left++) {
                if (len == 2) {
                    judge[left][left + 1] = ca[left] == ca[left + 1];
                } else if (judge[left + 1][left + len - 1 - 1] && ca[left] == ca[left + len - 1]) {
                    judge[left][left + len - 1] = true;
                }
            }
        }
        lc131Helper(0, judge, s);
        String[][] resArr = new String[lc131Result.size()][];
        for (int i = 0; i < lc131Result.size(); i++) {
            resArr[i] = lc131Result.get(i).toArray(new String[lc131Result.get(i).size()]);
        }
        return resArr;
    }

    private void lc131Helper(int idx, boolean[][] judge, String s) {
        if (idx == judge.length) {
            lc131Result.add(new ArrayList<>(lc131Tmp));
            return;
        }
        for (int len = 1; idx + len - 1 < judge.length; len++) {
            if (judge[idx][idx + len - 1]) {
                lc131Tmp.add(s.substring(idx, idx + len));
                lc131Helper(idx + len, judge, s);
                lc131Tmp.remove(lc131Tmp.size() - 1);
            }
        }
    }


    // LC792 ** 桶思想
    public int numMatchingSubseqBucket(String s, String[] words) {
        int result = 0;
        Map<Character, List<List<Character>>> bucket = new HashMap<>();
        for (String w : words) {
            bucket.putIfAbsent(w.charAt(0), new LinkedList<>());
            List<Character> bucketItem = new LinkedList<>();
            for (char c : w.toCharArray()) bucketItem.add(c);
            bucket.get(w.charAt(0)).add(bucketItem);
        }
        for (char c : s.toCharArray()) {
            Set<Character> set = new HashSet<>(bucket.keySet());
            for (char key : set) {
                if (c != key) continue;
                List<List<Character>> items = bucket.get(key);
                ListIterator<List<Character>> it = items.listIterator();
                while (it.hasNext()) {
                    List<Character> seq = it.next();
                    it.remove();
                    seq.remove(0);
                    if (seq.size() == 0) result++;
                    else {
                        bucket.putIfAbsent(seq.get(0), new LinkedList<>());
                        if (seq.get(0) == key) {
                            it.add(seq);
                        } else {
                            bucket.get(seq.get(0)).add(seq);
                        }
                    }
                }
                if (bucket.get(key).size() == 0) bucket.remove(key);
            }
        }
        return result;
    }

    // LC1055 **
    public int shortestWay(String source, String target) {
        int tIdx = 0, result = 0;
        char[] cs = source.toCharArray(), ct = target.toCharArray();
        while (tIdx < ct.length) {
            int sIdx = 0;
            int pre = tIdx;
            while (tIdx < ct.length && sIdx < cs.length) {
                if (ct[tIdx] == cs[sIdx]) tIdx++;
                sIdx++;
            }
            if (tIdx == pre) return -1;
            result++;
        }
        return result;
    }


    // LC1689
    public int minPartitions(String n) {
        int max = 0;
        for (char c : n.toCharArray()) {
            max = Math.max(max, c - '0');
        }
        return max;
    }

    // LC1962
    public int minStoneSum(int[] piles, int k) {
        int[] freq = new int[10001];
        for (int i : piles) freq[i]++;
        int sum = 0;
        for (int i = 10000; i >= 0; i--) {
            if (freq[i] == 0) continue;
            if (k > 0) {
                int minusTime = Math.min(k, freq[i]);
                freq[i] -= minusTime;
                freq[i - i / 2] += minusTime;
                k -= minusTime;
            }
            sum += i * freq[i];
        }
        return sum;
    }

    // LC301 **
    Set<String> lc301Result = new HashSet<>();

    public List<String> removeInvalidParentheses(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        // 多余的左右括号个数, 注意右括号多余当且仅当左边左括号不够匹配的时候
        int leftToRemove = 0, rightToRemove = 0;
        for (char c : ca) {
            if (c == '(') leftToRemove++;
            else if (c == ')') {
                if (leftToRemove == 0) rightToRemove++;
                else leftToRemove--;
            }
        }
        lc301Helper(0, ca, leftToRemove, rightToRemove, 0, 0, new StringBuilder());
        return new ArrayList<>(lc301Result);
    }

    private void lc301Helper(int curIdx, char[] ca,
            /*待删的左括号数*/
                             int leftToRemove, int rightToRemove,
            /*已删的左括号数*/
                             int leftCount, int rightCount,
                             StringBuilder sb) {
        if (curIdx == ca.length) {
            if (leftToRemove == 0 && rightToRemove == 0) {
                lc301Result.add(sb.toString());
            }
            return;
        }

        char c = ca[curIdx];
        if (c == '(' && leftToRemove > 0) { // 无视当前左括号
            lc301Helper(curIdx + 1, ca, leftToRemove - 1, rightToRemove, leftCount, rightCount, sb);
        }
        if (c == ')' && rightToRemove > 0) { // 无视当前右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove - 1, leftCount, rightCount, sb);
        }

        sb.append(c);
        if (c != '(' && c != ')') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount, sb);
        } else if (c == '(') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount + 1, rightCount, sb);
        } else if (c == ')' && rightCount < leftCount) { // 只有当当前已选择的左括号比右括号多才在此步选右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount + 1, sb);
        }
        sb.deleteCharAt(sb.length() - 1);
    }

    // LC1957
    public String makeFancyString(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        int n = ca.length;
        for (int i = 0; i < n; ) {
            int curIdx = i;
            char cur = ca[i];
            while (i + 1 < n && ca[i + 1] == cur) i++;
            int count = Math.min(i - curIdx + 1, 2);
            for (int j = 0; j < count; j++) {
                sb.append(cur);
            }
            i++;
        }
        return sb.toString();
    }

    // LC1540
    public boolean canConvertString(String s, String t, int k) {
        if (s.length() != t.length()) return false;
        // 第i次操作(从1算) 可以将s种之前未被操作过的下标j(从1算)的char+i
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        List<Integer> shouldChangeIdx = new ArrayList<>();
        for (int i = 0; i < cs.length; i++) {
            if (cs[i] != ct[i]) shouldChangeIdx.add(i);
        }
        int[] minSteps = new int[shouldChangeIdx.size()];
        for (int i = 0; i < shouldChangeIdx.size(); i++) {
            char sc = cs[shouldChangeIdx.get(i)], tc = ct[shouldChangeIdx.get(i)];
            minSteps[i] = (tc - 'a' + 26 - (sc - 'a')) % 26;
        }
        int[] freq = new int[27];
        for (int i : minSteps) freq[i]++;
        int max = 0;
        for (int i = 1; i <= 26; i++) {
            max = Math.max(max, i + (freq[i] - 1) * 26);
        }
        return max <= k;
    }

    // LC266
    public boolean canPermutePalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int oddCount = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddCount++;
        return oddCount <= 1;
    }

    // LC409
    public int longestPalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int even = 0, oddMax = 0, odd = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 0) even += freq[i];
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddMax = Math.max(oddMax, freq[i]);
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) odd += freq[i] - 1;
        if (oddMax == 0) return even;
        return odd + even + 1;
    }

    // LC1266
    public int minTimeToVisitAllPoints(int[][] points) {
        int x = points[0][0], y = points[0][1];
        int result = 0;
        for (int i = 1; i < points.length; i++) {
            int nx = points[i][0], ny = points[i][1];
            int deltaX = Math.abs(nx - x), deltaY = Math.abs(ny - y);
            int slash = Math.min(deltaX, deltaY);
            int line = Math.max(deltaX, deltaY) - slash;
            result += line + slash;
            x = nx;
            y = ny;
        }
        return result;
    }

    // LC1416
    Integer[] lc1416Memo;

    public int numberOfArrays(String s, int k) {
        int n = s.length();
        lc1416Memo = new Integer[n + 1];
        return lc1416Helper(0, s, k);
    }

    private int lc1416Helper(int cur, String s, int k) {
        final long mod = 1000000007l;
        if (cur == s.length()) return 1;
        if (lc1416Memo[cur] != null) return lc1416Memo[cur];
        int len = 1;
        long result = 0;
        while (cur + len <= s.length()) {
            long num = Long.parseLong(s.substring(cur, cur + len));
            if (String.valueOf(num).length() != len) break;
            if (num > k) break;
            if (num < 1) break;
            result += lc1416Helper(cur + len, s, k);
            result %= mod;
            len++;
        }
        return lc1416Memo[cur] = (int) (result % mod);
    }

    // LC1844
    public String replaceDigits(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            if (i % 2 == 0) sb.append(ca[i]);
            else sb.append((char) (ca[i - 1] + (ca[i] - '0')));
        }
        return sb.toString();
    }

    //
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}

class DSUArray {
    int[] father;
    int[] rank;
    int size;

    public DSUArray(int size) {
        this.size = size;
        father = new int[size];
        rank = new int[size];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public DSUArray() {
        this.size = Integer.MAX_VALUE >> 16;
        father = new int[Integer.MAX_VALUE >> 16];
        rank = new int[Integer.MAX_VALUE >> 16];
        Arrays.fill(father, -1);
        Arrays.fill(rank, -1);
    }

    public void add(int i) {
        if (i >= this.size || i < 0) return;
        if (father[i] == -1) {
            father[i] = i;
        }
        if (rank[i] == -1) {
            rank[i] = 1;
        }
    }

    public boolean contains(int i) {
        if (i >= this.size || i < 0) return false;
        return father[i] != -1;
    }

    public int find(int i) {
        if (i >= this.size || i < 0) return -1;
        int root = i;
        while (root < size && root >= 0 && father[root] != root) {
            root = father[root];
        }
        if (root == -1) return -1;
        while (father[i] != root) {
            int origFather = father[i];
            father[i] = root;
            rank[root]++;
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (j >= this.size || j < 0) return false;
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == -1 || jFather == -1) return false;
        if (iFather == jFather) return false;

        if (rank[iFather] >= rank[jFather]) {
            father[jFather] = iFather;
            rank[iFather] += rank[jFather];
        } else {
            father[iFather] = jFather;
            rank[jFather] += rank[iFather];
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        if (i >= this.size || i < 0) return false;
        if (i >= this.size || i < 0) return false;
        return find(i) == find(j);
    }


}

class DisjointSetUnion<T> {

    Map<T, T> father;
    Map<T, Integer> rank;

    public DisjointSetUnion() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(T i) {
        if (!father.containsKey(i)) {
            // 置初始父亲为自身
            // 之后判断连通分量个数时候, 遍历father, 找value==key的
            father.put(i, i);
        }
        if (!rank.containsKey(i)) {
            rank.put(i, 1);
        }
    }

    // 找父亲, 路径压缩
    public T find(T i) {
        //先找到根 再压缩
        T root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            T origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(T i, T j) {
        T iFather = find(i);
        T jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(T i, T j) {
        return find(i) == find(j);
    }

    public Map<T, Set<T>> getAllGroups() {
        Map<T, Set<T>> result = new HashMap<>();
        // 找出所有根
        for (T i : father.keySet()) {
            T f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<T> s = new HashSet<T>();
        for (T i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

    public boolean contains(T i) {
        return father.containsKey(i);
    }

}

