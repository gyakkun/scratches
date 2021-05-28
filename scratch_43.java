import javafx.util.*;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        int[] status = {1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0};
        int[] candies = {732, 320, 543, 300, 814, 568, 947, 685, 142, 111, 805, 233, 813, 306, 55, 1, 290, 944, 36, 592, 150, 596, 372, 299, 644, 445, 605, 202, 64, 807, 753, 731, 552, 766, 119, 862, 453, 136, 43, 572, 801, 518, 936, 408, 515, 215, 492, 738, 154};
        int[][] keys = {{42, 2, 24, 8, 39, 16, 46}, {20, 39, 46, 21, 32, 31, 43, 16, 12, 23, 3}, {21, 14, 30, 2, 11, 13, 27, 37, 4, 48}, {16, 17, 15, 6}, {31, 14, 3, 32, 35, 19, 42, 43, 44, 29, 25, 41}, {7, 39, 2, 3, 40, 28, 37, 35, 43, 22, 6, 23, 48, 10, 21, 11}, {27, 1, 37, 3, 45, 32, 30, 26, 16, 2, 35, 19, 31, 47, 5, 14}, {28, 35, 23, 17, 6}, {6, 39, 34, 22}, {44, 29, 36, 31, 40, 22, 9, 11, 17, 25, 1, 14, 41}, {39, 37, 11, 36, 17, 42, 13, 12, 7, 9, 43, 41}, {23, 16, 32, 37}, {36, 39, 21, 41}, {15, 27, 5, 42}, {11, 5, 18, 48, 25, 47, 17, 0, 41, 26, 9, 29}, {18, 36, 40, 35, 12, 33, 11, 5, 44, 14, 46, 7}, {48, 22, 11, 33, 14}, {44, 12, 3, 31, 25, 15, 18, 28, 42, 43}, {36, 9, 0, 42}, {1, 22, 3, 24, 9, 11, 43, 8, 35, 5, 41, 29, 40}, {15, 47, 32, 28, 33, 31, 4, 43}, {1, 11, 6, 37, 28}, {46, 20, 47, 32, 26, 15, 11, 40}, {33, 45, 26, 40, 12, 3, 16, 18, 10, 28, 5}, {14, 6, 4, 46, 34, 9, 33, 24, 30, 12, 37}, {45, 24, 18, 31, 32, 39, 26, 27}, {29, 0, 32, 15, 7, 48, 36, 26, 33, 31, 18, 39, 23, 34, 44}, {25, 16, 42, 31, 41, 35, 26, 10, 3, 1, 4, 29}, {8, 11, 5, 40, 9, 18, 10, 16, 26, 30, 19, 2, 14, 4}, {}, {0, 20, 17, 47, 41, 36, 23, 42, 15, 13, 27}, {7, 15, 44, 38, 41, 42, 26, 19, 5, 47}, {}, {37, 22}, {21, 24, 15, 48, 33, 6, 39, 11}, {23, 7, 3, 29, 10, 40, 1, 16, 6, 8, 27}, {27, 29, 25, 26, 46, 15, 16}, {33, 40, 10, 38, 13, 19, 17, 23, 32, 39, 7}, {35, 3, 39, 18}, {47, 11, 27, 23, 35, 26, 43, 4, 22, 38, 44, 31, 1, 0}, {}, {18, 43, 46, 9, 15, 3, 42, 31, 13, 4, 12, 39, 22}, {42, 45, 47, 18, 26, 41, 38, 9, 0, 35, 8, 16, 29, 36, 31}, {3, 20, 29, 12, 46, 41, 23, 4, 9, 27}, {19, 33}, {32, 18}, {17, 28, 7, 35, 6, 22, 4, 43}, {41, 31, 20, 28, 35, 32, 24, 23, 0, 33, 18, 39, 29, 30, 16}, {43, 47, 46}};
        int[][] containedBoxes = {{14}, {}, {26}, {4, 47}, {}, {6}, {39, 43, 46}, {30}, {}, {}, {0, 3}, {}, {}, {}, {}, {27}, {}, {}, {}, {}, {12}, {}, {}, {41}, {}, {31}, {20, 29}, {13, 35}, {18}, {10, 40}, {}, {38}, {}, {}, {19}, {5}, {}, {}, {11}, {1}, {15}, {}, {}, {}, {24}, {}, {}, {}, {}};
        int[] initialBoxes = {2, 7, 8, 9, 16, 17, 21, 22, 23, 25, 28, 32, 33, 34, 36, 37, 42, 44, 45, 48};

        System.err.println(s.maxCandies(status, candies, keys, containedBoxes, initialBoxes));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1298 Hard BFS
    // 给你n个盒子，每个盒子的格式为[status, candies, keys, containedBoxes]，其中：
    //
    // 状态字status[i]：整数，如果box[i]是开的，那么是 1，否则是 0。
    // 糖果数candies[i]: 整数，表示box[i] 中糖果的数目。
    // 钥匙keys[i]：数组，表示你打开box[i]后，可以得到一些盒子的钥匙，每个元素分别为该钥匙对应盒子的下标。
    // 内含的盒子containedBoxes[i]：整数，表示放在box[i]里的盒子所对应的下标。
    // 给你一个initialBoxes 数组，表示你现在得到的盒子，你可以获得里面的糖果，也可以用盒子里的钥匙打开新的盒子，还可以继续探索从这个盒子里找到的其他盒子。
    //
    // 请你按照上述规则，返回可以获得糖果的 最大数目。
    //
    // 每个盒子最多被一个盒子包含。
    public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        Set<Integer> acquiredKey = new HashSet<>();
        Deque<Integer> boxVisitQueue = new LinkedList<>();
//        Set<Integer> visited = new HashSet<>();
        int[] noKeyCounter = new int[status.length];
        int ans = 0;
        for (int i : initialBoxes) {
            boxVisitQueue.offer(i);
        }
        while (!boxVisitQueue.isEmpty()) {
            int tmpBoxIdx = boxVisitQueue.poll();
//            if (visited.contains(tmpBoxIdx)) {
//                continue;
//            }
            if (status[tmpBoxIdx] == 1 || acquiredKey.contains(tmpBoxIdx)) {
                ans += candies[tmpBoxIdx];
                for (int j : containedBoxes[tmpBoxIdx]) {
                    boxVisitQueue.offer(j);
                }
                for (int j : keys[tmpBoxIdx]) {
                    acquiredKey.add(j);
                }
//                visited.add(tmpBoxIdx);
            } else {
                noKeyCounter[tmpBoxIdx]++;
                if (noKeyCounter[tmpBoxIdx] > status.length) {
                    break;
                }
                boxVisitQueue.offer(tmpBoxIdx);
            }
        }
        return ans;
    }

    // LC477 Solution
    public int totalHammingDistance(int[] nums) {
        int ans = 0, n = nums.length;
        for (int i = 0; i < 30; i++) {
            int c = 0;
            for (int val : nums) {
                c += (val >> i) & 1;
            }
            ans += c * (n - c);
        }
        return ans;
    }

    // LC874
    public int robotSim(int[] commands, int[][] obstacles) {
        // -2 左转90度
        // -1 右转90度
        int x = 0, y = 0;
        int direct = 0; // 0 - 北, 1 - 东 , 2 - 南, 3 - 西
        int[][] step = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int ans = 0;
        Set<Pair<Integer, Integer>> obSet = new HashSet<>();
        for (int[] i : obstacles) {
            obSet.add(new Pair<>(i[0], i[1]));
        }
        for (int c : commands) {
            if (c < 0) {
                if (c == -2) {
                    direct = (direct + 4 - 1) % 4;
                } else if (c == -1) {
                    direct = (direct + 1) % 4;
                }
            } else {
                for (int j = 0; j < c; j++) {
                    if (!obSet.contains(new Pair<>(x + step[direct][0], y + step[direct][1]))) {
                        x += step[direct][0];
                        y += step[direct][1];
                        ans = Math.max(ans, x * x + y * y);
                    } else {
                        break;
                    }
                }
            }
        }
        return ans;
    }
}