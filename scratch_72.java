import javafx.util.Pair;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        // System.err.println(s.reachableNodes(new int[][]{{0, 1, 10}, {0, 2, 1}, {1, 2, 2}}, 6, 3));
        System.err.println(s.reachableNodes(new int[][]{{4, 21, 114}, {18, 25, 139}, {3, 22, 244}, {22, 26, 193}, {18, 22, 98}, {1, 24, 6}, {17, 18, 42}, {8, 25, 151}, {5, 28, 265}, {2, 22, 138}, {9, 20, 126}, {0, 8, 152}, {22, 28, 39}, {8, 27, 241}, {11, 29, 147}, {6, 23, 22}, {24, 26, 274}, {21, 27, 20}, {15, 18, 8}, {1, 19, 0}, {0, 25, 164}, {1, 22, 97}, {15, 21, 19}, {13, 16, 13}, {18, 28, 141}, {14, 20, 21}, {14, 26, 60}, {10, 13, 223}, {11, 20, 93}, {5, 8, 8}, {11, 14, 288}, {7, 28, 280}, {5, 23, 191}, {17, 19, 228}, {12, 17, 278}, {7, 16, 103}, {9, 17, 188}, {24, 29, 293}, {20, 29, 18}, {13, 25, 259}, {19, 22, 136}, {21, 26, 276}, {6, 21, 113}, {23, 25, 12}, {18, 27, 155}, {24, 25, 279}, {7, 24, 165}, {22, 23, 72}, {2, 8, 204}, {5, 6, 166}, {16, 19, 166}, {3, 9, 71}, {19, 28, 66}, {9, 12, 3}, {5, 16, 291}, {20, 26, 226}, {16, 21, 271}, {4, 15, 136}, {16, 27, 71}, {9, 21, 142}, {11, 23, 293}, {8, 22, 262}, {25, 27, 219}, {13, 27, 204}, {16, 23, 129}, {2, 6, 172}, {24, 27, 228}, {5, 25, 72}, {17, 24, 20}, {2, 25, 221}, {19, 23, 145}, {16, 20, 199}, {14, 21, 86}, {23, 24, 213}, {17, 20, 260}, {18, 29, 181}, {6, 14, 1}, {6, 9, 245}, {8, 19, 67}, {16, 26, 140}, {9, 25, 26}, {26, 28, 119}, {10, 12, 268}, {9, 23, 149}, {21, 25, 214}, {2, 28, 135}, {10, 17, 149}, {14, 24, 82}, {15, 26, 203}, {6, 28, 60}, {2, 24, 272}, {6, 19, 253}, {0, 27, 76}, {3, 28, 154}, {5, 18, 287}, {3, 12, 256}, {18, 21, 31}, {23, 26, 122}, {8, 17, 113}, {14, 28, 264}, {7, 23, 289}, {12, 28, 232}, {0, 17, 193}, {9, 14, 79}, {8, 20, 79}, {6, 8, 134}, {6, 15, 123}, {3, 11, 212}, {0, 1, 125}, {4, 9, 266}, {2, 7, 30}, {18, 20, 44}, {14, 25, 229}, {0, 2, 265}, {14, 23, 159}, {5, 10, 167}, {3, 25, 174}, {10, 23, 243}, {4, 11, 253}, {12, 26, 95}, {11, 16, 120}, {7, 9, 218}, {15, 24, 208}, {14, 22, 158}, {7, 10, 90}, {3, 17, 209}, {2, 12, 232}, {10, 20, 204}, {1, 29, 275}, {26, 27, 9}, {6, 7, 74}, {7, 22, 60}, {15, 17, 62}, {19, 24, 12}, {23, 28, 194}, {19, 21, 176}, {6, 18, 55}, {5, 9, 165}, {20, 28, 84}, {8, 23, 240}, {16, 17, 208}, {4, 28, 148}, {13, 23, 207}, {13, 15, 265}, {14, 17, 181}, {12, 15, 108}, {6, 17, 53}, {6, 12, 144}, {1, 3, 2}, {2, 10, 80}, {5, 20, 158}, {19, 20, 164}, {13, 20, 53}, {10, 16, 118}, {25, 26, 142}, {1, 7, 255}, {9, 29, 61}, {6, 13, 110}, {18, 26, 276}, {27, 28, 109}, {15, 25, 164}, {17, 27, 156}, {21, 29, 275}, {10, 18, 284}, {12, 21, 85}, {16, 28, 181}, {0, 11, 222}, {0, 9, 14}, {5, 29, 226}, {1, 18, 117}, {12, 27, 195}, {14, 18, 118}, {12, 14, 57}, {17, 28, 197}, {9, 22, 17}, {4, 8, 171}, {0, 10, 158}, {10, 29, 6}, {25, 29, 202}, {14, 16, 149}, {9, 16, 74}, {15, 23, 161}, {19, 27, 196}, {6, 22, 186}, {20, 25, 213}, {3, 10, 66}, {3, 27, 275}, {15, 29, 149}, {16, 25, 83}, {4, 26, 179}, {14, 19, 26}, {20, 23, 5}, {10, 26, 76}, {20, 24, 255}, {7, 29, 31}, {2, 23, 160}, {17, 21, 224}, {13, 22, 173}, {13, 19, 69}, {3, 18, 147}, {7, 21, 124}, {12, 29, 35}, {26, 29, 106}, {10, 21, 298}, {9, 24, 14}, {9, 10, 46}, {18, 23, 256}, {3, 16, 257}, {23, 29, 32}, {17, 26, 254}, {13, 28, 260}, {14, 27, 145}, {0, 7, 52}, {9, 11, 149}, {21, 24, 7}, {15, 28, 231}, {3, 23, 58}, {10, 28, 99}, {3, 4, 177}, {5, 26, 196}, {0, 6, 71}, {13, 14, 115}, {12, 25, 177}, {3, 14, 80}, {16, 24, 258}, {7, 19, 157}, {13, 26, 195}, {8, 12, 257}, {6, 16, 24}, {5, 17, 249}, {0, 18, 79}, {20, 21, 62}, {5, 7, 205}, {0, 5, 129}, {11, 12, 225}, {15, 22, 27}, {5, 22, 188}, {2, 21, 144}, {25, 28, 223}, {9, 15, 7}, {18, 24, 21}, {1, 20, 196}, {10, 22, 299}, {4, 14, 33}, {8, 26, 75}, {19, 25, 15}, {4, 20, 245}, {0, 13, 32}, {22, 29, 215}, {23, 27, 113}, {1, 23, 160}, {1, 6, 112}, {2, 4, 62}, {8, 18, 255}, {24, 28, 20}, {11, 22, 113}, {2, 11, 236}, {21, 28, 151}, {2, 20, 156}, {3, 29, 33}, {22, 24, 63}, {4, 6, 220}, {0, 12, 94}, {22, 27, 222}, {11, 13, 180}, {7, 15, 209}, {21, 22, 90}, {11, 27, 125}, {4, 10, 256}, {5, 14, 57}, {28, 29, 22}, {12, 24, 241}, {7, 26, 259}, {12, 23, 53}, {2, 17, 245}, {2, 27, 69}, {6, 24, 238}, {8, 10, 207}, {9, 26, 139}, {15, 19, 118}, {9, 28, 261}, {7, 8, 64}, {4, 24, 176}, {14, 29, 56}, {4, 25, 280}, {14, 15, 30}, {5, 13, 282}, {17, 25, 269}, {8, 11, 291}, {4, 29, 217}, {5, 15, 284}, {11, 21, 4}, {11, 15, 109}, {9, 13, 158}, {8, 9, 170}, {11, 18, 8}, {5, 24, 261}, {12, 20, 41}, {16, 29, 41}, {27, 29, 22}},
                172,
                30));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + " ms.");
    }

    // LC882 Hard TLE **
    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        Map<Integer, Map<Integer, Integer>> farMostOnEdge = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> edgeMap = new HashMap<>();
        Set<Integer> visited = new HashSet<>();
        visited.add(0);
        for (int[] e : edges) {
            edgeMap.putIfAbsent(e[0], new HashMap<>());
            edgeMap.putIfAbsent(e[1], new HashMap<>());
            edgeMap.get(e[0]).put(e[1], e[2]);
            edgeMap.get(e[1]).put(e[0], e[2]);

            farMostOnEdge.putIfAbsent(e[0], new HashMap<>());
            farMostOnEdge.putIfAbsent(e[1], new HashMap<>());
            farMostOnEdge.get(e[0]).put(e[1], 0);
            farMostOnEdge.get(e[1]).put(e[0], 0);
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(i -> -i[2])); // [from, to, /*nth always 0*/, remain]
        Map<Integer, Integer> zeroNext = edgeMap.getOrDefault(0, new HashMap<>());
        for (Map.Entry<Integer, Integer> e : zeroNext.entrySet()) {
            pq.offer(new int[]{0, e.getKey(), maxMoves});
        }
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int from = p[0], to = p[1], remain = p[2];
            int pointsOnEdge = edgeMap.get(from).get(to);
            visited.add(from);
            int farMostEverOnTheEdge = farMostOnEdge.get(from).get(to);
            if (farMostEverOnTheEdge >= remain) continue;
            if (remain <= pointsOnEdge) {
                farMostOnEdge.get(from).put(to, remain);
                continue;
            } else if (/* pointsOnEdge < */remain <= pointsOnEdge + 1) {
                farMostOnEdge.get(from).put(to, pointsOnEdge);
                visited.add(to);
                continue;
            } else {
                farMostOnEdge.get(from).put(to, pointsOnEdge);
                int newFrom = to, newRemain = remain - (pointsOnEdge + 1);
                for (Map.Entry<Integer, Integer> e : edgeMap.get(newFrom).entrySet()) {
                    int newNext = e.getKey();
                    if (visited.contains(newNext) && farMostOnEdge.get(newFrom).get(newNext) >= newRemain) {
                        continue;
                    }
                    pq.offer(new int[]{newFrom, newNext, newRemain});
                }
            }
        }
        int result = visited.size();
        for (int[] e : edges) {
            int a = e[0], b = e[1], points = e[2];
            int fromAToB = farMostOnEdge.get(a).get(b), fromBToA = farMostOnEdge.get(b).get(a);
            result += Math.min(fromAToB + fromBToA, points);
        }

        return result;
    }

    private int getEdgeId(int a, int b, int occupiedBits) { // n vertexes
        return a | (b << occupiedBits);
    }

    private int[] getEdgeVertexes(int id, int occupiedBits) {
        return new int[]{id & ((1 << occupiedBits) - 1), id >> occupiedBits};
    }

    // LC799 DP **
    public double champagneTower(int poured, int queryRow, int queryGlass) {
        double[] row = {poured};
        for (int i = 1; i <= queryRow; i++) {
            double[] nr = new double[i + 1];
            for (int j = 0; j < row.length; j++) {
                double volume = row[j];
                if (volume > 1D) {
                    nr[j] += (volume - 1) / 2;
                    nr[j + 1] += (volume - 1) / 2;
                }
            }
            row = nr;
        }
        return Math.min(1d, row[queryGlass]);
    }

    // LC792
    public int numMatchingSubseq(String s, String[] words) {
        char[] ca = s.toCharArray();
        char[][] wca = new char[words.length][];
        int result = 0;
        for (int i = 0; i < words.length; i++) wca[i] = words[i].toCharArray();
        Deque<int[]>[] qq = new Deque[128];
        for (int i = 'a'; i <= 'z'; i++) qq[i] = new LinkedList<>();
        for (int i = 0; i < wca.length; i++) {
            qq[wca[i][0]].offer(new int[]{i, 0});
        }
        for (char c : ca) {
            int qs = qq[c].size();
            for (int i = 0; i < qs; i++) {
                int[] p = qq[c].poll();
                int whichWord = p[0];
                int idxOnWord = p[1];
                if (idxOnWord == wca[whichWord].length - 1) {
                    result++;
                } else {
                    p[1]++;
                    qq[wca[whichWord][p[1]]].offer(p);
                }
            }
        }
        return result;
    }

    // LC790 DP
    Integer[] memo;

    public int numTilings(int n) {
        memo = new Integer[n + 3];
        memo[0] = 0;
        memo[1] = 1;
        memo[2] = 2;
        memo[3] = 5;
        return helper(n);
    }

    public int helper(int n) {
        if (memo[n] != null) {
            return memo[n];
        }
        long result = 0l;
        for (int i = 1; i <= 3; i++) {
            int left = i, right = n - i;
            long tmp = helper(left) * helper(right);
            tmp %= 1000000007L;
            result += tmp;
            result %= 1000000007L;
        }
        return memo[n] = (int) result;
    }

    // LC864 Hard
    int[][] lc864Dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int shortestPathAllKeys(String[] grid) {
        int m = grid.length, n = grid[0].length();
        char[][] mat = new char[m][];
        for (int i = 0; i < m; i++) mat[i] = grid[i].toCharArray();
        int[][][] coordinate = new int[128][][];
        int row = -1, col = -1;
        Set<Character> remain = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                char c = mat[i][j];
                if (c == '@') {
                    row = i;
                    col = j;
                } else if (Character.isLowerCase(c)) { // Key
                    if (coordinate[c] == null) {
                        coordinate[c] = new int[2][];
                    }
                    coordinate[c][0] = new int[]{i, j};
                    remain.add(c);
                } else if (Character.isUpperCase(c)) { // Lock
                    char upper = Character.toLowerCase(c);
                    if (coordinate[upper] == null) {
                        coordinate[upper] = new int[2][];
                    }
                    coordinate[upper][1] = new int[]{i, j};
                }
            }
        }
        int r = lc864Helper(mat, remain, coordinate, row, col, 0);
        if (r >= Integer.MAX_VALUE / 2) return -1;
        return r;
    }

    private int lc864Helper(char[][] grid, Set<Character> remain, int[][][] coordinate, int r, int c, int prevSteps) {
        if (remain.isEmpty()) {
            return 0;
        }
        int m = grid.length, n = grid[0].length;
        // BFS 求出所有可达的钥匙的位置
        Deque<Integer> q = new LinkedList<>();
        q.offer(r * n + c);
        List<Pair<Character, Integer>> reachable = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        int layer = -1;
        outer:
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int p = q.poll();
                if (visited.contains(p)) continue;
                visited.add(p);
                int row = p / n, col = p % n;
                if (Character.isLowerCase(grid[row][col])) {
                    if (remain.contains(grid[row][col])) {
                        reachable.add(new Pair<>(grid[row][col], layer));
                        if (reachable.size() == remain.size()) break outer;
                    }
                }
                for (int[] d : lc864Dirs) {
                    int nr = row + d[0], nc = col + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] != '#' && !Character.isUpperCase(grid[nr][nc])) {
                        int next = nr * n + nc;
                        if (visited.contains(next)) continue;
                        q.offer(next);
                    }
                }
            }
        }
        if (reachable.isEmpty()) {
            return Integer.MAX_VALUE / 2;
        }
        int result = Integer.MAX_VALUE / 2;
        for (Pair<Character, Integer> p : reachable) {
            Character ch = p.getKey();
            int currentSteps = p.getValue();
            int currentRow = coordinate[ch][0][0], currentCol = coordinate[ch][0][1];
            int unlockedRow = coordinate[ch][1][0], unlockedCol = coordinate[ch][1][1];

            grid[unlockedRow][unlockedCol] = '.';
            remain.remove(ch);

            result = Math.min(result, currentSteps + lc864Helper(grid, remain, coordinate, currentRow, currentCol, currentSteps + prevSteps));

            grid[unlockedRow][unlockedCol] = Character.toUpperCase(ch);
            remain.add(ch);

        }

        return result;
    }

    // LC1235 TBD
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        List<int[]> l = new ArrayList<>(n);
        TreeSet<Integer> startTS = new TreeSet<>(), endTS = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            startTS.add(startTime[i]);
            endTS.add(endTime[i]);
        }
        // 离散化
        Map<Integer, Integer> startTimeMap = new HashMap<>(), endTimeMap = new HashMap<>();
        List<Integer> startList = startTS.stream().toList(), endList = endTS.stream().toList();
        for (int i = 0; i < startList.size(); i++) {
            startTimeMap.put(startList.get(i), i);
        }
        for (int i = 0; i < endList.size(); i++) {
            endTimeMap.put(endList.get(i), i);
        }
        for (int i = 0; i < n; i++) {
            l.add(new int[]{startTimeMap.get(startTime[i]), endTimeMap.get(endTime[i]), profit[i]});
        }
        l.sort(Comparator.comparingInt(i -> i[0]));
        Integer[] memo = new Integer[startTS.size() + endTS.size()];
        BiFunction<Integer, Integer, Integer> helper = (startTimeIdx, nextStartTimeIdx) -> {
            // 返回在当前下标开始时间(含)的情况下, 最多能获得多少利润
            return -1;
        };
        return helper.apply(0, null);
    }

    // LC1768
    public String mergeAlternately(String word1, String word2) {
        int n = Math.min(word1.length(), word2.length());
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append(word1.charAt(i));
            sb.append(word2.charAt(i));
        }
        if (word1.length() > word2.length()) {
            sb.append(word1.substring(n));
        } else if (word2.length() > word1.length()) {
            sb.append(word2.substring(n));
        }
        return sb.toString();
    }

    // LC2156 Hard ** 滚动哈希 需要及其熟悉模算术 TAG: Rabin-Karp
    public String subStrHash(String s, int power, int modulo, int k, int hashValue) {
        int n = s.length();
        int result = -1;
        long hashing, powering;

        String reversed = new StringBuilder(s).reverse().toString();
        char[] ca = reversed.toCharArray();

        hashing = (ca[0] - 'a' + 1) % modulo;
        powering = 1L;
        for (int i = 1; i < k; i++) {
            powering *= power;
            powering %= modulo;
            hashing *= power;
            hashing += ca[i] - 'a' + 1;
            hashing %= modulo;
        }
        if (hashing == hashValue) {
            // result = k - 1 - k + 1;
            result = 0;
        }
        for (int i = k; i < n; i++) {
            hashing -= ((long) (ca[i - k] - 'a' + 1) * powering) % modulo;
            hashing += modulo;
            hashing *= power;
            hashing += ca[i] - 'a' + 1;
            hashing %= modulo;
            if (hashing == hashValue) {
                result = i - k + 1;
            }
        }
        return new StringBuilder(reversed.substring(result, result + k)).reverse().toString();
    }

    // LC2294 ** 纸笔试下 排序后不会出现分组不当导致的错划分组使得答案次佳的情况
    public int partitionArray(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) return 1;
        Arrays.sort(nums);
        int min = nums[0], result = 1;
        for (int i = 0; i < n; i++) {
            if (nums[i] > min + k) {
                result++;
                min = nums[i];
            }
        }
        return result;
    }

    // LC1346
    public boolean checkIfExist(int[] arr) {
        Map<Integer, Set<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            m.putIfAbsent(arr[i], new HashSet<>());
            m.get(arr[i]).add(i);
        }
        for (int i : arr) {
            if (i == 0 && m.get(i).size() > 1) {
                return true;
            }
            if (i != 0 && i % 2 == 0 && m.containsKey(i / 2)) {
                return true;
            }
        }
        return false;
    }

    // LCP50
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] o : operations) {
            int x = o[0], y = o[1];
            int origX = gem[x], origY = gem[y];
            gem[y] = origY + origX / 2;
            gem[x] = origX - origX / 2;
        }
        Arrays.sort(gem);

        return gem[gem.length - 1] - gem[0];
    }

    // LC779
    public int kthGrammar(int n, int k) {
        int actualN = n - 1, actualK = k - 1;
        List<Integer> remain = new ArrayList<>();
        while (actualN >= 0) {
            remain.add(actualK % 2);
            actualK /= 2;
            actualN--;
        }
        int s = remain.size(), cur = 0;
        int[] zeroOne = {0, 1}, oneZero = {1, 0};
        for (int i = s - 1; i >= 0; i--) {
            int r = remain.get(i);
            int next = -1;
            if (cur == 0) {
                next = zeroOne[r];
            } else {
                next = oneZero[r];
            }
            cur = next;
        }
        return cur;
    }

    // LCP49 Hard ** 楼教主解法
    public long ringGame(long[] challenge) {
        int n = challenge.length;
        List<Pair<Integer, Long>> idxScoreList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            idxScoreList.add(new Pair<>(i, challenge[i]));
        }
        idxScoreList.sort(Comparator.comparingLong(i -> -i.getValue()));
        Function<Integer, Integer> getNext = i -> (i + 1) % n;
        Function<Integer, Integer> getPrev = i -> (i - 1 + n) % n;
        Function<Long, Boolean> check = initVal -> {
            long ongoingVal = initVal;
            BitSet visited = new BitSet(n);
            for (Pair<Integer, Long> p : idxScoreList) { // 从最大的开始遍历所有起点
                int idx = p.getKey();
                long necessaryScore = p.getValue();
                if (ongoingVal < necessaryScore) {
                    continue;
                }
                if (visited.get(idx)) {
                    continue;
                }
                long mergedScore = ongoingVal | necessaryScore;
                visited.set(idx);
                int leftPtr = idx, rightPtr = idx;
                while (true) {
                    if (getNext.apply(rightPtr) == leftPtr) return true;
                    if (mergedScore >= challenge[getPrev.apply(leftPtr)]) {
                        leftPtr = getPrev.apply(leftPtr);
                        mergedScore |= challenge[leftPtr];
                        visited.set(leftPtr);
                    } else if (mergedScore >= challenge[getNext.apply(rightPtr)]) {
                        rightPtr = getNext.apply(rightPtr);
                        mergedScore |= challenge[rightPtr];
                        visited.set(rightPtr);
                    } else {
                        break;
                    }
                }
            }
            return false;
        };

        long result = 0;
        for (int i = 63; i >= 0; i--) {
            long initVal = (result | (1L << i)) - 1;
            if (!check.apply(initVal)) {
                result |= (1L << i);
            }
        }
        return result;
    }


    // LC2311 **
    public int longestSubsequence(String s, int k) {
        int sLen = s.length(), kLen = Integer.SIZE - Integer.numberOfLeadingZeros(k);
        if (sLen < kLen) return sLen;
        int alignWithK = Integer.parseInt(s.substring(sLen - kLen), 2);
        int result = alignWithK > k ? kLen - 1 : kLen;
        int leadingZeros = (int) s.substring(0, sLen - kLen).chars().filter(i -> i == '0').count();
        return leadingZeros + result;
    }

    // LC2341
    public int[] numberOfPairs(int[] nums) {
        Map<Integer, List<Integer>> m = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity()));
        int a0 = m.values().stream().map(i -> i.size() / 2).reduce((a, b) -> a + b).get();
        int a1 = m.values().stream().map(i -> i.size() % 2).reduce((a, b) -> a + b).get();
        return new int[]{a0, a1};
    }

    // LC1262
    Integer[][] lc1262Memo;
    int[] lc1262Nums;

    public int maxSumDivThree(int[] nums) {
        lc1262Nums = nums;
        int n = nums.length;
        lc1262Memo = new Integer[n + 1][3];
        return lc864Helper(n - 1, 0);
    }

    private int lc864Helper(int idx, int targetRemain) {
        if (idx == 0) {
            if (lc1262Nums[idx] % 3 != targetRemain) return 0;
            return lc1262Nums[idx];
        }
        if (lc1262Memo[idx][targetRemain] != null) return lc1262Memo[idx][targetRemain];
        int result = 0;
        int currentRemain = lc1262Nums[idx] % 3, currentValue = lc1262Nums[idx];
        // Choose current
        int nextTargetRemain = (targetRemain - currentRemain + 3) % 3;
        int tmpChooseRightPart = lc864Helper(idx - 1, nextTargetRemain);
        int tmpResult = currentValue + tmpChooseRightPart;
        if (tmpResult % 3 == targetRemain) {
            result = Math.max(result, tmpResult);
        }
        // Don't choose current
        tmpResult = lc864Helper(idx - 1, targetRemain);
        if (tmpResult % 3 == targetRemain) {
            result = Math.max(result, tmpResult);
        }
        return lc1262Memo[idx][targetRemain] = result;
    }

    // LC2089
    public List<Integer> targetIndices(int[] nums, int target) {
        Arrays.sort(nums);
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) result.add(i);
        }
        return result;
    }

    // LC1383 ** Hard
    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        // Pair: <speed, efficiency>
        List<Pair<Integer, Integer>> employeeList = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            employeeList.add(new Pair<>(speed[i], efficiency[i]));
        }
        employeeList.sort(Comparator.comparingInt(i -> -i.getValue()));
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(i -> i.getKey()));
        long sumSpeed = 0L, result = 0L;
        for (int i = 0; i < n; i++) {
            Pair<Integer, Integer> minEfficiencyStaff = employeeList.get(i);
            int staffSpeed = minEfficiencyStaff.getKey(), staffEfficiency = minEfficiencyStaff.getValue();
            sumSpeed += staffSpeed;
            result = Math.max(result, sumSpeed * (long) staffEfficiency);
            pq.offer(minEfficiencyStaff);
            if (pq.size() == k) {
                Pair<Integer, Integer> p = pq.poll();
                sumSpeed -= p.getKey();
            }
        }
        return (int) (result % 1000000007L);
    }

    // LC1700
    public int countStudents(int[] students, int[] sandwiches) {
        int n = students.length;
        int count = 0;
        int[] remain = new int[2];
        for (int i : sandwiches) remain[i]++;
        while (count < n && remain[sandwiches[count]] > 0) {
            remain[sandwiches[count]]--;
            count++;
        }
        return n - count;
    }

    final static class LCS03 {// LCS 03
        Set<Integer> visited = new HashSet<>();
        char[][] matrix;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int finalResult = 0, row = 0, col = 0;
        boolean isTouchBoundary;

        public int largestArea(String[] grid) {
            matrix = new char[grid.length][];
            for (int i = 0; i < grid.length; i++) {
                matrix[i] = grid[i].toCharArray();
            }
            row = matrix.length;
            col = matrix[0].length;
            for (int i = 0; i < (row * col); i++) {
                isTouchBoundary = false;
                int result = lcs03Helper(i);
                if (!isTouchBoundary) finalResult = Math.max(result, finalResult);
            }
            return finalResult;
        }

        private int lcs03Helper(int i) {
            if (visited.contains(i)) return 0;
            visited.add(i);
            int r = i / col, c = i % col;
            if (r == 0 || r == row - 1 || c == 0 || c == col - 1 || matrix[r][c] == '0') isTouchBoundary = true;
            int result = 1;
            for (int[] d : directions) {
                int nr = r + d[0], nc = c + d[1];
                if (!(nr >= 0 && nr < row && nc >= 0 && nc < col)) continue;
                if (matrix[nr][nc] == '0') {
                    isTouchBoundary = true;
                    continue;
                }
                if (matrix[nr][nc] != matrix[r][c]) continue;
                int next = lcs03Helper(nr * col + nc);
                result += next;
            }
            return result;
        }
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
        this.size = 1 << 16;
        father = new int[1 << 16];
        rank = new int[1 << 16];
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

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (int i = 0; i < size; i++) {
            if (father[i] != -1) {
                int f = find(i);
                result.putIfAbsent(f, new HashSet<>());
                result.get(f).add(i);
            }
        }
        return result;
    }

    public int getNumOfGroups() {
        return getAllGroups().size();
    }

    public int getSelfGroupSize(int x) {
        return rank[find(x)];
    }

}
