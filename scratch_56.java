import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        String a = "0.000";
        a = a.replaceAll("\\.", "");

        System.out.println(Long.parseLong(a));

        System.out.println(s.minimizeError(new String[]{"176.482", "198.901", "36.649", "484.363", "39.106", "706.278", "237.805", "606.143", "909.029", "397.232", "291.582", "155.740", "504.388", "908.987", "783.835", "242.981", "750.261", "80.856", "392.332", "387.279", "119.694", "724.676", "106.957", "235.810", "202.762", "680.192", "228.342", "639.083", "581.189", "582.050", "214.249", "594.841", "280.911", "900.931", "29.227", "27.418", "294.318", "108.824", "248.889", "62.516", "691.337", "965.642", "937.107", "699.067", "999.482", "761.636", "894.648", "573.063", "253.617", "645.941", "101.146", "13.341", "350.921", "249.666", "185.062", "628.160", "466.778", "136.878", "799.233", "445.992", "781.561", "765.179", "715.129", "898.270", "829.027", "291.306", "960.678", "79.816", "271.794", "11.481", "91.196", "697.517", "549.232", "714.818", "808.141", "653.918", "40.628", "497.676", "722.596", "176.944", "598.296", "936.731", "562.226", "161.292", "103.156", "552.001", "827.166", "125.729", "172.795", "94.795", "458.770", "584.536", "192.888", "99.168", "80.858", "500.447", "873.452", "256.456", "685.043", "700.272", "696.424", "861.629", "70.851", "554.679", "785.407", "614.120", "336.342", "630.633", "849.062", "705.911", "899.603", "230.828", "912.370", "173.113", "550.592", "621.038", "189.125", "783.173", "180.648", "323.832", "379.300", "900.027", "345.017", "948.221", "30.584", "571.719", "336.961", "438.852", "539.295", "511.785", "733.012", "172.039", "656.239", "78.881", "272.224", "350.091", "134.352", "640.695", "561.174", "760.949", "731.220", "188.564", "800.484", "859.406", "765.421", "762.566", "913.382", "836.858", "425.125", "974.211", "501.163", "826.763", "984.652", "634.616", "251.411", "284.561", "577.928", "102.397", "230.549", "350.442", "960.192", "424.219", "856.516", "489.991", "674.620", "310.060", "970.275", "771.297", "527.096", "452.818", "61.604", "324.661", "504.833", "765.907", "594.536", "23.124", "836.986", "932.868", "962.289", "576.681", "659.346", "688.344", "809.890", "209.888", "899.789", "303.385", "782.461", "638.753", "937.564", "750.771", "206.505", "871.730", "242.697", "128.613", "382.760", "24.371", "995.515", "965.332", "23.977", "324.923", "699.972", "540.356", "733.206", "935.280", "726.212", "887.225", "733.830", "922.663", "507.836", "729.250", "561.031", "31.231", "55.667", "740.038", "917.949", "158.097", "466.824", "162.507", "11.590", "270.021", "813.393", "467.459", "777.197", "926.727", "416.902", "923.274", "296.852", "99.172", "715.083", "41.008", "743.795", "618.281", "18.793", "284.030", "627.872", "631.400", "428.253", "745.996", "874.448", "495.478", "585.807", "339.294", "351.812", "799.344", "883.373", "937.590", "362.691", "557.557", "648.369", "409.413", "982.339", "962.199", "202.624", "688.881", "294.993", "296.087", "423.256", "773.650", "241.432", "154.863", "269.898", "863.123", "202.014", "338.028", "30.240", "680.904", "747.531", "407.863", "811.076", "93.848", "641.073", "764.963", "838.325", "361.577", "234.191", "453.184", "227.976", "379.927", "414.814", "727.165", "315.090", "8.278", "498.603", "34.115", "161.057", "812.691", "709.270", "698.997", "303.160", "988.946", "876.278", "14.155", "916.828", "185.835", "77.105", "880.009", "157.053", "17.036", "457.516", "961.624", "703.730", "527.528", "125.302", "323.105", "517.306", "556.921", "977.464", "130.114", "980.489", "31.171", "141.576", "715.355", "749.697", "30.323", "498.228", "485.567", "885.160", "294.925", "651.530", "198.375", "722.456", "670.144", "381.699", "656.074", "208.039", "251.729", "758.336", "89.308", "564.766", "830.903", "117.039", "796.570", "754.498", "168.058", "260.893", "60.091", "355.720", "992.158", "460.914", "598.943", "309.637", "847.000", "655.260", "955.832", "622.572", "682.282", "301.806", "244.942", "750.907", "286.629", "180.590", "928.372", "734.978", "660.173", "465.119", "719.643", "561.499", "834.548", "731.571", "428.803", "244.184", "383.744", "840.801", "353.256", "20.204", "782.414", "523.003", "154.162", "589.404", "989.569", "637.187", "281.555", "303.581", "448.380", "345.457", "286.025", "735.056", "366.298", "528.389", "779.217", "198.358", "604.072", "769.896", "444.242", "733.766", "458.694", "436.066", "779.673", "345.949", "501.780", "922.513", "253.129", "786.147", "824.602", "222.812", "396.176", "66.596", "965.756", "213.505", "188.498", "230.595", "512.001", "86.509", "888.709", "525.894", "670.177", "292.822", "883.236", "716.733", "214.218", "618.535", "324.968", "79.676", "756.772", "847.676", "581.219", "752.641", "746.612", "455.466", "537.155", "987.375", "10.434", "728.776", "33.588", "632.033", "238.435", "193.073", "49.466", "93.625", "462.667", "214.527", "637.316", "852.521", "358.107", "740.765", "868.695", "496.777", "289.177", "851.652", "78.733", "232.050", "420.353", "286.678", "403.990", "231.101", "191.268", "475.749", "894.321", "801.280", "463.420", "221.988", "480.282", "558.432", "326.906", "126.569", "503.138", "385.178", "717.114", "16.319", "720.830", "129.174", "145.721", "145.473", "746.880", "474.946", "562.019", "973.694", "581.162", "602.072", "593.771", "712.666", "381.776", "879.224", "609.789", "918.613", "850.640", "64.367", "266.904", "939.194", "205.984", "782.930", "615.195", "831.299", "614.922", "270.153", "562.829", "755.024", "567.315", "595.991", "163.790", "501.046", "157.278", "799.262", "953.534", "540.850", "186.209", "472.951", "826.515", "641.681", "283.117"},
                252167));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1093
    public double[] sampleStats(int[] count) {
        // min max avg middle popular
        double[] result = new double[5];
        int total = 0;
        for (int i : count) total += i;
        if (total % 2 == 0) {
            // 找 half, half+1
            int first = -1, second = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half) {
                    first = i;
                }
                if (first != -1 && accu >= half + 1) {
                    second = i;
                    break;
                }
            }
            result[3] = (first + second + 0d) / 2d;
        } else {
            // 找half+1
            int first = -1;
            int half = total / 2;
            int accu = 0;
            for (int i = 0; i < 256; i++) {
                accu += count[i];
                if (first == -1 && accu >= half + 1) {
                    first = i;
                    break;
                }
            }
            result[3] = first;
        }
        for (int i = 0; i < 256; i++) {
            if (count[i] != 0) {
                result[0] = i;
                break;
            }
        }
        for (int i = 255; i >= 0; i--) {
            if (count[i] != 0) {
                result[1] = i;
                break;
            }
        }
        long sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += (count[i] + 0l) * (i + 0l);
        }
        result[2] = (sum + 0d) / (total + 0d);
        int popCount = 0, popNum = -1;
        for (int i = 0; i < 256; i++) {
            if (count[i] > popCount) {
                popNum = i;
                popCount = count[i];
            }
        }
        result[4] = popNum;
        return result;
    }

    // LC1058
    public String minimizeError(String[] prices, int target) {
        double max = 0d, min = 0d, sum = 0d;
        List<Double> priceList = new ArrayList<>(prices.length);
        for (String p : prices) {
            double d = Double.parseDouble(p);
            if (p.endsWith("000")) { // 整数会对shouldFloorCount造成影响, 导致shouldFloorCount少1, 最终的和会变大
                target -= (int) d;
                continue;
            }
            priceList.add(d);
            max += Math.ceil(d);
            min += Math.floor(d);
            sum += d;
        }
        if (max < (target + 0d)) return "-1";
        if (min > (target + 0d)) return "-1";
        int shouldFloorCount = (int) (max - target);
        if (shouldFloorCount == 0) {
            return String.format("%.3f", max - sum);
        }
        if (shouldFloorCount == prices.length) {
            return String.format("%.3f", sum - min);
        }
        PriorityQueue<Double> floorPq = new PriorityQueue<>(Comparator.comparingDouble(d -> -(d - Math.floor(d))));
        List<Double> shouldCeilList = new ArrayList<>(priceList.size() - shouldFloorCount);
        for (double d : priceList) {
            if (floorPq.size() < shouldFloorCount) {
                floorPq.offer(d);
            } else {
                double deltaOut = d - Math.floor(d);
                double peek = floorPq.peek();
                double deltaIn = peek - Math.floor(peek);
                if (deltaOut < deltaIn) {
                    floorPq.poll();
                    floorPq.offer(d);
                    shouldCeilList.add(peek);
                } else {
                    shouldCeilList.add(d);
                }
            }
        }
        double totalDelta = 0d;
        while (!floorPq.isEmpty()) {
            double p = floorPq.poll();
            totalDelta += p - Math.floor(p);
        }
        for (double d : shouldCeilList) {
            totalDelta += Math.ceil(d) - d;
        }
        return String.format("%.3f", totalDelta);
    }


    // LC1508 前缀和 + 暴力
    public int rangeSum(int[] nums, int n, int left, int right) {
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + nums[i];
        List<Integer> accu = new ArrayList<>((n + 1) * n / 2);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                accu.add(prefix[j + 1] - prefix[i]);
            }
        }
        Collections.sort(accu);
        long sum = 0, mod = 1000000007;
        for (int i = left - 1; i < right; i++) {
            sum += accu.get(i);
            sum %= mod;
        }
        return (int) (sum % mod);
    }

    // LC1078
    public String[] findOcurrences(String text, String first, String second) {
        List<String> result = new ArrayList<>();
        String[] words = text.split(" ");
        String prev = '\0' + "";
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            if (prev.equals(first) && cur.equals(second)) {
                if (i + 1 < words.length) {
                    result.add(words[i + 1]);
                }
            }
            prev = cur;
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1815 Try Simulate Anneal 模拟退火
    class Lc1815 {
        List<Integer> toSatisfy = new ArrayList<>();
        int max = 0;
        int n;
        int batchSize;

        public int maxHappyGroups(int batchSize, int[] groups) {
            if (batchSize == 1) return groups.length;
            this.batchSize = batchSize;
            int modZeroGroupCount = 0;
            for (int i : groups) {
                if (i % batchSize == 0) modZeroGroupCount++;
                else toSatisfy.add(i);
            }
            if (toSatisfy.size() < 2) return modZeroGroupCount + toSatisfy.size();
            n = toSatisfy.size();
            // 多次使用模拟退火
            for (int i = 0; i < 32; i++) {
                simulateAnneal();
            }
            return modZeroGroupCount + max;
        }

        private void simulateAnneal() {
            Collections.shuffle(toSatisfy); // 随机化
            for (double t = 1e6; t >= 1e-5; t *= 0.97d) {
                int i = (int) (n * Math.random()), j = (int) (n * Math.random());
                int prevScore = evaluate();
                // 交换
                int tmp = toSatisfy.get(i);
                toSatisfy.set(i, toSatisfy.get(j));
                toSatisfy.set(j, tmp);

                int nextScore = evaluate();
                int delta = nextScore - prevScore;
                if (delta < 0 && Math.pow(Math.E, (delta / t)) <= (double) (Math.random())) {
                    tmp = toSatisfy.get(i);
                    toSatisfy.set(i, toSatisfy.get(j));
                    toSatisfy.set(j, tmp);
                }
            }
        }

        private int evaluate() {
            // 评价函数 直接返回在当前排列下有多少组可以好评
            int result = 0, sum = 0;
            for (int i : toSatisfy) {
                sum += i;
                if (sum % batchSize == 0) {
                    result++; // 能收到这一组的好评
                    sum = 0;
                }
            }
            if (sum > 0) result++; // 总是能收到第一组的好评
            max = Math.max(max, result);
            return result;
        }

    }


    // LC286
    public void wallsAndGates(int[][] rooms) {
        int m = rooms.length, n = rooms[0].length;
        final int INF = Integer.MAX_VALUE;
        int[][] direction = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int layer = -1;
        boolean[][] visited = new boolean[m][n];
        Deque<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == 0) {
                    q.offer(new int[]{i, j});
                }
            }
        }
        while (!q.isEmpty()) {
            int qs = q.size();
            layer++;
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                if (visited[p[0]][p[1]]) continue;
                visited[p[0]][p[1]] = true;
                if (rooms[p[0]][p[1]] == INF) rooms[p[0]][p[1]] = layer;
                for (int[] d : direction) {
                    int nr = p[0] + d[0], nc = p[1] + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc] && rooms[nr][nc] != -1) {
                        q.offer(new int[]{nr, nc});
                    }
                }
            }
        }
        return;
    }

    // LC1976 **
    final long lc1976Inf = Long.MAX_VALUE / 2;
    Integer[] lc1976Memo;

    public int countPaths(int n, int[][] roads) {
        // 最短路的数量
        lc1976Memo = new Integer[n];
        long[][] matrix = new long[n][n];
        long[][] minDist = new long[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(matrix[i], lc1976Inf);
            Arrays.fill(minDist[i], lc1976Inf);
        }
        for (int[] r : roads) {
            matrix[r[0]][r[1]] = r[2];
            matrix[r[1]][r[0]] = r[2];
            minDist[r[0]][r[1]] = r[2];
            minDist[r[1]][r[0]] = r[2];
        }
        for (int i = 0; i < n; i++) {
            minDist[i][i] = 0;
        }
        // Floyd 求出任意两点间的最短距离
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (j != k) {
                        minDist[j][k] = minDist[k][j] = Math.min(minDist[j][k], minDist[j][i] + minDist[i][k]);
                    }
                }
            }
        }
        return lc1976Helper(0, minDist, matrix);
    }

    private int lc1976Helper(int cur, long[][] minDist, long[][] matrix) {
        if (cur == minDist.length - 1) return 1; // 已经到达最后一个节点
        if (lc1976Memo[cur] != null) return lc1976Memo[cur];
        int n = minDist.length;
        long result = 0;
        final long mod = 1000000007;
        for (int next = 0; next < n; next++) {
            if (matrix[cur][next] != lc1976Inf && minDist[0][cur] + minDist[next][n - 1] + matrix[cur][next] == minDist[0][n - 1]) {
                result += lc1976Helper(next, minDist, matrix);
                result %= mod;
            }
        }
        return lc1976Memo[cur] = (int) (result % mod);
    }

    // LC1017 **
    public String baseNeg2(int n) {
        StringBuilder sb = new StringBuilder();
        List<Integer> result = toBase(n, -2);
        for (int i : result) sb.append(i);
        return sb.toString();
    }

    public List<Integer> toBase(int num, int base) {
        if (num == 0) return Arrays.asList(0);
        List<Integer> result = new ArrayList<>();
        while (num != 0) {
            int r = ((num % base) + Math.abs(base)) % Math.abs(base);
            result.add(r);
            num -= r;
            num /= base;
        }
        Collections.reverse(result);
        return result;
    }

    // LC1256
    public String encode(int num) {
        // 0 -> ""
        // 2 -> 0
        // 3 -> 1
        // 4 -> 00
        // 5 -> 01
        // 6 -> 10
        // 7 -> 11
        // 8 -> 000
        if (num == 0) return "";
        num++;
        boolean flag = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            if ((num >> (32 - i - 1) & 1) == 1) {
                if (!flag) {
                    flag = true;
                    continue;
                }
            }
            if (flag) {
                sb.append(num >> (32 - i - 1) & 1);
            }
        }
        return sb.toString();
    }

    // LC1218
    Map<Integer, TreeSet<Integer>> lc1218IdxMap;
    Integer[] lc1218Memo;

    public int longestSubsequence(int[] arr, int difference) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        lc1218IdxMap = new HashMap<>();
        lc1218Memo = new Integer[n + 1];
        int max = 1;
        for (int i = 0; i < n; i++) {
            lc1218IdxMap.putIfAbsent(arr[i], new TreeSet<>());
            lc1218IdxMap.get(arr[i]).add(i);
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int result = 1 + lc1218Helper(i, difference, arr, visited);
                max = Math.max(max, result);
            }
        }
        return max;
    }

    private int lc1218Helper(int idx, int difference, int[] arr, boolean[] visited) {
        visited[idx] = true;
        if (lc1218Memo[idx] != null) return lc1218Memo[idx];
        int expected = arr[idx] + difference;
        if (lc1218IdxMap.get(expected) != null) {
            Integer nextIdx = lc1218IdxMap.get(expected).higher(idx);
            if (nextIdx != null) {
                return lc1218Memo[idx] = 1 + lc1218Helper(nextIdx, difference, arr, visited);
            }
        }
        return lc1218Memo[idx] = 0;
    }

    // JZOF II 102 LC494
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i : nums) sum += i;
        if (Math.abs(target) > sum) return 0;
        int OFFSET = sum, n = nums.length;
        int[][] dp = new int[2][OFFSET * 2 + 1];
        dp[0][OFFSET] = 1; // 加入0个数, 和为 (0+OFFSET) 的个数为0
        for (int i = 1; i <= n; i++) {
            for (int total = 0; total <= 2 * sum; total++) {
                int result = 0;
                // 背包问题
                if (total - nums[i - 1] >= 0) {
                    result += dp[(i - 1) % 2][total - nums[i - 1]];
                }
                if (total + nums[i - 1] <= 2 * OFFSET) {
                    result += dp[(i - 1) % 2][total + nums[i - 1]];
                }
                dp[i % 2][total] = result;
            }
        }
        return dp[n % 2][OFFSET + target];
    }

    // LC1087
    public String[] expand(String s) {
        List<String> l = braceExpansionII(s);
        return l.toArray(new String[l.size()]);
    }

    // LC1096 ** DFS
    public List<String> braceExpansionII(String expression) {
        Set<String> result = helper(expression);
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    private Set<String> helper(String chunk) {
        if (chunk.length() == 0) return new HashSet<>();
        Set<String> result = new HashSet<>();
        Set<String> peek = new HashSet<>();
        char[] ca = chunk.toCharArray();
        int i = 0, n = ca.length;
        while (i < n) {
            char c = ca[i];
            if (c == '{') {
                int numParenthesis = 1;
                int start = ++i; // 括号对内的起始下标(不包括括号)
                while (numParenthesis != 0) {
                    if (ca[i] == '{') numParenthesis++;
                    if (ca[i] == '}') numParenthesis--;
                    i++;
                }
                Set<String> next = helper(chunk.substring(start, i - 1));
                peek = merge(peek, next);
                continue;
            } else if (c == ',') {
                result.addAll(peek);
                peek.clear();
                i++;
                continue;
            } else { // 不会遍历到 '{'
                StringBuilder word = new StringBuilder();
                while (i < n && Character.isLetter(ca[i])) {
                    word.append(ca[i]);
                    i++;
                }
                Set<String> tmp = new HashSet<>();
                tmp.add(word.toString());
                peek = merge(peek, tmp);
            }
        }
        if (i == n) result.addAll(peek);
        return result;
    }

    private Set<String> merge(Set<String> prefix, Set<String> suffix) {
        if (suffix.size() == 0) return prefix;
        if (prefix.size() == 0) return suffix;
        Set<String> result = new HashSet<>();
        for (String p : prefix) {
            for (String s : suffix) {
                result.add(p + s);
            }
        }
        return result;
    }

    // ** BFS
    public List<String> braceExpansionIiBfs(String expression) {
        expression = "{" + expression + "}"; // 预防 "a,{b}c"这种情况
        Deque<String> q = new LinkedList<>();
        q.offer(expression);
        Set<String> result = new HashSet<>();
        while (!q.isEmpty()) {
            String p = q.poll();
            if (p.indexOf("{") < 0) {
                result.add(p);
                continue;
            }
            // ** 找最深的括号对
            int idx = 0, left = -1, right = -1;
            while (p.charAt(idx) != '}') {
                if (p.charAt(idx) == '{') left = idx;
                idx++;
            }
            right = idx;
            String prefix = p.substring(0, left);
            String suffix = p.substring(right + 1);
            String[] middle = p.substring(left + 1, right).split(",");

            for (String m : middle) {
                q.offer(prefix + m + suffix);
            }
        }
        List<String> l = new ArrayList<>(result);
        Collections.sort(l);
        return l;
    }

    // LC1807
    public String evaluate(String s, List<List<String>> knowledge) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        Map<String, String> map = new HashMap<>(knowledge.size());
        for (List<String> k : knowledge) {
            map.put(k.get(0), k.get(1));
        }
        int left = -1, right = -1;
        boolean inParenthesis = false;
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '(') {
                left = i;
                inParenthesis = true;
            } else if (c == ')') {
                right = i;
                inParenthesis = false;
                String key = s.substring(left + 1, right);
                sb.append(map.getOrDefault(key, "?"));
            } else {
                if (inParenthesis) continue;
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // LC814
    public TreeNode pruneTree(TreeNode root) {
        if (!subtreeHasOne(root)) return null;
        lc814Helper(root);
        return root;
    }

    private void lc814Helper(TreeNode root) {
        if (root == null) return;
        if (!subtreeHasOne(root.left)) {
            root.left = null;
        } else {
            lc814Helper(root.left);
        }
        if (!subtreeHasOne(root.right)) {
            root.right = null;
        } else {
            lc814Helper(root.right);
        }
    }

    private boolean subtreeHasOne(TreeNode root) {
        if (root == null) return false;
        if (root.val == 1) return true;
        return subtreeHasOne(root.left) || subtreeHasOne(root.right);
    }

    // LC306
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        // 选前两个数
        for (int i = 1; i <= n / 2; i++) {
            long first = Long.parseLong(num.substring(0, i));
            if (String.valueOf(first).length() != i) continue;
            for (int j = i + 1; j < n; j++) {
                long second = Long.parseLong(num.substring(i, j));
                if (String.valueOf(second).length() != j - i) continue;
                if (judge(first, second, j, num)) return true;
            }
        }
        return false;
    }

    private boolean judge(long first, long second, int idx, String num) {
        if (idx == num.length()) return true;
        long sum = first + second;
        if (num.indexOf(String.valueOf(sum), idx) != idx) return false;
        return judge(second, sum, idx + String.valueOf(sum).length(), num);
    }


    // LC311 矩阵乘法
    public int[][] multiply(int[][] mat1, int[][] mat2) {
        // axb mult bxc = axc
        int a = mat1.length, b = mat1[0].length, c = mat2[0].length;
        int[][] result = new int[a][c];
        for (int i = 0; i < a; i++) {
            for (int k = 0; k < b; k++) {
                if (mat1[i][k] == 0) continue;
                for (int j = 0; j < c; j++) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return result;
    }

    // LC259 ** Solution O(n^2)
    public int threeSumSmaller(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int result = 0;
        for (int i = 0; i < n - 2; i++) {
            result += twoSumSmaller(nums, i + 1, target - nums[i]);
        }
        return result;
    }

    private int twoSumSmaller(int[] nums, int startIdx, int target) {
        int result = 0;
        int left = startIdx, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                result += right - left;
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    // LC1243
    public List<Integer> transformArray(int[] arr) {
        int n = arr.length;
        List<Integer> prev = Arrays.stream(arr).boxed().collect(Collectors.toList());
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < n; i++) cur.add(-1);
        while (true) {
            cur = helper(prev);
            if (cur.equals(prev)) return cur;
            prev = cur;
        }
    }

    private List<Integer> helper(List<Integer> prev) {
        int n = prev.size();
        List<Integer> cur = new ArrayList<>();
        cur.add(prev.get(0));
        for (int i = 1; i < n - 1; i++) {
            // 假如一个元素小于它的左右邻居，那么该元素自增 1。
            // 假如一个元素大于它的左右邻居，那么该元素自减 1。
            if (prev.get(i) < prev.get(i - 1) && prev.get(i) < prev.get(i + 1)) {
                cur.add(prev.get(i) + 1);
            } else if (prev.get(i) > prev.get(i - 1) && prev.get(i) > prev.get(i + 1)) {
                cur.add(prev.get(i) - 1);
            } else {
                cur.add(prev.get(i));
            }
        }
        cur.add(prev.get(n - 1));
        return cur;
    }


    // Interview 17.09 LC264 UglyNumber 丑数
    public int getKthMagicNumber(int k) {
        // Prime Factor 3,5,7
        long[] factor = {3, 5, 7};
        PriorityQueue<Long> pq = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        pq.offer(1l);
        set.add(1l);
        long result = -1;
        for (int i = 0; i < k; i++) {
            long p = pq.poll();
            result = p;
            for (long f : factor) {
                if (set.add(f * p)) {
                    pq.offer(f * p);
                }
            }
        }
        return (int) result;
    }

    // LC365
    public boolean canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        Deque<int[]> q = new LinkedList<>();
        Set<Pair<Integer, Integer>> visited = new HashSet<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{jug1Capacity, jug2Capacity});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            Pair<Integer, Integer> pair = new Pair<>(p[0], p[1]);
            if (visited.contains(pair)) continue;
            visited.add(pair);
            if (p[0] == targetCapacity || p[1] == targetCapacity) return true;
            if (p[0] + p[1] == targetCapacity) return true;
            // 倒满一侧
            pair = new Pair<>(jug1Capacity, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{jug1Capacity, p[1]});
            }
            pair = new Pair<>(p[0], jug2Capacity);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], jug2Capacity});
            }
            // 倒掉一侧
            pair = new Pair<>(0, p[1]);
            if (!visited.contains(pair)) {
                q.offer(new int[]{0, p[1]});
            }
            pair = new Pair<>(p[0], 0);
            if (!visited.contains(pair)) {
                q.offer(new int[]{p[0], 0});
            }
            // 一侧倒向另一侧
            if (p[0] < jug1Capacity) {
                int jug1Empty = jug1Capacity - p[0];
                int jug2ToJug1 = Math.min(p[1], jug1Empty);
                pair = new Pair<>(p[0] + jug2ToJug1, p[1] - jug2ToJug1);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] + jug2ToJug1, p[1] - jug2ToJug1});
                }
            }
            if (p[1] < jug2Capacity) {
                int jug2Empty = jug2Capacity - p[1];
                int jug1ToJug2 = Math.min(p[0], jug2Empty);
                pair = new Pair<>(p[0] - jug1ToJug2, p[1] + jug1ToJug2);
                if (!visited.contains(pair)) {
                    q.offer(new int[]{p[0] - jug1ToJug2, p[1] + jug1ToJug2});
                }
            }
        }
        return false;
    }

    // LC439 ** Great Solution
    public String parseTernary(String expression) {
        int len = expression.length();
        int level = 0;
        for (int i = 1; i < len; i++) {
            if (expression.charAt(i) == '?') level++;
            if (expression.charAt(i) == ':') level--;
            if (level == 0) {
                return expression.charAt(0) == 'T' ?
                        parseTernary(expression.substring(2, i)) : parseTernary(expression.substring(i + 1));
            }
        }
        return expression;
    }

    // LC385
    public NestedInteger deserialize(String s) {
        NestedInteger root = new NestedInteger();
        if (s.charAt(0) != '[') {
            root.setInteger(Integer.parseInt(s));
            return root;
        }
        Deque<NestedInteger> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            char c = ca[i];
            if (c == '[') {
                NestedInteger next = new NestedInteger();
                stack.push(next);
            } else if (c == ']') {
                NestedInteger pop = stack.pop();
                if (sb.length() != 0) {
                    pop.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                if (!stack.isEmpty()) {
                    stack.peek().add(pop);
                    continue;
                } else {
                    return pop;
                }
            } else if (c == ',') {
                NestedInteger peek = stack.peek();
                if (sb.length() != 0) {
                    peek.add(new NestedInteger(Integer.parseInt(sb.toString())));
                    sb = new StringBuilder();
                }
                continue;
            } else {
                sb.append(c);
            }
        }
        return null;
    }
}

// LC385
class NestedInteger {
    // Constructor initializes an empty nested list.
    public NestedInteger() {

    }

    // Constructor initializes a single integer.
    public NestedInteger(int value) {

    }

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    public boolean isInteger() {
        return false;
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    public Integer getInteger() {
        return -1;
    }

    // Set this NestedInteger to hold a single integer.
    public void setInteger(int value) {
        ;
    }

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    public void add(NestedInteger ni) {
        ;
    }

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return empty list if this NestedInteger holds a single integer
    public List<NestedInteger> getList() {
        return null;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
    }
}


class Trie {
    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }


    public boolean addWord(String word) {
        if (search(word)) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) {
                cur.children.put(c, new TrieNode());
            }
            cur = cur.children.get(c);
            cur.path++;
        }
        cur.end++;
        return true;
    }

    public boolean remove(String word) {
        if (!search(word)) return false;
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (cur.children.get(c).path-- == 1) {
                cur.children.remove(c);
                return true;
            }
            cur = cur.children.get(c);
        }
        cur.end--;
        return true;
    }

    public boolean search(String word) {
        TrieNode target = getNode(word);
        return target != null && target.end > 0;
    }

    public boolean beginWith(String prefix) {
        return getNode(prefix) != null;
    }

    private TrieNode getNode(String prefix) {
        TrieNode cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return null;
            cur = cur.children.get(c);
        }
        return cur;
    }

}


class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    int end = 0;
    int path = 0;
}


// LC211
class WordDictionary {
    Trie trie;

    public WordDictionary() {
        trie = new Trie();
    }

    public void addWord(String word) {
        trie.addWord(word);
    }

    public boolean search(String word) {
        return searchHelper("", word);
    }

    private boolean searchHelper(String prefix, String suffix) {
        if (suffix.equals("")) return trie.search(prefix);
        StringBuilder sb = new StringBuilder(prefix);
        for (int i = 0; i < suffix.length(); i++) {
            if (suffix.charAt(i) != '.') {
                sb.append(suffix.charAt(i));
            } else {
                for (int j = 0; j < 26; j++) {
                    sb.append((char) ('a' + j));
                    if (!trie.beginWith(sb.toString())) {
                        sb.deleteCharAt(sb.length() - 1);
                        continue;
                    }
                    if (searchHelper(sb.toString(), suffix.substring(i + 1))) return true;
                    sb.deleteCharAt(sb.length() - 1);
                }
                // 一旦所有'.'的可能性都被尝试, 且无一匹配, 即可返回false
                return false;
            }
        }
        // 如果全程没有'.', 返回Trie中是否有这个word
        return trie.search(sb.toString());
    }
}