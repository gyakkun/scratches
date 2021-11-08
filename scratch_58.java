import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


//        System.out.println(s.maxRepOpt1("babbaaabbbbbaa"));

        System.out.println("jisafdousaiouoi".replaceAll("[aeiou]", "#"));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC966
    public String[] spellchecker(String[] wordlist, String[] queries) {
        Map<String, Integer> lowerCaseIdxMap = new HashMap<>();
        Map<String, Integer> deVowelIdxMap = new HashMap<>();
        Set<String> wordSet = new HashSet<>();
        for (int i = 0; i < wordlist.length; i++) {
            lowerCaseIdxMap.putIfAbsent(wordlist[i].toLowerCase(), i);
            String deVowel = wordlist[i].toLowerCase().replaceAll("[aeiou]", "#");
            deVowelIdxMap.putIfAbsent(deVowel, i);
            wordSet.add(wordlist[i]);
        }
        String[] result = new String[queries.length];

        for (int i = 0; i < queries.length; i++) {
            if (wordSet.contains(queries[i])) {
                result[i] = queries[i];
                continue;
            }
            String lcKey = queries[i].toLowerCase();
            if (lowerCaseIdxMap.containsKey(lcKey)) {
                result[i] = wordlist[lowerCaseIdxMap.get(lcKey)];
                continue;
            }
            String dvKey = queries[i].toLowerCase().replaceAll("[aeiou]", "#");
            if (deVowelIdxMap.containsKey(dvKey)) {
                result[i] = wordlist[deVowelIdxMap.get(dvKey)];
                continue;
            }
            result[i] = "";
        }
        return result;
    }

    // LC1156
    public int maxRepOpt1(String text) {
        Map<Character, List<int[]>> letterAppearIdxPairMap = new HashMap<>(26);
        for (char c = 'a'; c <= 'z'; c++) {
            letterAppearIdxPairMap.put(c, new ArrayList<>());
        }
        char[] ca = text.toCharArray();
        int idx = 0;
        while (idx < ca.length) {
            int start = idx;
            while (idx + 1 < ca.length && ca[idx + 1] == ca[idx]) idx++;
            int end = idx;
            letterAppearIdxPairMap.get(ca[idx]).add(new int[]{start, end});
            idx++;
        }
        int maxLen = 0;
        for (char c = 'a'; c <= 'z'; c++) {
            List<int[]> appear = letterAppearIdxPairMap.get(c);
            for (int i = 0; i < appear.size(); i++) {
                int[] cur = appear.get(i);
                int curLen = cur[1] - cur[0] + 1;
                maxLen = Math.max(maxLen, curLen);
                if (i + 1 < appear.size()) {
                    int[] next = appear.get(i + 1);
                    int nextLen = next[1] - next[0] + 1;
                    if (next[0] - cur[1] == 2) {
                        // 中间只隔了一个字母 首先考虑当前c只有两段的情况
                        maxLen = Math.max(maxLen, curLen + nextLen);
                        // 如果c有三段或以上, 则从第三段调取一个字母过来填充
                        if (appear.size() >= 3) {
                            maxLen = Math.max(maxLen, curLen + nextLen + 1);
                        }
                    } else if (next[0] - cur[1] > 2) {
                        // 如果两段之间有超过一个字母, 则最多能抽掉一个过来填充
                        maxLen = Math.max(maxLen, curLen + 1);
                    }
                }
            }
            // 反过来再试一遍
            for (int i = appear.size() - 1; i >= 0; i--) {
                int[] cur = appear.get(i);
                int curLen = cur[1] - cur[0] + 1;
                maxLen = Math.max(maxLen, curLen);
                if (i - 1 >= 0) {
                    int[] next = appear.get(i - 1);
                    int nextLen = next[1] - next[0] + 1;
                    if (next[0] - cur[1] == -2) { // 注意符号, 或者用绝对值
                        // 中间只隔了一个字母 首先考虑当前c只有两段的情况
                        maxLen = Math.max(maxLen, curLen + nextLen);
                        // 如果c有三段或以上, 则从第三段调取一个字母过来填充
                        if (appear.size() >= 3) {
                            maxLen = Math.max(maxLen, curLen + nextLen + 1);
                        }
                    } else if (next[0] - cur[1] < -2) { // 注意符号, 或者用绝对值
                        // 如果两段之间有超过一个字母, 则最多能抽掉一个过来填充
                        maxLen = Math.max(maxLen, curLen + 1);
                    }
                }
            }
        }
        return maxLen;
    }

    // Interview 05.02
    public String printBin(double num) {
        double eps = 1e-11;
        StringBuilder bin = new StringBuilder("0.");
        while (true) {
            if (Math.abs(num - 0d) < eps) break;
            num = num * 2d;
            if (num >= 1d) {
                bin.append("1");
                num -= 1d;
            } else {
                bin.append("0");
            }
            if (bin.length() > 32) return "ERROR";
        }
        return bin.toString();
    }

    // LC1162 ** 多源最短路
    public int maxDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> q = new LinkedList<>();
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int landCount = 0, seaCount = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) seaCount++;
                else {
                    landCount++;
                    q.offer(new int[]{i, j});
                }
            }
        }
        if (landCount == m * n || seaCount == m * n) return -1;
        final int INF = Integer.MAX_VALUE / 2;
        int[][] minDistance = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) minDistance[i][j] = INF;
            }
        }
        int layer = -1;
        while (!q.isEmpty()) {
            layer++;
            int qs = q.size();
            for (int i = 0; i < qs; i++) {
                int[] p = q.poll();
                int r = p[0], c = p[1];
                if (grid[r][c] == 0 && minDistance[r][c] != INF) continue;
                if (grid[r][c] == 0) {
                    minDistance[r][c] = layer;
                }
                for (int[] d : directions) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] == 1 || minDistance[nr][nc] != INF) {
                        continue;
                    }
                    q.offer(new int[]{nr, nc});
                }
            }
        }
        int maxDistance = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    maxDistance = Math.max(maxDistance, minDistance[i][j]);
                }
            }
        }
        return maxDistance;
    }

    // LC1267
    public int countServers(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] rowSum = new int[m], colSum = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowSum[i] += grid[i][j];
                colSum[j] += grid[i][j];
            }
        }

        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && (rowSum[i] > 1 || colSum[j] > 1)) count++;
            }
        }

        return count;
    }

    // LC299
    public String getHint(String secret, String guess) {
        // secret.length = guess.length
        int n = secret.length();
        int[] sFreq = new int[10], gFreq = new int[10];
        char[] cs = secret.toCharArray(), cg = guess.toCharArray();
        int exactly = 0;
        for (int i = 0; i < n; i++) {
            if (cs[i] == cg[i]) {
                exactly++;
                continue;
            }
            sFreq[cs[i] - '0']++;
            gFreq[cg[i] - '0']++;
        }
        int blur = 0;
        for (int i = 0; i < 10; i++) {
            blur += Math.min(sFreq[i], gFreq[i]);
        }
        return exactly + "A" + blur + "B";
    }

    // LC1090
    public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
        int n = values.length;
        List<int[]> idxLabelSet = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            idxLabelSet.add(new int[]{labels[i], values[i]});
        }
        Collections.sort(idxLabelSet, Comparator.comparingInt(o -> -o[1]));
        int totalCount = 0, sum = 0;
        int[] labelFreq = new int[20001];
        for (int[] p : idxLabelSet) {
            int label = p[0], val = p[1];
            if (labelFreq[label] == useLimit) continue;
            if (totalCount == numWanted) break;
            totalCount++;
            sum += val;
            labelFreq[label]++;
        }
        return sum;
    }
}