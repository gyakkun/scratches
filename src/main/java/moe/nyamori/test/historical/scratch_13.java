package moe.nyamori.test.historical;

import java.util.*;
import java.util.stream.Collectors;

class scratch_13 {
    public static void main(String[] args) {
        System.err.println(findCircleNum(new int[][]
                {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}
        ));
    }

    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        if(pairs.size()==0) return s;
        char[] sChar = s.toCharArray();
        int maxIndex = -1;
//        int minIndex = Integer.MAX_VALUE;
//        for (List<Integer> li : pairs) {
//            maxIndex = maxIndex > li.get(0) ? maxIndex : li.get(0);
//            maxIndex = maxIndex > li.get(1) ? maxIndex : li.get(1);
////            minIndex = minIndex < li.get(0) ? minIndex : li.get(0);
////            minIndex = minIndex < li.get(1) ? minIndex : li.get(1);
//        }
//        boolean[][] mtx = new boolean[maxIndex + 1][maxIndex + 1];
        Map<Integer, Map<Integer, Boolean>> mtx = new HashMap<>();
//        Map<Integer, Boolean>[] mtx = new HashMap<Integer,Boolean>()[s.length()];
        boolean[] visited = new boolean[s.length()];
        for (List<Integer> li : pairs) {
//            if (mtx.get(li.get(0)) == null) {
                mtx.putIfAbsent(li.get(0), new HashMap<>());
//            }
//            if (mtx.get(li.get(0)).get(li.get(1)) == null) {
                mtx.get(li.get(0)).putIfAbsent(li.get(1), true);
//            }
//            if (mtx.get(li.get(1)) == null) {
                mtx.putIfAbsent(li.get(1), new HashMap<>());
//            }
//            if (mtx.get(li.get(1)).get(li.get(0)) == null) {
                mtx.get(li.get(1)).putIfAbsent(li.get(0), true);
//            }

//            mtx[li.get(0)][li.get(1)] = true;
//            mtx[li.get(1)][li.get(0)] = true;
        }
        List<List<Integer>> groups = stackDFS(visited, mtx);
        for (List<Integer> g : groups) {
            g = g.stream().sorted().collect(Collectors.toList());
            int firstIdxInStr = g.get(0);
            int lastIdxInStr = g.get(g.size() - 1);
            char[] ca = new char[g.size()];
            for (int j = 0; j < g.size(); j++) {
                ca[j] = s.charAt(g.get(j));
            }
            Arrays.sort(ca);
            for (int j = 0; j < g.size(); j++) {
                sChar[g.get(j)] = ca[j];
            }
        }
        return new String(sChar);
    }

    // Stack DFS, to calculate link circuit
    public List<List<Integer>> stackDFS(boolean[] visited, Map<Integer, Map<Integer, Boolean>> mtx) {
        int len = visited.length;
        Stack<Integer> s = new Stack<>();
        List<List<Integer>> linkedIdx = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            s.clear();
            if (!visited[i]) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(i);
                s.push(i);
                visited[i] = true;
                while (!s.isEmpty()) {
                    int idx = s.pop();
                    for (int j = 0; j < len; j++) {
                        if (mtx.get(idx) != null && mtx.get(idx).get(j) != null && mtx.get(idx).get(j) && !visited[j]) {
                            s.push(j);
                            tmp.add(j);
                            visited[j] = true;
                        }
                    }
                }
                linkedIdx.add(tmp);
            }
        }
        return linkedIdx;
    }

    public void dfs(int idx, boolean[] visited, boolean[][] mtx) {
        for (int i = 0; i < visited.length; i++) {
            //      if(mtx[idx][i])
        }
    }


    public static int findCircleNum(int[][] isConnected) {

        int len = isConnected.length;
        boolean[] visited = new boolean[len];
        List<List<Integer>> groups = stackDFS(visited, isConnected);

        return groups.size();
    }

    // Stack DFS, to calculate link circuit
    public static List<List<Integer>> stackDFS(boolean[] visited, int[][] mtx) {
        int len = visited.length;
        Stack<Integer> s = new Stack<>();
        List<List<Integer>> linkedIdx = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            s.clear();
            if (!visited[i]) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(i);
                s.push(i);
                visited[i] = true;
                while (!s.isEmpty()) {
                    int idx = s.pop();
                    for (int j = 0; j < len; j++) {
                        if (mtx[idx][j] == 1 && !visited[j]) {
                            s.push(j);
                            tmp.add(j);
                            visited[j] = true;
                        }
                    }
                }
                linkedIdx.add(tmp);
            }
        }
        return linkedIdx;
    }

    // Stack DFS, to calculate link circuit
    public List<List<Integer>> stackDFS(boolean[] visited, boolean[][] mtx) {
        int len = visited.length;
        Stack<Integer> s = new Stack<>();
        List<List<Integer>> linkedIdx = new ArrayList<>();

        for (int i = 0; i < len; i++) {
            s.clear();
            if (!visited[i]) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(i);
                s.push(i);
                visited[i] = true;
                while (!s.isEmpty()) {
                    int idx = s.pop();
                    for (int j = 0; j < len; j++) {
                        if (mtx[idx][j] && !visited[j]) {
                            s.push(j);
                            tmp.add(j);
                            visited[j] = true;
                        }
                    }
                }
                linkedIdx.add(tmp);
            }
        }
        return linkedIdx;
    }


    public static void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % nums.length;
        if (k == 0) return;
        int temp;
        for (int i = 0; i < k; i++) {
            for (int j = n - 1; j > 0; j--) {
                temp = nums[j];
                nums[j] = nums[j - 1];
                nums[j - 1] = temp;
            }
        }
    }

    public static int findCircleNumOrig(int[][] isConnected) {
        int result = 0;
        int n = isConnected.length;
        if (n == 0) {
            return 0;
        }
        boolean[] isVisited = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if ((isConnected[i][j] == 1 && isConnected[j][k] == 1)) {
                        isConnected[i][k] = 1;
                        isConnected[k][i] = 1;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if ((isConnected[i][j] == 1 && isConnected[j][k] == 1)) {
                        isConnected[i][k] = 1;
                        isConnected[k][i] = 1;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            int ctr = 0;
            String str = "";
            for (int j = 0; j < n; j++) {
                if (!isVisited[j] && isConnected[i][j] == 1) {
                    isVisited[j] = true;
                    str = str + j + " ";
                    ctr++;
                }
            }
            if (ctr > 0) {
                result++;
                System.err.println(str);
            }
        }
        return result;
    }
}