import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;
        while((n = br.readLine())!=null) {
            int i = Integer.valueOf(n);
            System.out.println(numOfPerfectNum(i));
        }
    }

    // HJ56
    static Map<Integer, Set<Integer>> numFactorMap = new HashMap<>();
    static List<Integer> perfectNum = new LinkedList<>();
    static int hj56Result = 0;
    static int[] hj56Table = new int[]{0, 8128, 496, 28, 6};

    public static int numOfPerfectNumTable(int n) {
        for (int i = 1; i <= 4; i++) {
            if (n >= hj56Table[i]) {
                return 11 - i + 1;
            }
        }
        return 0;
    }

    public static int numOfPerfectNum(int n) {
        for (int i = 6; i <= n; i++) {
            calFactor(i);
        }
        Collections.reverse(perfectNum);
        System.out.println(perfectNum);
        return hj56Result;
    }

    private static void calFactor(int n) {
        int upperBound = (int) (Math.sqrt(n) + 1);
        Set<Integer> s = new HashSet<>();
        boolean[] visited = new boolean[upperBound + 1];
        for (int i = 1; i <= upperBound; i++) {
            if (!visited[i] && n % i == 0) {
                s.add(i);
                s.add(n / i);
                if (numFactorMap.containsKey(i)) {
                    Set<Integer> tmp = numFactorMap.get(i);
                    Iterator<Integer> it = tmp.iterator();
                    while (it.hasNext()) {
                        int j = it.next();
                        s.add(j);
                        visited[j] = true;
                    }
                }
//                if (numFactorMap.containsKey(n / i)) {
//                    Set<Integer> tmp = numFactorMap.get(n / i);
//                    Iterator<Integer> it = tmp.iterator();
//                    while (it.hasNext()) {
//                        int j = it.next();
//                        s.add(j);
//                        if (j <= upperBound) {
//                            visited[j] = true;
//                        }
//                    }
//                }
                visited[i] = true;
            }
        }
        numFactorMap.put(n, s);
        Iterator<Integer> it = s.iterator();
        int sumFactor = 0;
        while (it.hasNext()) {
            int i = it.next();
            if (i != n) sumFactor += i;
        }
        if (sumFactor == n) {
            hj56Result++;
            perfectNum.add(n);
        }
    }

    // PRIMEOJ 1002
    public static int primeoj1002(int[][] money) {
        int totalVotes = 0;
        for (int[] i : money) {
            totalVotes += i[0];
        }
        int[] dp = new int[totalVotes + 1];
        // dp[i][j] 表示贿赂前i个群友得到j张选票的最小花费
        // dp[i][j] = Math.min(dp[i-1][j], dp[i-1][j - money[i][0] ] + money[i][1])

        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int i = 1; i <= money.length; i++) {
            for (int j = totalVotes; j >= money[i - 1][0]; j--) {
                dp[j] = Math.min(dp[j], dp[j - money[i - 1][0]] + money[i - 1][1]);
            }
        }

        int result = Integer.MAX_VALUE;
        for (int i = (totalVotes + 1) / 2; i <= totalVotes; i++) {
            result = Math.min(dp[i], result);
        }
        return result;
    }
}