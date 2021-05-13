import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

class Scratch {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;
        n = br.readLine();
        int numOfPeople = Integer.valueOf(n);
        int[][] money = new int[numOfPeople][numOfPeople];
        for (int i = 0; i < numOfPeople; i++) {
            n = br.readLine();
            String[] a = n.split(" ");
            money[i][0] = Integer.valueOf(a[0]);
            money[i][1] = Integer.valueOf(a[1]);
        }
        System.out.println(primeoj1002(money));

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