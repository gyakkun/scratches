import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

class Scratch {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;
        while ((n = br.readLine()) != null) {
            int i = Integer.valueOf(n);
            n = br.readLine();
            Integer[] intArr = Arrays.stream(n.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
            System.out.println(arraySplit(intArr));
        }
    }

    // HJ93
    public static boolean arraySplit(Integer[] arr) {
        int sum = 0;
        List<Integer> threeMul = new ArrayList<>();
        List<Integer> fiveMul = new ArrayList<>();
        List<Integer> other = new LinkedList<>();
        int fiveSum = 0;
        int threeSum = 0;
        for (int i : arr) {
            sum += i;
            if (i % 5 == 0) {
                fiveMul.add(i);
                fiveSum += i;
            } else if (i % 3 == 0) {
                threeMul.add(i);
                threeSum += i;
            } else {
                other.add(i);
            }
        }
        if (sum % 2 != 0) return false;
        int half = sum / 2;
        int target = half - fiveSum; // 目标: 在Other里面找到和为target的组合
        return hj93Backtrack(other, target);
    }

    private static boolean hj93Backtrack(List<Integer> other, int target) {
        if (target == 0) {
            return true;
        }
        for (int i = 0; i < other.size(); i++) {
            int tmp = other.get(i);
            other.remove(i);
            if (hj93Backtrack(other, target - tmp)) {
                return true;
            }
            other.add(i, tmp);
        }
        return false;
    }

    // HJ76
    public static String Nicomachus(int m) {
        StringBuffer sb = new StringBuffer();
        // m * m * m = m (i0 + im-1) / 2
        // im-1 = i0 + (m-1) *2
        // m*m = i0+(m-1)
        // i0 = m*m-m+1
        int i0 = m * m - m + 1;
        for (int i = 0; i < m; i++) {
            sb.append(i0 + "+");
            i0 += 2;
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    // HJ56
    static Map<Integer, Set<Integer>> numFactorMap = new HashMap<>();
    static List<Integer> perfectNum = new LinkedList<>();
    static int hj56Result = 0;
    static int[] hj56Table = new int[]{0, 8128, 496, 28, 6};

    public static int numOfPerfectNumTable(int n) {
        for (int i = 1; i <= 4; i++) {
            if (n >= hj56Table[i]) {
                return 4 - i + 1;
            }
        }
        return 0;
    }

    public static int numOfPerfectNum(int n) {
        hj56Result = 0;
        for (int i = 6; i <= n; i++) {
            calFactor(i);
        }
        return hj56Result;
    }

    private static void calFactor(int n) {
        int upperBound = (int) (Math.sqrt(n) + 1);
        int sum = 1;
        for (int i = 2; i < upperBound; i++) {
            if (n % i == 0) {
                int de = n / i;
                if (de == i) {// 平方根
                    sum += de;
                } else {
                    sum += i;
                    sum += de;
                }
            }
        }

        if (sum == n) hj56Result++;
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