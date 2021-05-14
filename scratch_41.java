
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;

        while ((n = br.readLine()) != null) {
            List<Integer> twoPrime = closetPrimeToSumEven(Integer.valueOf(n));
            for (int i : twoPrime) {
                System.out.println(i);
            }
        }
    }

    // HJ60
    public static List<Integer> closetPrimeToSumEven(int n) { // n is a even number
        // 欧拉筛
        List<Integer> primeList = new ArrayList<>();
        Set<Integer> primeSet;
        Set<Integer> notPrime = new HashSet<>();
        for (int i = 2; i <= n; i++) {
            if (!notPrime.contains(i)) {
                primeList.add(i);
            }
            for (int j = 0; j < primeList.size(); j++) {
                if (i * primeList.get(j) > n) break;
                notPrime.add(i * primeList.get(j));
                if (i % primeList.get(j) == 0) break;
            }
        }
        primeSet = new HashSet<>(primeList);
        int minDiff = Integer.MAX_VALUE;
        int pr1 = -1, pr2 = -1;
        Iterator<Integer> it = primeSet.iterator();
        while (it.hasNext()) {
            int prime1 = it.next();
            if (primeSet.contains(n - prime1)) {
                if (Math.abs(n - prime1 - prime1) < minDiff) {
                    pr1 = prime1;
                    pr2 = n - prime1;
                    minDiff = Math.abs(n - prime1 - prime1);
                }
            }
        }
        List<Integer> result = new ArrayList<>(2);
        if (pr1 != -1 && pr2 != -1) {
            result.add(pr1);
            result.add(pr2);
        }
        Collections.sort(result);
        return result;
    }

    // HJ55
    public static int countSeven(int n) {
        int result = 0;
        for (int i = 1; i <= n; i++) {
            if (i % 7 == 0) {
                result++;
                continue;
            }
            int numDigit = countDigit(i);
            for (int j = 0; j < numDigit; j++) {
                if ((i / tenPow(j)) % 10 == 7) {
                    result++;
                    break;
                }
            }
        }
        return result;
    }

    // HJ7
    public static String roundBelowFive(String fl) {
        double d = Double.valueOf(fl);
        long round = Math.round(d);
        return String.format("%d", round);
    }

    // HJ92
    public static List<String> longestNumberSubstring(String str) {
        List<String> result = new ArrayList<>();
        int maxLen = 0;
        int curLen = 0;
        int curIdx = -1;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) >= '0' && str.charAt(i) <= '9') {
                if (curIdx == -1) curIdx = i;
                curLen++;
                if (curLen == maxLen) {
                    result.add(str.substring(curIdx, curIdx + curLen));
                } else if (curLen > maxLen) {
                    result.clear();
                    result.add(str.substring(curIdx, curIdx + curLen));
                    maxLen = curLen;
                }
            } else {
                curLen = 0;
                curIdx = -1;
            }
        }
        return result;
    }

    // HJ99
    public static int countSelfProtectNum(int n) {
        int ctr = 0;
        for (int i = 0; i <= n; i++) {
            int mod10 = i % 10;
            if (mod10 == 0 || mod10 == 1 || mod10 == 5 || mod10 == 6) {
                if (isSelfProtectNum(i)) ctr++;
            }
        }
        return ctr;
    }

    public static boolean isSelfProtectNum(int n) {
        int numDigit = countDigit(n);
        long sqr = (long) n * (long) n;
        if ((sqr - n) % 10 != 0) return false;
        long tenPow = tenPow(numDigit);
        return sqr % tenPow == n;
    }

    private static long tenPow(int n) {
        long result = 1;
        while (n-- != 0) {
            result *= 10;
        }
        return result;
    }

    private static int countDigit(long n) {
        if (n == 0l) return 1;
        int result = 0;
        while (n != 0) {
            result++;
            n /= 10;
        }
        return result;
    }

    // HJ86
    public static int longestContinuousOneInBit(int num) {
        int result = 0;
        int tmp = 0;
        for (int i = 0; i < Integer.SIZE; i++) {
            if (((num >> i) & 1) == 1) {
                tmp++;
                if (tmp > result) {
                    result = tmp;
                }
            } else {
                tmp = 0;
            }
        }
        return result;
    }

    // HJ103
    public static int LIS(Integer[] arr) {
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : arr) {
            Integer ceil = ts.ceiling(i);
            if (ceil != null) {
                ts.remove(ceil);
            }
            ts.add(i);
        }
        return ts.size();
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

class Calculator {

    public double calculateDouble(String expression) {
        return evalRpnDouble(tokenToRpn(tokenize(expression)));
    }

    public static Set<String> notNumberToken = new HashSet<String>() {{
        add("+");
        add("-");
        add("/");
        add("*");
        add("(");
        add(")");
    }};

    public static Set<String> operSet = new HashSet<String>() {{
        add("+");
        add("-");
        add("/");
        add("*");
    }};

    public static int getOperPriority(String oper) {
        switch (oper) {
            case "+":
            case "-":
                return 1;
            case "*":
            case "/":
                return 2;
            default:
                return -1;
        }
    }

    public List<String> tokenize(String expression) {
        expression = expression.replaceAll("\\{", "(");
        expression = expression.replaceAll("\\}", ")");
        expression = expression.replaceAll("\\[", "(");
        expression = expression.replaceAll("\\]", ")");

        expression = expression.replaceAll(" ", ""); // 空格
        expression = expression.replaceAll("\\(\\-", "(0-");  // (-10) 之类的可能会出问题
        expression = expression.replaceAll("\\(\\+", "(0+"); // (+10) 之类的可能会出问题
        expression = expression.replaceAll("\\((\\d+(\\.\\d+)?)\\)", "$1"); // (0.0) 之类的

        int i = 0;
        List<String> tokens = new ArrayList<>();
        if (expression.length() == 0) return tokens;

        do {
            if (notNumberToken.contains(expression.charAt(i) + "")) {
                tokens.add(expression.charAt(i) + "");
                i++;
            } else {
                StringBuffer sb = new StringBuffer();
                while (i < expression.length() && !notNumberToken.contains(expression.charAt(i) + "")) {
                    sb.append(expression.charAt(i));
                    i++;
                }
                tokens.add(sb.toString());
            }
        } while (i < expression.length());
        return tokens;
    }

    public List<String> tokenToRpn(List<String> tokens) {
        Deque<String> stack = new LinkedList<>();
        List<String> rpn = new ArrayList<>();
        for (String token : tokens) {
            if (!notNumberToken.contains(token)) {
                rpn.add(token);
            } else if (token.equals("(")) {
                stack.push(token);
            } else if (token.equals(")")) {
                while (!stack.isEmpty()) {
                    String topToken = stack.pop();
                    if (!topToken.equals("(")) {
                        rpn.add(topToken);
                    } else {
                        break;
                    }
                }
            } else {
                while (!stack.isEmpty() && getOperPriority(stack.peek()) >= getOperPriority(token)) {
                    rpn.add(stack.pop());
                }
                stack.push(token);
            }
        }
        while (!stack.isEmpty()) {
            rpn.add(stack.pop());
        }
        return rpn;
    }

    public double evalRpnDouble(List<String> rpn) {
        Deque<String> stack = new LinkedList<>();
        stack.push("0");
        for (String token : rpn) {
            if (operSet.contains(token)) {
                double second = Double.valueOf(stack.pop());
                double first = Double.valueOf(stack.pop());
                double tmp = 0d;
                switch (token) {
                    case "+":
                        tmp = first + second;
                        break;
                    case "-":
                        tmp = first - second;
                        break;
                    case "*":
                        tmp = first * second;
                        break;
                    case "/":
                        tmp = first / second;
                        break;
                    default:
                        ;
                }
                stack.push(String.valueOf(tmp));
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0d : Double.valueOf(stack.pop());

    }

}