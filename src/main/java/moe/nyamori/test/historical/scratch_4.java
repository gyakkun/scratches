package moe.nyamori.test.historical;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


class scratch_4 {

}

class NthPrime4 {

    private static NthPrime4 instance;

    // ~ 1e11
    private final int N = 320005;
    private long phi[][] = new long[10005][105];
    private long p2[] = new long[N];
    private long ans[] = new long[N];
    private int len;
    private boolean vis[] = new boolean[N];

    private static Map<Long, Long> primesList; // 存储已经找到的素数

    private NthPrime4() {
    }

    // 单例
    public static synchronized NthPrime4 getInstance() {
        if (instance == null) {
            instance = new NthPrime4();
            instance.init();
        }
        return instance;
    }

    private void init() {
        len = 0;
        for (int i = 2; i < N; i++) {
            if (!vis[i]) {
                for (int j = i + i; j < N; j += i) vis[j] = true;
                p2[len++] = i;
                ans[i] = ans[i - 1] + 1;
                continue;
            }
            ans[i] = ans[i - 1];
        }
        for (int i = 0; i <= 10000; i++) {
            phi[i][0] = i;
            for (int j = 1; j <= 100; j++) {
                phi[i][j] = phi[i][j - 1] - phi[(int) (i / p2[j - 1])][j - 1];
            }
        }
    }

    private long solve_phi(long m, long n) {
        if (n == 0) return m;
        if (p2[(int) (n - 1)] >= m) return 1;
        if (m <= 10000 && n <= 100) return phi[(int) m][(int) n];
        return solve_phi(m, n - 1) - solve_phi(m / p2[(int) (n - 1)], n - 1);
    }

    public long pi(long m) {
        if (m < (long) N) return ans[(int) m];

        long y = (int) Math.cbrt(m * 1.0);
        long n = ans[(int) y];
        long sum = solve_phi(m, n) + n - 1;

        for (long i = n; p2[(int) i] * p2[(int) i] <= m; i++) //参考博客中的范围有误
            sum = sum - pi(m / p2[(int) i]) + pi(p2[(int) i]) - 1;
        return sum;
    }

    public void setMaxInitTh(long max) {

        // 求出上限

        double upperBounce = (int) (
                Double.valueOf(max) * Math.log(max)
                        + Double.valueOf(max) * Math.log(Math.log(max))
        );

        long sqrtUB = (long) Math.sqrt(upperBounce) + 1;

        sqrtUB = sqrtUB < 10000 ? 10000 : sqrtUB;

        getPrimeArray(sqrtUB);
    }

    private void getPrimeArray(long maxTh) {
        primesList = new HashMap<>();
        primesList.put(1l, 2l);
        long primeCnt = 1;
        for (long i = 3; primeCnt < maxTh; i += 2) {
            boolean isPrime = true;
            // 埃筛
            for (long j = 1; j < primeCnt && primesList.get(j) * primesList.get(j) <= i; j++) {
                if (i % primesList.get(j) == 0) {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime) {
                primeCnt++;
                primesList.put(primeCnt, i);
            }
        }
    }

    public long getNthPrimeUsingFormulaStackOverFlow(long targetTh) {

        if (primesList.get(targetTh) != null) {
            return primesList.get(targetTh);
        }

        long CntPrime = targetTh - 1; //假设目前找到的素数的数目是maxTh - 1

        // a(n) = a(n) = n*(log n + log (log n))

        long n = targetTh;

        long aN = (long) (
                Double.valueOf(targetTh) * Math.log(targetTh)
                        + Double.valueOf(targetTh) * Math.log(Math.log(targetTh))
        );

        long startPiOne = System.currentTimeMillis();

        long piAN = pi(aN);

        long endPiOne = System.currentTimeMillis();

        System.err.println("Pi 1 consumes: " + (endPiOne - startPiOne) + " ms. ");

        long eN = piAN - n;

        long bN = (long) (Double.valueOf(aN) - Math.log(aN) * eN);  // 标的

        long lowerBounce = bN;
        if (lowerBounce % 2 == 0) lowerBounce++;

        long startPiTwo = System.currentTimeMillis();

        int lowerBounceCnt = (int) pi(lowerBounce);
        long delta = (targetTh - lowerBounceCnt);

        if (delta <= 0) {
            System.err.println("Target: " + targetTh);
            System.err.println("Current: " + lowerBounceCnt);
            System.err.println("Delta : " + delta);
            System.err.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");

            int ctr = 1;
            while (delta <= 0) {
                lowerBounce = (int) (Double.valueOf(aN) - ((double) 1 + ((double) ctr * 0.0001)) * Math.log(aN) * (double) eN);
                if (lowerBounce % 2 == 0) lowerBounce--;
                lowerBounceCnt = (int) pi(lowerBounce);
                delta = (targetTh - lowerBounceCnt);
                ctr++;
            }
            System.err.println("IN " + ctr + " PERCENTAGE");
            System.err.println("LBC " + lowerBounceCnt);
            System.err.println("Delta:  " + delta);
            System.err.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        } else {
            System.err.println("Target: " + targetTh);
            System.err.println("Current: " + lowerBounceCnt);
            System.err.println("Delta : " + delta);
            System.err.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");

            int ctr = 1;
            while (delta > 0) {
                int tmpLowerBounce = (int) (Double.valueOf(aN) - ((double) 1 - ((double) ctr * 0.0001)) * Math.log(aN) * (double) eN);
                if (tmpLowerBounce % 2 == 0) tmpLowerBounce--;
                int tmpLowerBounceCnt = (int) pi(tmpLowerBounce);
                if (tmpLowerBounceCnt >= targetTh) {
                    break;
                }
                lowerBounce = tmpLowerBounce;
                lowerBounceCnt = tmpLowerBounceCnt;
                delta = (targetTh - lowerBounceCnt);
                ctr++;
            }
            System.err.println("IN " + ctr + " PERCENTAGE");
            System.err.println("LBC " + lowerBounceCnt);
            System.err.println("Delta:  " + delta);
            System.err.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        }

        long endPiTwo = System.currentTimeMillis();

        System.err.println("Pi 2 consumes: " + (endPiTwo - startPiTwo) + " ms. ");

        CntPrime = lowerBounceCnt;

        long startSieve = System.currentTimeMillis();

        for (long i = lowerBounce; CntPrime < targetTh; i += 2) {
            boolean isPrime = true;
            // 因为非素数可以拆成素数的乘积，所以只需要考虑已经找到的素数
            for (long j = 1; j < CntPrime && primesList.get(j) * primesList.get(j) <= i; j++) {
                if (i % primesList.get(j) == 0) {
                    isPrime = false;
                    break; //跳出循环
                }
            }
            if (isPrime) {
                CntPrime++;
                primesList.put(CntPrime, i);
            }
        }


        long endSieve = System.currentTimeMillis();

        System.err.println("Sieve consumes: " + (endSieve - startSieve) + " ms. ");

        return primesList.get(targetTh);
    }

}


class Main4 {

    private static NthPrime4 nthPrime = NthPrime4.getInstance();

    public static void main(String[] args) {
        int[] input = new int[10];
        int qIdx = -1;
        int max = 0;
        Scanner scan = new Scanner(System.in);
        for (int i = 0; i < 10; i++) {
            if (scan.hasNextLine()) {
                String tmp = scan.nextLine();
                if (tmp.matches("^qq_group:\\d+$")) {
                    qIdx = i;
                    Pattern pattern = Pattern.compile("[^\\d]");
                    Matcher matcher = pattern.matcher(tmp);
                    tmp = matcher.replaceAll("").trim();
                }
                input[i] = Integer.valueOf(tmp);
                max = max > input[i] ? max : input[i];
            }
        }

        nthPrime.setMaxInitTh(max);

        long beginTime = System.currentTimeMillis();

        for (int i = 0; i < 10; i++) {
            if (i == qIdx) {
                System.out.println("qq_group:" + nthPrime.getNthPrimeUsingFormulaStackOverFlow(input[i]));
            } else {
                System.out.println(nthPrime.getNthPrimeUsingFormulaStackOverFlow(input[i]));
            }
        }
        long endTime = System.currentTimeMillis();
        System.err.println("Total time consumes: " + (endTime - beginTime) + " ms.");
    }

}