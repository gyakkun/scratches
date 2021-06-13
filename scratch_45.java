import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Scratch {

    // PRIME OJ 1000
    public static void main(String[] args) throws IOException {
        int[] input = new int[10];
        int qIdx = -1;
        int max = 0;
        int min = Integer.MAX_VALUE;
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        for (int i = 0; i < 10; i++) {
            if ((line = br.readLine()) != null) {
                if (line.matches("^qq_group:\\d+$")) {
                    qIdx = i;
                    Pattern pattern = Pattern.compile("[^\\d]");
                    Matcher matcher = pattern.matcher(line);
                    line = matcher.replaceAll("").trim();
                }
                input[i] = Integer.valueOf(line);
                max = Math.max(max, input[i]);
                min = Math.min(min, input[i]);
            }
        }

        NthPrime nthPrime = new NthPrime(max);

        for (int i = 0; i < 10; i++) {
            if (i == qIdx) {
                System.out.println("qq_group:" + nthPrime.getNthPrime(input[i]));
            } else {
                System.out.println(nthPrime.getNthPrime(input[i]));
            }
        }
    }

}

class NthPrime {
    int maxNth;
    TreeMap<Integer, Integer> result;
    Helper helper;

    public NthPrime(int maxNth) {
        this.maxNth = maxNth;
        helper = new Helper(maxNth);
        result = new TreeMap<>();
    }

    public int getNthPrime(int nth) {
        if (nth < helper.prime.length) return helper.prime[nth];
        if (result.containsKey(nth)) return result.get(nth);
        return calcNth(nth);
    }


    private int calcNth(int n) {
        if (result.containsKey(n)) return result.get(n);
        // 以下估计参考了 Wikipedia - 素数计数函数
        int lb = (int) (n * ((Math.log(n) + Math.log(Math.log(n))) - 1)); // nth prime 的上界
        int ub = lb + n; // 下届
        int approx = lb + (int) ((0.0 + n * Math.log(Math.log(n)) - 2 * n) / (Math.log(n))); // 一个估计
        int apPi = (int) helper.pi(approx);
        int low = lb, high = ub;
        if (apPi > n) high = approx;
        else low = approx;

        Map.Entry<Integer, Integer> ce = result.ceilingEntry(n);
        Map.Entry<Integer, Integer> fe = result.floorEntry(n);

        if (ce != null && ce.getValue() < high) high = ce.getValue();
        if (fe != null && fe.getValue() > low) low = fe.getValue();
        int target = n - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            int tmpPi = (int) helper.pi(mid);
            if (tmpPi == target) {
                lb = mid;
                break;
            } else if (tmpPi > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        if (lb % 2 == 1) lb--;
        int primeCount = (int) helper.pi(lb);
        // 开始从tmp筛, 埃筛
        for (int i = lb; i <= ub; i++) {
            boolean isPrime = true;
            for (int j = 1; j < helper.prime.length && helper.prime[j] * helper.prime[j] <= i; j++) {
                if (i % helper.prime[j] == 0) {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime) {
                result.put(++primeCount, i);
            }
            if (primeCount == n) break;
        }

        return result.get(n);
    }

    // Meissel-Lehmer 法求素数计数函数
    class Helper {
        int maxNth;
        long upper;
        int[] prime;
        int sqrtPc;
        int sqrtU;
        int[][] phiMemo;
        TreeMap<Long, Long> piCache;
        int[] piCacheArray;

        Helper(int n) {
            this.upper = (long) (n * Math.log(n) + n * Math.log(Math.log(n))); // 对第n个质数的最大值的估算
            this.maxNth = n;
            initPrime();
            sqrtPc = (int) Math.sqrt(2 * prime.length);
            sqrtU = (int) Math.sqrt(upper);
            phiMemo = new int[sqrtU + 1][sqrtPc + 1];
            initPhiMemo();
            piCache = new TreeMap<>();
        }

        private void initPrime() {
            // 求出sqrt(upper)以内的所有素数
            int up = Math.max(100, (int) Math.sqrt(upper) + 1);
            // 欧拉筛
            int[] pl = new int[1 + up / 2];
            boolean[] isNotPrime = new boolean[up + 1];
            isNotPrime[2] = false;
            int pc = 0;

            for (int i = 2; i <= up; i++) {
                if (!isNotPrime[i]) {
                    pl[++pc] = i;
                }
                for (int j = 1; j <= pc && pl[j] <= up / i; j++) {
                    isNotPrime[i * pl[j]] = true;
                    if (i % pl[j] == 0) break;
                }
            }
            prime = new int[pc + 1];
            System.arraycopy(pl, 0, prime, 0, pc + 1);
            piCacheArray = new int[prime[prime.length - 1] + 1];
            for (int i = 0; i <= prime[prime.length - 1]; i++) {
                piCacheArray[i] = (int) piLessThanPrimeMax(i);
            }
        }

        private void initPhiMemo() {
            for (int i = 0; i <= sqrtU; i++) {
                phiMemo[i][0] = i;
                for (int j = 1; j <= sqrtPc; j++) {
                    phiMemo[i][j] = phiMemo[i][j - 1] - phiMemo[i / prime[j]][j - 1];
                }
            }
        }

        private long phi(long m, int n) {
            if (n == 0) return m;
            if (m == 0) return 0;
            if (prime[n] >= m) return 1;
            if (m <= sqrtU && n <= sqrtPc) return phiMemo[(int) m][n];
            return phi(m, n - 1) - phi((int) (m / prime[n]), n - 1);
        }

        private long piLessThanPrimeMax(long x) {
            if (x == 1) return 0;
            if (x == 2) return 1;
            int low = 1, high = prime.length - 1;
            while (low < high) {
                int mid = low + (high - low + 1) / 2;
                if (prime[mid] <= x) {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            return low;
        }

        public long pi(long m) {
            if (m <= prime[prime.length - 1]) return piCacheArray[(int) m];
            if (piCache.containsKey(m)) return piCache.get(m);
            Map.Entry<Long, Long> ce = piCache.ceilingEntry(m);
            Map.Entry<Long, Long> fe = piCache.floorEntry(m);

            if (ce != null && fe != null && ce.getValue() == fe.getValue()) {
                return fe.getValue();
            }
            int y = (int) Math.cbrt(m);
            int n = piCacheArray[y];
            long result = phi(m, n) + n - 1;
            for (int i = n + 1; i < prime.length && prime[i] * prime[i] <= m; i++) {
                result -= (pi(m / prime[i]) - pi(prime[i]) + 1);
            }
            piCache.put(m, result);
            return result;
        }
    }

}