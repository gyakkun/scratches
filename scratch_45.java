import javax.xml.transform.Result;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Scratch {

    public static void main1(String[] args) {
        NthPrime nthPrime = new NthPrime(100000000);
        System.out.println(nthPrime.getNthPrime(22222222));

    }

    // PRIME OJ 1000
    public static void main(String[] args) {
        int[] input = new int[10];
        int qIdx = -1;
        int max = 0;
        int min = Integer.MAX_VALUE;
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
                max = Math.max(max, input[i]);
                min = Math.min(min, input[i]);
            }
        }
        long timing = System.currentTimeMillis();

        NthPrime nthPrime = new NthPrime(max);
//        nthPrime.getNthPrime(min);
//        nthPrime.getNthPrime(max);

        for (int i = 0; i < 10; i++) {
            if (i == qIdx) {
                System.out.println("qq_group:" + nthPrime.getNthPrime(input[i]));
            } else {
                System.out.println(nthPrime.getNthPrime(input[i]));
            }
        }
        timing = System.currentTimeMillis() - timing;
//        System.err.println("TIMING: " + timing + "ms.");
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

    long biSearTiming = 0;

    private int calcNth(int n) {
        if(result.containsKey(n)) return result.get(n);
        int lb = (int) (n * ((Math.log(n) + Math.log(Math.log(n))) - 1));
        int ub = lb + n;
        int approximate = lb + (int) ((0.0 + n * Math.log(Math.log(n)) - 2 * n) / (Math.log(n)));
        int apPi = (int) helper.pi(approximate);
        int low = lb, high = ub;
        if (apPi > n) high = approximate;
        else low = approximate;

        Map.Entry<Integer, Integer> he = result.ceilingEntry(n);
        Map.Entry<Integer, Integer> le = result.floorEntry(n);

        if (he != null && he.getValue() < high) high = he.getValue();
        if (le != null && le.getValue() > low) low = le.getValue();
        int target = n - 1;
        int tmp = -1;

        long origBST = biSearTiming;
        long origPC = helper.piCount;
        long timing = System.currentTimeMillis();
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int tmpPi = (int) helper.pi(mid);
            if (tmpPi == target) {
                tmp = mid;
                break;
            } else if (tmpPi > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        timing = System.currentTimeMillis() - timing;
        biSearTiming += timing;
        if (tmp % 2 == 1) tmp--;
        int primeCount = (int) helper.pi(tmp);
        // 开始从tmp筛, 先使用埃筛
        for (int i = tmp; i <= ub; i++) {
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
//        System.err.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
//        System.err.println("BiSearchTiming: " + biSearTiming + "ms.");
//        System.err.println("piCount: " + helper.piCount + ".");
//        System.err.println("BiSearchTiming Delta: " + (biSearTiming - origBST) + "ms.");
//        System.err.println("piCount Delta: " + (helper.piCount - origPC) + ".");
//        System.err.println("avg pi consumes: " + ((biSearTiming - origBST + 0.0) / (helper.piCount - origPC + 0.0)) + ".");
//        System.err.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
//        System.err.println("");

        return result.get(n);
    }

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
            // 埃筛
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
            long timing = System.currentTimeMillis();
            for (int i = 0; i <= prime[prime.length - 1]; i++) {
                piCacheArray[i] = (int) piLessPrimeMax(i);
            }
            timing = System.currentTimeMillis() - timing;
//            System.err.println("init pi cache timing: " + timing + "ms.");
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
            long result = phi(m, n - 1) - phi((int) (m / prime[n]), n - 1);
            if (m <= sqrtU && n <= sqrtPc) {
                phiMemo[(int) m][n] = (int) result;
            }
            return result;
        }

        private long piLessPrimeMax(long x) { // 在 prime[prime.length-1] 以内求pai(x), 二分法找到第一个小于等于x的下标
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

        long piCount = 0;

        public long pi(long m) {
            piCount++;
            if (m <= prime[prime.length - 1]) return piCacheArray[(int) m];
            if (piCache.containsKey(m)) return piCache.get(m);
            Map.Entry<Long, Long> he = piCache.higherEntry(m);
            Map.Entry<Long, Long> le = piCache.lowerEntry(m);

            if (he != null && le != null && he.getValue() == le.getValue()) {
                return le.getValue();
            }
            int y = (int) Math.cbrt(m);
            int n = (int) piLessPrimeMax(y);
            long result = phi(m, n) + n - 1;
            for (int i = n + 1; i < prime.length && prime[i] * prime[i] <= m; i++) {
                result -= (pi(m / prime[i]) - pi(prime[i]) + 1);
            }
            piCache.put(m, result);
            return result;
        }

    }

}