import java.util.Map;
import java.util.TreeMap;

class Scratch {

    // PRIME OJ 1000

    public static void main(String[] args) {
        long timing = System.currentTimeMillis();

        NthPrime ntp = new NthPrime(100000000);
        System.out.println(ntp.getNthPrime(1));
        System.out.println(ntp.getNthPrime(100000000));
        System.out.println(ntp.getNthPrime(99999997));
        System.out.println(ntp.getNthPrime(99999999));
        System.out.println(ntp.getNthPrime(99999998));
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
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
        int lb = (int) (n * ((Math.log(n) + Math.log(Math.log(n))) - 1));
        int ub = lb + n;
        Map.Entry<Integer, Integer> he = result.higherEntry(n);
        Map.Entry<Integer, Integer> le = result.lowerEntry(n);

        int low = lb, high = ub;
        if (he != null) high = he.getValue();
        if (le != null) low = le.getValue();
        int target = n - 1;
        int tmp = -1;
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
        return result.get(n);
    }

    class Helper {
        int maxNth;
        long upper;
        int[] prime;
        int sqrtPc;
        int sqrtU;
        Integer[][] phiMemo;
        TreeMap<Long, Long> piCache;

        Helper(int n) {
            this.upper = (long) (n * Math.log(n) + n * Math.log(Math.log(n))); // 对第n个质数的最大值的估算
            this.maxNth = n;
            initPrime();
            sqrtPc = (int) Math.sqrt(2 * prime.length);
            sqrtU = (int) Math.sqrt(upper);
            phiMemo = new Integer[sqrtU + 1][sqrtPc + 1];
            phi(sqrtU, sqrtPc);
            piCache = new TreeMap<>();
        }

        private void initPrime() {
            // 求出sqrt(upper)以内的所有素数
            int up = (int) (1.5 * Math.max(20, (int) Math.sqrt(upper) + 1));
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
        }

        private long phi(long m, int n) {
            if (n == 0) return m;
            if (m == 0) return 0;
            if (m <= sqrtU && n <= sqrtPc && phiMemo[(int) m][n] != null) return phiMemo[(int) m][n];
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


        public long pi(long m) {
            if (m <= prime[prime.length - 1]) return piLessPrimeMax(m);
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