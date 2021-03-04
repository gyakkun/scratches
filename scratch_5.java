import java.util.Scanner;

class Scratch {

    public static int[] MxPF = new int[1000000];//最大质因子表
    public static int[] P = new int[100000];//P质数表(P[1]=2)
    public static int Pn;  // Pn:质数个数
    public static long[] P2 = new long[100000];//P2[i]=P[i]*P[i] 加速计算

    public static void Init() {//初始化质数表
        for (int i = 0; i < MxPF.length; i++) {
            MxPF[i] = -1;
        }
        Pn = 0;
        for (int i = 2; i <= 120000; ++i) {
            if ((~MxPF[i])!=0) {
                continue;
            }
            P[++Pn] = i;
            P2[Pn] = (long) P[Pn] * P[Pn];
            for (int j = i; j <= 120000; j += i) {
                MxPF[j] = i;
            }
        }
    }

    public static long GetPI(long n) {//返回有多少个小于等于n的质数
        assert (n > 0);
        assert (n < P2[Pn]);
        int m = GetK(n);
        return (GetD(n, m) + m - 2);
    }

    public static int GetK(long n) {//求最小的k使得 P[k]^2 > n
        int L = 1, R = Pn, M;//[L,R]  first ^2 > n
        while ((L ^ R) != 0) {
            M = (L + R) >> 1;
            if (P2[M] > n) {
                R = M;
            } else {
                L = M + 1;
            }
        }
        return L;
    }

    public static long GetD(long n, int m) {
        assert (m <= Pn);
        //到达边界直接返回
        if (P[m] > n || n == 1) {
            return 1;
        }
        if (m == 1) {
            return n;
        }

        //不是边界就递归
        int k = GetK(n);
        if (m > k) {
            return GetD(n, k) - (m - k);
        } else {
            return GetD(n, m - 1) - GetD(n / P[m - 1], m - 1);
        }
    }

    public static void main(String[] args) {
        Init();
        Scanner scan = new Scanner(System.in);
        if (scan.hasNextLine()) {
            String tmp = scan.nextLine();
            System.out.println("Start counting pi(" + tmp + ")");
            long startTime = System.currentTimeMillis();
            long answer = GetPI(Long.parseLong(tmp));
            long endTime = System.currentTimeMillis();
            System.out.println("Pi(" + tmp + ") = " + answer + " .");
            System.out.println("Time is " + (endTime - startTime ) + " ms .");
        }
    }
}