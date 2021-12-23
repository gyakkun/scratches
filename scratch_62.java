import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.longestDupSubstring("asdfdsafdsaqwuieroudsajkasdaisdufoiueqoiurioqwjfojndsaknadhfjkahgkjhfgskjhfasd"));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }


    // LC1044
    final long mod = 1000000007l;
    final long base1 = 29;
    final long base2 = 31;
    String s;

    public String longestDupSubstring(String s) {
        this.s = s;
        char[] ca = s.toCharArray();
        int lo = 0, hi = s.length() - 1;
        String result = "";
        String tmp = null;
        while (lo < hi) { // 找最大值
            int mid = lo + (hi - lo + 1) / 2;
            if ((tmp = helper(ca, mid)) != null) {
                result = tmp;
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }

    private String helper(char[] ca, int len) {
        Set<Integer> m1 = new HashSet<>();
        Set<Integer> m2 = new HashSet<>();
        long hash1 = 0, hash2 = 0, accu1 = 1, accu2 = 1;
        for (int i = 0; i < len; i++) {
            hash1 *= base1;
            hash1 %= mod;
            hash2 *= base2;
            hash2 %= mod;
            hash1 += ca[i] - 'a';
            hash1 %= mod;
            hash2 += ca[i] - 'a';
            hash2 %= mod;
            accu1 *= base1;
            accu1 %= mod;
            accu2 *= base2;
            accu2 %= mod;
        }
        m1.add((int) hash1);
        m2.add((int) hash2);
        for (int i = len; i < ca.length; i++) {
            String victim = s.substring(i - len + 1, i + 1);
            hash1 = (((hash1 * base1 - accu1 * (ca[i - len] - 'a')) % mod) + mod + ca[i] - 'a') % mod;
            hash2 = (((hash2 * base2 - accu2 * (ca[i - len] - 'a')) % mod) + mod + ca[i] - 'a') % mod;
            if (m1.contains((int) hash1)
                    && m2.contains((int) hash2)
            ) {
                return victim;
            }
            m1.add((int) hash1);
            m2.add((int) hash2);
        }
        return null;
    }
}