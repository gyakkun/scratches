import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.minStoneSum(new int[]{5, 4, 9}, 2));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1962
    public int minStoneSum(int[] piles, int k) {
        int[] freq = new int[10001];
        for (int i : piles) freq[i]++;
        int sum = 0;
        for (int i = 10000; i >= 0; i--) {
            if (freq[i] == 0) continue;
            if (k > 0) {
                int minusTime = Math.min(k, freq[i]);
                freq[i] -= minusTime;
                freq[i - i / 2] += minusTime;
                sum += i * freq[i];
                k -= minusTime;
                continue;
            }
            sum += i * freq[i];
        }
        return sum;
    }

    // LC301 **
    Set<String> lc301Result = new HashSet<>();

    public List<String> removeInvalidParentheses(String s) {
        char[] ca = s.toCharArray();
        int n = ca.length;
        // 多余的左右括号个数, 注意右括号多余当且仅当左边左括号不够匹配的时候
        int leftToRemove = 0, rightToRemove = 0;
        for (char c : ca) {
            if (c == '(') leftToRemove++;
            else if (c == ')') {
                if (leftToRemove == 0) rightToRemove++;
                else leftToRemove--;
            }
        }
        lc301Helper(0, ca, leftToRemove, rightToRemove, 0, 0, new StringBuilder());
        return new ArrayList<>(lc301Result);
    }

    private void lc301Helper(int curIdx, char[] ca,
            /*待删的左括号数*/int leftToRemove, int rightToRemove,
            /*已删的左括号数*/int leftCount, int rightCount,
                             StringBuilder sb) {
        if (curIdx == ca.length) {
            if (leftToRemove == 0 && rightToRemove == 0) {
                lc301Result.add(sb.toString());
            }
            return;
        }

        char c = ca[curIdx];
        if (c == '(' && leftToRemove > 0) { // 无视当前左括号
            lc301Helper(curIdx + 1, ca, leftToRemove - 1, rightToRemove, leftCount, rightCount, sb);
        }
        if (c == ')' && rightToRemove > 0) { // 无视当前右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove - 1, leftCount, rightCount, sb);
        }

        sb.append(c);
        if (c != '(' && c != ')') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount, sb);
        } else if (c == '(') {
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount + 1, rightCount, sb);
        } else if (c == ')' && rightCount < leftCount) { // 只有当当前已选择的左括号比右括号多才在此步选右括号
            lc301Helper(curIdx + 1, ca, leftToRemove, rightToRemove, leftCount, rightCount + 1, sb);
        }
        sb.deleteCharAt(sb.length() - 1);
    }

    // LC1957
    public String makeFancyString(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        int n = ca.length;
        for (int i = 0; i < n; ) {
            int curIdx = i;
            char cur = ca[i];
            while (i + 1 < n && ca[i + 1] == cur) i++;
            int count = Math.min(i - curIdx + 1, 2);
            for (int j = 0; j < count; j++) {
                sb.append(cur);
            }
            i++;
        }
        return sb.toString();
    }

    // LC1540
    public boolean canConvertString(String s, String t, int k) {
        if (s.length() != t.length()) return false;
        // 第i次操作(从1算) 可以将s种之前未被操作过的下标j(从1算)的char+i
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        List<Integer> shouldChangeIdx = new ArrayList<>();
        for (int i = 0; i < cs.length; i++) {
            if (cs[i] != ct[i]) shouldChangeIdx.add(i);
        }
        int[] minSteps = new int[shouldChangeIdx.size()];
        for (int i = 0; i < shouldChangeIdx.size(); i++) {
            char sc = cs[shouldChangeIdx.get(i)], tc = ct[shouldChangeIdx.get(i)];
            minSteps[i] = (tc - 'a' + 26 - (sc - 'a')) % 26;
        }
        int[] freq = new int[27];
        for (int i : minSteps) freq[i]++;
        int max = 0;
        for (int i = 1; i <= 26; i++) {
            max = Math.max(max, i + (freq[i] - 1) * 26);
        }
        return max <= k;
    }

    // LC266
    public boolean canPermutePalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int oddCount = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddCount++;
        return oddCount <= 1;
    }

    // LC409
    public int longestPalindrome(String s) {
        int[] freq = new int[256];
        char[] ca = s.toCharArray();
        for (char c : ca) {
            freq[c]++;
        }
        int even = 0, oddMax = 0, odd = 0;
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 0) even += freq[i];
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) oddMax = Math.max(oddMax, freq[i]);
        for (int i = 0; i < 256; i++) if (freq[i] % 2 == 1) odd += freq[i] - 1;
        if (oddMax == 0) return even;
        return odd + even + 1;
    }

    // LC1266
    public int minTimeToVisitAllPoints(int[][] points) {
        int x = points[0][0], y = points[0][1];
        int result = 0;
        for (int i = 1; i < points.length; i++) {
            int nx = points[i][0], ny = points[i][1];
            int deltaX = Math.abs(nx - x), deltaY = Math.abs(ny - y);
            int slash = Math.min(deltaX, deltaY);
            int line = Math.max(deltaX, deltaY) - slash;
            result += line + slash;
            x = nx;
            y = ny;
        }
        return result;
    }

    // LC1416
    Integer[] lc1416Memo;

    public int numberOfArrays(String s, int k) {
        int n = s.length();
        lc1416Memo = new Integer[n + 1];
        return lc1416Helper(0, s, k);
    }

    private int lc1416Helper(int cur, String s, int k) {
        final long mod = 1000000007l;
        if (cur == s.length()) return 1;
        if (lc1416Memo[cur] != null) return lc1416Memo[cur];
        int len = 1;
        long result = 0;
        while (cur + len <= s.length()) {
            long num = Long.parseLong(s.substring(cur, cur + len));
            if (String.valueOf(num).length() != len) break;
            if (num > k) break;
            if (num < 1) break;
            result += lc1416Helper(cur + len, s, k);
            result %= mod;
            len++;
        }
        return lc1416Memo[cur] = (int) (result % mod);
    }

    // LC1844
    public String replaceDigits(String s) {
        StringBuilder sb = new StringBuilder();
        char[] ca = s.toCharArray();
        for (int i = 0; i < ca.length; i++) {
            if (i % 2 == 0) sb.append(ca[i]);
            else sb.append((char) (ca[i - 1] + (ca[i] - '0')));
        }
        return sb.toString();
    }

    //
}