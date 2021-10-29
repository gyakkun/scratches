import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.isSelfCrossing(new int[]{1, 1, 3, 2, 1, 1}));


        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1983
    public int widestPairOfIndices(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int prefix1 = 0, prefix2 = 0;
        int result = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            prefix1 += nums1[i];
            prefix2 += nums2[i];
            int diff = prefix1 - prefix2;
            map.putIfAbsent(diff, i);
            if (map.containsKey(prefix1 - prefix2)) {
                result = Math.max(result, i - map.get(diff));
            }
        }
        return result;
    }

    // LC335 **
    public boolean isSelfCrossing(int[] distance) {
        for (int i = 3; i < distance.length; i++) {
            if (i >= 3
                    && distance[i] >= distance[i - 2]
                    && distance[i - 1] <= distance[i - 3])
                return true;
            else if (i >= 4
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 1] == distance[i - 3])
                return true;
            else if (i >= 5
                    && distance[i] + distance[i - 4] >= distance[i - 2]
                    && distance[i - 5] + distance[i - 1] >= distance[i - 3]
                    && distance[i - 2] > distance[i - 4]
                    && distance[i - 3] > distance[i - 1])
                return true;
        }
        return false;
    }

    // LC869
    public boolean reorderedPowerOf2(int n) {
        if (n == 0) return false;
        List<Integer> power2List = new ArrayList<>(31);
        for (int i = 0; i < 31; i++) power2List.add(1 << i);
        int[][] freqList = new int[31][10];
        for (int i = 0; i < 31; i++) {
            int power2 = power2List.get(i);
            int[] freq = new int[10];
            while (power2 != 0) {
                freq[power2 % 10]++;
                power2 /= 10;
            }
            freqList[i] = freq;
        }
        int[] thisFreq = new int[10];
        int dummy = n;
        while (dummy != 0) {
            thisFreq[dummy % 10]++;
            dummy /= 10;
        }
        outer:
        for (int i = 0; i < 31; i++) {
            for (int j = 0; j < 10; j++) {
                if (freqList[i][j] != thisFreq[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    // JZOF II 086 LC131
    List<List<String>> lc131Result;
    List<String> lc131Tmp;

    public String[][] partition(String s) {
        lc131Result = new ArrayList<>();
        lc131Tmp = new ArrayList<>();
        int n = s.length();
        boolean[][] judge = new boolean[n][n];
        char[] ca = s.toCharArray();
        for (int i = 0; i < n; i++) judge[i][i] = true;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left + len - 1 < n; left++) {
                if (len == 2) {
                    judge[left][left + 1] = ca[left] == ca[left + 1];
                } else if (judge[left + 1][left + len - 1 - 1] && ca[left] == ca[left + len - 1]) {
                    judge[left][left + len - 1] = true;
                }
            }
        }
        lc131Helper(0, judge, s);
        String[][] resArr = new String[lc131Result.size()][];
        for (int i = 0; i < lc131Result.size(); i++) {
            resArr[i] = lc131Result.get(i).toArray(new String[lc131Result.get(i).size()]);
        }
        return resArr;
    }

    private void lc131Helper(int idx, boolean[][] judge, String s) {
        if (idx == judge.length) {
            lc131Result.add(new ArrayList<>(lc131Tmp));
            return;
        }
        for (int len = 1; idx + len - 1 < judge.length; len++) {
            if (judge[idx][idx + len - 1]) {
                lc131Tmp.add(s.substring(idx, idx + len));
                lc131Helper(idx + len, judge, s);
                lc131Tmp.remove(lc131Tmp.size() - 1);
            }
        }
    }


    // LC792 ** 桶思想
    public int numMatchingSubseqBucket(String s, String[] words) {
        int result = 0;
        Map<Character, List<List<Character>>> bucket = new HashMap<>();
        for (String w : words) {
            bucket.putIfAbsent(w.charAt(0), new LinkedList<>());
            List<Character> bucketItem = new LinkedList<>();
            for (char c : w.toCharArray()) bucketItem.add(c);
            bucket.get(w.charAt(0)).add(bucketItem);
        }
        for (char c : s.toCharArray()) {
            Set<Character> set = new HashSet<>(bucket.keySet());
            for (char key : set) {
                if (c != key) continue;
                List<List<Character>> items = bucket.get(key);
                ListIterator<List<Character>> it = items.listIterator();
                while (it.hasNext()) {
                    List<Character> seq = it.next();
                    it.remove();
                    seq.remove(0);
                    if (seq.size() == 0) result++;
                    else {
                        bucket.putIfAbsent(seq.get(0), new LinkedList<>());
                        if (seq.get(0) == key) {
                            it.add(seq);
                        } else {
                            bucket.get(seq.get(0)).add(seq);
                        }
                    }
                }
                if (bucket.get(key).size() == 0) bucket.remove(key);
            }
        }
        return result;
    }

    // LC1055 **
    public int shortestWay(String source, String target) {
        int tIdx = 0, result = 0;
        char[] cs = source.toCharArray(), ct = target.toCharArray();
        while (tIdx < ct.length) {
            int sIdx = 0;
            int pre = tIdx;
            while (tIdx < ct.length && sIdx < cs.length) {
                if (ct[tIdx] == cs[sIdx]) tIdx++;
                sIdx++;
            }
            if (tIdx == pre) return -1;
            result++;
        }
        return result;
    }


    // LC1689
    public int minPartitions(String n) {
        int max = 0;
        for (char c : n.toCharArray()) {
            max = Math.max(max, c - '0');
        }
        return max;
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
                k -= minusTime;
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
            /*待删的左括号数*/
                             int leftToRemove, int rightToRemove,
            /*已删的左括号数*/
                             int leftCount, int rightCount,
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
