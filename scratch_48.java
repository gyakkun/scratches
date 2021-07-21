import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.maxScore(new int[]{1, 7, 3, 8, 9, 2, 1, 2, 666, 29, 13, 12, 16, 93}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1799 **
    int lc1799Result;
    int[][] lc1799GcdCache;
    Integer[] lc1799Memo;

    public int maxScore(int[] nums) {
        int n = nums.length / 2;
        int allMask = (1 << (2 * n)) - 1;
        lc1799Result = (1 + n) * n / 2;
        lc1799GcdCache = new int[n * 2][n * 2];
        lc1799Memo = new Integer[1 << (nums.length)];
        for (int i = 0; i < n * 2; i++) {
            for (int j = i + 1; j < n * 2; j++) {
                lc1799GcdCache[i][j] = lc1799GcdCache[j][i] = gcd(nums[i], nums[j]);
            }
        }
        return lc1799Helper(nums, 0, allMask);
    }

    // 注意DFS应该携带什么信息, 不应该携带什么信息, 不要把当前状态(比如这里的score)放进函数入参, 而应该动态计算, 动态更新 (见highlight)
    private int lc1799Helper(int[] nums, int curMask, int allMask) {
        if (lc1799Memo[curMask] != null) return lc1799Memo[curMask];
        lc1799Memo[curMask] = 0;
        int selectable = allMask ^ curMask;
        for (int subset = selectable; subset != 0; subset = (subset - 1) & selectable) {
            if (Integer.bitCount(subset) == 2) {
                int[] select = new int[2];
                int ctr = 0;
                for (int i = 0; i < nums.length; i++) {
                    if (((subset >> i) & 1) == 1) {
                        select[ctr++] = i;
                    }
                    if (ctr == 2) break;
                }
                int newMask = subset ^ curMask;
                lc1799Memo[curMask] = Math.max(lc1799Memo[curMask],
                        lc1799Helper(nums, newMask, allMask) + lc1799GcdCache[select[0]][select[1]] * Integer.bitCount(newMask) / 2); // Highlight
            }
        }
        return lc1799Memo[curMask];
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC844
    public boolean backspaceCompare(String s, String t) {
        Deque<Character> ss = new LinkedList<>();
        Deque<Character> ts = new LinkedList<>();
        for (char c : s.toCharArray()) {
            if (c != '#') {
                ss.push(c);
            } else {
                if (!ss.isEmpty()) ss.pop();
            }
        }
        for (char c : t.toCharArray()) {
            if (c != '#') {
                ts.push(c);
            } else {
                if (!ts.isEmpty()) ts.pop();
            }
        }
        if (ss.size() != ts.size()) return false;
        while (!ss.isEmpty()) {
            if (ss.pollLast() != ts.pollLast()) return false;
        }
        return true;
    }

    // LC650 **
    public int minSteps(int n) {
        if (n == 1) return 0;
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                if (i % j == 0) {
                    dp[i] = dp[j] + dp[i / j];
                }
            }
        }
        return dp[n];
    }

    // LC1255
    int lc1255Max;

    public int maxScoreWords(String[] words, char[] letters, int[] score) {
        lc1255Max = 0;
        int[] usable = new int[26];
        for (char c : letters) {
            usable[c - 'a']++;
        }
        boolean[] canAdd = new boolean[words.length];
        for (int i = 0; i < words.length; i++) {
            int[] tmp = new int[26];
            System.arraycopy(usable, 0, tmp, 0, 26);
            boolean flag = true;
            for (char c : words[i].toCharArray()) {
                tmp[c - 'a']--;
                if (tmp[c - 'a'] < 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) canAdd[i] = true;
        }
        List<String> addableWords = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (canAdd[i]) addableWords.add(words[i]);
        }
        int[] addableScores = new int[addableWords.size()];
        for (int i = 0; i < addableWords.size(); i++) {
            for (char c : addableWords.get(i).toCharArray()) {
                addableScores[i] += score[c - 'a'];
            }
        }

        lc1255Backtrack(0, 0, usable, addableWords, addableScores);

        return lc1255Max;
    }

    private void lc1255Backtrack(int curIdx, int curScore, int[] curUsable, List<String> addableWords, int[] addableScores) {
        if (curIdx == addableWords.size()) {
            lc1255Max = Math.max(lc1255Max, curScore);
            return;
        }
        for (int i = curIdx; i < addableWords.size(); i++) {
            subCurWordFreq(curUsable, addableWords, i);
            if (!isCanUse(curUsable)) {
                addBackCurWordFreq(curUsable, addableWords, i);
                lc1255Backtrack(i + 1, curScore, curUsable, addableWords, addableScores);
            } else {
                lc1255Backtrack(i + 1, curScore + addableScores[i], curUsable, addableWords, addableScores);
                addBackCurWordFreq(curUsable, addableWords, i);
            }
        }
    }

    private boolean isCanUse(int[] curUsable) {
        for (int j = 0; j < 26; j++) {
            if (curUsable[j] < 0) {
                return false;
            }
        }
        return true;
    }

    private void addBackCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']++;
        }
    }

    private void subCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']--;
        }
    }

    // LC1277
    public int countSquares(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1]; // dp[i][j] 表示以matrix[i][j]为右下角的最大正方形边长
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    dp[i + 1][j + 1] = 0;
                } else {
                    dp[i + 1][j + 1] = 1 + Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]);
                }
                result += dp[i + 1][j + 1];
            }
        }
        return result;
    }

    // LC147
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return head;
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode lastSorted = head, cur = head.next;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode prev = dummy;
                while (prev.next.val <= cur.val) {
                    prev = prev.next;
                }
                lastSorted.next = cur.next;
                cur.next = prev.next;
                prev.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    // JSOF 52 LC160
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode aPtr = headA, bPtr = headB;
        int aLen = 0, bLen = 0;
        while (aPtr != null && aPtr.next != null) {
            aLen++;
            aPtr = aPtr.next;
        }
        while (bPtr != null && bPtr.next != null) {
            bLen++;
            bPtr = bPtr.next;
        }
        if (aPtr != bPtr) return null;
        ListNode fast = aLen > bLen ? headA : headB;
        ListNode slow = fast == headA ? headB : headA;
        int aheadStep = Math.abs(aLen - bLen);
        while (aheadStep != 0) {
            fast = fast.next;
            aheadStep--;
        }
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
