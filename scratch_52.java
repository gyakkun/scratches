import javafx.util.Pair;

import java.util.Comparator;
import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();


        System.out.println(s.permutation("sde"));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC798 Hard ** 差分数组 学习分析方法
    // https://leetcode-cn.com/problems/smallest-rotation-with-highest-score/solution/chai-fen-shu-zu-by-sssz-qdut/
    public int bestRotation(int[] nums) {
        // 数组向左平移轮转
        int n = nums.length;
        int[] diff = new int[n + 1];
        // [0,Math.min(i,i-arr[i])], [i+1,Math.min(i-arr[i]+arr.len),arr.len-1)] 为可以得分的范围
        for (int i = 0; i < n; i++) {
            diff[0]++;
            int r1 = Math.min(i, i - nums[i]) + 1;
            if (r1 >= 0)
                diff[r1]--;
            diff[i + 1]++;
            int r2 = Math.min(i - nums[i] + n, n - 1) + 1;
            if (r2 >= 0 && r2 <= n)
                diff[r2]--;
        }
        int accumulate = 0, max = 0, maxIdx = -1;
        for (int i = 0; i < n; i++) {
            accumulate += diff[i];
            if (accumulate > max) {
                max = accumulate;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Interview 17.14
    public int[] smallestK(int[] arr, int k) {
        quickSelect.topK(arr, arr.length - k);
        return Arrays.copyOfRange(arr, 0, k);
    }

    // Interview 08.08 全排列
    List<String> iv0808Result;

    public String[] permutation(String S) {
        iv0808Result = new ArrayList<>();
        iv0808Dfs(S.toCharArray(), 0);
        return iv0808Result.toArray(new String[iv0808Result.size()]);
    }

    private void iv0808Dfs(char[] ca, int cur) {
        if (cur == ca.length) {
            iv0808Result.add(new String(ca));
        }
        Set<Character> set = new HashSet<>();
        for (int i = cur; i < ca.length; i++) {
            if (!set.contains(ca[i])) {
                set.add(ca[i]);
                char tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
                iv0808Dfs(ca, cur + 1);
                tmp = ca[i];
                ca[i] = ca[cur];
                ca[cur] = tmp;
            }
        }
    }

    // LC905
    public int[] sortArrayByParity(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] % 2 == 1) {
                int tmp = nums[right];
                nums[right] = nums[left];
                nums[left] = tmp;
                left--;
                right--;
            }
            left++;
        }
        return nums;
    }

    // JZOF II 056 LC653
    List<Integer> lc653List = new ArrayList<>();

    public boolean findTarget(TreeNode root, int k) {
        lc653Inorder(root);
        int left = 0, right = lc653List.size() - 1;
        while (left < right) {
            if (lc653List.get(left) + lc653List.get(right) > k) {
                right--;
            } else if (lc653List.get(left) + lc653List.get(right) < k) {
                left++;
            } else {
                return true;
            }
        }
        return false;
    }

    private void lc653Inorder(TreeNode root) {
        if (root == null) return;
        lc653Inorder(root.left);
        lc653List.add(root.val);
        lc653Inorder(root.right);
    }

    // LC1611 Hard ** Reverse Gray Code
    public int minimumOneBitOperations(int n) {
        // Function<Integer, Integer> gray = orig -> orig ^ (orig >> 1);
        int orig = 0;
        while (n != 0) {
            orig ^= n;
            n >>= 1;
        }
        return orig;
    }

    // LC468
    public String validIPAddress(String IP) {
        final String NEITHER = "Neither", IPV4 = "IPv4", IPV6 = "IPv6";
        if (checkIpv4(IP)) return IPV4;
        if (checkIpv6(IP)) return IPV6;
        return NEITHER;
    }

    private boolean checkIpv4(String ip) {
        String[] groups = ip.split("\\.", -1);
        if (groups.length != 4) return false;
        for (String w : groups) {
            for (char c : w.toCharArray()) if (!Character.isDigit(c)) return false;
            if (w.length() > 3 || w.length() == 0) return false;
            if (w.charAt(0) == '0' && w.length() != 1) return false;
            if (Integer.valueOf(w) > 255 || Integer.valueOf(w) < 0) return false;
        }
        return true;
    }

    private boolean checkIpv6(String ip) {
        String[] groupsByTwoColon = ip.split("::", -1);
        if (groupsByTwoColon.length > 2) return false; // 注意LC题目这里要改成>1, 因为题目不允许双冒号表示
        String[] groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length > 8) return false;
        for (String w : groupsByOneColon) {
            if (w.length() > 4) return false;
            String wlc = w.toLowerCase();
            for (char c : wlc.toCharArray()) {
                if (!Character.isLetter(c) && !Character.isDigit(c)) return false;
                if (Character.isLetter(c) && c > 'f') return false;
            }
        }
        int numColon = groupsByOneColon.length - 1;
        if (numColon != 7) {
            StringBuilder sb = new StringBuilder();
            int remain = 7 - numColon + 1;
            for (int i = 0; i < remain; i++) {
                sb.append(":0");
            }
            sb.append(":");
            ip = ip.replaceFirst("::", sb.toString());
        }
        groupsByOneColon = ip.split(":", -1);
        if (groupsByOneColon.length != 8) return false;
        long ipv6Left = 0, ipv6Right = 0;
        for (int i = 0; i < 8; i++) {
            String w = groupsByOneColon[i];
            int part = 0;
            if (w.equals("")) {
                if (i == 7) return false;
            } else {
                part = Integer.parseInt(w, 16);
            }
            if (i < 4) {
                ipv6Left = (ipv6Left << 16) | part;
            } else {
                ipv6Right = (ipv6Right << 16) | part;
            }
        }
        return true;
    }

    // LC751 **
    public List<String> ipToCIDR(String ip, int n) {
        final int INT_MASK = 0xffffffff;
        long start = ipStrToInt(ip);
        long end = start + n - 1;
        List<String> result = new ArrayList<>();
        while (n > 0) {
            int numTrailingZero = Long.numberOfTrailingZeros(start);
            int mask = 0, bitsInCidr = 1;
            while (bitsInCidr < n && mask < numTrailingZero) {
                bitsInCidr <<= 1;
                mask++;
            }
            if (bitsInCidr > n) {
                bitsInCidr >>= 1;
                mask--;
            }
            result.add(longToIpStr(start) + "/" + (32 - mask));
            start += bitsInCidr;
            n -= bitsInCidr;
        }
        return result;
    }

    private int bitLength(long x) {
        if (x == 0) return 1;
        return Long.SIZE - Long.numberOfLeadingZeros(x);
    }

    private String longToIpStr(long ipLong) {
        return String.format("%d.%d.%d.%d", (ipLong >> 24) & 0xff, (ipLong >> 16) & 0xff, (ipLong >> 8) & 0xff, ipLong & 0xff);
    }

    private long ipStrToInt(String ip) {
        String[] split = ip.split("\\.");
        long result = 0;
        for (String s : split) {
            int i = Integer.valueOf(s);
            result = (result << 8) | i;
        }
        return result;
    }


    // LC356
    public boolean isReflected(int[][] points) {
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        for (int[] p : points) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
        }
        int midXTimes2 = (minX + maxX);
        Set<Pair<Integer, Integer>> all = new HashSet<>();
        for (int[] p : points) {
            all.add(new Pair<>(p[0], p[1]));
        }
        Set<Pair<Integer, Integer>> s = new HashSet<>();
        for (Pair<Integer, Integer> p : all) {
            if ((double) p.getKey() == ((double) midXTimes2 / 2)) continue;
            if (!s.remove(new Pair<>(p.getKey(), p.getValue()))) {
                s.add(new Pair<>(midXTimes2 - p.getKey(), p.getValue()));
            }
        }
        return s.size() == 0;
    }

    // LC336 **
    public List<List<Integer>> palindromePairs(String[] words) {
        Set<Pair<Integer, Integer>> result = new HashSet<>();
        int wLen = words.length;
        String[] rWords = new String[wLen];
        Map<String, Integer> rWordIdx = new HashMap<>();
        for (int i = 0; i < wLen; i++) {
            rWords[i] = new StringBuilder(words[i]).reverse().toString();
            rWordIdx.put(rWords[i], i);
        }
        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            int len = cur.length();
            if (len == 0) continue;
            for (int j = 0; j <= len; j++) { // 注意边界, 为了取到空串, 截取长度可以去到len, 同时为了去重用到Set<Pair<>>
                if (checkPal(cur, j, len)) {
                    int leftId = rWordIdx.getOrDefault(cur.substring(0, j), -1);
                    if (leftId != -1 && leftId != i) {
                        result.add(new Pair<>(i, leftId));
                    }
                }
                if (checkPal(cur, 0, j)) {
                    int rightId = rWordIdx.getOrDefault(cur.substring(j), -1);
                    if (rightId != -1 && rightId != i) {
                        result.add(new Pair<>(rightId, i));
                    }
                }
            }
        }
        List<List<Integer>> listResult = new ArrayList<>(result.size());
        for (Pair<Integer, Integer> p : result) {
            listResult.add(Arrays.asList(p.getKey(), p.getValue()));
        }
        return listResult;
    }

    private boolean checkPal(String s, int startIdx, int endIdxExclude) {
        if (startIdx > endIdxExclude) return false;
        if (startIdx == endIdxExclude) return true;
        int len = endIdxExclude - startIdx;
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(startIdx + i) != s.charAt(endIdxExclude - 1 - i)) return false;
        }
        return true;
    }

    // LC747
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int[] idxMap = new int[101];
        for (int i = 0; i < nums.length; i++) {
            idxMap[nums[i]] = i;
        }
        Arrays.sort(nums);
        if (nums[nums.length - 1] >= nums[nums.length - 2] * 2) return idxMap[nums[nums.length - 1]];
        return -1;
    }

    // LC1224
    public int maxEqualFreq(int[] nums) {
        Map<Integer, Integer> numFreqMap = new HashMap<>();
        for (int i : nums) {
            numFreqMap.put(i, numFreqMap.getOrDefault(i, 0) + 1);
        }
        Map<Integer, Set<Integer>> freqIntSetMap = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : numFreqMap.entrySet()) {
            freqIntSetMap.putIfAbsent(e.getValue(), new HashSet<>());
            freqIntSetMap.get(e.getValue()).add(e.getKey());
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            // 当前元素
            int curEle = nums[i];

            // 看该不该删除

            // 情况1: freqMap.keySet.size > 2 此时删除哪个都没用
            if (freqIntSetMap.keySet().size() > 2) {
                continue;
            } else if (freqIntSetMap.keySet().size() == 2) {
                // 情况2: size == 2 时候, 看哪个set 的size ==1
                Iterator<Integer> it = freqIntSetMap.keySet().iterator();
                int freq1 = it.next(), freq2 = it.next();
                int smallFreq = freq1 < freq2 ? freq1 : freq2;
                int largeFreq = smallFreq == freq1 ? freq2 : freq1;
                Set<Integer> smallFreqSet = freqIntSetMap.get(smallFreq), largeFreqSet = freqIntSetMap.get(largeFreq);
                // 如果两个set都有超过一个元素, 则删除哪个元素都没用
                if (smallFreqSet.size() != 1 && largeFreqSet.size() != 1) {
                    continue;
                } else {
                    Set<Integer> oneEleSet = smallFreqSet.size() == 1 ? smallFreqSet : largeFreqSet;
                    Set<Integer> anotherSet = oneEleSet == smallFreqSet ? largeFreqSet : smallFreqSet;

                    int oneEle = oneEleSet.iterator().next();
                    int eleFreq = numFreqMap.get(oneEle);
                    int anotherFreq = eleFreq == smallFreq ? largeFreq : smallFreq;

                    // 情况1： 这个元素的当前频率是1
                    if (eleFreq == 1) return i + 1;
                        // 情况2: 当前元素的频率比另一个频率大1
                    else if (eleFreq == anotherFreq + 1) return i + 1;
                        // 特判一下 111 22 这种情况, 即两个freq的set的大小都是1
                        // 前面只判断了2不能删除, 没有判断1能不能删除, 此处补充判断一次
                    else if (anotherSet.size() == 1) {
                        if (anotherFreq == 1) return i + 1;
                        else if (anotherFreq == eleFreq + 1) return i + 1;
                    }
                    // 否则没办法 只能删除当前元素
                }
            }

            // 若没有找到该删除的 就删除当前元素
            int curFreq = numFreqMap.get(curEle);
            int nextFreq = curFreq - 1;
            numFreqMap.put(curEle, nextFreq);
            freqIntSetMap.get(curFreq).remove(nums[i]);
            if (freqIntSetMap.get(curFreq).size() == 0) freqIntSetMap.remove(curFreq);
            if (nextFreq != 0) {
                freqIntSetMap.putIfAbsent(nextFreq, new HashSet<>());
                freqIntSetMap.get(nextFreq).add(nums[i]);
            } else {
                numFreqMap.remove(nums[i]);
            }

        }
        return nums.length;
    }

    // JZOF 22
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // LC1705
    public int eatenApples(int[] apples, int[] days) {
        // pq 存数对 [i,j], i表示苹果数量, j表示过期时间
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        int n = apples.length;
        int result = 0;
        int i = 0;
        do {
            if (i < n) {
                if (apples[i] == 0 && days[i] == 0) {
                    ;
                } else if (apples[i] != 0) {
                    pq.offer(new int[]{apples[i], days[i] + i});
                }
            }
            if (!pq.isEmpty()) {
                int[] entry = null;
                do {
                    int[] p = pq.poll();
                    if (i >= p[1]) continue;
                    entry = p;
                    break;
                } while (!pq.isEmpty());
                if (entry != null) {
                    entry[0]--;
                    result++;
                    if (entry[0] > 0) pq.offer(entry);
                }
            }
            i++;
        } while (!pq.isEmpty() || i < n);
        return result;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Trie {
    Trie[] children = new Trie[26];
    boolean isEnd = false;

    public void addWord(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new Trie();
            }
            cur = cur.children[c - 'a'];
        }
        cur.isEnd = true;
    }

    public boolean startsWith(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return true;
    }

    public boolean search(String word) {
        Trie cur = this;
        for (char c : word.toCharArray()) {
            if (cur.children[c - 'a'] == null) return false;
            cur = cur.children[c - 'a'];
        }
        return cur.isEnd;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class quickSelect {
    static Random r = new Random();

    public static int topK(int[] arr, int topK) {
        return helper(arr, 0, arr.length - 1, topK);
    }

    private static Integer helper(int[] arr, int start, int end, int topK) {
        if (start == end && start == arr.length - topK) return arr[start];
        if (start >= end) return null;
        int randPivot = r.nextInt(end - start + 1) + start;
        if (arr[start] != arr[randPivot]) {
            int o = arr[start];
            arr[start] = arr[randPivot];
            arr[randPivot] = o;
        }
        int pivotVal = arr[start];
        int left = start, right = end;
        while (left < right) {
            while (left < right && arr[right] > pivotVal) {
                right--;
            }
            if (left < right) {
                arr[left] = arr[right];
                left++;
            }
            while (left < right && arr[left] < pivotVal) {
                left++;
            }
            if (left < right) {
                arr[right] = arr[left];
                right--;
            }
        }
        arr[left] = pivotVal;
        if (left == arr.length - topK) return arr[left];
        Integer leftResult = helper(arr, start, left - 1, topK);
        if (leftResult != null) return leftResult;
        Integer rightResult = helper(arr, right + 1, end, topK);
        if (rightResult != null) return rightResult;
        return null;
    }
}