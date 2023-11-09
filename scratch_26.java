
import java.util.*;


class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.mergeIntervalDiffArr(new int[][]{{1, 4}, {0, 0}}));
    }

    // LC56
    public int[][] mergeInterval(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        List<Integer> first = new ArrayList<>(), second = new ArrayList<>();
        first.add(intervals[0][0]);
        second.add(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > second.get(second.size() - 1)) {
                first.add(intervals[i][0]);
                second.add(intervals[i][1]);
            } else {
                second.set(second.size() - 1, Math.max(intervals[i][1], second.get(second.size() - 1)));
            }
        }
        int[][] result = new int[first.size()][2];
        for (int i = 0; i < first.size(); i++) {
            result[i][0] = first.get(i);
            result[i][1] = second.get(i);
        }
        return result;
    }

    public int[][] mergeIntervalDiffArr(int[][] intervals) {
        int max = 10;
        int[] diff = new int[max * 2 + 3];
        for (int[] i : intervals) {
            if (i[0] != i[1]) {
                // 整体右移1
                diff[i[0] * 2 + 1 + 1]++;
                diff[i[1] * 2 + 1 + 1]--;
            } else {
                // 开区间
                diff[i[0] * 2 + 1 + 1]++;
                diff[i[0] * 2 + 1 + 1 + 1]--;
            }
        }
        for (int i = 1; i < max * 2 + 3; i++) {
            diff[i] += diff[i - 1];
        }
        boolean inOnes = false;
        List<Integer> first = new ArrayList<>();
        List<Integer> second = new ArrayList<>();
        for (int i = 0; i < max * 2 + 3; i++) {
            if (inOnes) {
                if (diff[i] == 0) {
                    inOnes = false;
                    second.add(i - 2);
                } else {
                    continue;
                }
            } else {
                if (diff[i] > 0) {
                    inOnes = true;
                    first.add(i - 1);
                } else {
                    continue;
                }
            }
        }

        int[][] ans = new int[Math.max(first.size(), second.size())][2];
        for (int i = 0; i < first.size(); i++) {
            ans[i][0] = first.get(i) / 2;
            ans[i][1] = second.get(i) / 2;

        }
        return ans;
    }

    // 1,4 7,8 9,10
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    //

    public ListNode middleNode(ListNode head) {
        if (head.next == null) return head;
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public int getNextHappy(int n) {
        int result = 0;
        while (n != 0) {
            int mod = n % 10;
            result += mod * mod;
            n /= 10;
        }
        return result;
    }

    public boolean isHappy(int n) {
        int fast = n;
        int slow = n;
        do {
            if (fast == 1 || getNextHappy(fast) == 1) return true;
            slow = getNextHappy(slow);
            fast = getNextHappy(getNextHappy(fast));
        } while (slow != fast);
        return false;
    }

    class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }

    }

    // LC142
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode slow = head, fast = head;
        while (fast != null) {
            slow = slow.next;
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                return null;
            }
            if (fast == slow) {
                ListNode ptr = head;
                while (ptr != slow) {
                    ptr = ptr.next;
                    slow = slow.next;
                }
                return ptr;
            }
        }
        return null;
    }


    public void sortColors(int[] nums) {
        int n = nums.length;
        int p0 = 0, p1 = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                int temp = nums[i];
                nums[i] = nums[p1];
                nums[p1] = temp;
                ++p1;
            } else if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                if (p0 < p1) {
                    temp = nums[i];
                    nums[i] = nums[p1];
                    nums[p1] = temp;
                }
                ++p0;
                ++p1;
            }
        }
    }

    //     n1 * n2 < Integer.MAX_VALUE
    // <=> n1 < Integer.MAX_VALUE / n2

    // LC713
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int prod = 1, ans = 0, left = 0;
        for (int right = 0; right < nums.length; right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            ans += right - left + 1;
        }
        return ans;
    }

    // 当 nums[left] * nums[left + 1] ..nums[right-1] * nums[right] 都满足 < k 时，
    // 那么 nums[left + 1] ..nums[right-1] * nums[right] 和 nums[left+2] ..nums[right-1] * nums[right] ...... 都满足。
    // 其个数一共 right - left + 1个

    public boolean wontOverflow(int i, int j) {
        return i < Integer.MAX_VALUE / j;
    }

    public int threeSumClosestBinarySearch(int[] nums, int k) {
        int result = Integer.MAX_VALUE;
        int n = nums.length;
        if (n < 3) return 0;
        Arrays.sort(nums);

        for (int first = 0; first < n; first++) {
            // first转移至下一个不同的数
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }


            int second = first + 1, third = n - 1;
            while (second < third) {
                int sum = nums[first] + nums[second] + nums[third];
                if (sum == k) return k;
                if (Math.abs(result - k) > Math.abs(sum - k)) {
                    result = sum;
                }
                if (sum > k) {
                    int nextThird = third - 1;
                    while (nextThird > second && nums[nextThird] == nums[third]) {
                        nextThird--;
                    }
                    third = nextThird;
                } else {
                    int nextSecond = second - 1;
                    while (nextSecond < third && nums[nextSecond] == nums[second]) {
                        nextSecond++;
                    }
                    second = nextSecond;
                }
            }

        }

        return result;
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int best = 10000000;

        // 枚举 a
        for (int i = 0; i < n; ++i) {
            // 保证和上一次枚举的元素不相等
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            // 使用双指针枚举 b 和 c
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                // 如果和为 target 直接返回答案
                if (sum == target) {
                    return target;
                }
                // 根据差值的绝对值来更新答案
                if (Math.abs(sum - target) < Math.abs(best - target)) {
                    best = sum;
                }
                if (sum > target) {
                    // 如果和大于 target，移动 c 对应的指针
                    int k0 = k - 1;
                    // 移动到下一个不相等的元素
                    while (j < k0 && nums[k0] == nums[k]) {
                        --k0;
                    }
                    k = k0;
                } else {
                    // 如果和小于 target，移动 b 对应的指针
                    int j0 = j + 1;
                    // 移动到下一个不相等的元素
                    while (j0 < k && nums[j0] == nums[j]) {
                        ++j0;
                    }
                    j = j0;
                }
            }
        }
        return best;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    public List<List<Integer>> threeSumBinarySearch(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        int n = nums.length;
        if (n < 3) return result;
        Arrays.sort(nums);

        for (int left = 0; left < n && nums[left] <= 0; ) {
            int innerLeft = left, innerRight = n - 1;
            while (innerLeft < innerRight && nums[innerRight] >= 0) {
                int target = 0 - nums[innerLeft] - nums[innerRight];
                if (binarySearch(nums, innerLeft + 1, innerRight - 1, target) != -1) {
                    List<Integer> tmpResult = new ArrayList<>(3);
                    tmpResult.add(nums[innerLeft]);
                    tmpResult.add(target);
                    tmpResult.add(nums[innerRight]);
                    result.add(tmpResult);
                }

                for (; innerRight > innerLeft; ) {
                    if (innerRight - 1 > innerLeft && nums[innerRight - 1] != nums[innerRight]) {
                        innerRight--;
                        break;
                    } else {
                        innerRight--;
                    }
                }
            }
            for (; left < n; ) {
                if (left + 1 < n && nums[left + 1] != nums[left]) {
                    left++;
                    break;
                } else {
                    left++;
                }
            }
        }
        return result;
    }

    public int binarySearch(int[] arr, int low, int high, int target) {
        if (low > high) return -1;
        int left = low, right = high;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    public int binarySearchClosetHigh(int[] arr, int low, int high, int target) {
        if (low > high) return -1;
        int left = low, right = high;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (arr[right] >= target) return -1;
        return left;
    }

    public int binarySearchClosetLow(int[] arr, int low, int high, int target) {
        if (low > high) return -1;
        int left = low, right = high;
        while (left < right) {
            int mid = (left + right + 1) / 2;
            if (arr[mid] <= target) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        if (arr[left] > target) return -1;
        return left;
    }

    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n < 2) return n;
        int numOfUnique = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) numOfUnique++;
        }
        for (int i = 1, j = 1; i < numOfUnique; j++) {
            if (nums[j] != nums[j - 1]) {
                nums[i] = nums[j];
                i++;
            }
        }
        return numOfUnique;
    }

    public List<List<Integer>> pairSums(int[] nums, int target) {
        Map<Integer, Integer> ctr = new HashMap<>();
        List<List<Integer>> result = new LinkedList<>();
        for (int i : nums) {
            ctr.put(i, ctr.getOrDefault(i, 0) + 1);
        }
        for (int i : nums) {
            if (ctr.containsKey(i) && ctr.get(i) > 0 && ctr.containsKey(target - i) && ctr.get(target - i) > 0) {
//                result.add(new ArrayList<>())
                List<Integer> tmpResult = new ArrayList<>(2);
                tmpResult.add(i);
                ctr.put(i, ctr.get(i) - 1);
                if (ctr.get(target - i) == 0) {
                    ctr.put(i, ctr.get(i) + 1);
                    continue;
                }
                tmpResult.add(target - i);
                result.add(tmpResult);
                ctr.put(target - i, ctr.get(target - i) - 1);
                if (ctr.get(i) == 0) ctr.remove(i);
                if (ctr.get(target - i) == 0) ctr.remove(target - i);
            }
        }
        return result;
    }

    public boolean isMonotonic(int[] A) {
        if (A.length == 1) return true;
        int n = A.length;
        boolean increaseFlag = false;
        int i = 1;
        for (; i < n; i++) {
            if (A[i - 1] != A[i]) break;
        }
        if (i == n || i == n - 1) return true;
        if (A[i] > A[i - 1]) increaseFlag = true;
        if (increaseFlag) {
            for (; i < n; i++) {
                if (A[i] < A[i - 1]) return false;
            }
        } else {
            for (; i < n; i++) {
                if (A[i] > A[i - 1]) return false;
            }
        }
        return true;
    }

    // LC395
    public int longestSubstring(String s, int k) {
        if (s.length() == 0) return 0;
//        Map<Character, Integer> m = new LinkedHashMap<>();
        int[] m = new int[26];
        for (char c : s.toCharArray()) {
            m[c - 'a']++;
//            m.put(c, m.getOrDefault(c, 0) + 1);
        }
        int result = 0;
        for (int i = 0; i < 26; i++) {
            if (m[i] < k) {
                String[] sArr = s.split(String.valueOf((char) ('a' + i)));
                for (String innerS : sArr) {
                    result = Math.max(result, longestSubstring(innerS, k));
                }
                return result;
            }
        }
        return s.length();
    }

    // LC3
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int result = 0;
        Map<Character, Integer> m = new HashMap<>();
        int left = 0, right = 0;
        while (right < n) {
            if (m.containsKey(s.charAt(right))) {
                int origLeft = left;
                left = m.get(s.charAt(right)) + 1;
                for (int i = origLeft; i < left; i++) {
                    m.remove(s.charAt(i));
                }
            }
            m.put(s.charAt(right), right);
            result = Math.max(result, right - left + 1);
            right++;
        }
        return result;
    }

    // LC904
    public int totalFruit(int[] tree) {
        int result = 0;
        int n = tree.length;

        int left = 0, right = 0;
        Set<Integer> selectedType = new HashSet<>();

        while (right < n) {
            selectedType.add(tree[right]);
            if (selectedType.size() > 2) {
                for (int i = right - 1; i >= left; i--) {
                    if (tree[i] != tree[right - 1]) {
                        left = right = i + 1;
                        break;
                    }
                }
                selectedType.clear();
            } else {
                right++;
            }
            result = Math.max(result, right - left);
        }

        return result;
    }

    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        List<Integer> result = new ArrayList<>(puzzles.length);
        Map<Integer, Integer> m = new HashMap<>();
        for (String w : words) {
            int freq = getBitmask(w);
            m.put(freq, m.getOrDefault(freq, 0) + 1);
        }
        for (String p : puzzles) {
            int tmpResult = 0;
            List<Integer> s = getBinarySubset(p);
            for (int i : s) {
                tmpResult += m.getOrDefault(i, 0);
            }
            result.add(tmpResult);
        }
        return result;
    }

    public List<Integer> getBinarySubset(String puzzle) {
        List<Integer> result = new ArrayList<>();
        result.add(getBitmask(puzzle));
        String firstLetter = puzzle.substring(0, 1);
        String lastSix = puzzle.substring(1, 7);
        int firstLetterBitmask = getBitmask(firstLetter);
        int lastSixBitmask = getBitmask(lastSix);
        int subset = lastSixBitmask;
        while (subset != 0) {
            subset = (subset - 1) & lastSixBitmask;
            result.add((subset | firstLetterBitmask));
        }
        return result;
    }

    public boolean isBinarySubset(int puzzle, int word) {
        int nor = puzzle | word;
        return nor - puzzle == 0;
    }

    public int getBitmask(String s) {
        int result = 0;
        for (char a : s.toCharArray()) {
            result |= (0x01 << (a - 'a'));
        }
        return result;
    }


    // LC340
    // 至多包含 K 个不同字符的最长子串长度
    public int kDistinctSubArrayMaxLen(String s, int k) {
        int n = s.length();
        int result = 0;
        Map<Character, Integer> m = new HashMap<>();
        for (int left = 0, right = 0; right < n; right++) {
            m.put(s.charAt(right), m.getOrDefault(s.charAt(right), 0) + 1);
            while (left < n && m.size() > k) {
                if (m.containsKey(s.charAt(left))) {
                    m.put(s.charAt(left), m.get(s.charAt(left)) - 1);
                    if (m.get(s.charAt(left)) == 0) {
                        m.remove(s.charAt(left));
                    }
                }
                left++;
            }
            result = Math.max(result, right - left + 1);
        }
        return result;
    }


    // LC325
    // 给定一个数组 nums 和一个目标值 k，找到和等于 k 的最长子数组长度。如果不存在任意一个符合要求的子数组，则返回 0。
    public int sumKSubArrayMaxLen(int[] arr, int target) {
        int n = arr.length;
        int[] prefix = new int[n];
        prefix[0] = arr[0];
        int maxLen = 0;
        // Map, key为前缀和的值, 如果有相同值, 则存最左侧一个, 以使得子数组长度最大
        // value为该值的坐标, 用putIfAbsent方法
        Map<Integer, Integer> m = new HashMap<>();

        // 初始化前缀和数组
        for (int i = 1; i < n; i++) {
            prefix[i] = prefix[i - 1] + arr[i];
        }

        m.put(0, -1);

        for (int i = 0; i < n; i++) {
            int right = prefix[i];
            m.putIfAbsent(right, i);
            // if(right-left == target) -> left = right - target
            if (m.containsKey(right - target)) {
                maxLen = Math.max(maxLen, i - m.get(right - target));
            }
        }
        return maxLen;
    }
}