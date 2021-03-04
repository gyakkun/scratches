import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();

        System.err.println(s.checkPossibility(new int[]{4, 2, 3}));

    }

    public boolean checkPossibility(int[] nums) {
        int n = nums.length;
        int ctr = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] > nums[i + 1]) {
                ctr++;
                if (ctr == 2) {
                    return false;
                }
                if (i > 0 && nums[i + 1] < nums[i - 1]) {
                    nums[i + 1] = nums[i];
                }
            }
        }
        return true;
    }

    //    1 <= s.length <= 10^5
    //    1 <= maxLetters <= 26
    //    1 <= minSize <= maxSize <= min(26, s.length)
    //    s只包含小写英文字母。

    //    Check out the constraints, (maxSize <=26).
    //    This means you can explore all substrings in O(n * 26).
    //    Find the Maximum Number of Occurrences of a Substring with bruteforce.

    public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {


        return 0;
    }

    public int search(String message, String pattern) {
        Map<Integer, Map<Character, Integer>> dfa = getDfa(pattern);
        char[] m = message.toCharArray();
        int j = 0;
        for (int i = 0; i < m.length; i++) {
            j = dfa.get(j).getOrDefault(m[i], 0);
            if (j == pattern.length()) {
                return i - pattern.length() + 1;
            }
        }
        return -1;
    }

    public int searchWithDfa(String message, int patternLen, Map<Integer, Map<Character, Integer>> dfa) {
        int mLen = message.length();
        int j = 0;
        for (int i = 0; i < mLen; i++) {
            j = dfa.get(j).getOrDefault(message.charAt(i), 0);
            if (j == patternLen) {
                return i - patternLen + 1;
            }
        }
        return -1;
    }


    private Map<Integer, Map<Character, Integer>> getDfa(String pattern) {
        char[] pat = pattern.toCharArray();
        Set<Character> s = new HashSet<>();
        for (char c : pat) {
            s.add(c);
        }
        Map<Integer, Map<Character, Integer>> dfa = new HashMap<>();
        dfa.put(0, new HashMap<>());
        dfa.get(0).put(pat[0], 1);
        int x = 0;
        for (int i = 1; i < pat.length; i++) {
            dfa.putIfAbsent(i, new HashMap<>());
            dfa.get(i).put(pat[i], i + 1);
            for (char c : s) {
                if (c != pat[i]) {
                    dfa.get(i).put(c, dfa.get(x).getOrDefault(c, 0));
                }
            }
            x = dfa.get(x).getOrDefault(pat[i], 0);
        }
        return dfa;
    }


    List<TreeNode> res = new ArrayList<>();

    public List<TreeNode> delNodes(TreeNode root, int[] toDelete) {
        Set<Integer> hashset = new HashSet<>();
        for (int i : toDelete) hashset.add(i);
        if (!hashset.contains(root.val)) res.add(root);
        DFS(root, hashset);
        return res;
    }

    public TreeNode DFS(TreeNode root, Set<Integer> hashset) {
        if (root == null) return null;
        root.left = DFS(root.left, hashset);
        root.right = DFS(root.right, hashset);
        if (hashset.contains(root.val)) {
            if (root.left != null) res.add(root.left);
            if (root.right != null) res.add(root.right);
            root = null;
        }
        return root;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }


    public int equalSubstring(String s, String t, int maxCost) {
        int len = s.length();
        int[] eachDistance = new int[len];
        int left = -1, right = -1;
        for (int i = 0; i < len; i++) {
            eachDistance[i] = charDistance(s.charAt(i), t.charAt(i));
        }
        for (int i = 0; i < len; i++) {
            if (eachDistance[i] <= maxCost) {
                left = right = i;
                break;
            }
        }
        if (left == -1) return 0;
        int maxLen = 1;
        int currentCost = eachDistance[left];

        while (right + 1 < len) {
            if (currentCost + eachDistance[right + 1] <= maxCost) {
                currentCost += eachDistance[right + 1];
                right++;
                maxLen = Math.max(right - left + 1, maxLen);
            } else {
                currentCost -= eachDistance[left];
                left++;
            }
            if (left == right) {
                right++;
                left++;
                if (right < len) {
                    currentCost = eachDistance[right];
                } else {
                    break;
                }
            }
        }
        return maxLen;
    }

    private int charDistance(char i, char j) {
        return Math.abs(i - j);
    }

    public int countDigitOne(int n) {
        int[] maxOfEachDigitNum = new int[10];
        int[] cumulative = new int[10];
        cumulative[1] = maxOfEachDigitNum[1] = 1;
        for (int i = 2; i < 10; i++) {
            maxOfEachDigitNum[i] = tenPower(i - 1) + 9 * cumulative[i - 1];
            cumulative[i] = cumulative[i - 1] + maxOfEachDigitNum[i];
        }
        int numOfDigit = howManyDigits(n);
        int idx = tenPower(numOfDigit - 1) - 1; // 通项公式的第几项
        int result = cumulative[numOfDigit - 1]; // 通项公式当前的结果
        int currentAddableTenPowerPower = howManyDigits(n - idx) - 1; // 当前可以加的10的幂的幂次, 比如2333, 开始时是999, 可以加10^3->1999
        // 它可以由 numOfDigit(n-idx)-1得到

        while (n - idx >= 10) {
            int nextStart = idx + 1;
            int nextEnd = idx + tenPower(currentAddableTenPowerPower);
            idx = nextEnd;
            int howManyOnesInNextStart = countOneByString(nextStart);
            result += howManyOnesInNextStart * tenPower(currentAddableTenPowerPower);
            result += cumulative[currentAddableTenPowerPower];
            currentAddableTenPowerPower = howManyDigits(n - idx) - 1;
        }

        if (idx == n) return result;

        while (++idx <= n) {
            result += countOneByString(idx);
        }
        return result;
    }

    private int tenPower(int n) {
        int result = 1;
        while (n-- != 0) {
            result *= 10;
        }
        return result;
    }

    private int countOneByString(int n) {
        char[] num = String.valueOf(n).toCharArray();
        int result = 0;
        for (int i = 0; i < num.length; i++) {
            if (num[i] == '1') {
                result++;
            }
        }
        return result;
    }

    private int howManyDigits(int n) {
        if (n == 0) return 1;
        int result = 0;
        while (n != 0) {
            n /= 10;
            result++;
        }
        return result;
    }


    /*
     * O(nlog(max-min))
     */
    public double findMaxAverage(int[] nums, int k) {
        if (nums.length < k) return -1.0;
        double maxVal = Double.MIN_VALUE;
        double minVal = Double.MAX_VALUE;
        // 寻找最值
        for (double n : nums) {
            maxVal = Math.max(maxVal, n);
            minVal = Math.min(minVal, n);
        }

        double err = 1e-5;
        // 用二分法查找平均值x，对于每个x去检查是否有子数组的平均值大于x
        while (maxVal - minVal > err) {
            double mid = (maxVal + minVal) / 2.0;
            if (check(nums, mid, k))
                minVal = mid;
            else
                maxVal = mid;
        }
        return maxVal;
    }


    private boolean check(int[] nums, double key, int k) {
        double sum = 0, prev = 0, minSum = 0;
        for (int i = 0; i < k; ++i) {
            sum += nums[i] - key;
        }
        if (sum >= 0) {
            return true;
        }
        for (int i = k; i < nums.length; ++i) {
            sum += nums[i] - key;
            prev += nums[i - k] - key;
            minSum = Math.min(prev, minSum);
            if (sum >= minSum) {
                return true;
            }
        }
        return false;
    }

}


