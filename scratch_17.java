import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        int[] a = new int[]{1, 2, 6, 3, 0, 7, 1, 7, 1, 9, 7, 5, 6, 6, 4, 4, 0, 0, 6, 3};
//        System.err.println(addToArrayForm(a, 516));
        System.err.println(test());

    }

    public static int test() {
        int[] a = new int[]{1, 2, 3, 4, 5, 9};
        List<Integer> l = Arrays.stream(a).boxed().collect(Collectors.toList());
        return search(l, 9);
    }

    // LC881
    public int numRescueBoats2(int[] people, int limit) {
        Arrays.sort(people);
        int i = 0, j = people.length - 1;
        int ans = 0;

        while (i <= j) {
            ans++;
            if (people[i] + people[j] <= limit)
                i++;
            j--;
        }

        return ans;
    }

    // LC881
    public static int numRescueBoats(int[] people, int limit) {
        Arrays.sort(people);
        List<Integer> l = Arrays.stream(people).boxed().collect(Collectors.toList());
        int tmpIdx = -1;
        int result = 0;
        while (!l.isEmpty()) {
            tmpIdx = search(l, limit - l.get(0));
            if (tmpIdx != -1 && tmpIdx != 0) {
                l.remove(tmpIdx);
                if (!l.isEmpty()) {
                    l.remove(0);
                }
            } else {
                l.remove(0);
            }
            result++;
        }
        return result;
    }


    public static int search(List<Integer> li, int lessOrEqualThan) {
        int l = 0;
        int r = li.size() - 1;
        int mid = -1;
        while (l < r) {
            mid = (l + r + 1) >> 1;
            if (li.get(mid) <= lessOrEqualThan) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        if (li.get(l) > lessOrEqualThan) return -1;
        return l;
    }

    public static List<Integer> addToArrayForm(int[] A, int K) {

        long a = 0;
        for (int i = 0; i < A.length; i++) {
            a = a * 10 + A[i];
        }
        long k = K;
        a = a + k;
        List<Integer> result = new ArrayList<>();
        if (a == 0l) {
            result = new ArrayList<>();
            result.add(0);
            return result;
        }
        while (a != 0) {
            result.add((int) (a % 10));
            a = a / 10;
        }
        Collections.reverse(result);
        return result;
    }

    public int[] smallerNumbersThanCurrent(int[] nums) {
        final int n = nums.length;
        List<Integer> l = Arrays.stream(nums).boxed().collect(Collectors.toList());
        ArrayList<Integer> a = new ArrayList<>(l);
//        a.sort(Comparator.naturalOrder());
        Collections.sort(a);
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = a.indexOf(nums[i]);
        }
        return result;
    }

    // LC926
    public static int minFlipsMonoIncr(String S) {
        // 对于一个位置 我们希望知道自己及其前面有多少个1, 自己及后面有多少个0
        int len = S.length();
        ArrayList<Integer> howManyOnesBefore = new ArrayList<>(len);
        ArrayList<Integer> howManyZerosAfter = new ArrayList<>(len);
        ArrayList<Integer> flipTimes = new ArrayList<>(len);
        System.err.println("size" + howManyZerosAfter.size());
        if (S.charAt(0) == '1') {
            howManyOnesBefore.add(0, 1);
        } else {
            howManyOnesBefore.add(0, 0);
        }
        if (S.charAt(len - 1) == '0') {
//            howManyZerosAfter.add(len - 1, 1);
            howManyZerosAfter.add(0, 1);
        } else {
            howManyZerosAfter.add(0, 0);

        }
        for (int i = 1; i < len; i++) {
            // 顺序
            if (S.charAt(i) == '1') {
                howManyOnesBefore.add(i, howManyOnesBefore.get(i - 1) + 1);
            } else {
                howManyOnesBefore.add(i, howManyOnesBefore.get(i - 1));
            }
            // 逆序
            if (S.charAt(len - 1 - i) == '0') {
                howManyZerosAfter.add(i, howManyZerosAfter.get(i - 1) + 1);
            } else {
                howManyZerosAfter.add(i, howManyZerosAfter.get(i - 1));
            }
        }
        Collections.reverse(howManyZerosAfter);
        for (int i = 0; i < len; i++) {
            flipTimes.add(i, howManyOnesBefore.get(i) - 1 + howManyZerosAfter.get(i));
        }
        return Collections.min(flipTimes);
    }

    // LC1653
    public static int minimumDeletions(String s) {
        int len = s.length();
        // 包括自己有多少个A
        int[] howManyA = new int[s.length()];
        int[] howManyDelete = new int[s.length()];
        int min = Integer.MAX_VALUE;
        howManyA[0] = s.charAt(0) == 'a' ? 1 : 0;
        for (int i = 1; i < len; i++) {
            howManyA[i] = howManyA[i - 1] + (s.charAt(i) == 'a' ? 1 : 0);
        }
//        if (howManyA[len - 1] == 0 || howManyA[len - 1] == len) {
//            return 0;
//        }
        // 删除多少个使得右边全部是b (不删除自己)
        for (int i = 0; i < len; i++) {
            howManyDelete[i] = (i + 1) - howManyA[i] + (howManyA[len - 1] - howManyA[i]);
            min = Math.min(min, howManyDelete[i]);
        }
        min = Math.min(min, len - howManyA[len - 1]);
        min = Math.min(min, howManyA[len - 1]);
        return min;
    }
}

class Solution {
    int n;
    Set<Integer> s;
    Random r;

    public Solution(int N, int[] blacklist) {
        this.n = N;
        this.r = new Random();
        this.s = new HashSet<>();
        for (int i : blacklist) {
            s.add(i);
        }

    }

    public int pick() {
        int tmpResult = r.nextInt(n);
        while (s.contains(tmpResult)) {
            tmpResult = r.nextInt(n);
        }
        return tmpResult;
    }


}