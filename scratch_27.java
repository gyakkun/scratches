import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.intervalIntersection(new int[][]{{0, 2}, {5, 10}, {13, 23}, {24, 25}},
                new int[][]{{1, 5}, {8, 12}, {15, 24}, {25, 26}}));
    }

    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> result = new ArrayList<>(nums.length / 2);
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if(nums[ Math.abs(nums[i]-1) ]<0){
                result.add(Math.abs(nums[i] - 1));
            } else {
                nums[Math.abs(nums[i] - 1)] *= -1;
            }
        }
        return result;
    }

    public int findDuplicate(int[] nums) {
        Set<Integer> s = new HashSet<>();
        for (int i : nums) {
            if (s.contains(i)) return i;
            s.add(i);
        }
        return -1;
    }


    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    public int missingNumber(int[] nums) {
        int n = nums.length;
        int missing = n;
        for (int i = 0; i < n; i++) {
            missing ^= i ^ nums[i];
        }
        return missing;
    }

    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int n = intervals.length;
        Arrays.sort(intervals, Comparator.comparingInt((a) -> a[1]));
        int right = intervals[0][1];
        int selected = 1;
        for (int i = 1; i < n; i++) {
            if (intervals[i][0] >= right) {
                selected++;
                right = intervals[i][1];
            }
        }
        return n - selected;
    }

    public int[][] intervalIntersection(int[][] A, int[][] B) {
        List<int[]> ans = new ArrayList();
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            // Let's check if A[i] intersects B[j].
            // lo - the startpoint of the intersection
            // hi - the endpoint of the intersection
            int lo = Math.max(A[i][0], B[j][0]);
            int hi = Math.min(A[i][1], B[j][1]);
            // 如果存在交集
            if (lo <= hi)
                ans.add(new int[]{lo, hi});
            // Remove the interval with the smallest endpoint
            // 如果上面已经根据当前最小右端点的位置删除了某个区间, 则向右移
            // 动态更新A,B中的最小右端点所在的坐标, 确保A[i]和B[j]中存在当前最小右端点。
            if (A[i][1] < B[j][1])
                i++;
            else
                j++;
        }
        return ans.toArray(new int[ans.size()][]);
    }

    public int[][] insertInterval(int[][] intervals, int[] newInterval) {
        boolean isPlaced = false;
        int left = newInterval[0];
        int right = newInterval[1];
        List<int[]> result = new ArrayList<>();
        for (int[] i : intervals) {
            if (i[0] > right) {
                if (!isPlaced) {
                    result.add(new int[]{left, right});
                    isPlaced = true;
                }
                result.add(i);
            } else if (i[1] < left) {
                result.add(i);
            } else {
                left = Math.min(left, i[0]);
                right = Math.min(right, i[1]);
            }
        }
        if (!isPlaced) {
            result.add(new int[]{left, right});
        }
        return result.toArray(new int[result.size()][]);
    }

    public int[] countBits(int num) {
        int[] result = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            result[i] = Integer.bitCount(i);
        }
        return result;
    }
}