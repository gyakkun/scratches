import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
//        System.err.println(s.countRangeSum(new int[]{-2, 5, -1}, -2, 2));
        int[] ia = new int[]{9, 4, 2, 10, 100};
        System.err.println(s.findKthLargest(new int[]{3, 2, 1, 5, 6, 4}, 2));
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        int n = nums.length;
        for (int i = 0; i < k; i++) {
            pq.add(nums[i]);
        }
        for (int i = k; i < n; i++) {
            if (nums[i] < pq.peek()) continue;
            else {
                pq.poll();
                pq.add(nums[i]);
            }
        }
        return pq.peek();
    }

    public int[] topKFrequent(int[] nums, int k) {
        int[] result = new int[k];
        Map<Integer, Integer> m = new HashMap<>();
        List<int[]> l = new ArrayList<>();
        for (int i : nums) {
            m.put(i, m.getOrDefault(i, 0) + 1);
        }
        for (int i : m.keySet()) {
            l.add(new int[]{i, m.get(i)});
        }
        l.sort((o1, o2) -> (m.get(o2) - m.get(o1)));
        for (int i = 0; i < k; i++) {
            result[i] = l.get(i)[0];
        }
        return result;
    }

    public String frequencySort(String s) {
        Map<Character, Integer> m = new HashMap<>();
        List<int[]> l = new ArrayList<>();
        for (char c : s.toCharArray()) {
            m.put(c, m.getOrDefault(c, 0) + 1);
        }
        for (char c : m.keySet()) {
            l.add(new int[]{c - 'a', m.get(c)});
        }
        l.sort(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }
        });
        StringBuffer sb = new StringBuffer();
        for (int[] ia : l) {
            for (int i = 0; i < ia[1]; i++) {
                sb.append((char) (ia[0] + 'a'));
            }
        }
        return sb.toString();
    }

    // LC658
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                int diff = Math.abs(o1 - x) - Math.abs(o2 - x);
                return diff == 0 ? o1 - o2 : diff;
            }
        });
        List<Integer> result = new LinkedList<>();
        for (int i : arr) {
            minHeap.add(i);
        }
        while (k-- != 0) {
            result.add(minHeap.poll());
        }
        Collections.sort(result);
        return result;
    }

    // https://www.geeksforgeeks.org/maximum-distinct-elements-removing-k-elements/
    public int maxDistinctNum(int arr[], int k) {
        int n = arr.length;
        int result = 0;
        Map<Integer, Integer> m = new HashMap<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1])); // min-heap
        for (int i : arr) {
            m.put(i, m.getOrDefault(i, 0) + 1);
        }
        for (int i : m.keySet()) {
            if (m.get(i) == 1)
                result++;
            else
                pq.add(new int[]{i, m.get(i)});
        }
        while (k > 0 && !pq.isEmpty()) {
            int[] i = pq.poll();
            if (i[1] - 1 == 1) {
                result++;
            } else {
                pq.add(new int[]{i[0], i[1] - 1});
            }
            k--;
        }
        return result - k;
    }

    // Merge sort
    public static List<Integer> sort(List<Integer> list) {
        if (list.size() <= 1)
            return list;
        int mid = list.size() / 2;
        List<Integer> left = new LinkedList<>();  // 定义左侧List
        List<Integer> right = new LinkedList<>(); // 定义右侧List
        // 以下兩個循環把 list 分為左右兩個 List
        for (int i = 0; i < mid; i++)
            left.add(list.get(i));
        for (int j = mid; j < list.size(); j++)
            right.add(list.get(j));
        left = sort(left);
        right = sort(right);
        return merge(left, right);
    }

    private static List<Integer> merge(List<Integer> left, List<Integer> right) {
        List<Integer> temp = new LinkedList<>();
        while (left.size() > 0 && right.size() > 0) {
            if (left.get(0) <= right.get(0)) {
                temp.add(left.get(0));
                left.remove(0);
            } else {
                temp.add(right.get(0));
                right.remove(0);
            }
        }
        if (left.size() > 0) {
            for (int i = 0; i < left.size(); i++)
                temp.add(left.get(i));
        }
        if (right.size() > 0) {
            for (int i = 0; i < right.size(); i++)
                temp.add(right.get(i));
        }
        return temp;
    }

    // LC327 LC solution

    public int countRangeSumLC(int[] nums, int lower, int upper) {
        long s = 0;
        long[] sum = new long[nums.length + 1];
        for (int i = 0; i < nums.length; ++i) {
            s += nums[i];
            sum[i + 1] = s;
        }
        return countRangeSumRecursive(sum, lower, upper, 0, sum.length - 1);
    }

    public int countRangeSumRecursive(long[] sum, int lower, int upper, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = countRangeSumRecursive(sum, lower, upper, left, mid);
            int n2 = countRangeSumRecursive(sum, lower, upper, mid + 1, right);
            int ret = n1 + n2;

            // 首先统计下标对的数量 相当于在归并排序前做hook
            int i = left;
            int l = mid + 1;
            int r = mid + 1;
            while (i <= mid) {
                while (l <= right && sum[l] - sum[i] < lower) {
                    l++;
                }
                while (r <= right && sum[r] - sum[i] <= upper) {
                    r++;
                }
                ret += r - l;
                i++;
            }

            // 随后合并两个排序数组
            int[] sorted = new int[right - left + 1];
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = (int) sum[p2++];
                } else if (p2 > right) {
                    sorted[p++] = (int) sum[p1++];
                } else {
                    if (sum[p1] < sum[p2]) {
                        sorted[p++] = (int) sum[p1++];
                    } else {
                        sorted[p++] = (int) sum[p2++];
                    }
                }
            }
            for (int j = 0; j < sorted.length; j++) {
                sum[left + j] = sorted[j];
            }
            return ret;
        }
    }


    // LC327
    public int countRangeSum(int[] nums, int lower, int upper) {
        int n = nums.length;
        long[] prefix = new long[nums.length + 1];
        prefix[0] = 0;
        int result = 0;
        for (int i = 1; i < n + 1; i++) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }

        for (int i = 0; i < n + 1; i++) {
            for (int j = i + 1; j < n + 1; j++) {
                long tmpSum = prefix[j] - prefix[i];
                if (tmpSum <= upper && tmpSum >= lower) {
                    result++;
                }
            }
        }
        return result;
    }
}