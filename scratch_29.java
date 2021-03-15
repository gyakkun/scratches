import java.util.*;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
//        System.err.println(s.countRangeSum(new int[]{-2, 5, -1}, -2, 2));
        int[] ia = new int[]{9, 4, 2, 10, 100};

        int[] i = new int[]{1, 5};
        int[] j = new int[]{2, 3, 5};
        int[] k = new int[]{4, 6};
        List<Integer> ii = new LinkedList<>(Arrays.stream(i).boxed().collect(Collectors.toList()));
        List<Integer> jj = new LinkedList<>(Arrays.stream(j).boxed().collect(Collectors.toList()));
        List<Integer> kk = new LinkedList<>(Arrays.stream(k).boxed().collect(Collectors.toList()));
        List<List<Integer>> l = new LinkedList<>();
        l.add(ii);
        l.add(jj);
        l.add(kk);
        s.smallestRange(l);

    }


    // LC632, hard
    public int[] smallestRange(List<List<Integer>> nums) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        int[] result = new int[2];
        PriorityQueue<List<Integer>> pq = new PriorityQueue<>(new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                return o1.get(0) - o2.get(0);
            }
        });

        for (List<Integer> num : nums) {
            pq.add(num);
            max = Math.max(num.get(0), max);
            min = Math.min(num.get(0), min);
        }
        result[0] = min;
        result[1] = max;

        // 1       5
        //   2 3   5
        //       4   6
        //         5
        //
        while (true) {
            List<Integer> tmp = pq.poll();
            if (tmp.size() == 1) break;
            tmp.remove(0);
            pq.offer(tmp);

            min = pq.peek().get(0);
            max = Math.max(max, tmp.get(0));
            if (max - min < result[1] - result[0]) {
                result[0] = min;
                result[1] = max;
            }
        }
        return result;
    }

    // LC474 m zero , n one
    public int findMaxForm(String[] strs, int m, int n) {
        int[] zeroCounts = new int[strs.length];
        int[] oneCounts = new int[strs.length];
        for (int i = 0; i < strs.length; i++) {
            for (char c : strs[i].toCharArray()) {
                if (c == '0') {
                    zeroCounts[i]++;
                }
            }
            oneCounts[i] = strs[i].length() - zeroCounts[i];
        }



        return -1;
    }

    // 经典背包
//    public int findMaxValue(int[] weight, int[] value, int maxWeight){
//        // dp[i][j] 表示将前i件物品装进限重为j的背包中能获取的最大价值
//        // dp[i][j] = Math.max(dp[i-1][j],dp[i-1][j-weight]+value[i])
//
//
//    }


    //  Definition for singly-linked list.
    public static class ListNode {
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

    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -o));
        for (int[] i : matrix) {
            for (int j : i) {
                if (pq.size() < k) {
                    pq.offer(j);
                } else {
                    if (j > pq.peek()) {
                        break;
                    } else {
                        pq.poll();
                        pq.offer(j);
                    }
                }
            }
        }
        return pq.peek();
    }

    public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length - 1);
    }

    public ListNode merge(ListNode[] lists, int l, int r) {
        if (l == r) {
            return lists[l];
        }
        if (l > r) {
            return null;
        }
        int mid = (l + r) >> 1;
        return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    public static ListNode mergeTwoLists(ListNode first, ListNode second) {
        ListNode newHead;
        ListNode firstHead = first;
        ListNode secondHead = second;
        ListNode head;

        if (first != null && second != null) {
            newHead = first.val < second.val ? first : second;
        } else if (first == null) {
            return second;
        } else if (second == null) {
            return first;
        } else {
            return null;
        }
        head = newHead;

        while (firstHead != null && secondHead != null) {
            if (firstHead.val < secondHead.val) {
                ListNode tmpNext = firstHead.next;
                head.next = firstHead;
                head = head.next;
                firstHead = tmpNext;
            } else {
                ListNode tmpNext = secondHead.next;
                head.next = secondHead;
                head = head.next;
                secondHead = tmpNext;
            }
        }

        if (firstHead == null) {
            while (secondHead != null) {
                ListNode tmpNext = secondHead.next;
                head.next = secondHead;
                head = head.next;
                secondHead = tmpNext;
            }
        }
        if (secondHead == null) {
            while (firstHead != null) {
                ListNode tmpNext = firstHead.next;
                head.next = firstHead;
                head = head.next;
                firstHead = tmpNext;
            }
        }

        // first  1 4 5
        // second 1 3 4
        //
        // target 1 1 3 4 4 5


        return newHead;
    }

    public int[][] kClosest(int[][] points, int k) {
        int[][] result = new int[k][2];
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }
        });

        for (int i = 0; i < k; i++) {
            pq.add(new int[]{i, points[i][0] * points[i][0] + points[i][1] * points[i][1]});
        }

        for (int i = k; i < points.length; i++) {
            int tmpDistance = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            if (tmpDistance > pq.peek()[1]) continue;
            else {
                pq.poll();
                pq.add(new int[]{i, tmpDistance});
            }
        }

        int i = 0;
        while (!pq.isEmpty()) {
            result[i++] = points[pq.poll()[0]];
        }
        return result;

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