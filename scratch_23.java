import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.subarraysWithKDistinct(new int[]{1, 2, 1, 3, 4}, 3));
    }


    // LC448
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            nums[(nums[i] - 1) % n] += n;
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                result.add(i + 1);
            }
        }
        return result;
    }

    // LC1052
    public int maxSatisfied(int[] customers, int[] grumpy, int x) {
        int time = customers.length;
        int totalSatisfyTime = 0;
        int maxBearTime = Integer.MIN_VALUE;
        int tmpTotal = 0;
        for (int i = 0; i < time; i++) {
            if (grumpy[i] == 0) {
                totalSatisfyTime += customers[i];
            }
            if (i < x) {
                tmpTotal += grumpy[i] * customers[i];
            } else if (i < time) {
                tmpTotal -= grumpy[i - x] * customers[i - x];
                tmpTotal += grumpy[i] * customers[i];
            }
            if (tmpTotal > maxBearTime) {
                maxBearTime = tmpTotal;
            }
        }
        return totalSatisfyTime + maxBearTime;
    }

    // LC567
    public boolean checkInclusion(String s1, String s2) {
        int len = s1.length();
        int[] s1Freq = getCharFreq(s1);
        int[] tmpFreq;
        for (int i = 0; i <= s2.length() - len; i++) {
            String tmp = s2.substring(i, i + len);
            tmpFreq = getCharFreq(tmp);
            if (checkSameCharFreq(s1Freq, tmpFreq)) {
                return true;
            }
        }

        return false;
    }

    public int[] getCharFreq(String s) {
        int i1[] = new int[26];
        for (char a : s.toCharArray()) {
            i1[a - 'a']++;
        }
        return i1;
    }

    public boolean checkSameCharFreq(int[] i1, int[] i2) {
        for (int i = 0; i < 26; i++) {
            if (i1[i] != i2[i]) return false;
        }
        return true;
    }

    // LC992
    public int subarraysWithKDistinct(int[] A, int K) {
        int n = A.length;
        int leftK = 0, leftKm1 = 0, right = 0;
        int ctrK = 0, ctrKm1 = 0;
        int result = 0;
        // m 记录最后一次出现的位置
        Map<Integer, Integer> mK = new HashMap<>(n + 1);
        Map<Integer, Integer> mKm1 = new HashMap<>(n + 1);
        int tmpK, tmpKm1;

        while (right < n) {
            mK.putIfAbsent(A[right], 0);
            tmpK = mK.get(A[right]);
            if (tmpK == 0) {
                ctrK++;
            }
            mK.put(A[right], tmpK + 1);

            mKm1.putIfAbsent(A[right], 0);
            tmpKm1 = mKm1.get(A[right]);
            if (tmpKm1 == 0) {
                ctrKm1++;
            }
            mKm1.put(A[right], tmpKm1 + 1);

            while (ctrK > K) {
                tmpK = mK.get(A[leftK]);
                mK.put(A[leftK], tmpK - 1);
                if (tmpK - 1 == 0) {
                    ctrK--;
                }
                leftK++;
            }

            while (ctrKm1 > (K - 1)) {
                tmpKm1 = mKm1.get(A[leftKm1]);
                mKm1.put(A[leftKm1], tmpKm1 - 1);
                if (tmpKm1 - 1 == 0) {
                    ctrKm1--;
                }
                leftKm1++;
            }
            result += leftKm1 - leftK;
            right++;
        }

        return result;
    }

    // LC766
    public boolean isToeplitzMatrix(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) return false;
            }
        }

        return true;
    }

    // LC978
    public int maxTurbulenceSize(int[] arr) {
        int left = 0, right = 2;
        int max = 0;
        int n = arr.length;
        int currentLen = 0;
        boolean initFlag = true;
        if (n == 0) return 0;
        if (n < 3) return 1;
        int rM1, rM2;
        while (right < n) {
            rM1 = arr[right] - arr[right - 1];
            rM2 = arr[right - 1] - arr[right - 2];
            if ((rM1 > 0 && rM2 < 0) || (rM1 < 0 && rM2 > 0)) {
                if (initFlag) {
                    currentLen = 3;
                    initFlag = false;
                } else {
                    currentLen++;
                }
                max = Math.max(max, currentLen);
                right++;
            } else if ((rM1 == 0 && rM2 != 0) || (rM1 != 0 && rM2 == 0)) {
                max = Math.max(max, 2);
                left = right - 1;
                right++;
                initFlag = true;
            } else {
                max = Math.max(max, currentLen);
                left = right - 1;
                right++;
                initFlag = true;
            }
        }

        return Math.max(1, max);
    }
}

class KthLargest {

    private int k;
    private PriorityQueue<Integer> pq;

    public KthLargest(int k, int[] nums) {
        this.k = k;
//        this.pq = new PriorityQueue<>(Comparator.naturalOrder()); // 最小堆
        this.pq = new PriorityQueue<>(); // 默认最小堆 (堆顶最小)
        for (int i : nums) {
            pq.add(i);
        }
        while (pq.size() > k) {
            pq.poll();
        }
    }

    public int add(int val) {
        pq.offer(val);
        while (pq.size() > k) {
            pq.poll();
        }
        return pq.peek();
    }
}