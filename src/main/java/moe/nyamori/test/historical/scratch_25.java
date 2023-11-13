package moe.nyamori.test.historical;

import java.util.*;

class scratch_25 {
    public static void main(String[] args) {
        scratch_25 s = new scratch_25();
        System.err.println(s.longestOnes(new int[]{1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0}, 2));
    }

    // LC1438
    public int longestSubarrayAVL(int[] nums, int limit) {
        TreeMap<Integer, Integer> map = new TreeMap<Integer, Integer>();
        int n = nums.length;
        int left = 0, right = 0;
        int ret = 0;
        while (right < n) {
            map.put(nums[right], map.getOrDefault(nums[right], 0) + 1);
            while (map.lastKey() - map.firstKey() > limit) {
                map.put(nums[left], map.get(nums[left]) - 1);
                if (map.get(nums[left]) == 0) {
                    map.remove(nums[left]);
                }
                left++;
            }
            ret = Math.max(ret, right - left + 1);
            right++;
        }
        return ret;
    }

    // LC1438
    public int longestSubarray(int[] nums, int limit) {
        int n = nums.length;
        int left = 0, right = 0;
        int innerMax = Integer.MIN_VALUE, innerMin = Integer.MAX_VALUE;
        int result = 0;
        while (right < n) {
            boolean isMaxUpdate = false, isMinUpdate = false;

            if (nums[right] > innerMax) {
                innerMax = nums[right];
                isMaxUpdate = true;
            }

            if (nums[right] < innerMin) {
                innerMin = nums[right];
                isMinUpdate = true;
            }

            if (isMaxUpdate) {
                // 检查更新后的最大最小值的差值是否小于等于limit

                if (innerMax - innerMin > limit) {
                    // 如果超出, 则找到一个left, 使得新的[left,right]中的值都大于等于innerMax-limit
                    // n >= innerMax-limit
                    // n+limit >= innerMax
                    // 也就是新的最小值也要>=innerMax-limit
                    // 从右往左遍历, 若找到则更新新的innerMin和left
                    int formerLeft = left;
                    innerMin = innerMax;
                    for (int i = right; i >= formerLeft; i--) {
                        if (innerMax - nums[i] <= limit) {
                            left = i;
                            innerMin = Math.min(innerMin, nums[i]);
                        } else {
                            break;
                        }
                    }
                }
            }

            if (isMinUpdate) {
                if (innerMax - innerMin > limit) {
                    int formerLeft = left;
                    innerMax = innerMin;
                    for (int i = right; i >= formerLeft; i--) {
                        if (nums[i] - innerMin <= limit) {
                            left = i;
                            innerMax = Math.max(innerMax, nums[i]);
                        } else {
                            break;
                        }
                    }
                }
            }
            result = Math.max(result, right - left + 1);
            right++;
        }
        return result;
    }

    // LC697
    public int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> occurCtr = new HashMap<>();
        Set<Integer> maxOccurSet = new HashSet<>();
        int n = nums.length;
        int maxOccur = 0;
        int shortestLen = Integer.MAX_VALUE;
        for (int i : nums) {
            occurCtr.putIfAbsent(i, 0);
            occurCtr.put(i, occurCtr.get(i) + 1);
        }
        for (int i : occurCtr.keySet()) {
            int tmp = occurCtr.get(i);
            if (maxOccur < tmp) {
                maxOccur = tmp;
                maxOccurSet.clear();
                maxOccurSet.add(i);
            } else if (maxOccur == tmp) {
                maxOccurSet.add(i);
            }
        }
        for (int i : maxOccurSet) {
            int left = 0, right = n - 1;
            for (int j = 0; j < n; j++) {
                if (nums[j] == i) {
                    left = j;
                    break;
                }
            }
            for (int j = n - 1; j >= 0; j--) {
                if (nums[j] == i) {
                    right = j;
                    break;
                }
            }
            int tmpLen = right - left + 1;
            shortestLen = Math.min(tmpLen, shortestLen);
        }
        return shortestLen;
    }

    // LC867
    public int[][] transpose(int[][] matrix) {
        int[][] result = new int[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    // LC1004
    public int longestOnes(int[] A, int K) {
        int n = A.length;
        int max = 0;
        int[] prefix = new int[n + 1];
//        prefix[0] = A[0];
        for (int i = 1; i <= n; i++) {
            prefix[i] = (1 - A[i - 1]) + prefix[i - 1];
        }
        for (int i = 0; i < n; i++) {
            // 对于每个A[i]==1 (原来是0), 其对应的P[i]期望找到一个P[j],使得P[i]-P[j]=K
            int left = binarySearch(prefix, prefix[i + 1] - K);
            if (left != -1) {
                max = Math.max(max, i - left + 1);
            }
        }
        return max;

    }

    public int binarySearch(int[] arr, int target) {
        int n = arr.length;
        if (n == 0) return -1;
        int low = 0, high = n - 1;
        while (low < high) {
            //这里进行的是取低位， 也是为了使得循环可以正确退出，防止死循环
            int mid = (low + high) / 2;
            if (arr[mid] < target) {
                low = mid + 1;
            } else { //a[mid] >= key
                high = mid;   //因为mid也满足情况
            }
        }
        //这里进行检查的原因参考上面的标注
        if (arr[high] >= target) {
            return high;
        } else {
            return -1;
        }
    }

    // LC995
    public int minKBitFlips(int[] A, int K) {
        int n = A.length;
        int[] diff = new int[n + 1];
        int result = 0;
        int cumulativeSum = 0; // 表示当前位置翻转的次数, 偶数次相当于没有翻转,
        diff[0] = 1 - A[0];  // 初始化为A[0]项需要翻转的次数
        diff[K] -= diff[0];  // 若[0]需要翻转, 则[K]需要减一
        //    0 1 1 0 1 1
        // 1) 0 0 0 0 0 0
        // 1)
        for (int i = 0; i < n; i++) {
            cumulativeSum += diff[i];
            if ((A[i] + cumulativeSum) % 2 == 0) {
                if (i + K > n) {
                    return -1;
                }
                result++;
                diff[i]++;
                cumulativeSum++;
                diff[i + K]--;
            }
        }
        return result;
    }

    // LC995
    public int minKBitFlipsOrig(int[] A, int K) {
        int n = A.length;
        int result = 0;
        for (int i = 0; i < n - K + 1; i++) {
            if (A[i] == 1) continue;
            result++;
            for (int j = 0; j < K; j++) {
                A[i + j] = A[i + j] ^ 1;
            }
        }

        for (int i = n - K; i < n; i++) {
            if (A[i] == 0) return -1;
        }
        return result;

    }

    // LC832
    public int[][] flipAndInvertImage(int[][] A) {
        for (int[] row : A) {
            for (int i = 0; i < row.length / 2; i++) {
                int tmp = row[i];
                row[i] = row[row.length - i - 1];
                row[row.length - i - 1] = tmp;
                row[i] = 1 - row[i];
                row[row.length - i - 1] = 1 - row[row.length - i - 1];
            }
            if (row.length % 2 == 1) {
                row[row.length / 2] = 1 - row[row.length / 2];
            }
        }

        return A;
    }

    // LC566
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int origM = nums.length;
        int origN = nums[0].length;
        if (origM * origN != r * c) return nums;
        int[][] result = new int[r][c];

        for (int i = 0; i < r * c; i++) {
            int m = i / origN;
            int n = i % origN;
            int j = i / c;
            int k = i % c;
            result[j][k] = nums[m][n];
        }
        return result;
    }

    // LC561
    public int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int result = 0;
        for (int i = 0; i < nums.length; i += 2) {
            result += nums[i];
        }
        return result;
    }

    // LC485
    public int findMaxConsecutiveOnes(int[] nums) {
        int max = 0;
        int currentLen = 0;
        int right = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                right = i;
                break;
            }
        }
        while (right < n) {
            if (nums[right] == 1) {
                currentLen++;
                right++;
                max = Math.max(max, currentLen);
            } else {
                currentLen = 0;
                right++;
            }
        }
        return max;
    }

    // LC765
    public int minSwapsCouples(int[] row) {

        DisjointSetUnion14 dsu = new DisjointSetUnion14();
        int n = row.length;
        int result = 0;

        for (int i = 0; i < n; i += 2) {
            // 如果相邻的两个是cp, 则必然满足以下任意一个
            // 1) 第一个是偶数 第二个是第一个+1
            // 2) 第一个是奇数 第二个是第一个-1
            // 否则 两个不是cp
            // 对于不是cp的 计算出其所属的N,
            // 由于计数从0开始, 则row中的项目直接/2得到其对应的N
            // 将N加入dsu, merge
            if ((row[i] % 2 == 0 && row[i + 1] == row[i] + 1) || (row[i] % 2 == 1 && row[i + 1] == row[i] - 1))
                continue;
            dsu.add(row[i] / 2);
            dsu.add(row[i + 1] / 2);
            dsu.merge(row[i] / 2, row[i + 1] / 2);
        }

        Map<Integer, Set<Integer>> groups = dsu.getAllGroups();
        for (Set<Integer> i : groups.values()) {
            result += i.size() - 1;
        }
        return result;
    }
}

class DisjointSetUnion25 {

    Map<Integer, Integer> father;
    Map<Integer, Integer> rank;

    public DisjointSetUnion25() {
        father = new HashMap<>();
        rank = new HashMap<>();
    }

    public void add(int i) {
        // 置初始父亲为自身
        // 之后判断连通分量个数时候, 遍历father, 找value==key的
        father.putIfAbsent(i, i);
        // 初始秩为1
        rank.putIfAbsent(i, 1);
    }

    // 找父亲, 路径压缩
    public int find(int i) {
        //先找到根 再压缩
        int root = i;
        while (father.get(root) != root) {
            root = father.get(root);
        }
        // 找到根, 开始对一路上的子节点进行路径压缩
        while (father.get(i) != root) {
            int origFather = father.get(i);
            father.put(i, root);
            // 更新秩, 按照节点数
            rank.put(root, rank.get(root) + 1);
            i = origFather;
        }
        return root;
    }

    public boolean merge(int i, int j) {
        int iFather = find(i);
        int jFather = find(j);
        if (iFather == jFather) return false;
        // 按秩合并
        if (rank.get(iFather) >= rank.get(jFather)) {
            father.put(jFather, iFather);
            rank.put(iFather, rank.get(jFather) + rank.get(iFather));
        } else {
            father.put(iFather, jFather);
            rank.put(jFather, rank.get(jFather) + rank.get(iFather));
        }
        return true;
    }

    public boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> getAllGroups() {
        Map<Integer, Set<Integer>> result = new HashMap<>();
        // 找出所有根
        for (Integer i : father.keySet()) {
            int f = find(i);
            result.putIfAbsent(f, new HashSet<>());
            result.get(f).add(i);
        }
        return result;
    }

    public int getNumOfGroups() {
        Set<Integer> s = new HashSet<Integer>();
        for (Integer i : father.keySet()) {
            s.add(find(i));
        }
        return s.size();
    }

}