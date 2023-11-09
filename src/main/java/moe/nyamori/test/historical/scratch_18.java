package moe.nyamori.test.historical;

import java.util.*;

class scratch_18 {
    public static void main(String[] args) {
        int[][] arr = {{2, 1}, {3, 1}, {4, 1}, {1, 5}};
        //              1,0   2,0   3,0   0,4
        scratch_18 s = new scratch_18();
        System.err.println(s.minNumberOfSemesters(5, arr, 2));
    }

    public TreeNode8 lowestCommonAncestor(TreeNode8 root, TreeNode8 p, TreeNode8 q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode8 left = lowestCommonAncestor(root.left, p, q);
        TreeNode8 right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left != null) {
            return left;
        } else {
            return right;
        }
    }


    Map<Integer, TreeNode8> parent = new HashMap<Integer, TreeNode8>();
    Set<Integer> visited = new HashSet<Integer>();

    public void dfs(TreeNode8 root) {
        if (root.left != null) {
            parent.put(root.left.val, root);
            dfs(root.left);
        }
        if (root.right != null) {
            parent.put(root.right.val, root);
            dfs(root.right);
        }
    }

    public TreeNode8 lowestCommonAncestorMap(TreeNode8 root, TreeNode8 p, TreeNode8 q) {
        dfs(root);
        while (p != null) {
            visited.add(p.val);
            p = parent.get(p.val);
        }
        while (q != null) {
            if (visited.contains(q.val)) {
                return q;
            }
            q = parent.get(q.val);
        }
        return null;
    }


    public String reverseLeftWords(String s, int n) {
        return s.substring(n).concat(s.substring(0, n));
    }

//    给你一个整数n表示某所大学里课程的数目，编号为1到n，数组dependencies中，dependencies[i] = [xi, yi] 表示一个先修课的关系，也就是课程xi必须在课程yi之前上。同时你还有一个整数k。
//
//    在一个学期中，你 最多可以同时上 k门课，前提是这些课的先修课在之前的学期里已经上过了。
//
//    Use backtracking with states (bitmask, degrees) where bitmask represents the set of courses,
//    if the ith bit is 1 then the ith course was taken, otherwise, you can take the ith course.
//    Degrees represent the degree for each course (nodes in the graph).

//    Note that you can only take nodes (courses) with degree = 0 and it is optimal at every step
//    in the backtracking take the maximum number of courses limited by k.

//    请你返回上完所有课最少需要多少个学期。题目保证一定存在一种上完所有课的方式。

    int minNumberOfSemestersAnswer = Integer.MAX_VALUE;

    public int minNumberOfSemesters(int n, int[][] dependencies, int k) {
        this.minNumberOfSemestersAnswer = n;
        if (dependencies.length == 0) {
            if (n % k == 0) return n / k;
            else return (n / k) + 1;
        }
        Set<Integer> bs = new HashSet<>(n);
        Set<Integer> noDepCourses = new HashSet<>();

        for (int i = 0; i < n; i++) {
            bs.add(i);
            noDepCourses.add(i);
        }
        // 入度
        int[] degree = new int[n];
        int[] degreeDelay = new int[n];
        Map<Integer, Set<Integer>> dependencySetMap = new HashMap<>();
        for (int[] i : dependencies) {
            dependencySetMap.putIfAbsent(i[0] - 1, new HashSet<>());
            dependencySetMap.get(i[0] - 1).add(i[1] - 1);
            degree[i[1] - 1]++;
            degreeDelay[i[1] - 1]++;
            noDepCourses.remove(i[1] - 1);
        }

        int currentTaking = noDepCourses.size() % k;
        int currentSemester = noDepCourses.size() / k;
        for (int i : noDepCourses) {
            bs.remove(i);
            if (dependencySetMap.containsKey(i)) {
                for (int j : dependencySetMap.get(i)) {
                    degree[j]--;
                    degreeDelay[j]--;
                }
            }
        }
        backtrack(bs, degree, degreeDelay, k, currentTaking, currentSemester, dependencySetMap);

        return this.minNumberOfSemestersAnswer;
    }

    public void backtrack(Set<Integer> bs, int[] degree, int[] degreeDelay, int maxK, int currentTaking, int currentSemester, Map<Integer, Set<Integer>> dependencySetMap) {
        if (bs.isEmpty()) {
            this.minNumberOfSemestersAnswer = Math.min(currentSemester, this.minNumberOfSemestersAnswer);
            return;
        }

        for (int i = 0; i < degree.length; i++) {
            if (currentSemester > this.minNumberOfSemestersAnswer) return;
            boolean thisSemesterShouldOver = true;
            int currentTakingBak = currentTaking;
            int currentSemesterBak = currentSemester;
            int[] degreeBak = new int[degree.length];
            for (int j = 0; j < degree.length; j++) {
                degreeBak[j] = degree[j];
            }
            if (bs.contains(i) && degree[i] == 0 && currentTaking < maxK) {
                bs.remove(i);
                currentTaking++;
                if (dependencySetMap.containsKey(i)) {
                    for (int j : dependencySetMap.get(i)) {
                        degreeDelay[j]--;
                    }
                }
            } else {
                continue;
            }
            if (currentTaking == maxK) {
                currentSemester++;
                for (int j = 0; j < degree.length; j++) {
                    degree[j] = degreeDelay[j];
                }
                currentTaking = 0;
            } else {
                Set<Integer> leftCourses = new HashSet<>();
                for (int j = 0; j < degree.length; j++) {
                    if (bs.contains(j)) leftCourses.add(j);
                }
                if (!leftCourses.isEmpty()) {
                    for (int u : leftCourses) {
                        if (degree[u] == 0) {
                            thisSemesterShouldOver = false;
                            break;
                        }
                    }
                }
                if (thisSemesterShouldOver) {
                    currentSemester++;
                    for (int j = 0; j < degree.length; j++) {
                        degree[j] = degreeDelay[j];
                    }
                    currentTaking = 0;
                }
            }
            backtrack(bs, degree, degreeDelay, maxK, currentTaking, currentSemester, dependencySetMap);
            bs.add(i);
            if (dependencySetMap.containsKey(i)) {
                for (int j : dependencySetMap.get(i)) {
                    degreeDelay[j]++;
                }
            }
            for (int j = 0; j < degree.length; j++) {
                degree[j] = degreeBak[j];
            }
            currentTaking = currentTakingBak;
            currentSemester = currentSemesterBak;
        }
    }


    public TreeNode8 lowestCommonAncestorMy(TreeNode8 root, TreeNode8 p, TreeNode8 q) {
        // FIFO: 先加根, 然后遍历子节点, 如果两个节点都被遍历到, 则这是一个公共节点, 退出队列用ans存起来, 并将其左右子节点加入队列
        // 如果左子树遍历到两个节点, 则更新ans, 清空队列, 加入左子树的左右儿子; 否则遍历右子树; 若都没有遍历到, 则返回ans

        TreeNode8 ans = root;
        Queue<TreeNode8> fifo = new LinkedList<>();

        fifo.add(root);
        while (!fifo.isEmpty()) {
            TreeNode8 top = fifo.peek();
            if (preOrder(top, p) && preOrder(top, q)) {
                if (top.left != null) {
                    fifo.add(top.left);
                }
                if (top.right != null) {
                    fifo.add(top.right);
                }
                ans = fifo.poll();
            } else {
                fifo.poll();
            }
        }
        return ans;

    }

    private boolean preOrder(TreeNode8 root, TreeNode8 t) {
        if (root.val == t.val) return true;
        return (root.left != null ? preOrder(root.left, t) : false) || (root.right != null ? preOrder(root.right, t) : false);
    }

    public int[] fairCandySwap(int[] A, int[] B) {
        int[] result = new int[2];

        int sumA, sumB;
        sumA = sumB = 0;
        Set<Integer> s = new HashSet<>();
        for (int i : A) sumA += i;
        for (int i : B) {
            sumB += i;
            s.add(i);
        }
        int diff = (sumA - sumB) / 2;
        for (int i : A) {
            if (s.contains(i - diff)) {
                result[0] = i;
                result[1] = i - diff;
                return result;
            }
        }
        return result;
    }

    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int maxLen = 0, count = 0;
        for (int i = 0; i < nums.length; i++) {
            count = count + (nums[i] == 1 ? 1 : -1);
            if (map.containsKey(count)) {
                maxLen = Math.max(maxLen, i - map.get(count));
            } else {
                map.put(count, i);
            }
        }
        return maxLen;
    }

}


class TreeNode18 {
    int val;
    TreeNode8 left;
    TreeNode8 right;

    TreeNode18(int x) {
        val = x;
    }
}
