import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.subarraySum(new int[]{1, 1, 1}, 2));
    }

    // LC115

    // bottom up with memory search

    // func numDistinct(s string, t string) int {
    //	sLen, tLen := len(s), len(t)
    //
    //	memo := make([][]int, sLen) // 创建一个memo数组存储计算过的子问题
    //	for i := range memo {
    //		memo[i] = make([]int, tLen)
    //		for j := 0; j < tLen; j++ {
    //			memo[i][j] = -1
    //		}
    //	}
    //	var helper func(i, j int) int
    //	helper = func(i, j int) int { // 从开头到s[i]的子串中，出现『从开头到t[i]的子串』的 次数
    //		if j < 0 { // base case 当j指针越界，此时t为空串，s不管是不是空串，匹配方式数都是1
    //			return 1
    //		}
    //		if i < 0 { // base case i指针越界，此时s为空串，t不是，s怎么也匹配不了t，方式数0
    //			return 0
    //		}
    //		if memo[i][j] != -1 { // 计算过的子问题的解，直接从memo中拿出来返回
    //			return memo[i][j]
    //		}
    //		if s[i] == t[j] { // t[j]被匹配掉，对应helper(i-1, j-1)，不被匹配掉对应helper(i-1, j)
    //			memo[i][j] = helper(i-1, j) + helper(i-1, j-1)
    //		} else {
    //			memo[i][j] = helper(i-1, j) //
    //		}
    //		return memo[i][j] // 返回当前递归子问题的解
    //	}
    //
    //	return helper(sLen-1, tLen-1) //从开头到s[sLen-1]的子串中，出现『从开头到t[tLen-1]的子串』的次数
    //}

    // LC115 top down

    public int numDistinctTopDown(String s, String t) {
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        // dp[i][j] 表示 在s的前i个字符中, 出现的(t的前j个字符的子序列)的个数
        // dp[i][j] =
        //  1) s[i] == t[j], dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
        //  2) s[i] != t[j], dp[i][j] = dp[i-1][j]
        //
        // dp[i][0] = 1
        // dp[0][j] = 0
        for (int i = 0; i <= s.length(); i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[s.length()][t.length()];
    }

    // LC115 bottom up

    public int numDistinctBottomUp(String s, String t) {
        int[][] memo = new int[s.length()][t.length()];
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                memo[i][j] = -1;
            }
        }
        return numDistinctRecursive(s.length() - 1, t.length() - 1, memo, s, t);
    }

    private int numDistinctRecursive(int i, int j, int[][] memo, String s, String t) {
        // 目标: 返回s从0到i中, 出现(t从0到j的子序列)的次数
        if (j < 0) {
            return 1;
        }
        if (i < 0) {
            return 0;
        }
        if (memo[i][j] != -1) return memo[i][j];
        if (s.charAt(i) == t.charAt(j)) {
            memo[i][j] = numDistinctRecursive(i - 1, j - 1, memo, s, t) + numDistinctRecursive(i - 1, j, memo, s, t);
        } else {
            memo[i][j] = numDistinctRecursive(i - 1, j, memo, s, t);
        }
        return memo[i][j];
    }

    // LC560
    public int subarraySum(int[] nums, int k) {
        int[] prefix = new int[nums.length + 1];
        Map<Integer, Integer> m = new HashMap<>();
        int result = 0;
        for (int i = 0; i < nums.length; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }
        for (int i = 0; i <= nums.length; i++) {
            // 目标 : prefix[i] - prefix[j] = k , j<i
            // <=>   prefix[i] - k = prefix[j]
            // 考虑 i 的递增性, 每次判断完后往合集里加入i即可
            result += m.getOrDefault(prefix[i] - k, 0);
            m.put(prefix[i], m.getOrDefault(prefix[i], 0) + 1);
        }
        return result;
    }
}

class SolutionDFS {
    // 存储有向图
    List<List<Integer>> edges;
    // 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    int[] visited;
    // 用数组来模拟栈，下标 n-1 为栈底，0 为栈顶
    int[] result;
    // 判断有向图中是否有环
    boolean valid = true;
    // 栈下标
    int index;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<>());
        }
        visited = new int[numCourses];
        result = new int[numCourses];
        index = numCourses - 1;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        // 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for (int i = 0; i < numCourses && valid; ++i) {
            if (visited[i] == 0) {
                dfs(i);
            }
        }
        if (!valid) {
            return new int[0];
        }
        // 如果没有环，那么就有拓扑排序
        return result;
    }

    public void dfs(int u) {
        // 将节点标记为「搜索中」
        visited[u] = 1;
        // 搜索其相邻节点
        // 只要发现有环，立刻停止搜索
        for (int v : edges.get(u)) {
            // 如果「未搜索」那么搜索相邻节点
            if (visited[v] == 0) {
                dfs(v);
                if (!valid) {
                    return;
                }
            }
            // 如果「搜索中」说明找到了环
            else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        // 将节点标记为「已完成」
        visited[u] = 2;
        // 将节点入栈
        result[index--] = u;
    }
}

class SolutionBFS {
    // 存储有向图
    List<List<Integer>> edges;
    // 存储每个节点的入度
    int[] indeg;
    // 存储答案
    int[] result;
    // 答案下标
    int index;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<>());
        }
        indeg = new int[numCourses];
        result = new int[numCourses];
        index = 0;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
            ++indeg[info[0]];
        }

        Queue<Integer> queue = new LinkedList<>();
        // 将所有入度为 0 的节点放入队列中
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        while (!queue.isEmpty()) {
            // 从队首取出一个节点
            int u = queue.poll();
            // 放入答案中
            result[index++] = u;
            for (int v : edges.get(u)) {
                --indeg[v];
                // 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
                if (indeg[v] == 0) {
                    queue.offer(v);
                }
            }
        }

        if (index != numCourses) {
            return new int[0];
        }
        return result;
    }
}
