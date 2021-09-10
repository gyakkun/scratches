import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        System.out.println(s.numberOfWeeks(new int[]{16,7,5,3}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC1953 TLE
    public long numberOfWeeks(int[] milestones) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(o -> -milestones[o]));
        int prevWeek = -1, n = milestones.length;
        for (int i = 0; i < n; i++) pq.offer(i);
        int count = 0;
        while (!pq.isEmpty()) {
            int thisWeek = pq.poll();
            milestones[thisWeek]--;
            if (prevWeek != -1 && milestones[prevWeek] != 0) pq.offer(prevWeek);
            prevWeek = thisWeek;
            count++;
            // 检查
            // if (pq.isEmpty()) break;
        }
        return count;
    }


    //    给你 n 个项目，编号从 0 到 n - 1 。同时给你一个整数数组 milestones ，其中每个 milestones[i] 表示第 i 个项目中的阶段任务数量。
    //
    //    你可以按下面两个规则参与项目中的工作：
    //
    //    每周，你将会完成 某一个 项目中的 恰好一个 阶段任务。你每周都 必须 工作。
    //    在 连续的 两周中，你 不能 参与并完成同一个项目中的两个阶段任务。
    //    一旦所有项目中的全部阶段任务都完成，或者仅剩余一个阶段任务都会导致你违反上面的规则，那么你将 停止工作 。注意，由于这些条件的限制，你可能无法完成所有阶段任务。
    //
    //    返回在不违反上面规则的情况下你 最多 能工作多少周。

    // LC249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<>();
        Map<Integer, Map<String, Integer>> m = new HashMap<>();
        for (String s : strings) {
            m.putIfAbsent(s.length(), new HashMap<>());
            Map<String, Integer> inner = m.get(s.length());
            inner.put(s, inner.getOrDefault(s, 0) + 1);
        }

        for (Map<String, Integer> s : m.values()) {
            while (!s.isEmpty()) {
                String w = s.keySet().iterator().next();
                // 构造
                List<String> list = new ArrayList<>();
                char[] ca = w.toCharArray();
                for (int i = 0; i < 26; i++) {
                    for (int j = 0; j < ca.length; j++) {
                        ca[j] = (char) (((ca[j] - 'a' + 1) % 26) + 'a');
                    }
                    String built = new String(ca);
                    if (s.containsKey(built)) {
                        int count = s.get(built);
                        s.remove(built);
                        for (int j = 0; j < count; j++)
                            list.add(built);
                    }
                }
                result.add(list);
            }
        }
        return result;
    }
}