import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

        String[][] smtx = {{"b"}, {"f"}, {"f", "r"}, {"f", "r", "g"}, {"f", "r", "g", "c"}, {"f", "r", "g", "c", "r"}, {"f", "o"}, {"f", "o", "x"}, {"f", "o", "x", "t"}, {"f", "o", "x", "d"}, {"f", "o", "l"}, {"l"}, {"l", "q"}, {"c"}, {"h"}, {"h", "t"}, {"h", "o"}, {"h", "o", "d"}, {"h", "o", "t"}};
        List<List<String>> sll = new ArrayList<>(smtx.length);
        for (String[] sa : smtx) sll.add(Arrays.asList(sa));
        Lc1948 lc1948 = new Lc1948();
        System.out.println(lc1948.deleteDuplicateFolder(sll));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // Interview 04.10
    public boolean checkSubTree(TreeNode t1, TreeNode t2) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(t1);
        while (!q.isEmpty()) {
            TreeNode p = q.poll();
            if (checkSubTreeHelper(p, t2)) {
                return true;
            }
            if (p.left != null) q.offer(p.left);
            if (p.right != null) q.offer(p.right);
        }
        return false;
    }

    private boolean checkSubTreeHelper(TreeNode root, TreeNode target) {
        if (root == null && target == null) return true;
        else if (root == null || target == null) return false;
        if (root.val != target.val) {
            return false;
        }
        return checkSubTreeHelper(root.left, target.left) && checkSubTreeHelper(root.right, target.right);
    }

    // LC1948 **
    static class Lc1948 {
        class Node {
            String serial = "";
            // 用TreeMap保证序列化后的出现顺序唯一
            TreeMap<String, Node> children = new TreeMap<>();
        }

        public List<List<String>> deleteDuplicateFolder(List<List<String>> paths) {
            Node root = new Node();
            Map<Node, List<String>> nodeListStringMap = new HashMap<>();
            List<List<String>> result = new LinkedList<>();
            for (List<String> p : paths) {
                Node cur = root;
                for (int i = 0; i < p.size(); i++) {
                    cur.children.putIfAbsent(p.get(i), new Node());
                    cur = cur.children.get(p.get(i));
                }
                nodeListStringMap.put(cur, p);
            }
            getSerial(root);
            // BFS
            Deque<Node> q = new LinkedList<>();
            Map<String, Integer> serialCountMap = new HashMap<>();
            q.offer(root);
            while (!q.isEmpty()) {
                Node p = q.poll();
                serialCountMap.put(p.serial, serialCountMap.getOrDefault(p.serial, 0) + 1);
                for (String s : p.children.keySet()) {
                    q.offer(p.children.get(s));
                }
            }
            // DFS
            Deque<Node> stack = new LinkedList<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                Node p = stack.pop();
                if (!p.serial.equals("()") && serialCountMap.get(p.serial) > 1) {
                    continue;
                }
                if (nodeListStringMap.containsKey(p)) {
                    result.add(nodeListStringMap.get(p));
                }
                for (String s : p.children.keySet()) {
                    stack.push(p.children.get(s));
                }
            }
            return result;
        }

        private void getSerial(Node node) {
            if (node.children.size() == 0) {
                node.serial = "()";
                return;
            }
            for (String s : node.children.keySet()) {
                getSerial(node.children.get(s));
            }
            for (String s : node.children.keySet()) {
                node.serial += "(" + s + node.children.get(s).serial + ")";
            }
        }

    }

    // LC1943 **
    public List<List<Long>> splitPainting(int[][] segments) {
        Map<Integer, Long> diff = new HashMap<>();
        for (int[] intv : segments) {
            diff.putIfAbsent(intv[0], 0L);
            diff.putIfAbsent(intv[1], 0L);
            diff.put(intv[0], diff.get(intv[0]) + intv[2]);
            diff.put(intv[1], diff.get(intv[1]) - intv[2]);
        }
        List<Integer> keyArr = new ArrayList<>(diff.keySet());
        Collections.sort(keyArr);
        long prev = 0, accumulative = 0;
        List<List<Long>> result = new ArrayList<>();
        for (int i : keyArr) {
            if (accumulative != 0) result.add(Arrays.asList(prev, (long) i, accumulative));
            accumulative += diff.get(i);
            prev = i;
        }
        return result;
    }

    // LC657
    public boolean judgeCircle(String moves) {
        int[] count = new int[2];
        for (char c : moves.toCharArray()) {
            if (c == 'U') count[0]++;
            else if (c == 'D') count[0]--;
            else if (c == 'L') count[1]++;
            else if (c == 'R') count[1]--;
        }
        return count[0] == 0 && count[1] == 0;
    }

    // LCP 01
    public int game(int[] guess, int[] answer) {
        int result = 0;
        for (int i = 0; i < 3; i++) if (guess[i] == answer[i]) result++;
        return result;
    }

    // LC1487
    public String[] getFolderNames(String[] names) {
        Map<String, Integer> m = new HashMap<>();
        int n = names.length;
        String[] result = new String[n];
        for (int i = 0; i < n; i++) {
            if (m.containsKey(names[i])) {
                int count = m.get(names[i]);
                while (m.containsKey(names[i] + '(' + count + ')')) {
                    count++;
                }
                result[i] = names[i] + '(' + count + ')';

                m.put(names[i], count + 1);
                m.put(result[i], 1);
            } else {
                result[i] = names[i];
                m.put(names[i], 1);
            }
        }
        return result;
    }

    // LC671
    TreeSet<Integer> lc671Ts;

    public int findSecondMinimumValue(TreeNode root) {
        lc671Ts = new TreeSet<>();
        lc671Helper(root);
        if (lc671Ts.size() <= 1) return -1;
        return lc671Ts.last();
    }

    private void lc671Helper(TreeNode root) {
        if (root == null) return;
        if (lc671Ts.contains(root.val)) {
            ;
        } else if (lc671Ts.size() < 2) {
            lc671Ts.add(root.val);
        } else {
            if (root.val < lc671Ts.last()) {
                lc671Ts.remove(lc671Ts.last());
                lc671Ts.add(root.val);
            }
        }
        lc671Helper(root.left);
        lc671Helper(root.right);
    }

    // LC1713 **
    public int minOperations(int[] target, int[] arr) {
        Map<Integer, Integer> targetValIdxMap = new HashMap<>();
        for (int i = 0; i < target.length; i++) targetValIdxMap.put(target[i], i);
        List<Integer> arrList = new ArrayList<>(arr.length);
        for (int i = 0; i < arr.length; i++) {
            if (targetValIdxMap.containsKey(arr[i])) arrList.add(targetValIdxMap.get(arr[i]));
        }
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i : arrList) {
            Integer ceiling = ts.ceiling(i);
            if (ceiling != null) {
                ts.remove(ceiling);
            }
            ts.add(i);
        }
        return target.length - ts.size();
    }

    // LC1743
    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, List<Integer>> m = new HashMap<>();
        for (int[] p : adjacentPairs) {
            m.putIfAbsent(p[0], new ArrayList<>(2));
            m.putIfAbsent(p[1], new ArrayList<>(2));
            m.get(p[0]).add(p[1]);
            m.get(p[1]).add(p[0]);
        }

        int end = -1;
        for (int i : m.keySet()) {
            if (m.get(i).size() == 1) {
                end = i;
                break;
            }
        }
        int[] result = new int[adjacentPairs.length + 1];
        result[0] = end;
        result[1] = m.get(result[0]).get(0);
        for (int i = 2; i < result.length; i++) {
            List<Integer> prevAdj = m.get(result[i - 1]);
            result[i] = result[i - 2] == prevAdj.get(0) ? prevAdj.get(1) : prevAdj.get(0);
        }
        return result;
    }

    // Interview 16.19
    public int[] pondSizes(int[][] land) {
        int[][] directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        int totalRow = land.length, totalCol = land[0].length;
        DisjointSetUnion dsu = new DisjointSetUnion();
        for (int i = 0; i < totalRow; i++) {
            for (int j = 0; j < totalCol; j++) {
                int curId = getMatrixId(i, j, totalRow, totalCol);
                if (land[i][j] == 0) {
                    dsu.add(curId);
                    for (int[] dir : directions) {
                        int targetRow = i + dir[0];
                        int targetCol = j + dir[1];
                        if (checkPoint(targetRow, targetCol, totalRow, totalCol) && land[targetRow][targetCol] == 0) {
                            dsu.add(getMatrixId(targetRow, targetCol, totalRow, totalCol));
                            dsu.merge(curId, getMatrixId(targetRow, targetCol, totalRow, totalCol));
                        }
                    }
                }
            }
        }
        Map<Integer, Set<Integer>> groups = dsu.allGroups();
        List<Integer> result = new ArrayList<>(groups.size());
        for (int i : groups.keySet()) {
            result.add(groups.get(i).size());
        }
        Collections.sort(result);
        int[] res = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            res[i] = result.get(i);
        }
        return res;
    }

    private boolean checkPoint(int targetRow, int targetCol, int totalRow, int totalCol) {
        return !(targetRow >= totalRow || targetRow < 0 || targetCol >= totalCol || targetCol < 0);
    }

    private int getMatrixId(int targetRow, int targetCol, int totalRow, int totalCol) {
        return targetRow * totalCol + targetCol;
    }

    // LC500
    public String[] findWords(String[] words) {
        List<String> result = new ArrayList<>(words.length);
        String[] kb = new String[]{"qwertyuiop", "asdfghjkl", "zxcvbnm"};
        boolean[][] kbCheck = new boolean[3][26];
        int kbCtr = 0;
        for (String s : kb) {
            for (char c : s.toCharArray()) {
                kbCheck[kbCtr][c - 'a'] = true;
            }
            kbCtr++;
        }
        for (String word : words) {
            boolean flag = false;
            for (int i = 0; i < 3; i++) {
                boolean flag2 = true;
                for (char c : word.toLowerCase().toCharArray()) {
                    if (!kbCheck[i][c - 'a']) {
                        flag2 = false;
                        break;
                    }
                }
                if (flag2) {
                    result.add(word);
                    break;
                }
            }
        }
        return result.toArray(new String[result.size()]);
    }

    // LC1893
    public boolean isCovered(int[][] ranges, int left, int right) {
        boolean[] check = new boolean[51];
        int min = 51, max = -1;
        for (int[] r : ranges) {
            min = Math.min(min, r[0]);
            max = Math.max(max, r[1]);
            for (int i = r[0]; i <= r[1]; i++) {
                check[i] = true;
            }
        }
        if (left < min || right > max) return false;
        for (int i = left; i <= right; i++) if (!check[i]) return false;
        return true;
    }

    // LC713
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int left = 0, prod = 1, result = 0, n = nums.length;
        for (int right = 0; right < n; right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            result += right - left + 1;
        }
        return result;
    }

    // LC209
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];
        int left = 0, right = 0;
        int min = Integer.MAX_VALUE;
        while (left <= right && right != n) {
            if (prefix[right + 1] - prefix[left] < target) {
                right++;
                continue;
            } else {
                if (right - left + 1 < min) {
                    min = right - left + 1;
                }
                left++;
            }
        }
        if (min == Integer.MAX_VALUE) return 0;
        return min;
    }

    // LC5
    public String longestPalindrome(String s) {
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        boolean[][] dp = new boolean[n][n];
        // dp[i][j] 表示 [i,j]是不是回文串
        // dp[i][j] = true iff dp[i+1][j-1]==true && cArr[i]==cArr[j]
        for (int i = 0; i < n; i++) dp[i][i] = true;
        int max = 1;
        int maxLeft = 0, maxRight = 0;
        for (int len = 2; len <= n; len++) {
            for (int left = 0; left < n; left++) {
                int right = left + len - 1;
                if (right >= n) break;
                if (cArr[left] == cArr[right]) {
                    if (len == 2) dp[left][right] = true;
                    else {
                        dp[left][right] = dp[left + 1][right - 1];
                    }
                }
                if (len > max && dp[left][right]) {
                    maxLeft = left;
                    maxRight = right;
                }
            }
        }
        return s.substring(maxLeft, maxRight + 1);
    }

    // LC3
    public int lengthOfLongestSubstring(String s) {
        char[] cArr = s.toCharArray();
        int n = cArr.length;
        Map<Character, Integer> lastOccur = new HashMap<>();
        int left = 0, right = 0;
        int max = 0;
        while (right != n) {
            if (lastOccur.containsKey(cArr[right])) {
                int end = lastOccur.get(cArr[right]);
                for (int i = left; i <= end; i++) {
                    lastOccur.remove(cArr[i]);
                }
                left = end + 1;
            }
            lastOccur.put(cArr[right], right);
            max = Math.max(max, right - left + 1);
            right++;
        }
        return max;
    }

    // LC720 You sure this is EASY???
    int lc720MaxLen = 0;
    String lc720Result = "";
    StringBuilder lc720Sb = new StringBuilder();

    public String longestWord(String[] words) {
        Trie trie = new Trie();
        for (String word : words) trie.addWord(word);
        TrieNode root = trie.root;
        lc720Helper(root);
        return lc720Result;
    }

    private void lc720Helper(TrieNode root) {
        for (char c : root.children.keySet()) {
            if (root.children.get(c).isEnd) {
                lc720Sb.append(c);
                if (lc720Sb.length() > lc720MaxLen) {
                    lc720MaxLen = lc720Sb.length();
                    lc720Result = lc720Sb.toString();
                } else if (lc720Sb.length() == lc720MaxLen && lc720Sb.toString().compareTo(lc720Result) < 0) {
                    lc720Result = lc720Sb.toString();
                }
                lc720Helper(root.children.get(c));
                lc720Sb.deleteCharAt(lc720Sb.length() - 1);
            }
        }
    }

    // LC968 **
    enum lc968Status {
        NO_NEED,
        NEED,
        HAS_CAMERA
    }

    int lc968Result;

    public int minCameraCover(TreeNode root) {
        lc968Result = 0;
        if (lc968Helper(root) == lc968Status.NEED) lc968Result++;
        return lc968Result;
    }

    private lc968Status lc968Helper(TreeNode root) {
        if (root == null) {
            return lc968Status.NO_NEED;
        }
        lc968Status left = lc968Helper(root.left), right = lc968Helper(root.right);
        // 如果子结点中有一个需要相机, 则该节点需要放置相机, 否则子节点会不被覆盖
        if (left == lc968Status.NEED || right == lc968Status.NEED) {
            lc968Result++;
            return lc968Status.HAS_CAMERA;
        }
        // 如果子节点中有一个拥有相机, 则父节点被覆盖, 不需要相机, 否则需要相机
        return (left == lc968Status.HAS_CAMERA || right == lc968Status.HAS_CAMERA) ? lc968Status.NO_NEED : lc968Status.NEED;
    }

    // LC866
    public int primePalindrome(int n) {
        // 前 663,961 个素数的回文素数表, 借用nthPrime打出来, 超过这个数就耗时太长了
        Integer[] table = {2, 3, 5, 7, 11, 101, 131, 151, 181, 191, 313, 353, 373, 383, 727, 757, 787, 797, 919, 929, 10301, 10501, 10601, 11311, 11411, 12421, 12721, 12821, 13331, 13831, 13931, 14341, 14741, 15451, 15551, 16061, 16361, 16561, 16661, 17471, 17971, 18181, 18481, 19391, 19891, 19991, 30103, 30203, 30403, 30703, 30803, 31013, 31513, 32323, 32423, 33533, 34543, 34843, 35053, 35153, 35353, 35753, 36263, 36563, 37273, 37573, 38083, 38183, 38783, 39293, 70207, 70507, 70607, 71317, 71917, 72227, 72727, 73037, 73237, 73637, 74047, 74747, 75557, 76367, 76667, 77377, 77477, 77977, 78487, 78787, 78887, 79397, 79697, 79997, 90709, 91019, 93139, 93239, 93739, 94049, 94349, 94649, 94849, 94949, 95959, 96269, 96469, 96769, 97379, 97579, 97879, 98389, 98689, 1003001, 1008001, 1022201, 1028201, 1035301, 1043401, 1055501, 1062601, 1065601, 1074701, 1082801, 1085801, 1092901, 1093901, 1114111, 1117111, 1120211, 1123211, 1126211, 1129211, 1134311, 1145411, 1150511, 1153511, 1160611, 1163611, 1175711, 1177711, 1178711, 1180811, 1183811, 1186811, 1190911, 1193911, 1196911, 1201021, 1208021, 1212121, 1215121, 1218121, 1221221, 1235321, 1242421, 1243421, 1245421, 1250521, 1253521, 1257521, 1262621, 1268621, 1273721, 1276721, 1278721, 1280821, 1281821, 1286821, 1287821, 1300031, 1303031, 1311131, 1317131, 1327231, 1328231, 1333331, 1335331, 1338331, 1343431, 1360631, 1362631, 1363631, 1371731, 1374731, 1390931, 1407041, 1409041, 1411141, 1412141, 1422241, 1437341, 1444441, 1447441, 1452541, 1456541, 1461641, 1463641, 1464641, 1469641, 1486841, 1489841, 1490941, 1496941, 1508051, 1513151, 1520251, 1532351, 1535351, 1542451, 1548451, 1550551, 1551551, 1556551, 1557551, 1565651, 1572751, 1579751, 1580851, 1583851, 1589851, 1594951, 1597951, 1598951, 1600061, 1609061, 1611161, 1616161, 1628261, 1630361, 1633361, 1640461, 1643461, 1646461, 1654561, 1657561, 1658561, 1660661, 1670761, 1684861, 1685861, 1688861, 1695961, 1703071, 1707071, 1712171, 1714171, 1730371, 1734371, 1737371, 1748471, 1755571, 1761671, 1764671, 1777771, 1793971, 1802081, 1805081, 1820281, 1823281, 1824281, 1826281, 1829281, 1831381, 1832381, 1842481, 1851581, 1853581, 1856581, 1865681, 1876781, 1878781, 1879781, 1880881, 1881881, 1883881, 1884881, 1895981, 1903091, 1908091, 1909091, 1917191, 1924291, 1930391, 1936391, 1941491, 1951591, 1952591, 1957591, 1958591, 1963691, 1968691, 1969691, 1970791, 1976791, 1981891, 1982891, 1984891, 1987891, 1988891, 1993991, 1995991, 1998991, 3001003, 3002003, 3007003, 3016103, 3026203, 3064603, 3065603, 3072703, 3073703, 3075703, 3083803, 3089803, 3091903, 3095903, 3103013, 3106013, 3127213, 3135313, 3140413, 3155513, 3158513, 3160613, 3166613, 3181813, 3187813, 3193913, 3196913, 3198913, 3211123, 3212123, 3218123, 3222223, 3223223, 3228223, 3233323, 3236323, 3241423, 3245423, 3252523, 3256523, 3258523, 3260623, 3267623, 3272723, 3283823, 3285823, 3286823, 3288823, 3291923, 3293923, 3304033, 3305033, 3307033, 3310133, 3315133, 3319133, 3321233, 3329233, 3331333, 3337333, 3343433, 3353533, 3362633, 3364633, 3365633, 3368633, 3380833, 3391933, 3392933, 3400043, 3411143, 3417143, 3424243, 3425243, 3427243, 3439343, 3441443, 3443443, 3444443, 3447443, 3449443, 3452543, 3460643, 3466643, 3470743, 3479743, 3485843, 3487843, 3503053, 3515153, 3517153, 3528253, 3541453, 3553553, 3558553, 3563653, 3569653, 3586853, 3589853, 3590953, 3591953, 3594953, 3601063, 3607063, 3618163, 3621263, 3627263, 3635363, 3643463, 3646463, 3670763, 3673763, 3680863, 3689863, 3698963, 3708073, 3709073, 3716173, 3717173, 3721273, 3722273, 3728273, 3732373, 3743473, 3746473, 3762673, 3763673, 3765673, 3768673, 3769673, 3773773, 3774773, 3781873, 3784873, 3792973, 3793973, 3799973, 3804083, 3806083, 3812183, 3814183, 3826283, 3829283, 3836383, 3842483, 3853583, 3858583, 3863683, 3864683, 3867683, 3869683, 3871783, 3878783, 3893983, 3899983, 3913193, 3916193, 3918193, 3924293, 3927293, 3931393, 3938393, 3942493, 3946493, 3948493, 3964693, 3970793, 3983893, 3991993, 3994993, 3997993, 3998993, 7014107, 7035307, 7036307, 7041407, 7046407, 7057507, 7065607, 7069607, 7073707, 7079707, 7082807, 7084807, 7087807, 7093907, 7096907, 7100017, 7114117, 7115117, 7118117, 7129217, 7134317, 7136317, 7141417, 7145417, 7155517, 7156517, 7158517, 7159517, 7177717, 7190917, 7194917, 7215127, 7226227, 7246427, 7249427, 7250527, 7256527, 7257527, 7261627, 7267627, 7276727, 7278727, 7291927, 7300037, 7302037, 7310137, 7314137, 7324237, 7327237, 7347437, 7352537, 7354537, 7362637, 7365637, 7381837, 7388837, 7392937, 7401047, 7403047, 7409047, 7415147, 7434347, 7436347, 7439347, 7452547, 7461647, 7466647, 7472747, 7475747, 7485847, 7486847, 7489847, 7493947, 7507057, 7508057, 7518157, 7519157, 7521257, 7527257, 7540457, 7562657, 7564657, 7576757, 7586857, 7592957, 7594957, 7600067, 7611167, 7619167, 7622267, 7630367, 7632367, 7644467, 7654567, 7662667, 7665667, 7666667, 7668667, 7669667, 7674767, 7681867, 7690967, 7693967, 7696967, 7715177, 7718177, 7722277, 7729277, 7733377, 7742477, 7747477, 7750577, 7758577, 7764677, 7772777, 7774777, 7778777, 7782877, 7783877, 7791977, 7794977, 7807087, 7819187, 7820287, 7821287, 7831387, 7832387, 7838387, 7843487, 7850587, 7856587, 7865687, 7867687, 7868687, 7873787, 7884887, 7891987, 7897987, 7913197, 7916197, 7930397, 7933397, 7935397, 7938397, 7941497, 7943497, 7949497, 7957597, 7958597, 7960697, 7977797, 7984897, 7985897, 7987897, 7996997, 9002009, 9015109, 9024209, 9037309, 9042409, 9043409, 9045409, 9046409, 9049409, 9067609, 9073709, 9076709, 9078709, 9091909, 9095909, 9103019, 9109019, 9110119, 9127219, 9128219, 9136319, 9149419, 9169619, 9173719, 9174719, 9179719, 9185819, 9196919, 9199919, 9200029, 9209029, 9212129, 9217129, 9222229, 9223229, 9230329, 9231329, 9255529, 9269629, 9271729, 9277729, 9280829, 9286829, 9289829, 9318139, 9320239, 9324239, 9329239, 9332339, 9338339, 9351539, 9357539, 9375739, 9384839, 9397939, 9400049, 9414149, 9419149, 9433349, 9439349, 9440449, 9446449, 9451549, 9470749, 9477749, 9492949, 9493949, 9495949, 9504059, 9514159, 9526259, 9529259, 9547459, 9556559, 9558559, 9561659, 9577759, 9583859, 9585859, 9586859, 9601069, 9602069, 9604069, 9610169, 9620269, 9624269, 9626269, 9632369, 9634369, 9645469, 9650569, 9657569, 9670769, 9686869, 9700079, 9709079, 9711179, 9714179, 9724279, 9727279, 9732379, 9733379, 9743479, 9749479, 9752579, 9754579, 9758579, 9762679, 9770779, 9776779, 9779779, 9781879, 9782879, 9787879, 9788879, 9795979, 9801089, 9807089, 9809089, 9817189, 9818189, 9820289, 9822289, 9836389, 9837389, 9845489, 9852589, 9871789, 9888889, 9889889, 9896989, 9902099, 9907099, 9908099, 9916199, 9918199, 9919199, 9921299, 9923299, 9926299, 9927299, 9931399, 9932399, 9935399, 9938399, 9957599, 9965699, 9978799, 9980899, 9981899, 9989899, 100030001};
        TreeSet<Integer> ts = new TreeSet<>(Arrays.asList(table));
        return ts.ceiling(n);
    }

    private void printPrimePalindromeFrom1e7to2e8() {
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973, 10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211, 10223, 10243, 10247, 10253, 10259, 10267, 10271, 10273, 10289, 10301, 10303, 10313, 10321, 10331, 10333, 10337, 10343, 10357, 10369, 10391, 10399, 10427, 10429, 10433, 10453, 10457, 10459, 10463, 10477, 10487, 10499, 10501, 10513, 10529, 10531, 10559, 10567, 10589, 10597, 10601, 10607, 10613, 10627, 10631, 10639, 10651, 10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729, 10733, 10739, 10753, 10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853, 10859, 10861, 10867, 10883, 10889, 10891, 10903, 10909, 10937, 10939, 10949, 10957, 10973, 10979, 10987, 10993, 11003, 11027, 11047, 11057, 11059, 11069, 11071, 11083, 11087, 11093, 11113, 11117, 11119, 11131, 11149, 11159, 11161, 11171, 11173, 11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261, 11273, 11279, 11287, 11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383, 11393, 11399, 11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489, 11491, 11497, 11503, 11519, 11527, 11549, 11551, 11579, 11587, 11593, 11597, 11617, 11621, 11633, 11657, 11677, 11681, 11689, 11699, 11701, 11717, 11719, 11731, 11743, 11777, 11779, 11783, 11789, 11801, 11807, 11813, 11821, 11827, 11831, 11833, 11839, 11863, 11867, 11887, 11897, 11903, 11909, 11923, 11927, 11933, 11939, 11941, 11953, 11959, 11969, 11971, 11981, 11987, 12007, 12011, 12037, 12041, 12043, 12049, 12071, 12073, 12097, 12101, 12107, 12109, 12113, 12119, 12143, 12149, 12157, 12161, 12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251, 12253, 12263, 12269, 12277, 12281, 12289, 12301, 12323, 12329, 12343, 12347, 12373, 12377, 12379, 12391, 12401, 12409, 12413, 12421, 12433, 12437, 12451, 12457, 12473, 12479, 12487, 12491, 12497, 12503, 12511, 12517, 12527, 12539, 12541, 12547, 12553, 12569, 12577, 12583, 12589, 12601, 12611, 12613, 12619, 12637, 12641, 12647, 12653, 12659, 12671, 12689, 12697, 12703, 12713, 12721, 12739, 12743, 12757, 12763, 12781, 12791, 12799, 12809, 12821, 12823, 12829, 12841, 12853, 12889, 12893, 12899, 12907, 12911, 12917, 12919, 12923, 12941, 12953, 12959, 12967, 12973, 12979, 12983, 13001, 13003, 13007, 13009, 13033, 13037, 13043, 13049, 13063, 13093, 13099, 13103, 13109, 13121, 13127, 13147, 13151, 13159, 13163, 13171, 13177, 13183, 13187, 13217, 13219, 13229, 13241, 13249, 13259, 13267, 13291, 13297, 13309, 13313, 13327, 13331, 13337, 13339, 13367, 13381, 13397, 13399, 13411, 13417, 13421, 13441, 13451, 13457, 13463, 13469, 13477, 13487, 13499, 13513, 13523, 13537, 13553, 13567, 13577, 13591, 13597, 13613, 13619, 13627, 13633, 13649, 13669, 13679, 13681, 13687, 13691, 13693, 13697, 13709, 13711, 13721, 13723, 13729, 13751, 13757, 13759, 13763, 13781, 13789, 13799, 13807, 13829, 13831, 13841, 13859, 13873, 13877, 13879, 13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933, 13963, 13967, 13997, 13999, 14009, 14011, 14029, 14033, 14051, 14057, 14071, 14081, 14083, 14087, 14107};
        // 回文根
        int proot = 1000;
        for (; proot <= 2000; proot++) {
            // 事实: 不存在偶数位数的回文素数, 即所有回文素数都是奇数位数的, 所以在回文根1000-9999的8位回文整数中不存在回文素数
            // 为此, 在两4位回文根中间插入0~9, 构成9位回文素数
            for (int j = 0; j < 10; j++) {
                String spr = String.valueOf(proot);
                spr = spr + String.valueOf(j) + new StringBuilder(spr).reverse().toString();
                int p = Integer.valueOf(spr);
                boolean flag = true;
                for (int i : primes) {
                    if (p % i == 0) {
                        flag = false;
                        break;
                    }
                }
                if (flag) System.out.println(p);
            }
        }

    }

    // LC846 LC1296
    public boolean isNStraightHand(int[] hand, int groupSize) {
        if (hand.length % groupSize != 0) return false;
        TreeMap<Integer, Integer> m = new TreeMap<>();
        for (int i : hand) m.put(i, m.getOrDefault(i, 0) + 1);
        while (!m.isEmpty()) {
            int firstKey = m.firstKey();
            for (int i = 0; i < groupSize; i++) {
                if (!m.containsKey(firstKey + i)) {
                    return false;
                }
                m.put(firstKey + i, m.get(firstKey + i) - 1);
                if (m.get(firstKey + i) == 0) m.remove(firstKey + i);
            }
        }
        return true;
    }

    // LC1302
    public int deepestLeavesSum(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        List<TreeNode> thisLayer = new LinkedList<>();
        while (!q.isEmpty()) {
            thisLayer.clear();
            int qSize = q.size();
            for (int i = 0; i < qSize; i++) {
                TreeNode p = q.poll();
                thisLayer.add(p);
                if (p.left != null) {
                    q.offer(p.left);
                }
                if (p.right != null) {
                    q.offer(p.right);
                }
            }
        }
        int sum = 0;
        for (TreeNode t : thisLayer) {
            sum += t.val;
        }
        return sum;
    }

    // LC138
    class LC138 {

        public Node copyRandomList(Node head) {
            Node dummy = new Node(-1);
            dummy.next = head;
            Node cur = head;
            Map<Node, Node> m = new HashMap<>();
            m.put(null, null);
            while (cur != null) {
                Node t = new Node(cur.val);
                m.put(cur, t);
                cur = cur.next;
            }

            cur = head;
            while (cur != null) {
                m.get(cur).next = m.get(cur.next);
                m.get(cur).random = m.get(cur.random);
                cur = cur.next;
            }

            return m.get(head);
        }

        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }

    }

    // LC1051
    public int heightChecker(int[] heights) {
        int[] freq = new int[101];
        for (int i : heights) freq[i]++;
        int result = 0;
        for (int i = 1, j = 0; i <= 100; i++) {
            while (freq[i]-- != 0) {
                if (heights[j++] != i) result++;
            }
        }
        return result;
    }

    // LC805 **
    public boolean splitArraySameAverage(int[] nums) {
        Arrays.sort(nums); // 避免出现{18,0,16,2}的被hack情况
        int n = nums.length;
        long sum = 0;
        for (int i : nums) sum += i;
        long[] arr = new long[n];
        for (int i = 0; i < n; i++) {
            arr[i] = ((long) nums[i]) * ((long) n) - sum;
        }
        long[] left = new long[n / 2];
        long[] right = new long[n - left.length];
        for (int i = 0; i < left.length; i++) {
            left[i] = arr[i];
        }
        for (int i = left.length; i < n; i++) {
            right[i - left.length] = arr[i];
        }

        // 找0
        Map<Integer, Set<Integer>> leftSumMaskMap = new HashMap<>();
        for (int subset = (1 << left.length) - 1; subset != 0; subset--) {
            int tmp = 0;
            for (int i = 0; i < left.length; i++) {
                if (((subset >> i) & 1) == 1) {
                    tmp += arr[i];
                }
            }
            leftSumMaskMap.putIfAbsent(tmp, new HashSet<>());
            leftSumMaskMap.get(tmp).add(subset);
        }
        for (int subset = (1 << right.length) - 1; subset != 0; subset--) {
            int tmp = 0;
            for (int i = 0; i < right.length; i++) {
                if (((subset >> i) & 1) == 1) {
                    tmp += arr[i + left.length];
                }
            }
            if (leftSumMaskMap.containsKey(-tmp)) {
                for (int mask : leftSumMaskMap.get(-tmp)) {
                    if (Integer.bitCount(mask) + Integer.bitCount(subset) != n) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // LC1799 **
    int lc1799Result;
    int[][] lc1799GcdCache;
    Integer[] lc1799Memo;

    public int maxScore(int[] nums) {
        int n = nums.length / 2;
        int allMask = (1 << (2 * n)) - 1;
        lc1799Result = (1 + n) * n / 2;
        lc1799GcdCache = new int[n * 2][n * 2];
        lc1799Memo = new Integer[1 << (nums.length)];
        for (int i = 0; i < n * 2; i++) {
            for (int j = i + 1; j < n * 2; j++) {
                lc1799GcdCache[i][j] = lc1799GcdCache[j][i] = gcd(nums[i], nums[j]);
            }
        }
        return lc1799Helper(nums, 0, allMask);
    }

    // 注意DFS应该携带什么信息, 不应该携带什么信息, 不要把当前状态(比如这里的score)放进函数入参, 而应该动态计算, 动态更新 (见highlight)
    private int lc1799Helper(int[] nums, int curMask, int allMask) {
        if (lc1799Memo[curMask] != null) return lc1799Memo[curMask];
        lc1799Memo[curMask] = 0;
        int selectable = allMask ^ curMask;
        for (int subset = selectable; subset != 0; subset = (subset - 1) & selectable) {
            if (Integer.bitCount(subset) == 2) {
                int[] select = new int[2];
                int ctr = 0;
                for (int i = 0; i < nums.length; i++) {
                    if (((subset >> i) & 1) == 1) {
                        select[ctr++] = i;
                    }
                    if (ctr == 2) break;
                }
                int newMask = subset ^ curMask;
                lc1799Memo[curMask] = Math.max(lc1799Memo[curMask],
                        lc1799Helper(nums, newMask, allMask) + lc1799GcdCache[select[0]][select[1]] * Integer.bitCount(newMask) / 2); // Highlight
            }
        }
        return lc1799Memo[curMask];
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // LC844
    public boolean backspaceCompare(String s, String t) {
        Deque<Character> ss = new LinkedList<>();
        Deque<Character> ts = new LinkedList<>();
        for (char c : s.toCharArray()) {
            if (c != '#') {
                ss.push(c);
            } else {
                if (!ss.isEmpty()) ss.pop();
            }
        }
        for (char c : t.toCharArray()) {
            if (c != '#') {
                ts.push(c);
            } else {
                if (!ts.isEmpty()) ts.pop();
            }
        }
        if (ss.size() != ts.size()) return false;
        while (!ss.isEmpty()) {
            if (ss.pollLast() != ts.pollLast()) return false;
        }
        return true;
    }

    // LC650 **
    public int minSteps(int n) {
        if (n == 1) return 0;
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                if (i % j == 0) {
                    dp[i] = dp[j] + dp[i / j];
                }
            }
        }
        return dp[n];
    }

    // LC1255
    int lc1255Max;

    public int maxScoreWords(String[] words, char[] letters, int[] score) {
        lc1255Max = 0;
        int[] usable = new int[26];
        for (char c : letters) {
            usable[c - 'a']++;
        }
        boolean[] canAdd = new boolean[words.length];
        for (int i = 0; i < words.length; i++) {
            int[] tmp = new int[26];
            System.arraycopy(usable, 0, tmp, 0, 26);
            boolean flag = true;
            for (char c : words[i].toCharArray()) {
                tmp[c - 'a']--;
                if (tmp[c - 'a'] < 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) canAdd[i] = true;
        }
        List<String> addableWords = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (canAdd[i]) addableWords.add(words[i]);
        }
        int[] addableScores = new int[addableWords.size()];
        for (int i = 0; i < addableWords.size(); i++) {
            for (char c : addableWords.get(i).toCharArray()) {
                addableScores[i] += score[c - 'a'];
            }
        }

        lc1255Backtrack(0, 0, usable, addableWords, addableScores);

        return lc1255Max;
    }

    private void lc1255Backtrack(int curIdx, int curScore, int[] curUsable, List<String> addableWords, int[] addableScores) {
        if (curIdx == addableWords.size()) {
            lc1255Max = Math.max(lc1255Max, curScore);
            return;
        }
        for (int i = curIdx; i < addableWords.size(); i++) {
            subCurWordFreq(curUsable, addableWords, i);
            if (!isCanUse(curUsable)) {
                addBackCurWordFreq(curUsable, addableWords, i);
                lc1255Backtrack(i + 1, curScore, curUsable, addableWords, addableScores);
            } else {
                lc1255Backtrack(i + 1, curScore + addableScores[i], curUsable, addableWords, addableScores);
                addBackCurWordFreq(curUsable, addableWords, i);
            }
        }
    }

    private boolean isCanUse(int[] curUsable) {
        for (int j = 0; j < 26; j++) {
            if (curUsable[j] < 0) {
                return false;
            }
        }
        return true;
    }

    private void addBackCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']++;
        }
    }

    private void subCurWordFreq(int[] curUsable, List<String> addableWords, int i) {
        for (char c : addableWords.get(i).toCharArray()) {
            curUsable[c - 'a']--;
        }
    }

    // LC1277
    public int countSquares(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1]; // dp[i][j] 表示以matrix[i][j]为右下角的最大正方形边长
        int result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    dp[i + 1][j + 1] = 0;
                } else {
                    dp[i + 1][j + 1] = 1 + Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]);
                }
                result += dp[i + 1][j + 1];
            }
        }
        return result;
    }

    // LC147
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return head;
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode lastSorted = head, cur = head.next;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode prev = dummy;
                while (prev.next.val <= cur.val) {
                    prev = prev.next;
                }
                lastSorted.next = cur.next;
                cur.next = prev.next;
                prev.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    // JSOF 52 LC160
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode aPtr = headA, bPtr = headB;
        int aLen = 0, bLen = 0;
        while (aPtr != null && aPtr.next != null) {
            aLen++;
            aPtr = aPtr.next;
        }
        while (bPtr != null && bPtr.next != null) {
            bLen++;
            bPtr = bPtr.next;
        }
        if (aPtr != bPtr) return null;
        ListNode fast = aLen > bLen ? headA : headB;
        ListNode slow = fast == headA ? headB : headA;
        int aheadStep = Math.abs(aLen - bLen);
        while (aheadStep != 0) {
            fast = fast.next;
            aheadStep--;
        }
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}

class ListNode {
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

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Trie {

    public TrieNode root;

    public Trie() {
        this.root = new TrieNode();
    }

    public boolean addWord(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            cur.children.putIfAbsent(c, new TrieNode());
            cur = cur.children.get(c);
        }
        if (cur.isEnd) return false;
        return cur.isEnd = true;
    }

    public boolean beginWith(String prefix) {
        TrieNode cur = root;
        for (char c : prefix.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return true;
    }

    public boolean search(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            if (!cur.children.containsKey(c)) return false;
            cur = cur.children.get(c);
        }
        return cur.isEnd;
    }

}

class TrieNode {
    Map<Character, TrieNode> children;
    boolean isEnd;

    public TrieNode() {
        children = new HashMap<>();
        isEnd = false;
    }
}

class DisjointSetUnion {
    Map<Integer, Integer> parent = new HashMap<>();

    public boolean add(int i) {
        if (parent.containsKey(i)) return false;
        parent.put(i, i);
        return true;
    }

    // 找最终父节点
    public int find(int i) {
        int cur = i;
        while (parent.get(cur) != cur) {
            cur = parent.get(cur);
        }
        int finalParent = cur;
        cur = i;
        // 路径压缩
        while (parent.get(cur) != finalParent) {
            int origParent = parent.get(cur);
            parent.put(cur, finalParent);
            cur = origParent;
        }
        return finalParent;
    }

    public boolean merge(int i, int j) {
        int ip = find(i);
        int jp = find(j);
        if (ip == jp) return false;
        parent.put(ip, jp);
        return true;
    }

    public boolean isConnect(int i, int j) {
        return find(i) == find(j);
    }

    public Map<Integer, Set<Integer>> allGroups() {
        for (int i : parent.keySet()) {
            find(i);
        }
        Map<Integer, Set<Integer>> result = new HashMap<>();
        for (int i : parent.keySet()) {
            result.putIfAbsent(parent.get(i), new HashSet<>());
            result.get(parent.get(i)).add(i);
        }
        return result;
    }
}