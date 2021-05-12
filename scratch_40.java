import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {
    public static void main(String[] args) throws IOException {
//        System.out.println(new Calculator().calculate("3+2*{1+2*[-4/(8-6)+7]}"));

    }

    public static void main1(String[] args) throws IOException {
//        System.out.println(learnEnglish(969150, false));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;
        Calculator cal = new Calculator();

        while ((n = br.readLine()) != null) {
            System.out.println(cal.calculate(n));
        }


//        while ((n = br.readLine()) != null) {
//            System.out.println(ipToLong(n));
//            System.out.println(longToIp(br.readLine()));
//        }

//        Integer[] intArr = Arrays.stream(n.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);


//        System.out.println(twentyFourPoints(n));
//        while ((n = br.readLine()) != null) {
//            int minLen = Integer.valueOf(br.readLine());
//            System.out.println(maxGcRatio(n, minLen));

//            String ip1S = br.readLine().trim();
//            String ip2S = br.readLine().trim();
////            Integer[] mask = Arrays.stream(str.trim().split("\\.")).map(Integer::valueOf).toArray(Integer[]::new);
//            Integer[] ip1 = Arrays.stream(ip1S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
//            Integer[] ip2 = Arrays.stream(ip2S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
//            int num = numOfWeights(n, ip1, ip2);
//            System.out.println(num);
//        }
    }

    // HJ37
    public static int fib(int nth) {
        int cur = 1, next = 1;
        if (nth <= 2) return 1;
        for (int i = 2; i <= nth; i++) {
            int tmp = cur + next;
            cur = next;
            next = tmp;
        }
        return cur;
    }

    static Integer[] fibMemo;

    public static int fibRecursive(int i) {
        fibMemo = new Integer[i + 1];
        return fibHelper(i);
    }

    public static int fibHelper(int i) {
        if (i <= 2) return 1;
        if (fibMemo[i] != null) return fibMemo[i];
        fibMemo[i] = fibHelper(i - 1) + fibHelper(i - 2);
        return fibMemo[i];
    }

    // HJ43 走迷宫
    public static List<int[]> shortestMazePath(Integer[][] oldMaze) {
        int rowNum = oldMaze.length;
        int colNum = oldMaze[0].length;
        Integer[][] maze = new Integer[rowNum + 1][colNum + 1];
        for (int i = 1; i <= rowNum; i++) {
            System.arraycopy(oldMaze[i - 1], 0, maze[i], 1, colNum);
        }
        maze[1][1] = 1;

        Deque<int[]> queue = new LinkedList<>();
        int[][] directions = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        queue.push(new int[]{1, 1});
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int row = point[0];
            int col = point[1];
            if (row == rowNum && col == colNum) break;

            for (int[] dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];
                if (newRow > 0 && newRow <= rowNum && newCol > 0 && newCol <= colNum && maze[newRow][newCol] == 0) {
                    maze[newRow][newCol] = row * 100 + col; // 对100取模得走到这里得列, 除以100得行
                    queue.offer(new int[]{newRow, newCol});
                    if (newRow == rowNum && newCol == colNum) {
                        break;
                    }
                }
            }
        }
        int row = maze[maze.length - 1][maze[0].length - 1] / 100;
        int col = maze[maze.length - 1][maze[0].length - 1] % 100;
        List<int[]> result = new ArrayList<>();
        result.add(new int[]{rowNum, colNum});
        while (maze[row][col] != 1) {
            int newRow = maze[row][col] / 100;
            int newCol = maze[row][col] % 100;
            result.add(new int[]{row, col});
            row = newRow;
            col = newCol;
        }
        result.add(new int[]{1, 1});
        Collections.reverse(result);
        return result;
    }


    // HJ33 IP 地址 转整数
    public static long ipToLong(String ipAdd) {
        String[] split = ipAdd.split("\\.");
        long result = 0;
        for (String i : split) {
            int part = Integer.valueOf(i);
            result = (result << 8) ^ part;
        }
        return result;
    }

    // 整数转IP地址
    public static String longToIp(String ipLong) {
        long ip = Long.valueOf(ipLong);
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < 4; i++) {
            long part = ip - ((ip >> 8) << 8);
            ip = ip >> 8;
            sb.insert(0, ".");
            sb.insert(0, part);
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }


    // HJ35 蛇形矩阵
    public static void snakeMtx(int n) {
        int total = (n + 1) * n / 2;
        int ctr = 1;
        int[][] mtx = new int[n][n];
        int row = 0, col = 0;
        boolean goingUp = false;
        while (ctr <= total) {
            mtx[row][col] = ctr++;
            if (goingUp) {
                row--;
                col++;
                goingUp = !(row == 0);
            } else {
                row = col + 1;
                col = 0;
                goingUp = true;
            }
        }
        for (int[] i : mtx) {
            for (int j = 0; j < i.length; j++) {
                if (i[j] == 0) break;
                if (j + 1 >= i.length || i[j + 1] == 0) {
                    System.out.print(i[j] + "\n");
                } else {
                    System.out.print(i[j] + " ");
                }
            }
        }
        return;
    }


    // HJ20
    //1.长度超过8位
    //2.包括大小写字母.数字.其它符号,以上四种至少三种
    //3.不能有相同长度大于2的子串重复
    public static boolean validPassword(String pw) {
        if (pw.length() <= 8) return false;
        String noLowerCase = pw.replaceAll("[a-z]", "");
        String noUpperCase = pw.replaceAll("[A-Z]", "");
        String noDigit = pw.replaceAll("\\d", "");
        int lowCaseLen = pw.length() - noLowerCase.length();
        int upperCaseLen = pw.length() - noUpperCase.length();
        int digitLen = pw.length() - noDigit.length();
        int otherLen = pw.length() - lowCaseLen - upperCaseLen - digitLen;
        int ctr = 0;
        if (lowCaseLen > 0) ctr++;
        if (upperCaseLen > 0) ctr++;
        if (digitLen > 0) ctr++;
        if (otherLen > 0) ctr++;
        if (ctr < 3) return false;
        for (int i = 0; i < pw.length() - 3; i++) {
            String word = pw.substring(i, i + 3);
            int nextIdx = pw.lastIndexOf(word);
            if (nextIdx != i) return false;
        }
        return true;
    }

    // HJ61 m个苹果 n个盆子
    static Integer[][] hj61Memo;

    public static int placeApple(int m, int n) {
        hj61Memo = new Integer[m + 1][n + 1];
        return placeAppleHelper(m, n);
    }


    public static int placeAppleHelper(int m, int n) {
        if (m == 0 || n == 1) {
            return 1;
        }
        if (hj61Memo[m][n] != null) {
            return hj61Memo[m][n];
        }
        int result = -1;
        if (m < n) {
            result = placeAppleHelper(m, m);
        } else {
            result = placeAppleHelper(m, n - 1) + placeAppleHelper(m - n, n);
        }
        hj61Memo[m][n] = result;
        return result;
    }

    // HJ32 最长回文子串

    public static int longestPalindromeDp(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        String ans = "";
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = true;
                } else if (l == 1) {
                    dp[i][j] = (s.charAt(i) == s.charAt(j));
                } else {
                    dp[i][j] = (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.length()) {
                    ans = s.substring(i, i + l + 1);
                }
            }
        }
        return ans.length();
    }

    static int lpMaxLen = 0;
    static int lpMaxStartIdx = -1;
    static int lpMaxEndIdx = -1;

    public static String longestPalindrome(String s) {
        int n = s.length();
        Boolean[][] memo = new Boolean[n + 1][n + 1];
        longestPalindromeHelper(s, 0, n - 1, memo);
        return s.substring(lpMaxStartIdx, lpMaxEndIdx + 1);
    }

    private static boolean longestPalindromeHelper(String s, int start, int end, Boolean[][] memo) {
        if (start == end) {
            memo[start][end] = true;
            if (lpMaxLen == 0) {
                lpMaxLen = 1;
                lpMaxStartIdx = start;
                lpMaxStartIdx = end;
            }
            return true;
        }
        if (start > end) return false;
        if (memo[start][end] != null) {
            return memo[start][end];
        }
        boolean result = false;
        if (start + 1 == end) {
            result = s.charAt(start) == s.charAt(end);
        } else {
            longestPalindromeHelper(s, start + 1, end, memo);
            longestPalindromeHelper(s, start, end - 1, memo);
            longestPalindromeHelper(s, start + 1, end - 1, memo);
            result = s.charAt(start) == s.charAt(end) && memo[start + 1][end - 1];
        }
        if (result && (end - start + 1) > lpMaxLen) {
            lpMaxLen = end - start + 1;
            lpMaxStartIdx = start;
            lpMaxEndIdx = end;
        }
        memo[start][end] = result;
        return result;
    }


    // HJ82
    public static List<Integer> egypt(int num, int den) {
        List<Integer> result = new LinkedList<>();
        int gcd = GCD(num, den);
        num /= gcd;
        den /= gcd;
        while (num != 1) {
            double d = (double) den / (double) num;
            int maxEgypt = (int) Math.ceil(d);
            result.add(maxEgypt);
            // 通分
            int newNum = maxEgypt * num - den;
            int newDen = den * maxEgypt;
            gcd = GCD(newNum, newDen);
            num = newNum / gcd;
            den = newDen / gcd;
        }
        result.add(den);
        return result;
    }

    // HJ80
    static Set<Integer> s = new HashSet<>();

    public static String mergeInt(Integer[] arr) {
        for (int i : arr) {
            s.add(i);
        }
        List<Integer> l = new ArrayList<>(s);
        StringBuffer sb = new StringBuffer();
        for (int i : l) {
            sb.append(i);
        }
        return sb.toString();
    }

    // HJ77 Solution 非常精妙
    static Deque<String> stack1 = new LinkedList<>();
    static Deque<String> stack2 = new LinkedList<>();
    static List<String> result = new LinkedList<>();

    public static void hj77dfs(String status) {
        if (stack1.isEmpty() && stack2.isEmpty()) {
            result.add(status.trim());
            return;
        }
        if (!stack2.isEmpty()) {
            String tmp = stack2.pop();
            hj77dfs(status + " " + tmp);
            stack2.push(tmp);
        }
        if (!stack1.isEmpty()) {
            String tmp = stack1.pop();
            stack2.push(tmp);
            hj77dfs(status);
            stack2.pop();
            stack1.push(tmp);
        }
    }

    // HJ44 解数独
    public static boolean sudoku(int[][] board, int left) {
        if (left == 0) {
            for (int[] i : board) {
                for (int j = 0; j < 9; j++) {
                    if (j == 8) {
                        System.out.print(i[j] + "\n");
                    } else {
                        System.out.print(i[j] + " ");
                    }
                }
            }
            return true;
        }

        Set<Integer> notAllow = new HashSet<>();
        int i = 0, j = 0;
        for (i = 0; i < 9; i++) {
            boolean flag = false;
            for (j = 0; j < 9; j++) {
                if (board[i][j] == 0) {
                    flag = true;
                    break;
                }
            }
            if (flag) break;
        }
        if (i == 9 || j == 9) return false;
        int row = i, col = j;
        for (int k = 0; k < 9; k++) {
            notAllow.add(board[i][k]);
            notAllow.add(board[k][j]);
        }
        i /= 3;
        j /= 3;
        for (int k = 0; k < 3; k++) {
            for (int n = 0; n < 3; n++) {
                notAllow.add(board[i * 3 + k][j * 3 + n]);
            }
        }
        List<Integer> allow = new ArrayList<>();
        for (i = 1; i <= 9; i++) {
            if (!notAllow.contains(i)) allow.add(i);
        }
        if (allow.size() == 0) return false;

        for (int num : allow) {
            board[row][col] = num;
            if (sudoku(board, left - 1)) {
                return true;
            }
            board[row][col] = 0;
        }

        return false;
    }

    // HJ5 进制转换, 16进制转10进制
    public static int hexToDec(String hexStr) {
        return Integer.valueOf(hexStr.replaceAll("0x", ""), 16);
    }

    // HJ89
    final static Map<String, Integer> poker = new HashMap<String, Integer>() {{
        put("A", 1);
        put("2", 2);
        put("3", 3);
        put("4", 4);
        put("5", 5);
        put("6", 6);
        put("7", 7);
        put("8", 8);
        put("9", 9);
        put("10", 10);
        put("J", 11);
        put("Q", 12);
        put("K", 13);
        put("JOKER", -1);
    }};
    final static Map<Integer, String> pokerReverse = new HashMap<Integer, String>() {{
        put(1, "A");
        put(2, "2");
        put(3, "3");
        put(4, "4");
        put(5, "5");
        put(6, "6");
        put(7, "7");
        put(8, "8");
        put(9, "9");
        put(10, "10");
        put(11, "J");
        put(12, "Q");
        put(13, "K");
    }};

    public static String twentyFourPoints(String s) {
        s = s.toUpperCase();
        String[] pokers = s.trim().split(" ");
        if (pokers.length != 4) return "ERROR";
        int[] cards = new int[4];
        for (int i = 0; i < 4; i++) {
            if (!poker.containsKey(pokers[i])) return "ERROR";
            cards[i] = poker.get(pokers[i]);
            if (cards[i] == -1) {
                return "ERROR";
            }
        }
        Calculator cal = new Calculator();
        String[] operPerm = new String[64];
        char[] opers = new char[]{'+', '-', '*', '/'};
        int ctr = 0;
        for (int i = 0; i < 4; i++) {
            StringBuffer sb = new StringBuffer();
            sb.append(opers[i]);
            for (int j = 0; j < 4; j++) {
                sb.append(opers[j]);
                for (int k = 0; k < 4; k++) {
                    sb.append(opers[k]);
                    operPerm[ctr++] = sb.toString();
                    sb.deleteCharAt(2);
                }
                sb.deleteCharAt(1);
            }
            sb.deleteCharAt(0);
        }
        List<int[]> numPerm = new ArrayList<>(24);
        permNum(cards, numPerm, 0, 3);
        for (String operP : operPerm) {
            for (int[] numP : numPerm) {
                int result = numP[0];
                for (int i = 0; i < 3; i++) {
                    switch (operP.charAt(i)) {
                        case '+':
                            result += numP[i + 1];
                            break;
                        case '-':
                            result -= numP[i + 1];
                            break;
                        case '*':
                            result *= numP[i + 1];
                            break;
                        case '/':
                            result /= numP[i + 1];
                            break;
                        default:
                    }
                }
                if (result == 24) {
                    StringBuffer sb = new StringBuffer();
                    for (int i = 0; i < 3; i++) {
                        sb.append(pokerReverse.get(numP[i]));
                        sb.append(operP.charAt(i));
                    }
                    sb.append(pokerReverse.get(numP[3]));
                    return sb.toString();
                }
            }
        }

        return "NONE";
    }

    private static void permNum(int[] cards, List<int[]> numPerm, int start, int end) {
        if (start == end) {
            int[] result = new int[4];
            System.arraycopy(cards, 0, result, 0, 4);
            numPerm.add(result);
        }
        for (int i = start; i <= end; i++) {
            int tmp = cards[start];
            cards[start] = cards[i];
            cards[i] = tmp;
            permNum(cards, numPerm, start + 1, end);
            tmp = cards[start];
            cards[start] = cards[i];
            cards[i] = tmp;
        }
    }

    // HJ107 立方根 不使用库函数, 留1位小数
    public static double cubeRoot(double d) {
        boolean negFlag = d < 0;
        if (negFlag) d = -d;
        double left = 0d;
        double right = d;
        while (right - left > 0.01d) {
            double mid = (left + right) / 2;
            if (mid * mid * mid > d) {
                right = mid;
            } else {
                left = mid;
            }
        }
        left = (double) ((int) (left * 10)) / 10;
        right = (double) ((int) (right * 10)) / 10;
        double result = left;
        for (double i = left - 1d; i <= right + 1d; i += 0.1d) {
            if (Math.abs(i * i * i - d) < Math.abs(result * result * result - d)) {
                result = i;
            }
        }
        return negFlag ? -result : result;
    }


    // HJ108 最小公倍数
    public static int LCM(int a, int b) {
        return a * b / GCD(a, b);
    }

    public static int GCD(int a, int b) {
        return a % b == 0 ? b : GCD(b, a % b);
    }

    // HJ16 背包
    public static int shoppingList(int N, int m, int[] price, int[] importance, int[] mainPartNum) {
        // 注意mainPartNum表示的主键下标是从1开始算的
        // 如果要买归类为附件的物品，必须先买该附件所属的主件。每个主件可以有 0 个、 1 个或 2 个附件。
        // 总价格N, 总个数m, N是10的整数倍, 作处理
        int n = price.length;
        N /= 10;
        for (int i = 0; i < n; i++) {
            price[i] /= 10;
        }

        // 处理附件, 将 主 / 主 - 附1 / 主 - 附2 / 主 - 附1+2 分为4个商品, 有各自的价格和重要性,
        // 4个商品为1组, 每组只能选一个转移, 考虑使用下标和取模来限制转移
        Map<Integer, List<Integer>> mainPartSubPartMap = new HashMap<>();
        // 结构 key - 主件idx, value - 附件id list, 容量为2
        for (int i = 0; i < n; i++) {
            if (mainPartNum[i] == 0) {
                mainPartSubPartMap.putIfAbsent(i, new ArrayList<>(2));
            } else {
                mainPartSubPartMap.putIfAbsent(mainPartNum[i] - 1, new ArrayList<>(2));
                mainPartSubPartMap.get(mainPartNum[i] - 1).add(i);
            }
        }
        // 构造新的price value 数组
        int newLen = mainPartSubPartMap.keySet().size() * 4;
        int[] newPrice = new int[newLen], newValue = new int[newLen];
        int ctr = 0;
        for (int i : mainPartSubPartMap.keySet()) {
            // mod4 的 0,1,2,3 分别对应 主 / 主 - 附1 / 主 - 附2 / 主 - 附1+2
            newPrice[ctr] = price[i];
            newValue[ctr] = price[i] * importance[i];
            if (mainPartSubPartMap.get(i).size() == 0) {
                ;
            } else if (mainPartSubPartMap.get(i).size() == 1) {
                newPrice[ctr + 1] = newPrice[ctr + 2] = newPrice[ctr + 3]
                        = price[i] + price[mainPartSubPartMap.get(i).get(0)];
                newValue[ctr + 1] = newValue[ctr + 2] = newValue[ctr + 3]
                        = price[i] * importance[i] + price[mainPartSubPartMap.get(i).get(0)] * importance[mainPartSubPartMap.get(i).get(0)];
            } else if (mainPartSubPartMap.get(i).size() == 2) {
                newPrice[ctr + 1] = price[i] + price[mainPartSubPartMap.get(i).get(0)];
                newPrice[ctr + 2] = price[i] + price[mainPartSubPartMap.get(i).get(1)];
                newPrice[ctr + 3] = newPrice[ctr + 1] + newPrice[ctr + 2] - price[i];

                newValue[ctr + 1] = price[i] * importance[i] + price[mainPartSubPartMap.get(i).get(0)] * importance[mainPartSubPartMap.get(i).get(0)];
                newValue[ctr + 2] = price[i] * importance[i] + price[mainPartSubPartMap.get(i).get(1)] * importance[mainPartSubPartMap.get(i).get(1)];
                newValue[ctr + 3] = newValue[ctr + 1] + newValue[ctr + 2] - price[i] * importance[i];
            }
            ctr += 4;
        }
        int[] dp = new int[N + 1];
        // dp[i][j] 表示购买前i项商品在限价为j的情况下能得到的最大价值
        // 其中 若 i%4==0, 则在i,i+1,i+2,i+3四件商品中只能选一件购买

        for (int i = 1; i <= newLen; i += 4) {
            for (int j = N; j >= 0; j--) {
                int tmpMaxValue = dp[j];
                for (int k = 0; k < 4; k++) {
                    if (j - newPrice[i + k - 1] >= 0) {
                        int pv = Math.max(dp[j], dp[j - newPrice[i + k - 1]] + newValue[i + k - 1]);
                        tmpMaxValue = Math.max(tmpMaxValue, pv);
                    }
                }
                dp[j] = tmpMaxValue;
            }
        }
        return dp[N] * 10;
    }

    // HJ28 匈牙利算法
    public static int primePartner(Integer[] arr) {
        int n = arr.length;
        Arrays.sort(arr);
        int max = arr[arr.length - 1];
        int maxM1 = arr.length >= 2 ? arr[arr.length - 2] : arr[arr.length - 1];
        int primeRange = max + maxM1;
        int primeCtr = 0;
        boolean[] notPrimeMark = new boolean[primeRange + 1];
        List<Integer> primes = new ArrayList<>();
        for (int i = 2; i <= primeRange; i++) {
            if (!notPrimeMark[i]) {
                primes.add(i);
            }
            for (int j = 0; j < primes.size(); j++) {
                if (i * primes.get(j) > primeRange) {
                    break;
                }
                notPrimeMark[i * primes.get(j)] = true;
                if (i % primes.get(j) == 0) {
                    break;
                }
            }
        }
        Set<Integer> primeSet = new HashSet<>(primes);

        boolean[][] edgeMtx = new boolean[n][n];
        boolean[] rightVisited = new boolean[n];
        Integer[] rightLeftMap = new Integer[n]; // 右侧对应的左侧元素
        int result = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (primeSet.contains(arr[i] + arr[j])) {
                    edgeMtx[i][j] = true;
                    edgeMtx[j][i] = true;
                }
            }
        }
        for (int leftIdx = 0; leftIdx < n; leftIdx++) {
            Arrays.fill(rightVisited, false);
            if (match(leftIdx, n, edgeMtx, rightVisited, rightLeftMap)) {
                result++;
            }
        }
        return result / 2;

    }

    public static boolean match(int leftIdx, int n, boolean[][] edgeMtx, boolean[] rightVisited, Integer[] rightLeftMap) {
        for (int rightIdx = 0; rightIdx < n; rightIdx++) {
            if (edgeMtx[leftIdx][rightIdx] && !rightVisited[rightIdx]) {
                rightVisited[rightIdx] = true;
                if (rightLeftMap[rightIdx] == null || match(rightLeftMap[rightIdx], n, edgeMtx, rightVisited, rightLeftMap)) {
                    rightLeftMap[rightIdx] = leftIdx;
                    return true;
                }
            }
        }
        return false;
    }

    // HJ6 分解质因数
    public static List<Integer> primeFactors(int i) {
        List<Integer> result = new LinkedList<>();
        int factor = 2;
        while (i != 0) {
            while (i % factor == 0) {
                result.add(factor);
                i /= factor;
            }
            factor++;
            if (factor > i) break;
        }
        return result;
    }

    // HJ22
    public static int softDrink(int n) {
        int result = 0;
        // 3瓶子换1瓶
        while (n / 3 != 0) {
            result += n / 3;
            n = n - (n / 3) * 3 + (n / 3);
        }
        if (n == 2) result++;
        return result;
    }

    // HJ24 使用最佳LIS算法
    public static int chorus(Integer[] heights) {
        int n = heights.length;
        TreeSet<Integer> left = new TreeSet<>();
        TreeSet<Integer> right = new TreeSet<>();
        int[] leftLis = new int[n];
        int[] rightLis = new int[n];
        for (int pivot = 0; pivot < n; pivot++) {
            Integer ceilLeft = left.ceiling(heights[pivot]);
            Integer ceilRight = right.ceiling(heights[n - 1 - pivot]);
            if (ceilLeft != null) {
                left.remove(ceilLeft);
            }
            if (ceilRight != null) {
                right.remove(ceilRight);
            }
            left.add(heights[pivot]);
            right.add(heights[n - 1 - pivot]);
            leftLis[pivot] = left.size();
            rightLis[n - 1 - pivot] = right.size();
        }
        int max = 0;
        for (int i = 0; i < n; i++) {
            max = Math.max(leftLis[i] + rightLis[i], max);
        }
        return (n - max + 1);
    }


    // HJ63 算最高GC比例的子串, 定义的应该是最小子串长度而不是固定子串长度
    public static String maxGcRatio(String gene, int minLen) {
        int[] prefix = new int[gene.length() + 1];
        for (int i = 1; i <= gene.length(); i++) {
            if (gene.charAt(i - 1) == 'G' || gene.charAt(i - 1) == 'C') {
                prefix[i] = prefix[i - 1] + 1;
            } else {
                prefix[i] = prefix[i - 1];
            }
        }
        int startIdx = 0, max = 0, maxLen = minLen;
        double maxRatio = 0d;
        for (int i = minLen; i <= gene.length(); i++) {
            for (int j = 0; j <= (gene.length() - i); j++) {
                int count = prefix[j + i] - prefix[j];
                double ratio = (double) count / (double) i;
                if (ratio > maxRatio) {
                    maxRatio = ratio;
                    startIdx = j;
                    maxLen = i;
                }
            }
        }

        return gene.substring(startIdx, startIdx + maxLen);
    }

    // HJ45
    public static int howBeautiful(String str) {
        str = str.toLowerCase();
        Map<Character, Integer> freqMap = new HashMap<>();
        for (int i = 0; i < 26; i++) {
            freqMap.put((char) ('a' + i), 0);
        }
        for (char c : str.toCharArray()) {
            freqMap.put(c, freqMap.get(c) + 1);
        }
        List<Map.Entry<Character, Integer>> freqList = new ArrayList<>(26);
        for (Map.Entry<Character, Integer> entry : freqMap.entrySet()) {
            freqList.add(entry);
        }
        freqList.sort(new Comparator<Map.Entry<Character, Integer>>() {
            @Override
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }
        });
        int result = 0;
        for (int i = 0; i < 26; i++) {
            result += (26 - i) * freqList.get(i).getValue();
        }
        return result;

    }

    // HJ42
    final static String[] numEng = new String[]{null, "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    final static String[] numTenToTwenty = new String[]{"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    final static String[] numTensEng = new String[]{null, "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

    public static String learnEnglish(int i, boolean isLastTuple) {
        StringBuffer sb = new StringBuffer();
        if (i == 0) return "";
        else if (i > 0 && i < 1000) {
            int ones = i % 10;
            int tens = (i / 10) % 10;
            int hundreds = (i / 100) % 10;
            if (hundreds != 0) {
                sb.append(numEng[hundreds]);
                sb.append(" hundred");
                if (tens == 0 && ones == 0) {
                    return sb.toString();
                } else {
                    sb.append(" and ");
                }
            }
            if (tens == 0) {
                sb.append(numEng[ones]);
            } else if (tens == 1) {
                sb.append(numTenToTwenty[ones]);
            } else {
                sb.append(numTensEng[tens]);
                if (ones != 0) {
                    sb.append(" ");
                    sb.append(numEng[ones]);
                }
            }
            if (hundreds == 0 && tens == 0 && isLastTuple) {
                sb.insert(0, "and ");
            }
            return sb.toString();
        } else if (i >= 1000 && i < 1000000) {
            return (" " + learnEnglish(i / 1000, false) + " thousand " + learnEnglish(i % 1000, true) + " ").trim();
        } else if (i >= 1000000 && i < 1000000000) {
            return (" " + learnEnglish(i / 1000000, false) + " million " + learnEnglish(i % 1000000, (i % 1000000) < 1000) + " ").trim();
        } else {
            return "Error";
        }
    }

    // HJ41
    public static int numOfWeights(int n, Integer[] eachWeight, Integer[] eachCount) {
        Set<Integer> s = new HashSet<>();
        s.add(0);
        for (int i = 0; i < n; i++) {
            Set<Integer> tmp = new HashSet<>(s);
            Iterator<Integer> it = tmp.iterator();
            while (it.hasNext()) {
                int next = it.next();
                for (int j = 1; j <= eachCount[i]; j++) {
                    s.add(next + eachWeight[i] * j);
                }
            }
        }
        return s.size();
    }
//    public static void main(String[] args) {
//        Scanner in = new Scanner(System.in);
//        // 注意 hasNext 和 hasNextLine 的区别
//        while (in.hasNextInt()) { // 注意 while 处理多个 case
//            int a = in.nextInt();
//            int b = in.nextInt();
//            System.out.println(a + b);
//        }
//    }
}

// HJ54
class Calculator {
    public int calculate(String s) {
        return (int) evalRPN(toRPN(decodeExpression(s)));
    }

    public List<String> toRPN(List<String> express) {
        List<String> rpn = new LinkedList<>();
        Deque<String> stack = new LinkedList<>();
        Set<String> notNumber = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
            add("(");
            add(")");
        }};
        String tmp;
        for (String token : express) {
            if (!notNumber.contains(token)) {
                rpn.add(token);
            } else if (token.equals("(")) {
                stack.push(token);
            } else if (token.equals(")")) {
                while (!(tmp = stack.pop()).equals("(")) {
                    rpn.add(tmp);
                }

            } else {
                while (!stack.isEmpty() && getOperPriority(stack.peek()) >= getOperPriority(token)) {
                    rpn.add(stack.pop());
                }
                stack.push(token);
            }
        }
        while (!stack.isEmpty()) {
            rpn.add(stack.pop());
        }
        return rpn;
    }

    private List<String> decodeExpression(String express) {
        express = express.replaceAll("\\{", "(");
        express = express.replaceAll("\\}", ")");
        express = express.replaceAll("\\[", "(");
        express = express.replaceAll("\\]", ")");

        express = express.replaceAll("\\ ", "");
        express = express.replaceAll("\\(\\+", "(0+");
        express = express.replaceAll("\\(\\-", "(0-");
        express = express.replaceAll("\\((\\d+\\.?\\d*)\\)", "$1");
        List<String> result = new LinkedList<>();
        int i = 0;
        StringBuffer sb;
        do {
            if ((express.charAt(i) < '0' || express.charAt(i) > '9') && express.charAt(i) != '.') {
                result.add(express.charAt(i) + "");
                i++;
            } else {
                sb = new StringBuffer();
                while (i < express.length() && ((express.charAt(i) >= '0' && express.charAt(i) <= '9') || express.charAt(i) == '.')) {
                    sb.append(express.charAt(i));
                    i++;
                }
                result.add(sb.toString());
            }
        } while (i < express.length());
        return result;
    }

    private int getOperPriority(String oper) {
        switch (oper) {
            case "+":
            case "-":
                return 1;
            case "*":
            case "/":
                return 2;
            default:
                return -1;
        }
    }

    // LC150 逆波兰表达式
    public int evalRPN(List<String> tokens) {
        Deque<String> stack = new LinkedList<>();
        stack.push("0");
        Set<String> oper = new HashSet<String>() {{
            add("+");
            add("-");
            add("/");
            add("*");
        }};
        for (String token : tokens) {
            if (oper.contains(token)) {
                int a = Integer.parseInt(stack.pop());
                int b = Integer.parseInt(stack.pop());
                int tmp;
                switch (token) {
                    case "+":
                        tmp = a + b;
                        break;
                    case "-":
                        tmp = b - a;
                        break;
                    case "/":
                        tmp = b / a;
                        break;
                    case "*":
                        tmp = a * b;
                        break;
                    default:
                        tmp = 0;
                }
                stack.push(String.valueOf(tmp));
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
    }
}
