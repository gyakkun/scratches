import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {
    public static void main(String[] args) throws IOException {

        int[] price = new int[]{800, 400, 300, 400, 500};
        int[] importance = new int[]{2, 5, 5, 3, 2};
        int[] mainPartNum = new int[]{0, 1, 1, 0, 0};
        int N = 1000;
        int m = 5;

        System.out.println(shoppingList(N, m, price, importance, mainPartNum));
    }

    public static void main2(String[] args) throws IOException {
//        System.out.println(learnEnglish(969150, false));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;
        n = br.readLine();
        Integer[] intArr = Arrays.stream(n.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
        int N = intArr[0];
        int m = intArr[1];
        int[] price = new int[m];
        int[] importance = new int[m];
        int[] mainPartNum = new int[m];


        for (int i = 0; i < m; i++) {
            String arr = br.readLine();
            intArr = Arrays.stream(arr.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
            price[i] = intArr[0];
            importance[i] = intArr[1];
            mainPartNum[i] = intArr[2];
        }
        System.out.println(shoppingList(N, m, price, importance, mainPartNum));

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
