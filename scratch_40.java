import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {
    public static void main1(String[] args) throws IOException {
        System.out.println(maxGcRatio("AAAACCACCCCACAAAAA", 5));
    }

    public static void main(String[] args) throws IOException {
//        System.out.println(learnEnglish(969150, false));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String n;

        n = br.readLine();
        Calculator calculator = new Calculator();
        System.out.println(calculator.calculate(n));

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
            return (" " + learnEnglish(i / 1000000, false) + " million " + learnEnglish(i % 1000000, true) + " ").trim();
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
