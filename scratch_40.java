import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {
    public static void main(String[] args) throws IOException {
//        System.out.println(learnEnglish(969150, false));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String str;
        while ((str = br.readLine()) != null) {
            System.out.println(learnEnglish(Integer.valueOf(str), Integer.valueOf(str) < 1000));
//            int n = Integer.valueOf(str);
//            String ip1S = br.readLine().trim();
//            String ip2S = br.readLine().trim();
////            Integer[] mask = Arrays.stream(str.trim().split("\\.")).map(Integer::valueOf).toArray(Integer[]::new);
//            Integer[] ip1 = Arrays.stream(ip1S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
//            Integer[] ip2 = Arrays.stream(ip2S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
//            int num = numOfWeights(n, ip1, ip2);
//            System.out.println(num);
        }
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


