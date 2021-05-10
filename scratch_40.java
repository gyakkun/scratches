import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class Scratch {
    public static void main(String[] args) throws IOException {
//        System.out.println(numOfWeights(2, new Integer[]{1, 2}, new Integer[]{2, 1}));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String str;
        while ((str = br.readLine()) != null) {
            int n = Integer.valueOf(str);
            String ip1S = br.readLine().trim();
            String ip2S = br.readLine().trim();
//            Integer[] mask = Arrays.stream(str.trim().split("\\.")).map(Integer::valueOf).toArray(Integer[]::new);
            Integer[] ip1 = Arrays.stream(ip1S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
            Integer[] ip2 = Arrays.stream(ip2S.trim().split(" ")).map(Integer::valueOf).toArray(Integer[]::new);
            int num = numOfWeights(n, ip1, ip2);
            System.out.println(num);
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


