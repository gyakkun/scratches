import java.util.Currency;
import java.util.LinkedList;

class Scratch {
    public static void main(String[] args) {
        solution("0123");
    }

    private static void solution(String s) {
        LinkedList<Integer> list = new LinkedList<>();
        backtrack(list, s, 0);
        for (int j = 0; j < list.size(); j++) {
            System.out.println(list.get(j));
        }
    }

    private static boolean backtrack(LinkedList<Integer> list, String s, int index) {
        if (index == s.length()) {
            return list.size() >= 3;
        }

        for (int i = index; i < s.length(); i++) {

            if (i > index && s.charAt(index) == '0') {
                break;
            }

            long curLong = Long.valueOf(s.substring(index, i+1));

            if (curLong > Integer.MAX_VALUE) {
                break;
            }

            int cur = (int) curLong;

            if (list.size() >= 2) {
                int sum = list.get(list.size() - 1) + list.get(list.size() - 2);
                if (cur > sum) {
                    break;
                } else if (cur < sum) {
                    continue;
                }
            }


            list.add(cur);

            if (backtrack(list, s, i + 1)) {
                return true;
            }

            list.remove(list.size() - 1);

        }
        return false;
    }
}