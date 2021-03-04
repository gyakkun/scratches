import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

class Scratch {
    public static void main(String[] args) {
        System.err.println(judgePalindrome(""));
    }

    public List<Integer> filterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
        //restaurants[i] = [id, rating, veganFriendly, price, distance]
        return Arrays.asList(restaurants).stream()
                .filter(ints -> !((veganFriendly == 1) ? ints[2] != 1 : false || ints[3] > maxPrice || ints[4] > maxDistance))
                .sorted((o1, o2) -> (o1[1] == o2[1]) ? (o2[0] - o1[0]) : (o2[1] - o1[1]))
                .map(e -> e[0]).collect(Collectors.toList());
    }

    public String longestPalindrome(String s) {
        return "";
    }

    private static boolean judgePalindrome(String s) {
        if (s.equals("")) return false;
        if (s.length() == 1) return true;
        int halfLen = s.length() >> 1;
        int len = s.length();
        for (int i = 0; i < halfLen; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }
}