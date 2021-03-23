class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.minDelToMakePalindrome("abababa"));

    }

    // Minimum Deletions in a String to make it a Palindrome，怎么删掉最少字符构成回文
    // https://www.geeksforgeeks.org/minimum-number-deletions-make-string-palindrome/
    public int minDelToMakePalindrome(String s) {
        int n = s.length();

        return n - 1;
    }
}