import java.util.*;

class Main {
    private static int minCharge = 500000001;
    private static int totalVotes = 0;
    private static int halfVotes = 0;
    private static int[] dp;
    private static int[] price;
    private static int[] eachNumOfVote;

    public static void main(String[] args) {
        int num = 0;
        Scanner scan = new Scanner(System.in);
        if (scan.hasNextLine()) {
            String tmp = scan.nextLine();
            num = Integer.valueOf(tmp);
            price = new int[num + 1];
            eachNumOfVote = new int[num + 1];
            price[0] = eachNumOfVote[0] = 0;
        }
        for (int i = 0; i < num; i++) {
            if (scan.hasNextLine()) {
                String tmp = scan.nextLine();
                String[] nums = tmp.split(" ");
                eachNumOfVote[i + 1] = Integer.valueOf(nums[0]);
                price[i + 1] = Integer.valueOf(nums[1]);
                totalVotes += eachNumOfVote[i + 1];
            }
        }
        halfVotes = (totalVotes + 1) / 2;
        //DP
        dp = new int[totalVotes + 1];
        for (int i = 0; i <= totalVotes; i++) {
            dp[i] = 500000001;
        }
        dp[0] = 0;
        for (int i = 1; i <= (price.length - 1); i++) {
            for (int j = totalVotes; j >= eachNumOfVote[i]; j--) {
                dp[j] = Math.min(dp[j], dp[j - eachNumOfVote[i]] + price[i]);
            }
        }
        for (int i = halfVotes; i <= totalVotes; i++) {
            if (dp[i] < minCharge) {
                minCharge = dp[i];
            }
        }
        System.out.println(minCharge);
    }
}