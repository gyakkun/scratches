package moe.nyamori.test.historical;

import java.util.*;

class scratch_3 {


    public static void main(String[] args) {
//        printIntArray(wiggleSort(new int[]{1, 2, 3, 4, 5, 6}));
        System.out.println(monotoneIncreasingDigits(1234));
    }

    public static int[] toIntArr(int N){
        int numOfDigit = 0;
        int nCopy = N;
        while (N != 0) {
            N /= 10;
            numOfDigit++;
        }
        N = nCopy;
        int[] digitArr = new int[numOfDigit];
        for (int i = 0; i < numOfDigit; i++) {
            int currentIdx = numOfDigit - i - 1;
            digitArr[currentIdx] = N % 10;
            N /= 10;
        }
        return digitArr;
    }
    public static boolean judgeLegal(int[] digitArr){
        for(int i = 1;i<digitArr.length;i++){
            if(digitArr[i-1]<digitArr[i]){
                return false;
            }
        }
        return true;
    }

    public static int monotoneIncreasingDigits(int N) {
        int numOfDigit = 0;
        int nCopy = N;
        int origHSB = 0;
        while (N != 0) {
            N /= 10;
            numOfDigit++;
        }
        N = nCopy;
        int[] digitArr = new int[numOfDigit];
        for (int i = 0; i < numOfDigit; i++) {
            int currentIdx = numOfDigit - i - 1;
            digitArr[currentIdx] = N % 10;
            N /= 10;
        }
        origHSB = digitArr[0];

        int firstNotSatisfyDigitIdx = -1;
        int fromThisDigitAllNineIdx = 0;

        for (int i = 1; i < numOfDigit; i++) {
            if (digitArr[i - 1] > digitArr[i]) {
                firstNotSatisfyDigitIdx = i;
                break;
            }
        }

        if (firstNotSatisfyDigitIdx != -1) {
            for (int i = firstNotSatisfyDigitIdx; i > 0; i--) {
                if (digitArr[i - 1] > digitArr[i]) {
                    digitArr[i - 1] = digitArr[i - 1] - 1;
                    fromThisDigitAllNineIdx = i;
                }
            }

            for (int i = fromThisDigitAllNineIdx; i > 0 && i < numOfDigit; i++) {
                digitArr[i] = 9;
            }

        }


        int result = 0;
        for (int i = 0; i < numOfDigit; i++) {
            result = result * 10 + digitArr[i];
        }

        return result;
    }


    public static void printIntArray(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.print("\r\n");
    }

    public static int[] wiggleSort(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] < nums[j]) {
                    int tmp = nums[i];
                    nums[i] = nums[j];
                    nums[j] = tmp;
                }
            }
        }
        for (int i = 0; i < (nums.length) / 2; i++) {
            if (i % 2 == 0) {
                int tmp = nums[i];
                nums[i] = nums[nums.length - i - 1];
                nums[nums.length - i - 1] = tmp;
            }
        }
        return nums;
    }


    public static List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();

        Map<String, List<String>> m = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            char[] tmp = strs[i].toCharArray();
            Arrays.sort(tmp);
            String sorted = new String(tmp);
            if (!m.containsKey(sorted)) {
                List<String> ls = new ArrayList<>();
                ls.add(strs[i]);
                m.put(sorted, ls);
            } else {
                m.get(sorted).add(strs[i]);
            }
        }

        for (String s : m.keySet()) {
            result.add(m.get(s));
        }
        return result;
    }

    public static String predictPartyVictory(String senate) {
        Queue<Integer> qr = new LinkedList<>();
        Queue<Integer> qd = new LinkedList<>();
        int ssize = senate.length();
        for (int i = 0; i < ssize; i++) {
            if (senate.charAt(i) == 'R') {
                qr.offer(i);
            } else {
                qd.offer(i);
            }
        }
        while (!qr.isEmpty() && !qd.isEmpty()) {
            if (qr.peek() < qd.peek()) {
                qd.poll();
                qr.offer(qr.poll() + ssize);
            } else {
                qr.poll();
                qd.offer(qd.poll() + ssize);
            }
        }
        return qr.isEmpty() ? "Dire" : "Radiant";
    }
}