import sun.text.resources.cldr.es.FormatData_es_419;

import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();
        System.err.println(s.letterCombinations("2"));
        // 6-8-7+(1+6)
        // 6 8 - 7 - 1 6 + +
        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC17
    public List<String> letterCombinations(String digits) {
        Map<Character, Character[]> m = new HashMap<>();
        m.put('2', new Character[]{'a', 'b', 'c'});
        m.put('3', new Character[]{'d', 'e', 'f'});
        m.put('4', new Character[]{'g', 'h', 'i'});
        m.put('5', new Character[]{'j', 'k', 'l'});
        m.put('6', new Character[]{'m', 'n', 'o'});
        m.put('7', new Character[]{'p', 'q', 'r', 's'});
        m.put('8', new Character[]{'t', 'u', 'v'});
        m.put('9', new Character[]{'w', 'x', 'y', 'z'});
        List<String> result = new LinkedList<>();
        for (char c : digits.toCharArray()) {
            int size = result.size();
            if (size != 0) {
                for (int i = 0; i < size; i++) {
                    for (char innerC : m.get(c)) {
                        result.add(result.get(i) + innerC);
                    }
                }
                for (int i = 0; i < size; i++) {
                    result.remove(0);
                }
            } else {
                for (char innerC : m.get(c)) {
                    result.add(String.valueOf(innerC));
                }
            }
        }
        return result;
    }

    // LC13
    public int romanToInt(String s) {

        // 字符          数值
        //  I             1
        //  V             5
        //  X             10
        //  L             50
        //  C             100
        //  D             500
        //  M             1000

        // I可以放在V(5) 和X(10) 的左边，来表示 4 和 9。
        // X可以放在L(50) 和C(100) 的左边，来表示 40 和90。
        // C可以放在D(500) 和M(1000) 的左边，来表示400 和900。

        Map<Character, Integer> m = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};

        int pre = m.get(s.charAt(0));
        int sum = 0;
        int i = 1;
        while (i < s.length()) {
            int cur = m.get(s.charAt(i++));
            if (pre < cur) {
                sum -= pre;
            } else {
                sum += pre;
            }
            pre = cur;
        }
        sum += pre;
        return sum;
    }

    // LC11
    public int maxArea(int[] height) {
        int result = Integer.MIN_VALUE;
        int left = 0, right = height.length - 1;
        while (left < right) {
            result = Math.max(result, (right - left) * Math.min(height[right], height[left]));
            if (height[left] <= height[right]) {
                left++;
            } else {
                right--;
            }
            // 假设两侧高度x<=y, 宽为t 面积min(x,y)*t = xt
            // 移动y -> y1
            // 面积min(x,y1)*(t-1)
            //  1) if y1<=y, min(x,y1) <= min(x,y) -> 面积必定更小
            //  2) if y1>y , min(x,y1) = x, 又因为t-1<t, 面积必定更小
            // 因此无论如何, 小的一端(x) 已经无法再作为一端边界取得更大的面积, 只能相向移动
        }
        return result;
    }

    // My Eval
    public double myEval(String expression) {
        return evalRPN(toRPN(decodeExpression(expression)));
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
//                while (!(tmp = stack.pop()).equals("(")) {
//                    rpn.add(tmp);
//                }
                while (!stack.isEmpty()) {
                    tmp = stack.pop();
                    if (tmp.equals("(")) {
                        break;
                    } else {
                        rpn.add(tmp);
                    }
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
        express = express.replaceAll(" ", "")
                .replaceAll("\\(\\+", "(0+")
                .replaceAll("\\(-", "(0-")
                .replaceAll("\\((\\d+(\\.\\d+)?)\\)", "$1");
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
    public double evalRPN(List<String> tokens) {
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
                double a = Double.parseDouble(stack.pop());
                double b = Double.parseDouble(stack.pop());
                double tmp;
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
        return stack.isEmpty() ? 0d : Double.parseDouble(stack.pop());
    }

    // LC14
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        StringBuffer sb = new StringBuffer();
        sb.append(strs[0]);
        for (int i = 1; i < strs.length; i++) {
            if (sb.length() == 0) return "";
            if (sb.length() > strs[i].length()) sb.delete(strs[i].length(), sb.length());
            for (int j = 0; j < strs[i].length(); j++) {
                if (j + 1 > sb.length()) break;
                if (strs[i].charAt(j) != sb.charAt(j)) {
                    sb.delete(j, sb.length());
                    break;
                }
            }
        }
        return sb.toString();
    }

    // LC7, 不能使用long, 注意溢出判断
    public int reverse(int x) {
        if (x == 0) return 0;
        boolean negFlag = x < 0;
        if (x < 0) x = -x;
        int result = 0;
        while (x != 0) {
            // 溢出判断
            if (result > Integer.MAX_VALUE / 10) {
                return 0;
            }
            if (result * 10 > Integer.MAX_VALUE - x % 10) {
                return 0;
            }

            result = result * 10 + x % 10;
            x /= 10;
        }
        return negFlag ? -result : result;
    }

    // LC4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int totalLen = nums1.length + nums2.length;
        boolean oddFlag = totalLen % 2 == 1;
        int finalLen = totalLen / 2 + 1;
        List<Integer> l = new ArrayList<>(finalLen);
        int[] longer = nums1.length > nums2.length ? nums1 : nums2;
        int[] shorter = longer == nums1 ? nums2 : nums1;
        int shorterPtr = 0, longerPtr = 0;
        while (l.size() < finalLen) {
            if (shorterPtr != shorter.length && longerPtr != longer.length) {
                if (shorter[shorterPtr] < longer[longerPtr]) {
                    l.add(shorter[shorterPtr++]);
                } else {
                    l.add(longer[longerPtr++]);
                }
            } else if (shorterPtr == shorter.length) {
                l.add(longer[longerPtr++]);
            } else {
                l.add(shorter[shorterPtr++]);
            }
        }
        if (oddFlag) return l.get(l.size() - 1);
        return ((double) l.get(l.size() - 1) + (double) l.get(l.size() - 2)) / 2d;
    }

    // LC83
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                cur = cur.next;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }


    // LC82
    public ListNode deleteDuplicatesLC82(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;
                // 注意短路的始末
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

}