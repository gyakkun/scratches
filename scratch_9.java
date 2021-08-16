import java.util.Arrays;

class Scratch {
    public static void main(String[] args) {
        Scratch i = new Scratch();
        System.err.println(i.countArrangement(4));
    }

    // LC455 2020.12.25
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int result = 0;
        int idxS = 0;
        int nextIdxG = 0;
        for (; idxS < s.length; idxS++) {
            for (int j = nextIdxG; j < g.length; j++) {
                if (s[idxS] >= g[j]) {
                    result++;
                    nextIdxG = j + 1;
                    s[idxS] = -1;
                    break;
                }
            }
        }
        return result;
    }

    // LC526
    private int result = 0;

    public int countArrangement(int N) {
        int[] l = new int[N];
        for (int i = 0; i < N; i++) {
            l[i] = i + 1;
        }
        recursive(l, 0);
        return result;
    }

    public void recursive(int[] l, int index) {
        if (index == l.length) {
            if (judge(l, l.length)) {
                result++;
            }
            return;
        }
        for (int i = index; i < l.length; i++) {
            int tmp = l[i];
            l[i] = l[index];
            l[index] = tmp;
            if (judge(l, index)) {
                recursive(l, index + 1);
            }
            tmp = l[index];
            l[index] = l[i];
            l[i] = tmp;
        }
    }


    private boolean judge(int[] l, int idx) {
        for (int i = 0; i < idx; i++) {
            if (l[i] % (i + 1) != 0 && (i + 1) % l[i] != 0) {
                return false;
            }
        }
        return true;
    }
}