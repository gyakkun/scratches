import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Scratch {
    public static void main(String[] args) {
        Scratch i = new Scratch();
        System.err.println(i.countArrangement(4));
    }


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
//            if(s[idxS]==)
        }
        return result;
    }

    private List<int[]> result = new ArrayList<>();
    private int len;

    public int countArrangement(int N) {
        len = N;
        int[] l = new int[len];
        for (int i = 0; i < len; i++) {
            l[i] = i + 1;
        }
        recursive(l, 0);
        return result.size();
    }

    public void recursive(int[] l, int index) {
        if (judge(l, len)) {
            int[] ans = Arrays.copyOf(l, l.length);
            result.add(ans);
        }


        for (int i = index + 1; index < len; index++) {
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
//        if (l.size() != len) return false;
        for (int i = 0; i < idx; i++) {
            if (l[i] % (i + 1) != 0 && (i + 1) % l[i] != 0) {
                return false;
            }
        }
        return true;
    }
}