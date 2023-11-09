import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Random;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        System.err.println(s.canChoose(new int[][]{{1363850, 6300176, 8430440, -9635380, -1343994, -9365453, 5210548, -1702094, 8165619, 4988596, -1524607, -4244825, -7838619, -1604017, 8054294, 3277839}, {-9180987, 4743777, 9146280, -7908834, 1909925, 4434157}, {3981590}},
                new int[]{-1702094, -9635380, 5210548, 8165619, 8054294, 1363850, 6300176, 8430440, -9635380, -1343994, -9365453, 5210548, -1702094, 8165619, 4988596, -1524607, -4244825, -7838619, -1604017, 8054294, 3277839, -1343994, -1524607, 1363850, 6300176, 8165619, -9180987, 4743777, 9146280, -7908834}));
    }


    private static void testExtremeRandom() {
        var spotCheckBs = new BitSet(390016);
        spotCheckBs.set(0, spotCheckBs.size());
        var r = new Random();
        var a = new HashSet<Integer>();
        var b = new HashSet<Integer>();
        for (int i = 0; i < Long.SIZE; i++) {
            int s;
            do {
                s = r.nextInt(0, spotCheckBs.size());
            } while (!spotCheckBs.get(s));
            System.err.println("S: " + s);
            a.add(s);
            spotCheckBs.clear(s);
        }
        System.err.println("Remain zeros: " + (spotCheckBs.size() - spotCheckBs.cardinality()));
        for (int i = 0; i < Long.SIZE; i++) {
            int victim = 0;
            // do {
            //     victim = bs.nextClearBit(r.nextInt(0, bs.size()));
            // } while (victim >= bs.size());
            while ((victim = (spotCheckBs.nextClearBit(victim))) < spotCheckBs.size()) {
                spotCheckBs.set(victim);
                b.add(victim);
                System.err.println("V: " + victim);
            }
        }
        System.err.println(a.equals(b));
        if (!a.equals(b)) {
            throw new RuntimeException("a b should equal");
        }
    }

    public boolean canChoose(int[][] groups, int[] nums) {
        byte[] byteNums = helper(nums);
        int offset = 0;
        for (int[] p : groups) {
            byte[] pattern = helper(p);
            Sunday sunday = new Sunday(pattern, byteNums, offset);
            int nextOffset;
            while ((nextOffset = sunday.nextOffset()) > 0) {
                if (nextOffset % 4 != 0) continue;
                offset = sunday.offset + pattern.length - 1;
                break;
            }
            if (nextOffset < 0) return false;
        }
        return true;
    }

    private byte[] helper(int[] intArr) {
        int size = intArr.length;
        int byteArrSize = size * 4;
        byte[] result = new byte[byteArrSize];
        for (int i = 0; i < intArr.length; i++) {
            for (int j = 0; j < 4; j++) {
                result[i * 4 + 3 - j] = (byte) ((intArr[i] >> (j * 8)) & 0xFF);
            }
        }
        return result;
    }
}

class Sunday {

    int[] toSkip = new int[1 << Byte.SIZE];
    final byte[] pattern;
    final byte[] toMatch;
    int offset = 0;

    Sunday(byte[] pattern, byte[] toMatch) {
        this.pattern = pattern;
        this.toMatch = toMatch;
        init();
    }

    Sunday(byte[] pattern, byte[] toMatch, int initOffset) {
        this.pattern = pattern;
        this.toMatch = toMatch;
        this.offset = initOffset;
        init();
    }

    private void init() {
        Arrays.fill(toSkip, pattern.length);
        for (int i = 0; i < pattern.length; i++) {
            toSkip[0xFF & pattern[i]] = pattern.length - i;
        }
    }

    int nextOffset() {
        int cur = offset;
        outer:
        while (cur < toMatch.length) {
            for (int i = 0; i < pattern.length; i++) {
                if (cur + i >= toMatch.length) return -1;
                if (toMatch[cur + i] != pattern[i]) {
                    if (cur + pattern.length >= toMatch.length) {
                        offset = toMatch.length;
                        return -1;
                    }
                    cur += toSkip[toMatch[cur + pattern.length] & 0xFF];
                    continue outer;
                }
            }
            int result = cur;
            offset = cur + 1;
            return result;
        }
        return -1;
    }

}