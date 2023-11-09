package moe.nyamori.test.historical;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

class buffer2 {
    boolean[] mem;

    public boolean backtrack(String s, int start, Set<String> set) {
        if (s.length() == 0) return true;
        if (mem[start]) return false;
        for (int i = 0; i < s.length(); i++) {
            if (set.contains(s.substring(0, i + 1))) {
                if (backtrack(s.substring(i + 1), start + i + 1, set)) return true;
            }
        }
        mem[start] = true;
        return false;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        this.mem = new boolean[s.length()];
        Set<String> set = new HashSet<>(wordDict);
        return backtrack(s, 0, set);
    }

}