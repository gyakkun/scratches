import java.util.*;

class Scratch {
    public static void main(String[] args) {

    }


    // LC843 Minmax **
    public void findSecretWord(String[] wordlist, Master master) {
        int[][] H;
        int n = wordlist.length;
        H = new int[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                int match = 0;
                for (int k = 0; k < 6; ++k) {
                    if (wordlist[i].charAt(k) == wordlist[j].charAt(k))
                        match++;
                }
                H[i][j] = H[j][i] = match;
            }
        }

        Set<Integer> possible = new HashSet();
        Set<Integer> selected = new HashSet<>();
        for (int i = 0; i < n; ++i) {
            possible.add(i);
        }

        while (!possible.isEmpty()) {
            int guess = guess(possible, selected, H);
            int match = master.guess(wordlist[guess]);
            if (match == 6) return;
            Set<Integer> alterPossible = new HashSet<>();
            possible.remove(guess);
            for (int i : possible) {
                if (H[guess][i] == match) {
                    alterPossible.add(i);
                }
            }
            if (match == 0) {
                for (int i : possible) {
                    if (H[guess][i] != 0) {
                        selected.add(i);
                    }
                }
            }
            selected.add(guess);
            possible = alterPossible;
        }
    }

    private int guess(Set<Integer> possible, Set<Integer> selected, int[][] H) {
        int ansGuess = -1;
        Set<Integer> ansGroup = possible;

        for (int guess = 0; guess < H.length; guess++) {
            if (!selected.contains(guess)) {
                Set<Integer>[] groups = new Set[7];
                for (int j = 0; j < 7; j++) {
                    groups[j] = new HashSet<>();
                }

                for (int j : possible) {
                    if (j != guess) {
                        groups[H[guess][j]].add(j);
                    }
                }

                Set<Integer> maxGroup = groups[0];
                for (int i = 1; i < 7; i++) {
                    if (groups[i].size() > maxGroup.size()) { // 最大化
                        maxGroup = groups[i];
                    }
                }

                if (maxGroup.size() < ansGroup.size()) { // 最小化
                    ansGroup = maxGroup;
                    ansGuess = guess;
                }
            }
        }
        return ansGuess;
    }
}

// LC843
interface Master {
    public default int guess(String word) {
        return -1;
    }
}

// LC65 有限状态自动机
class IsNumber {
    static enum CharacterType {
        DIGIT,
        POINT,
        SIGN,
        EXP,
        INVALID
    }

    static enum INState {
        START,
        SIGNING,
        EMPTY_INT_DEC,
        INTEGERING,
        WITH_INT_DEC,
        DECIMALING,
        EXP,
        EXP_SIGN,
        EXPING
    }

    static CharacterType getCharacterType(char c) {
        if (Character.isDigit(c)) return CharacterType.DIGIT;
        if (c == '.') return CharacterType.POINT;
        if (c == 'e' || c == 'E') return CharacterType.EXP;
        if (c == '+' || c == '-') return CharacterType.SIGN;
        return CharacterType.INVALID;
    }

    static Map<INState, Map<CharacterType, INState>> DFA = new HashMap<INState, Map<CharacterType, INState>>() {{
        put(INState.START, new HashMap<CharacterType, INState>() {{
            put(CharacterType.SIGN, INState.SIGNING);
            put(CharacterType.POINT, INState.EMPTY_INT_DEC);
            put(CharacterType.DIGIT, INState.INTEGERING);
        }});
        put(INState.SIGNING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.POINT, INState.EMPTY_INT_DEC);
            put(CharacterType.DIGIT, INState.INTEGERING);
        }});
        put(INState.EMPTY_INT_DEC, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
        }});
        put(INState.INTEGERING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.INTEGERING);
            put(CharacterType.POINT, INState.WITH_INT_DEC);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.WITH_INT_DEC, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.DECIMALING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.DECIMALING);
            put(CharacterType.EXP, INState.EXP);
        }});
        put(INState.EXP, new HashMap<CharacterType, INState>() {{
            put(CharacterType.SIGN, INState.EXP_SIGN);
            put(CharacterType.DIGIT, INState.EXPING);
        }});
        put(INState.EXP_SIGN, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.EXPING);
        }});
        put(INState.EXPING, new HashMap<CharacterType, INState>() {{
            put(CharacterType.DIGIT, INState.EXPING);
        }});
    }};

    public static boolean isNumber(String s) {
        char[] cArr = s.toCharArray();
        INState state = INState.START;
        for (int i = 0; i < cArr.length; i++) {
            CharacterType t = getCharacterType(cArr[i]);
            if (!DFA.get(state).containsKey(t)) {
                return false;
            } else {
                state = DFA.get(state).get(t);
            }
        }
        return state == INState.INTEGERING || state == INState.WITH_INT_DEC || state == INState.DECIMALING || state == INState.EXPING;
    }
}