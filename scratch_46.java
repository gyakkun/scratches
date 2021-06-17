import java.util.HashMap;
import java.util.Map;

class Scratch {
    public static void main(String[] args) {

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