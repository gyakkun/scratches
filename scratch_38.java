import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class Scratch {
    public static void main(String[] args) {
        Scratch s = new Scratch();
        long timing = System.currentTimeMillis();

//        System.err.println(s.findWords(new char[][]{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}}, new String[]{"oath", "pea", "eat", "rain"}));
        System.err.println(s.findWordsMP(new char[][]{{'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'}}
                , new String[]{"lllllll", "fffffff", "ssss", "s", "rr", "xxxx", "ttt", "eee", "ppppppp", "iiiiiiiii", "xxxxxxxxxx", "pppppp", "xxxxxx", "yy", "jj", "ccc", "zzz", "ffffffff", "r", "mmmmmmmmm", "tttttttt", "mm", "ttttt", "qqqqqqqqqq", "z", "aaaaaaaa", "nnnnnnnnn", "v", "g", "ddddddd", "eeeeeeeee", "aaaaaaa", "ee", "n", "kkkkkkkkk", "ff", "qq", "vvvvv", "kkkk", "e", "nnn", "ooo", "kkkkk", "o", "ooooooo", "jjj", "lll", "ssssssss", "mmmm", "qqqqq", "gggggg", "rrrrrrrrrr", "iiii", "bbbbbbbbb", "aaaaaa", "hhhh", "qqq", "zzzzzzzzz", "xxxxxxxxx", "ww", "iiiiiii", "pp", "vvvvvvvvvv", "eeeee", "nnnnnnn", "nnnnnn", "nn", "nnnnnnnn", "wwwwwwww", "vvvvvvvv", "fffffffff", "aaa", "p", "ddd", "ppppppppp", "fffff", "aaaaaaaaa", "oooooooo", "jjjj", "xxx", "zz", "hhhhh", "uuuuu", "f", "ddddddddd", "zzzzzz", "cccccc", "kkkkkk", "bbbbbbbb", "hhhhhhhhhh", "uuuuuuu", "cccccccccc", "jjjjj", "gg", "ppp", "ccccccccc", "rrrrrr", "c", "cccccccc", "yyyyy", "uuuu", "jjjjjjjj", "bb", "hhh", "l", "u", "yyyyyy", "vvv", "mmm", "ffffff", "eeeeeee", "qqqqqqq", "zzzzzzzzzz", "ggg", "zzzzzzz", "dddddddddd", "jjjjjjj", "bbbbb", "ttttttt", "dddddddd", "wwwwwww", "vvvvvv", "iii", "ttttttttt", "ggggggg", "xx", "oooooo", "cc", "rrrr", "qqqq", "sssssss", "oooo", "lllllllll", "ii", "tttttttttt", "uuuuuu", "kkkkkkkk", "wwwwwwwwww", "pppppppppp", "uuuuuuuu", "yyyyyyy", "cccc", "ggggg", "ddddd", "llllllllll", "tttt", "pppppppp", "rrrrrrr", "nnnn", "x", "yyy", "iiiiiiiiii", "iiiiii", "llll", "nnnnnnnnnn", "aaaaaaaaaa", "eeeeeeeeee", "m", "uuu", "rrrrrrrr", "h", "b", "vvvvvvv", "ll", "vv", "mmmmmmm", "zzzzz", "uu", "ccccccc", "xxxxxxx", "ss", "eeeeeeee", "llllllll", "eeee", "y", "ppppp", "qqqqqq", "mmmmmm", "gggg", "yyyyyyyyy", "jjjjjj", "rrrrr", "a", "bbbb", "ssssss", "sss", "ooooo", "ffffffffff", "kkk", "xxxxxxxx", "wwwwwwwww", "w", "iiiiiiii", "ffff", "dddddd", "bbbbbb", "uuuuuuuuu", "kkkkkkk", "gggggggggg", "qqqqqqqq", "vvvvvvvvv", "bbbbbbbbbb", "nnnnn", "tt", "wwww", "iiiii", "hhhhhhh", "zzzzzzzz", "ssssssssss", "j", "fff", "bbbbbbb", "aaaa", "mmmmmmmmmm", "jjjjjjjjjj", "sssss", "yyyyyyyy", "hh", "q", "rrrrrrrrr", "mmmmmmmm", "wwwww", "www", "rrr", "lllll", "uuuuuuuuuu", "oo", "jjjjjjjjj", "dddd", "pppp", "hhhhhhhhh", "kk", "gggggggg", "xxxxx", "vvvv", "d", "qqqqqqqqq", "dd", "ggggggggg", "t", "yyyy", "bbb", "yyyyyyyyyy", "tttttt", "ccccc", "aa", "eeeeee", "llllll", "kkkkkkkkkk", "sssssssss", "i", "hhhhhh", "oooooooooo", "wwwwww", "ooooooooo", "zzzz", "k", "hhhhhhhh", "aaaaa", "mmmmm"}));

        timing = System.currentTimeMillis() - timing;
        System.err.println("TIMING: " + timing + "ms.");
    }

    // LC212 Hard, 多线程版本效果不好
    ConcurrentMap<String, Integer> lc212Result;
    HashSet<String> lc212ResultStr;
    int boardRow;
    int boardCol;

    public List<String> findWords(char[][] board, String[] words) {
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        lc212ResultStr = new HashSet<>();
        boardRow = board.length;
        boardCol = board[0].length;
        ForkJoinPool fjp = new ForkJoinPool();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                boolean[][] visited = new boolean[boardRow][boardCol];
                visited[i][j] = true;
                lc79Backtrack(board, trie, i, j, "" + board[i][j], visited, -1);
            }
        }

        return new LinkedList<>(lc212ResultStr);
    }

    public List<String> findWordsMP(char[][] board, String[] words) {
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        lc212Result = new ConcurrentHashMap<>();
        boardRow = board.length;
        boardCol = board[0].length;
        List<lc79> joblist = new LinkedList<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                lc79 job = new lc79(board, trie, i, j, "" + board[i][j], new boolean[boardRow][boardCol], -1);
                joblist.add(job);
                job.fork();
            }
        }
        for (lc79 job : joblist) {
            job.join();
        }

        return new LinkedList<>(lc212Result.keySet());
    }

    public class lc79 extends RecursiveAction {

        char[][] board;
        Trie trie;
        int curRow;
        int curCol;
        String curWord;
        boolean[][] visited;
        int direct;
        ReentrantReadWriteLock rwl;
        Lock rLVisited;
        Lock wLVisited;

        public lc79(char[][] board, Trie trie, int curRow, int curCol, String curWord, boolean[][] visited, int direct) {
            this.board = board;
            this.trie = trie;
            this.curRow = curRow;
            this.curCol = curCol;
            this.curWord = curWord;
            this.visited = visited;
            this.direct = direct;
        }

        public lc79(char[][] board, Trie trie, int curRow, int curCol, String curWord, boolean[][] visited, int direct, Lock rLVisited, Lock wLVisited) {
            this.board = board;
            this.trie = trie;
            this.curRow = curRow;
            this.curCol = curCol;
            this.curWord = curWord;
            this.visited = visited;
            this.direct = direct;
            this.rLVisited = rLVisited;
            this.wLVisited = wLVisited;
        }

        @Override
        protected void compute() {
            if (direct == -1) {
                this.rwl = new ReentrantReadWriteLock();
                this.rLVisited = rwl.readLock();
                this.wLVisited = rwl.writeLock();
                try {
                    wLVisited.lock();
                    visited[curRow][curCol] = true;
                } finally {
                    wLVisited.unlock();
                }
            }
            if (!trie.startsWith(curWord)) {
                return;
            }
            if (trie.search(curWord)) {
                lc212Result.putIfAbsent(curWord, 1);
            }
            List<lc79> joblist = new ArrayList<>(4);

            int[][] options = new int[][]{{curRow - 1, curCol, 1, 0}, {curRow + 1, curCol, 0, 1}, {curRow, curCol - 1, 3, 2}, {curRow, curCol + 1, 2, 3}};
            for (int[] option : options) {
                if (checkLegalPosition(option[0], option[1]) && direct != option[2]) {
                    try {
                        wLVisited.lock();
                        visited[option[0]][option[1]] = true;
                    } finally {
                        wLVisited.unlock();
                    }

                    lc79 job = new lc79(board, trie, option[0], option[1], curWord + board[option[0]][option[1]], visited, option[3], this.rLVisited, this.wLVisited);
                    joblist.add(job);
                    job.fork();

                    try {
                        wLVisited.lock();
                        visited[option[0]][option[1]] = false;
                    } finally {
                        wLVisited.unlock();
                    }
                }
            }
            for (lc79 job : joblist) {
                job.join();
            }

            return;
        }

        private boolean checkLegalPosition(int row, int col) {
            boolean flag1 = (row >= 0 && row < boardRow && col >= 0 && col < boardCol);
            if (!flag1) return false;
            boolean flag2 = false;
            try {
                rLVisited.lock();
                flag2 = visited[row][col];
            } finally {
                rLVisited.unlock();
            }
            return flag1 && !flag2;
        }
    }

    private void lc79Backtrack(char[][] board, Trie trie, int curRow, int curCol,
                               String curWord, boolean[][] visited, int direct) { // 0123 - 上下左右
        if (!trie.startsWith(curWord)) {
            return;
        }
        if (trie.search(curWord)) {
            lc212ResultStr.add(curWord);
        }
        int[][] options = new int[][]{{curRow - 1, curCol, 1, 0}, {curRow + 1, curCol, 0, 1}, {curRow, curCol - 1, 3, 2}, {curRow, curCol + 1, 2, 3}};
        for (int[] option : options) {
            if (checkLegalPosition(option[0], option[1], visited) && direct != option[2]) {
                visited[option[0]][option[1]] = true;

                lc79Backtrack(board, trie, option[0], option[1], curWord + board[option[0]][option[1]], visited, option[3]);

                visited[option[0]][option[1]] = false;
            }
        }
    }

    private boolean checkLegalPosition(int row, int col, boolean[][] visited) {
        boolean flag1 = (row >= 0 && row < boardRow && col >= 0 && col < boardCol);
        if (!flag1) return false;
        boolean flag2 = visited[row][col];
        return flag1 && !flag2;
    }

}

class Trie {
    TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {

            if (former.val == '#') former.val = word.charAt(i);

            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) {
                former.sibling = new TrieNode(word.charAt(i));
                former = former.sibling;
            }
            if (former.child == null) former.child = new TrieNode();
            former = former.child;
        }

        if (former.val == '#') former.val = word.charAt(i);

        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) {
            former.sibling = new TrieNode(word.charAt(i));
            former = former.sibling;
        }
        if (former.child == null) former.child = new TrieNode();
        former.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode former = root;
        int i;
        for (i = 0; i < word.length() - 1; i++) {
            former = former.searchSibling(word.charAt(i));
            if (former.val != word.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(word.charAt(i));
        if (former.val != word.charAt(i)) return false;
        return former.isEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode former = root;
        int i;
        for (i = 0; i < prefix.length() - 1; i++) {
            former = former.searchSibling(prefix.charAt(i));
            if (former.val != prefix.charAt(i)) return false;
            former = former.child;
        }
        former = former.searchSibling(prefix.charAt(i));
        if (former.val != prefix.charAt(i)) return false;
        return true;
    }
}

class TrieNode {
    Character val;
    Boolean isEnd;
    TrieNode child;
    TrieNode sibling;

    public TrieNode() {
        this.val = '#';
        this.isEnd = false;
    }

    public TrieNode(Character c) {
        this.val = c;
        this.isEnd = false;
    }

    public TrieNode searchSibling(Character c) {
        TrieNode former = this;
        while (former.sibling != null) {
            if (former.val == c) return former;
            former = former.sibling;
        }
        return former;
    }

    public TrieNode searchChildren(Character c) {
        TrieNode former = this;
        while (former.child != null) {
            if (former.val == c) return former;
            former = former.child;
        }
        return former;
    }
}