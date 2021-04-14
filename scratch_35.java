import java.util.*;

class Scratch {
    public static void main(String[] args) {
        Trie t = new Trie();
        t.insert("ben");
        System.err.println(t.search("ben"));
        t.insert("beneficiation");
        System.err.println(t.search("beneficiation"));
        System.err.println(t.search("ben"));

        return;
    }
}

// LC208 二叉树实现, 最后10000个操作的大算例 有两个会不一致??? "no", "ben"
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
        for (i = 0; i < word.length(); i++) {

            if (former.val == '#') former.val = word.charAt(i);

            TrieNode possibleSibling = former.searchSibling(word.charAt(i));
            if (possibleSibling.val != word.charAt(i)) {
                possibleSibling.sibling = new TrieNode(word.charAt(i));
                possibleSibling = possibleSibling.sibling;
                former = possibleSibling;
            }
            if (possibleSibling.child == null) possibleSibling.child = new TrieNode();
            if (i != word.length() - 1) former = possibleSibling.child;
        }
        former.isEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode former = root;
        for (int i = 0; i < word.length(); i++) {
            TrieNode possibleSibling = former.searchSibling(word.charAt(i));
            if (possibleSibling.val == '#' || possibleSibling.val != word.charAt(i)) return false;
            if (i == word.length() - 1) return possibleSibling.isEnd;
            former = possibleSibling.child;
        }
        return false;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode former = root;
        for (int i = 0; i < prefix.length(); i++) {
            TrieNode possibleSibling = former.searchSibling(prefix.charAt(i));
            if (possibleSibling.val == '#' || possibleSibling.val != prefix.charAt(i)) return false;
            if (i == prefix.length() - 1) return true;
            former = possibleSibling.child;
        }
        return false;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */

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

class TrieHM {
    Map<String, Boolean> m;

    /**
     * Initialize your data structure here.
     */
    public TrieHM() {
        m = new HashMap<>();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        for (int i = 0; i < word.length(); i++) {
            m.putIfAbsent(word.substring(0, i + 1), false);
        }
        m.put(word, true);
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        return m.getOrDefault(word, false);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        return m.containsKey(prefix);
    }
}