fun main() {
    fun sanitizeFilename(filename: String): String {
        // val reg = Regex("[#%&{}\<>?/ $!'":@+`|=]""
        return filename.replace("[#%&{}<>*$@`'+=:/\\\\?|\"]".toRegex(), "-")
    }
    System.err.println(sanitizeFilename("<>||???asfjdsajlfd''''"))
}