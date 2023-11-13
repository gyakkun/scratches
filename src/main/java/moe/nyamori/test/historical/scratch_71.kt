package moe.nyamori.test.historical

fun main() {
    fun sanitizeFilename(filename: String): String {
        // val reg = Regex("[#%&{}\<>?/ $!'":@+`|=]""
        return filename.replace("[#%&{}<>*$@`'+=:/\\\\?|\"]".toRegex(), "-")
    }
    System.err.println(sanitizeFilename("<>||???asfjdsajlfd''''"))
}