System.err.println(sanitizeFilename("<>||???asfjdsajlfd''''"))
fun sanitizeFilename(filename: String): String {
    // val reg = Regex("[#%&{}\<>?/ $!'":@+`|=]""
    return filename.replace("[#%&{}<>*$@`'+=:/\\\\?|\"]".toRegex(), "-")
}

