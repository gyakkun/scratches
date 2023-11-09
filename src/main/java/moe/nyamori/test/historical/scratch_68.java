package moe.nyamori.test.historical;

import java.io.*;
import java.util.Random;

class scratch_68 {
    // For Prime OJ 1003 Test
    public static void main(String[] args) throws IOException {
        File f = new File("C:\\Users\\Steve\\prime_oj_1003_test_case.txt");
        char nextLine = '\n';
        Random r = new Random();
        FileWriter fw = new FileWriter(f);
        int t = 300, max = 123456;
        fw.write("" + t);
        fw.write(nextLine);
        for (int i = 0; i < t; i++) {
            fw.write("" + max + " " + max);
            fw.write(nextLine);
            fw.write("" + 0 + " " + r.nextInt(max));
            fw.write(nextLine);
            for (int j = 1; j < max; j++) {
                fw.write("" + r.nextInt(max) + " " + r.nextInt(max));
                fw.write(nextLine);
            }
        }
        fw.close();
    }
}